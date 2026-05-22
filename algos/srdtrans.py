"""
SRDTrans — Spatial Redundancy Denoising Transformer
====================================================

Implements the architecture from Li et al., "Spatial redundancy transformer
for self-supervised fluorescence image denoising" (Nature Computational
Science, 2023, https://doi.org/10.1038/s43588-023-00568-2).
Official repository: https://github.com/cabooster/SRDTrans

What the paper describes (Methods + Supplementary Figs 21–22):
    1. Temporal encoder: compresses time T → T/r² without reducing
       spatial resolution. Built from temporal-only 1D conv blocks.
    2. Spatiotemporal Transformer Block (STB): operates on tubes of
       p × p × (T/r²) patches; tubes are flattened to tokens and run
       through (a) a temporal transformer block then (b) a spatial
       transformer block, both standard multi-head self-attention with
       learned position embeddings.
    3. Temporal decoder: upsamples T/r² back to T.
    4. Skip connections from encoder to decoder layers.

This is a paper-faithful INTERPRETATION of the architecture sketch in
the paper. The official repo's exact channel counts and layer ordering
may differ — for byte-equivalent behaviour use the official code.

What we use vs the paper
------------------------
  * Paper trains with SPATIAL REDUNDANCY SAMPLING (orthogonal masks
    producing input/target/target2 sub-stacks of H/2 × W/2 × T).
  * Our framework uses standard self-supervised Noise2Void blind-spot
    loss for ALL algos for consistency.
  * The ARCHITECTURE matches the paper's structural sketch.
  * The TRAINING is our standard self-supervised N2V flow.

If you want to reproduce the paper's published numbers exactly, that
requires the official repo's sampling pipeline.

API parity with the rest of our algos.
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runner import preprocessing as _prep


DEFAULT_NORMALIZATION = "p0.5_p99.5"
DEFAULT_TEMPORAL_TARGET = "temporal_median_2d"


def _default_norm():
    return _prep.resolve_normalization(DEFAULT_NORMALIZATION)


def compute_norm_params(stack: np.ndarray) -> dict:
    return _default_norm().compute_params(stack)


def normalize(data, params):
    return _default_norm().forward(data, params)


def denormalize(data, params):
    return _default_norm().inverse(data, params)


# ══════════════════════════════════════════════════════════════
# TEMPORAL ENCODER / DECODER (paper: time-axis compression only)
# ══════════════════════════════════════════════════════════════

class TemporalDownBlock(nn.Module):
    """Compresses time by factor r via stride-r conv on the time axis.

    Spatial dims stay the same; only the temporal dim shrinks. Implemented
    as Conv3d with stride (r, 1, 1).
    """

    def __init__(self, in_ch, out_ch, r=2):
        super().__init__()
        # Kernel = (r, 3, 3) so we use the local time + spatial context
        # while compressing time by r.
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=(r, 3, 3),
                                stride=(r, 1, 1), padding=(0, 1, 1),
                                bias=True)
        self.norm = nn.GroupNorm(min(4, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


# ══════════════════════════════════════════════════════════════
# SPATIOTEMPORAL TRANSFORMER BLOCK (STB)
# ══════════════════════════════════════════════════════════════

class MHSA(nn.Module):
    """Plain pre-norm multi-head self-attention with learnable PE."""

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                            batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class SpatiotemporalTransformerBlock(nn.Module):
    """
    STB: takes [B, C, T', H, W] feature maps and processes them through
    a temporal transformer (across T') followed by a spatial transformer
    (across H*W). Uses learned position embeddings on each.

    Implementation note: we do attention along the time axis (treating
    each (h, w) as a separate sequence of length T') and along the
    spatial axis (treating each t as a separate sequence of length H*W).
    Both are full attention; for large H*W the spatial branch dominates
    memory, so keep the input spatial dim small (this is why we apply
    STB AFTER temporal compression but BEFORE any spatial pooling).
    """

    def __init__(self, dim, num_heads=4, mlp_ratio=2.0,
                 max_t=64, max_hw=64*64):
        super().__init__()
        self.dim = dim
        self.temp_attn = MHSA(dim, num_heads, mlp_ratio)
        self.spat_attn = MHSA(dim, num_heads, mlp_ratio)
        # Learnable PE for time and space — sized for upper bounds.
        # If actual dims exceed these we interpolate to fit.
        self.t_pe = nn.Parameter(torch.zeros(1, max_t, dim))
        self.s_pe = nn.Parameter(torch.zeros(1, max_hw, dim))
        nn.init.trunc_normal_(self.t_pe, std=0.02)
        nn.init.trunc_normal_(self.s_pe, std=0.02)

    def _resize_pe(self, pe, target_len):
        """Linearly interpolate the position embedding to `target_len`."""
        if pe.shape[1] == target_len:
            return pe
        # pe: [1, L, C] → [1, C, L] for 1D interp → back
        x = pe.transpose(1, 2)
        x = F.interpolate(x, size=target_len, mode="linear",
                          align_corners=False)
        return x.transpose(1, 2)

    def forward(self, x):
        """x: [B, C, T', H, W]"""
        B, C, T, H, W = x.shape

        # ── Temporal attention ────────────────────────────
        # Tokens: each (h, w) is a separate sequence of length T.
        # Rearrange to [B*H*W, T, C]
        xt = x.permute(0, 3, 4, 2, 1).contiguous().view(B * H * W, T, C)
        t_pe = self._resize_pe(self.t_pe, T)
        xt = xt + t_pe
        xt = self.temp_attn(xt)
        # Back to [B, C, T, H, W]
        x = xt.view(B, H, W, T, C).permute(0, 4, 3, 1, 2).contiguous()

        # ── Spatial attention ────────────────────────────
        # Tokens: each t is a separate sequence of length H*W.
        # [B*T, H*W, C]
        N_sp = H * W
        xs = x.permute(0, 2, 3, 4, 1).contiguous().view(B * T, N_sp, C)
        s_pe = self._resize_pe(self.s_pe, N_sp)
        xs = xs + s_pe
        xs = self.spat_attn(xs)
        # Back to [B, C, T, H, W]
        x = xs.view(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return x


# ══════════════════════════════════════════════════════════════
# SRDTrans — full network
# ══════════════════════════════════════════════════════════════

class SRDTransNet(nn.Module):
    """
    Lightweight SRDTrans architecture:

        intro Conv3D(1 → embed_dim)
        TemporalDown(× n_time_levels)                  — compresses T
        STB × n_stb_blocks                             — global attention
        TemporalUp(× n_time_levels, with skips)         — restores T
        final Conv3D(embed_dim → 1)                    — predicts residual
    """

    def __init__(
        self,
        embed_dim: int = 32,
        n_time_levels: int = 2,        # compress T by 2^n_time_levels
        n_stb_blocks: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        time_compress_r: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_time_levels = n_time_levels
        self.r = time_compress_r
        self.divisor_t = self.r ** n_time_levels   # input T must be a multiple

        # Stem
        self.intro = nn.Conv3d(1, embed_dim, kernel_size=3, padding=1,
                                bias=True)

        # Temporal encoder
        self.t_down = nn.ModuleList([
            TemporalDownBlock(embed_dim, embed_dim, r=self.r)
            for _ in range(n_time_levels)
        ])

        # STB stack (operates at the compressed temporal scale)
        self.stbs = nn.ModuleList([
            SpatiotemporalTransformerBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
            )
            for _ in range(n_stb_blocks)
        ])

        # Temporal decoder: each level upsamples T by r, then fuses with
        # the skip at that level. The fuse Conv3d takes 2*embed_dim
        # (feat upsampled + skip) and produces embed_dim.
        self.t_up_expand = nn.ModuleList([
            nn.ConvTranspose3d(
                embed_dim, embed_dim, kernel_size=(self.r, 3, 3),
                stride=(self.r, 1, 1), padding=(0, 1, 1), bias=True,
            )
            for _ in range(n_time_levels)
        ])
        self.t_up_fuse = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=3,
                            padding=1, bias=True),
                nn.GroupNorm(min(4, embed_dim), embed_dim),
                nn.GELU(),
            )
            for _ in range(n_time_levels)
        ])

        # Final 3D conv to produce noise prediction
        self.out_conv = nn.Conv3d(embed_dim, 1, kernel_size=3, padding=1,
                                    bias=True)

    def forward(self, x):
        """x: [B, 1, T, H, W]"""
        identity = x
        _, _, T, H, W = x.shape

        # Pad time dim to a multiple of self.divisor_t
        pad_t = (self.divisor_t - T % self.divisor_t) % self.divisor_t
        if pad_t:
            mode = "reflect" if pad_t < T else "replicate"
            x = F.pad(x, (0, 0, 0, 0, 0, pad_t), mode=mode)
            identity_p = F.pad(identity, (0, 0, 0, 0, 0, pad_t), mode=mode)
        else:
            identity_p = identity

        # Stem
        feat = self.intro(x)            # [B, embed_dim, T_pad, H, W]

        # Encoder with skips
        skips = []
        for down in self.t_down:
            skips.append(feat)
            feat = down(feat)

        # STB stack at the compressed temporal scale
        for stb in self.stbs:
            feat = stb(feat)

        # Decoder with skips (reverse order)
        for up_expand, fuse, skip in zip(self.t_up_expand,
                                            self.t_up_fuse,
                                            reversed(skips)):
            # Up: ConvTranspose3d on the time dim only
            feat = up_expand(feat)
            # If the up restored time mismatches skip exactly, resize
            if feat.shape[2:] != skip.shape[2:]:
                feat = F.interpolate(feat, size=skip.shape[2:],
                                        mode="trilinear",
                                        align_corners=False)
            # Fuse with skip (concat channels → conv → embed_dim)
            feat = fuse(torch.cat([feat, skip], dim=1))

        # Output prediction (residual)
        noise_pred = self.out_conv(feat)
        out = identity_p - noise_pred

        # Strip pad
        if pad_t:
            out = out[:, :, :T, :, :]
        return out


# ══════════════════════════════════════════════════════════════
# 3D BLIND-SPOT + AUG + WINDOW
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume: torch.Tensor, mask_ratio: float = 0.015,
                radius: int = 2):
    D, H, W = volume.shape
    n_vox = D * H * W
    n_mask = max(int(n_vox * mask_ratio), 1)
    flat_idx = torch.randperm(n_vox, device=volume.device)[:n_mask]
    mz = flat_idx // (H * W)
    my = (flat_idx % (H * W)) // W
    mx = flat_idx % W
    original = volume[mz, my, mx].clone()
    dz = torch.randint(-radius, radius + 1, (n_mask,), device=volume.device)
    dy = torch.randint(-radius, radius + 1, (n_mask,), device=volume.device)
    dx = torch.randint(-radius, radius + 1, (n_mask,), device=volume.device)
    same = (dz == 0) & (dy == 0) & (dx == 0)
    dz[same] = 1
    nz = (mz + dz).clamp(0, D - 1)
    ny = (my + dy).clamp(0, H - 1)
    nx = (mx + dx).clamp(0, W - 1)
    masked = volume.clone()
    masked[mz, my, mx] = volume[nz, ny, nx]
    return masked, (mz, my, mx), original


def _augment_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


def _gaussian_window_3d(shape, sigma_frac=0.25, device="cpu"):
    windows = []
    for s in shape:
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = max(s * sigma_frac, 1.0)
        windows.append(torch.exp(-0.5 * ((coords - center) / sigma) ** 2))
    return (windows[0][:, None, None]
            * windows[1][None, :, None]
            * windows[2][None, None, :]).clamp(min=1e-6)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_self_supervised(stack, device, config=None, verbose=True):
    t0 = time.time()
    cfg = {
        "embed_dim":      32,
        "n_time_levels":  2,
        "n_stb_blocks":   2,
        "num_heads":      4,
        "mlp_ratio":      2.0,
        "time_compress_r": 2,
        "patch_d":        16,
        "patch_hw":       48,
        "batch_size":     1,
        "warmup_iters":   200,
        "n2v_iters":      2500,
        "lr":             2e-4,
        "mask_ratio":     0.015,
        "mask_radius":    2,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" SRDTrans: embed_dim={cfg['embed_dim']}, "
              f"n_time_levels={cfg['n_time_levels']}, "
              f"n_stb_blocks={cfg['n_stb_blocks']}, "
              f"heads={cfg['num_heads']}")
        print(f" Patch: {pd}x{phw}x{phw}, batch={bs}")
        print(f" Schedule: warmup={cfg['warmup_iters']}, "
              f"n2v={cfg['n2v_iters']}")
        print(f" Precision: fp32")

    norm_name = cfg.get("normalization", DEFAULT_NORMALIZATION)
    norm_strategy = _prep.resolve_normalization(norm_name)
    norm_params = norm_strategy.compute_params(stack)
    cfg["norm_params"] = norm_params
    cfg["__resolved_normalization"] = norm_strategy.name
    stack_norm = norm_strategy.forward(stack, norm_params)

    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    cfg["__resolved_temporal_target"] = tt_strategy.name
    if tt_strategy.returns != "2d":
        _tt = tt_strategy.compute(stack_norm)
        temporal_med = np.median(_tt, axis=0).astype(np.float32)
    else:
        temporal_med = tt_strategy.compute(stack_norm)

    stack_t = torch.from_numpy(stack_norm).float().to(device)
    tmed_t = torch.from_numpy(temporal_med).float().to(device)

    model = SRDTransNet(
        embed_dim=cfg["embed_dim"],
        n_time_levels=cfg["n_time_levels"],
        n_stb_blocks=cfg["n_stb_blocks"],
        num_heads=cfg["num_heads"],
        mlp_ratio=cfg["mlp_ratio"],
        time_compress_r=cfg["time_compress_r"],
    ).to(device)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f" Model params: {n_params:,}")

    def random_patch():
        t0_ = np.random.randint(0, max(F_total - pd, 1))
        y0 = np.random.randint(0, max(H - phw, 1))
        x0 = np.random.randint(0, max(W - phw, 1))
        d = min(pd, F_total); h = min(phw, H); w = min(phw, W)
        return (stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w],
                tmed_t[y0:y0+h, x0:x0+w])

    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Temporal-median warmup — "
                  f"{cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1)
        model.train(); rl = 0.0
        for it in range(cfg["warmup_iters"]):
            patches, targets = [], []
            for _ in range(bs):
                vol, tmed_crop = random_patch()
                aug = np.random.randint(0, 8)
                vol = _augment_3d(vol, aug)
                tmed_b = _augment_3d(
                    tmed_crop.unsqueeze(0).expand(vol.shape[0], -1, -1), aug,
                )
                patches.append(vol.unsqueeze(0))
                targets.append(tmed_b.unsqueeze(0))
            inp = torch.stack(patches, dim=0).to(device)
            tgt = torch.stack(targets, dim=0).to(device)
            opt.zero_grad()
            pred = model(inp)
            loss = F.l1_loss(pred, tgt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); rl += loss.item()
            if verbose and (it + 1) % 100 == 0:
                print(f"   {it+1:>5}/{cfg['warmup_iters']} "
                      f"loss={rl/100:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] 3D Noise2Void — {cfg['n2v_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"] * 0.5,
                                 weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6)
        model.train(); rl = 0.0
        for it in range(cfg["n2v_iters"]):
            all_orig, patches = [], []
            for _ in range(bs):
                vol, _ = random_patch()
                aug = np.random.randint(0, 8)
                vol = _augment_3d(vol, aug)
                masked, (mz, my, mx), orig = n2v_mask_3d(
                    vol, mask_ratio=cfg["mask_ratio"],
                    radius=cfg["mask_radius"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((mz, my, mx, orig))
            inp = torch.stack(patches, dim=0).to(device)
            opt.zero_grad()
            pred = model(inp)
            loss = torch.tensor(0.0, device=device)
            for b, (mz, my, mx, orig) in enumerate(all_orig):
                loss = loss + F.l1_loss(pred[b, 0, mz, my, mx], orig)
            loss = loss / bs
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); rl += loss.item()
            if verbose and (it + 1) % 250 == 0:
                lr_now = sch.get_last_lr()[0]
                print(f"   {it+1:>5}/{cfg['n2v_iters']} "
                      f"loss={rl/250:.6f} lr={lr_now:.2e} "
                      f"{time.time()-t0:.1f}s")
                rl = 0.0

    elapsed = time.time() - t0
    if verbose:
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return model, cfg


@torch.no_grad()
def denoise_stack(model, stack, config, device, verbose=True):
    model.eval()
    model = model.float()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    pd = min(config.get("patch_d", 16), F_total)
    phw = min(config.get("patch_hw", 48), H, W)
    r = config.get("time_compress_r", 2)
    n_lvl = config.get("n_time_levels", 2)
    div_t = r ** n_lvl
    pd = max((pd // div_t) * div_t, div_t)
    stride_d = max(pd // 2, div_t)
    stride_hw = max(phw // 2, 8)

    if verbose:
        print(f" Sliding window: patch={pd}x{phw}x{phw}, "
              f"stride={stride_d}x{stride_hw}x{stride_hw}")

    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    output_sum = torch.zeros(F_total, H, W, device=device)
    weight_sum = torch.zeros(F_total, H, W, device=device)
    gauss_win = _gaussian_window_3d((pd, phw, phw), sigma_frac=0.3, device=device)

    z_starts = list(range(0, max(F_total - pd, 0) + 1, stride_d))
    if not z_starts or z_starts[-1] + pd < F_total:
        z_starts.append(max(F_total - pd, 0))
    y_starts = list(range(0, max(H - phw, 0) + 1, stride_hw))
    if not y_starts or y_starts[-1] + phw < H:
        y_starts.append(max(H - phw, 0))
    x_starts = list(range(0, max(W - phw, 0) + 1, stride_hw))
    if not x_starts or x_starts[-1] + phw < W:
        x_starts.append(max(W - phw, 0))
    z_starts = sorted(set(z_starts))
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))
    total = len(z_starts) * len(y_starts) * len(x_starts)
    if verbose:
        print(f" Patches: {len(z_starts)}x{len(y_starts)}x{len(x_starts)} = {total}")
    t0 = time.time()
    done = 0
    for z0 in z_starts:
        z1 = min(z0 + pd, F_total); ad = z1 - z0
        for y0 in y_starts:
            y1 = min(y0 + phw, H); ah = y1 - y0
            for x0 in x_starts:
                x1 = min(x0 + phw, W); aw = x1 - x0
                patch = stack_t[z0:z1, y0:y1, x0:x1]
                if (ad < pd) or (ah < phw) or (aw < phw):
                    mode = ("reflect" if (pd - ad < ad and
                                           phw - ah < ah and
                                           phw - aw < aw)
                            else "replicate")
                    patch = F.pad(patch,
                                   (0, phw - aw, 0, phw - ah, 0, pd - ad),
                                   mode=mode)
                inp = patch.unsqueeze(0).unsqueeze(0).float()
                pred = model(inp).squeeze(0).squeeze(0).float()
                pred = pred[:ad, :ah, :aw]
                win = gauss_win[:ad, :ah, :aw]
                output_sum[z0:z1, y0:y1, x0:x1] += pred * win
                weight_sum[z0:z1, y0:y1, x0:x1] += win
                done += 1
                if verbose:
                    print(f"   {done}/{total} patches "
                          f"({100*done/total:.0f}%)  "
                          f"{time.time()-t0:.1f}s", end="\r")
    if verbose:
        print(f"\n Inference: {total} patches in {time.time()-t0:.1f}s")

    output = output_sum / weight_sum.clamp(min=1e-8)
    output = output.cpu().numpy()
    n_bad = int((~np.isfinite(output)).sum())
    if n_bad / max(output.size, 1) > 0.5:
        if verbose:
            print(f" WARNING: mostly NaN — falling back to noisy input.")
        return stack.astype(np.float32)
    if n_bad > 0:
        bad = ~np.isfinite(output)
        output[bad] = stack.astype(np.float32)[bad]

    output = norm_strategy.inverse(output, norm_params)
    in_lo, in_hi = float(stack.min()), float(stack.max())
    output = np.clip(output, in_lo, in_hi)
    return output


def save_checkpoint(model, config, path):
    torch.save({"model_state_dict": model.state_dict(),
                "config": config, "arch": "SRDTransNet"}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SRDTransNet(
        embed_dim=cfg.get("embed_dim", 32),
        n_time_levels=cfg.get("n_time_levels", 2),
        n_stb_blocks=cfg.get("n_stb_blocks", 2),
        num_heads=cfg.get("num_heads", 4),
        mlp_ratio=cfg.get("mlp_ratio", 2.0),
        time_compress_r=cfg.get("time_compress_r", 2),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
