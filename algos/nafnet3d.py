"""
NAFNet 3D — volumetric extension of the NAFNet architecture
============================================================

Adapts Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022,
https://arxiv.org/abs/2204.04676) to 3D denoising for the AI4Life-CIDC25
challenge. The 2D variant is in `nafnet2d.py`.

Changes from 2D to 3D
---------------------

  * All `Conv2d` → `Conv3d`, `LayerNorm2d` → `LayerNorm3d`
  * Depthwise conv: `kernel_size=3` over (D, H, W)
  * Channel attention uses `AdaptiveAvgPool3d(1)`
  * Downsamples use stride-2 `Conv3d` over all three dims; upsamples
    use `ConvTranspose3d` (no PixelShuffle3D in PyTorch core)
  * Reflect-pad must avoid `mode='reflect'` when the pad ≥ input dim
    along that axis; we fall back to `replicate` in that case
  * Self-supervised training uses 3D Noise2Void with full spatial+temporal
    masking, identical in shape to the other 3D algos in this framework

API parity with the rest of our algos.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runner import preprocessing as _prep

# ══════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════

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
# 3D building blocks
# ══════════════════════════════════════════════════════════════

class LayerNorm3d(nn.Module):
    """Channel-wise LayerNorm for [B, C, D, H, W] tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        # broadcast over [B, C, D, H, W]
        w = self.weight[None, :, None, None, None]
        b = self.bias[None, :, None, None, None]
        return x * w + b


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class SimpleChannelAttention3d(nn.Module):
    """SCA over [B, C, D, H, W] — pools all of (D, H, W) to 1."""

    def __init__(self, channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Conv3d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        return x * self.fc(self.gap(x))


class NAFBlock3d(nn.Module):
    """3D NAFBlock — same as 2D but with Conv3d / DWConv3d / LN3d / SCA3d."""

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2,
                 drop_out_rate: float = 0.0):
        super().__init__()
        c = channels
        dw_c = c * dw_expand
        ffn_c = c * ffn_expand

        self.norm1 = LayerNorm3d(c)
        self.conv1 = nn.Conv3d(c, dw_c, 1, bias=True)
        self.dwconv = nn.Conv3d(dw_c, dw_c, 3, padding=1, groups=dw_c, bias=True)
        self.gate1 = SimpleGate()
        self.sca = SimpleChannelAttention3d(dw_c // 2)
        self.conv2 = nn.Conv3d(dw_c // 2, c, 1, bias=True)

        self.norm2 = LayerNorm3d(c)
        self.conv3 = nn.Conv3d(c, ffn_c, 1, bias=True)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv3d(ffn_c // 2, c, 1, bias=True)

        self.drop1 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0 \
            else nn.Identity()
        self.drop2 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0 \
            else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1, 1)))

    def forward(self, x):
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.dwconv(h)
        h = self.gate1(h)
        h = self.sca(h)
        h = self.conv2(h)
        x = x + self.beta * self.drop1(h)

        h = self.norm2(x)
        h = self.conv3(h)
        h = self.gate2(h)
        h = self.conv4(h)
        x = x + self.gamma * self.drop2(h)
        return x


# ══════════════════════════════════════════════════════════════
# NAFNet3D — full U-Net topology
# ══════════════════════════════════════════════════════════════

class NAFNet3D(nn.Module):
    """3D NAFNet U-Net. Takes [B, 1, D, H, W] noisy patches and outputs
    [B, 1, D, H, W] denoised patches. Dims auto-padded to multiples of
    2**depth. Smaller defaults than 2D because 3D blocks are much heavier."""

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 16,
        enc_blocks: tuple = (1, 1, 2, 2),
        middle_blocks: int = 4,
        dec_blocks: tuple = (1, 1, 1, 1),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks)
        self.depth = len(enc_blocks)
        self.divisor = 2 ** self.depth

        self.intro = nn.Conv3d(in_channels, width, 3, padding=1, bias=True)
        c = width

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[
                NAFBlock3d(c, dw_expand, ffn_expand, drop_out_rate)
                for _ in range(n)
            ]))
            # Stride-2 conv downsample on all 3 dims; doubles channels
            self.downs.append(nn.Conv3d(c, c * 2, 2, stride=2, bias=True))
            c *= 2

        self.middle = nn.Sequential(*[
            NAFBlock3d(c, dw_expand, ffn_expand, drop_out_rate)
            for _ in range(middle_blocks)
        ])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            # ConvTranspose3d upsample (PyTorch has no PixelShuffle3d).
            # Halves channels.
            self.ups.append(nn.ConvTranspose3d(c, c // 2, 2, stride=2,
                                                bias=True))
            c //= 2
            self.decoders.append(nn.Sequential(*[
                NAFBlock3d(c, dw_expand, ffn_expand, drop_out_rate)
                for _ in range(n)
            ]))

        self.ending = nn.Conv3d(width, in_channels, 3, padding=1, bias=True)

    def forward(self, x):
        """x: [B, 1, D, H, W] → [B, 1, D, H, W] (denoised)."""
        identity = x

        # Pad all 3 dims to multiples of self.divisor.
        _, _, D, H, W = x.shape
        pd_ = (self.divisor - D % self.divisor) % self.divisor
        ph_ = (self.divisor - H % self.divisor) % self.divisor
        pw_ = (self.divisor - W % self.divisor) % self.divisor
        if pd_ or ph_ or pw_:
            # Use reflect if every pad is strictly less than the
            # corresponding dim, else fall back to replicate
            # (PyTorch F.pad reflect requires pad < dim).
            mode = "reflect" if (pd_ < D and ph_ < H and pw_ < W) else "replicate"
            x = F.pad(x, (0, pw_, 0, ph_, 0, pd_), mode=mode)
            identity_pad = F.pad(identity, (0, pw_, 0, ph_, 0, pd_), mode=mode)
        else:
            identity_pad = identity

        h = self.intro(x)
        skips = []
        for blocks, down in zip(self.encoders, self.downs):
            h = blocks(h)
            skips.append(h)
            h = down(h)

        h = self.middle(h)

        for blocks, up, skip in zip(self.decoders, self.ups,
                                      reversed(skips)):
            h = up(h)
            # Pad up to match skip if there's a residual off-by-one
            if h.shape[2:] != skip.shape[2:]:
                dd = skip.shape[2] - h.shape[2]
                dh = skip.shape[3] - h.shape[3]
                dw = skip.shape[4] - h.shape[4]
                h = F.pad(h, (0, max(dw, 0), 0, max(dh, 0), 0, max(dd, 0)))
                h = h[:, :, :skip.shape[2], :skip.shape[3], :skip.shape[4]]
            h = h + skip
            h = blocks(h)

        noise_pred = self.ending(h)
        out = identity_pad - noise_pred

        if pd_ or ph_ or pw_:
            out = out[:, :, :D, :H, :W]
        return out


# ══════════════════════════════════════════════════════════════
# 3D BLIND-SPOT MASKING (same as DVT — Noise2Void on 3D volume)
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume: torch.Tensor, mask_ratio: float = 0.015,
                radius: int = 2):
    """3D Noise2Void on [D, H, W]: returns (masked_vol, indices, originals)."""
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
        w = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
        windows.append(w)
    w3d = (windows[0][:, None, None]
           * windows[1][None, :, None]
           * windows[2][None, None, :])
    return w3d.clamp(min=1e-6)


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
):
    """Stage 0 — temporal-median warmup; Stage 1 — 3D N2V. Full fp32."""
    t0 = time.time()
    cfg = {
        "width":          16,
        "enc_blocks":     (1, 1, 2, 2),
        "middle_blocks":  4,
        "dec_blocks":     (1, 1, 1, 1),
        "dw_expand":      2,
        "ffn_expand":     2,
        "drop_out_rate":  0.0,
        "patch_d":        16,       # 3D blocks are much heavier than 2D
        "patch_hw":       64,
        "batch_size":     1,
        "warmup_iters":   300,
        "n2v_iters":      2500,
        "lr":             3e-4,
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
        print(f" NAFNet3D: width={cfg['width']}, "
              f"enc={cfg['enc_blocks']}, mid={cfg['middle_blocks']}, "
              f"dec={cfg['dec_blocks']}")
        print(f" Patch: {pd}x{phw}x{phw}, batch={bs}")
        print(f" Schedule: warmup={cfg['warmup_iters']}, "
              f"n2v={cfg['n2v_iters']}")
        print(f" Precision: fp32")

    # ── Normalize ──────────────────────────────────────────
    norm_name = cfg.get("normalization", DEFAULT_NORMALIZATION)
    norm_strategy = _prep.resolve_normalization(norm_name)
    norm_params = norm_strategy.compute_params(stack)
    cfg["norm_params"] = norm_params
    cfg["__resolved_normalization"] = norm_strategy.name
    stack_norm = norm_strategy.forward(stack, norm_params)
    if verbose:
        print(f" Norm [{norm_strategy.name}]: "
              f"shift={norm_params['shift']:.2f}, "
              f"scale={norm_params['scale']:.2f}, "
              f"range=[{stack_norm.min():.3f}, {stack_norm.max():.3f}]")

    # ── Temporal target ──────────────────────────────────
    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    cfg["__resolved_temporal_target"] = tt_strategy.name
    if tt_strategy.returns != "2d":
        _tt_3d = tt_strategy.compute(stack_norm)
        temporal_med = np.median(_tt_3d, axis=0).astype(np.float32)
    else:
        temporal_med = tt_strategy.compute(stack_norm)

    stack_t = torch.from_numpy(stack_norm).float().to(device)
    tmed_t = torch.from_numpy(temporal_med).float().to(device)

    model = NAFNet3D(
        in_channels=1,
        width=cfg["width"],
        enc_blocks=tuple(cfg["enc_blocks"]),
        middle_blocks=cfg["middle_blocks"],
        dec_blocks=tuple(cfg["dec_blocks"]),
        dw_expand=cfg["dw_expand"],
        ffn_expand=cfg["ffn_expand"],
        drop_out_rate=cfg["drop_out_rate"],
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

    # Stage 0 warmup
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Temporal-median warmup — "
                  f"{cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1)
        model.train()
        rl = 0.0
        for it in range(cfg["warmup_iters"]):
            patches, targets = [], []
            for _ in range(bs):
                vol, tmed_crop = random_patch()
                aug = np.random.randint(0, 8)
                vol = _augment_3d(vol, aug)
                tmed_crop_b = _augment_3d(
                    tmed_crop.unsqueeze(0).expand(vol.shape[0], -1, -1), aug,
                )
                patches.append(vol.unsqueeze(0))            # [1, d, h, w]
                targets.append(tmed_crop_b.unsqueeze(0))    # [1, d, h, w]
            inp = torch.stack(patches, dim=0).to(device)    # [B, 1, d, h, w]
            tgt = torch.stack(targets, dim=0).to(device)

            opt.zero_grad()
            pred = model(inp)
            loss = F.l1_loss(pred, tgt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()
            if verbose and (it + 1) % 100 == 0:
                print(f"   {it+1:>5}/{cfg['warmup_iters']} "
                      f"loss={rl/100:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # Stage 1 N2V
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] 3D Noise2Void — {cfg['n2v_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"] * 0.5,
                                 weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6)
        model.train()
        rl = 0.0
        for it in range(cfg["n2v_iters"]):
            all_orig = []
            patches = []
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
            opt.step()
            sch.step()
            rl += loss.item()
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


# ══════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE (3D)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(
    model: NAFNet3D,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    model.eval()
    model = model.float()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    pd = min(config.get("patch_d", 16), F_total)
    phw = min(config.get("patch_hw", 64), H, W)
    pd = max((pd // 16) * 16, 16)
    phw = max((phw // 16) * 16, 16)
    stride_d = max(pd // 2, 8)
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
                                          phw - aw < aw) else "replicate")
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
            print(f" WARNING: model output mostly NaN — falling back to "
                  f"noisy input.")
        return stack.astype(np.float32)
    if n_bad > 0:
        bad = ~np.isfinite(output)
        output[bad] = stack.astype(np.float32)[bad]

    output = norm_strategy.inverse(output, norm_params)
    in_lo, in_hi = float(stack.min()), float(stack.max())
    output = np.clip(output, in_lo, in_hi)
    return output


# ══════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({"model_state_dict": model.state_dict(),
                "config": config,
                "arch": "NAFNet3D"}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NAFNet3D(
        in_channels=1,
        width=cfg.get("width", 16),
        enc_blocks=tuple(cfg.get("enc_blocks", (1, 1, 2, 2))),
        middle_blocks=cfg.get("middle_blocks", 4),
        dec_blocks=tuple(cfg.get("dec_blocks", (1, 1, 1, 1))),
        dw_expand=cfg.get("dw_expand", 2),
        ffn_expand=cfg.get("ffn_expand", 2),
        drop_out_rate=cfg.get("drop_out_rate", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
