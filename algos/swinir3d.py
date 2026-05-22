"""
SwinIR 3D — Research extension of SwinIR to volumetric input
=============================================================

This is OUR EXTENSION of the SwinIR architecture (Liang et al. ICCVW 2021)
to volumetric (3D) inputs. It is NOT from any published paper. The 2D
variant in `swinir2d.py` is paper-faithful; this 3D variant adapts the
same building blocks to 3D windowed attention.

Changes from 2D to 3D
---------------------

  * Windows are 3D cubes window_size × window_size × window_size
  * Window partition/reverse handle [B, D, H, W, C] tensors
  * Relative position bias is over a (2W-1)^3 grid instead of (2W-1)^2
  * Convs at block ends are Conv3D
  * Reflect-pad falls back to replicate when pad ≥ dim (PyTorch requires
    pad < dim for reflect)

Defaults are smaller than SwinIR2D since 3D windowed attention is much
heavier (O(window_size^6) per window vs O(window_size^4) in 2D).

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
# 3D WINDOW PRIMITIVES
# ══════════════════════════════════════════════════════════════

def window_partition_3d(x: torch.Tensor, ws: int):
    """x: [B, D, H, W, C] → [num_windows*B, ws^3, C]"""
    B, D, H, W, C = x.shape
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(
        -1, ws * ws * ws, C
    )
    return windows


def window_reverse_3d(windows: torch.Tensor, ws: int,
                       D: int, H: int, W: int):
    """[num_windows*B, ws^3, C] → [B, D, H, W, C]"""
    B = int(windows.shape[0] / (D * H * W / (ws ** 3)))
    x = windows.view(B, D // ws, H // ws, W // ws,
                     ws, ws, ws, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias for 3D
        W = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * W - 1) ** 3, num_heads)
        )
        # 3D coords
        coords_d = torch.arange(W)
        coords_h = torch.arange(W)
        coords_w = torch.arange(W)
        coords = torch.stack(
            torch.meshgrid(coords_d, coords_h, coords_w, indexing="ij")
        )                                                # [3, W, W, W]
        coords_flat = coords.flatten(1)                  # [3, N]
        rel = (coords_flat[:, :, None]
               - coords_flat[:, None, :])                 # [3, N, N]
        rel = rel.permute(1, 2, 0).contiguous()           # [N, N, 3]
        rel[..., 0] += W - 1
        rel[..., 1] += W - 1
        rel[..., 2] += W - 1
        rel[..., 0] *= (2 * W - 1) ** 2
        rel[..., 1] *= (2 * W - 1)
        rel_index = rel.sum(-1)
        self.register_buffer("relative_position_index", rel_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """x: [B*nW, N, C] where N = ws^3"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                       + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class SwinTransformerLayer3D(nn.Module):
    def __init__(self, dim, num_heads, window_size: int = 4,
                 shift_size: int = 0, mlp_ratio: float = 2.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ws = window_size
        self.shift_size = shift_size
        assert 0 <= shift_size < window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def _shift_attn_mask(self, D, H, W, device):
        if self.shift_size == 0:
            return None
        img_mask = torch.zeros((1, D, H, W, 1), device=device)
        ws, ss = self.ws, self.shift_size
        slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        cnt = 0
        for ds in slices:
            for hs in slices:
                for sws in slices:
                    img_mask[:, ds, hs, sws, :] = cnt
                    cnt += 1
        mask_windows = window_partition_3d(img_mask, ws)
        mask_windows = mask_windows.view(-1, ws ** 3)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                            float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, D, H, W):
        """x: [B, D*H*W, C]"""
        B, L, C = x.shape
        assert L == D * H * W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size,) * 3,
                                  dims=(1, 2, 3))
        else:
            shifted = x

        x_w = window_partition_3d(shifted, self.ws)
        attn_mask = self._shift_attn_mask(D, H, W, x.device)
        attn_w = self.attn(x_w, mask=attn_mask)
        shifted = window_reverse_3d(attn_w, self.ws, D, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted, shifts=(self.shift_size,) * 3,
                             dims=(1, 2, 3))
        else:
            x = shifted

        x = x.view(B, D * H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB3D(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            SwinTransformerLayer3D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x, D, H, W):
        identity = x
        for layer in self.layers:
            x = layer(x, D, H, W)
        B, _, C = x.shape
        x_v = x.transpose(1, 2).view(B, C, D, H, W)
        x_v = self.conv(x_v)
        x = x_v.flatten(2).transpose(1, 2)
        return x + identity


# ══════════════════════════════════════════════════════════════
# SwinIR3D
# ══════════════════════════════════════════════════════════════

class SwinIR3D(nn.Module):
    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 32,
        depths: tuple = (2, 2, 2),
        num_heads: tuple = (2, 2, 2),
        window_size: int = 4,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.window_size = window_size
        assert len(depths) == len(num_heads)

        self.conv_first = nn.Conv3d(in_chans, embed_dim, 3, padding=1)
        self.body = nn.ModuleList([
            RSTB3D(dim=embed_dim, depth=d, num_heads=h,
                    window_size=window_size, mlp_ratio=mlp_ratio)
            for d, h in zip(depths, num_heads)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv3d(embed_dim, embed_dim, 3, padding=1)
        self.conv_last = nn.Conv3d(embed_dim, in_chans, 3, padding=1)

    def forward(self, x):
        """x: [B, 1, D, H, W]"""
        _, _, D, H, W = x.shape
        ws = self.window_size
        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_d or pad_h or pad_w:
            mode = "reflect" if (pad_d < D and pad_h < H and pad_w < W) \
                else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d), mode=mode)
        Dp, Hp, Wp = x.shape[-3:]

        f = self.conv_first(x)
        identity = f

        B, C = f.shape[:2]
        tokens = f.flatten(2).transpose(1, 2)
        for blk in self.body:
            tokens = blk(tokens, Dp, Hp, Wp)
        tokens = self.norm(tokens)
        f = tokens.transpose(1, 2).view(B, C, Dp, Hp, Wp)
        f = self.conv_after_body(f) + identity

        noise_pred = self.conv_last(f)
        out = x - noise_pred
        if pad_d or pad_h or pad_w:
            out = out[:, :, :D, :H, :W]
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
        "depths":         (2, 2, 2),
        "num_heads":      (2, 2, 2),
        "window_size":    4,
        "mlp_ratio":      2.0,
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
        print(f" SwinIR3D: embed_dim={cfg['embed_dim']}, "
              f"depths={cfg['depths']}, heads={cfg['num_heads']}, "
              f"window={cfg['window_size']}")
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

    model = SwinIR3D(
        in_chans=1,
        embed_dim=cfg["embed_dim"],
        depths=tuple(cfg["depths"]),
        num_heads=tuple(cfg["num_heads"]),
        window_size=cfg["window_size"],
        mlp_ratio=cfg["mlp_ratio"],
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

    ws = config.get("window_size", 4)
    pd = min(config.get("patch_d", 16), F_total)
    phw = min(config.get("patch_hw", 48), H, W)
    pd = max((pd // ws) * ws, ws)
    phw = max((phw // ws) * ws, ws)
    stride_d = max(pd // 2, ws)
    stride_hw = max(phw // 2, ws)

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
                "config": config, "arch": "SwinIR3D"}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SwinIR3D(
        in_chans=1,
        embed_dim=cfg.get("embed_dim", 32),
        depths=tuple(cfg.get("depths", (2, 2, 2))),
        num_heads=tuple(cfg.get("num_heads", (2, 2, 2))),
        window_size=cfg.get("window_size", 4),
        mlp_ratio=cfg.get("mlp_ratio", 2.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
