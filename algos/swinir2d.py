"""
SwinIR 2D — Swin Transformer for Image Restoration (denoising variant)
=======================================================================

Implements the architecture from Liang et al., "SwinIR: Image Restoration
Using Swin Transformer" (ICCVW 2021, https://arxiv.org/abs/2108.10257).
We use the *denoising* configuration: no upscaling head at the end, just
the shallow → deep → reconstruction pipeline producing same-resolution
output.

Architecture
------------

    1. Shallow feature extraction:  Conv3×3 → embed_dim channels
    2. Deep feature extraction:     K Residual Swin Transformer Blocks
                                       (RSTBs), each containing several
                                       Swin Transformer Layers (STLs).
                                       Plus a long residual skip from
                                       shallow features.
    3. Reconstruction:              Conv3×3 → 1 channel (for denoising)

Each RSTB:  (STL × N)  → Conv3×3  → + skip from RSTB input

Each STL:  Pre-norm windowed multi-head self-attention (W-MSA) + MLP,
            with relative position bias. Alternates regular and shifted
            windows (SW-MSA) across consecutive STLs.

This is the 2D per-frame variant — there's a sibling swinir3d.py that
extends the same ideas to volumetric input as a research extension
(not from any paper).

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
# WINDOW PARTITIONING (standard Swin primitives)
# ══════════════════════════════════════════════════════════════

def window_partition_2d(x: torch.Tensor, window_size: int):
    """
    x: [B, H, W, C] (channels-last for attention)
    Returns: [num_windows*B, window_size*window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size * window_size, C
    )
    return windows


def window_reverse_2d(windows: torch.Tensor, window_size: int,
                       H: int, W: int):
    """
    windows: [num_windows*B, window_size*window_size, C]
    Returns: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention2D(nn.Module):
    """W-MSA with relative position bias."""

    def __init__(self, dim: int, window_size: int, num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0.0,
                 proj_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = (window_size, window_size)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table — [(2Wh-1)*(2Ww-1), num_heads]
        Wh, Ww = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
        )
        # Index mapping for relative positions
        coords_h = torch.arange(Wh)
        coords_w = torch.arange(Ww)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w,
                                             indexing="ij"))     # [2, Wh, Ww]
        coords_flatten = coords.flatten(1)                       # [2, Wh*Ww]
        relative_coords = (coords_flatten[:, :, None]
                           - coords_flatten[:, None, :])         # [2, N, N]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += Wh - 1
        relative_coords[:, :, 1] += Ww - 1
        relative_coords[:, :, 0] *= 2 * Ww - 1
        relative_position_index = relative_coords.sum(-1)        # [N, N]
        self.register_buffer("relative_position_index",
                              relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """x: [B*nW, N, C]"""
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias
        Wh, Ww = self.window_size
        rpb = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Wh * Ww, Wh * Ww, -1)
        rpb = rpb.permute(2, 0, 1).contiguous()
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


class SwinTransformerLayer2D(nn.Module):
    """One STL: W-MSA or SW-MSA + MLP, with pre-norms and residuals."""

    def __init__(self, dim, num_heads, window_size: int = 8,
                 shift_size: int = 0, mlp_ratio: float = 2.0,
                 qkv_bias: bool = True, drop: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        assert 0 <= shift_size < window_size, "shift_size must be < window_size"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention2D(
            dim, window_size, num_heads, qkv_bias, attn_drop, drop
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop)

    def _shift_attn_mask(self, H, W, device):
        if self.shift_size == 0:
            return None
        # Build the shift-window mask exactly per Swin paper
        img_mask = torch.zeros((1, H, W, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for hs in h_slices:
            for ws in w_slices:
                img_mask[:, hs, ws, :] = cnt
                cnt += 1
        mask_windows = window_partition_2d(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size *
                                           self.window_size)
        attn_mask = (mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2))
        attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                            float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        """x: [B, H*W, C]"""
        B, L, C = x.shape
        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Shift if needed
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size,
                                                 -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Window partition
        x_windows = window_partition_2d(shifted_x, self.window_size)
        attn_mask = self._shift_attn_mask(H, W, x.device)
        attn_windows = self.attn(x_windows, mask=attn_mask)

        # Reverse windows
        shifted_x = window_reverse_2d(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size,
                                                self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block: N STLs then a conv with residual."""

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            SwinTransformerLayer2D(
                dim=dim, num_heads=num_heads, window_size=window_size,
                # Alternate non-shifted and shifted windows
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])
        # Conv at the end of the block (SwinIR-specific: residual conv path)
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, H, W):
        """x: [B, H*W, C]"""
        identity = x
        for layer in self.layers:
            x = layer(x, H, W)
        # Convert to image, apply conv, back to tokens
        B, _, C = x.shape
        x_img = x.transpose(1, 2).view(B, C, H, W)
        x_img = self.conv(x_img)
        x = x_img.flatten(2).transpose(1, 2)
        return x + identity


# ══════════════════════════════════════════════════════════════
# SWINIR (denoising)
# ══════════════════════════════════════════════════════════════

class SwinIR2D(nn.Module):
    """
    SwinIR for image denoising (1× — same resolution in/out).

    Inputs/outputs: [B, 1, H, W] grayscale frames.
    """

    def __init__(
        self,
        in_chans: int = 1,
        embed_dim: int = 60,
        depths: tuple = (4, 4, 4, 4),     # depth of each RSTB
        num_heads: tuple = (4, 4, 4, 4),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.window_size = window_size
        assert len(depths) == len(num_heads), \
            "depths and num_heads must have same length"

        # 1. Shallow features
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, padding=1)

        # 2. Deep features (stack of RSTBs)
        self.body = nn.ModuleList([
            RSTB(dim=embed_dim, depth=d, num_heads=h,
                 window_size=window_size, mlp_ratio=mlp_ratio)
            for d, h in zip(depths, num_heads)
        ])
        # Conv at the end of all RSTBs (SwinIR architecture)
        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, padding=1)

        # 3. Reconstruction (denoising = no upscale)
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, padding=1)

    def forward(self, x):
        """x: [B, C, H, W]"""
        # Pad H, W up to multiple of window_size
        _, _, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h or pad_w:
            mode = "reflect" if (pad_h < H and pad_w < W) else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        Hp, Wp = x.shape[-2:]

        # Shallow features
        f = self.conv_first(x)
        identity = f

        # Deep features as tokens
        B, C = f.shape[:2]
        tokens = f.flatten(2).transpose(1, 2)        # [B, Hp*Wp, C]
        for blk in self.body:
            tokens = blk(tokens, Hp, Wp)
        tokens = self.norm(tokens)
        f = tokens.transpose(1, 2).view(B, C, Hp, Wp)
        f = self.conv_after_body(f) + identity       # long skip

        # Reconstruction
        # Denoising: predict residual (output = input − predicted_noise)
        noise_pred = self.conv_last(f)
        out = x - noise_pred

        # Strip padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


# ══════════════════════════════════════════════════════════════
# 2D BLIND-SPOT MASKING (per-frame N2V)
# ══════════════════════════════════════════════════════════════

def n2v_mask_2d(frame: torch.Tensor, mask_ratio: float = 0.015,
                 radius: int = 2):
    H, W = frame.shape
    n_pix = H * W
    n_mask = max(int(n_pix * mask_ratio), 1)
    flat_idx = torch.randperm(n_pix, device=frame.device)[:n_mask]
    my = flat_idx // W
    mx = flat_idx % W
    original = frame[my, mx].clone()
    dy = torch.randint(-radius, radius + 1, (n_mask,), device=frame.device)
    dx = torch.randint(-radius, radius + 1, (n_mask,), device=frame.device)
    same = (dy == 0) & (dx == 0)
    dy[same] = 1
    ny = (my + dy).clamp(0, H - 1)
    nx = (mx + dx).clamp(0, W - 1)
    masked = frame.clone()
    masked[my, mx] = frame[ny, nx]
    return masked, (my, mx), original


def _augment_2d(frame: torch.Tensor, aug_id: int) -> torch.Tensor:
    if aug_id >= 4:
        frame = torch.flip(frame, dims=[1])
    k = aug_id % 4
    if k > 0:
        frame = torch.rot90(frame, k=k, dims=[0, 1])
    return frame


def _gaussian_window_2d(shape, sigma_frac=0.25, device="cpu"):
    windows = []
    for s in shape:
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = max(s * sigma_frac, 1.0)
        windows.append(torch.exp(-0.5 * ((coords - center) / sigma) ** 2))
    return (windows[0][:, None] * windows[1][None, :]).clamp(min=1e-6)


# ══════════════════════════════════════════════════════════════
# TRAINING (per-frame, two-stage)
# ══════════════════════════════════════════════════════════════

def train_self_supervised(stack: np.ndarray, device: torch.device,
                           config: dict = None, verbose: bool = True):
    """Stage 0 — per-frame warmup; Stage 1 — per-frame N2V. fp32."""
    t0 = time.time()
    cfg = {
        "embed_dim":      60,
        "depths":         (4, 4, 4, 4),
        "num_heads":      (4, 4, 4, 4),
        "window_size":    8,
        "mlp_ratio":      2.0,
        "patch_hw":       128,
        "batch_size":     2,
        "warmup_iters":   300,
        "n2v_iters":      3000,
        "lr":             2e-4,
        "mask_ratio":     0.015,
        "mask_radius":    2,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    phw = cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" SwinIR2D: embed_dim={cfg['embed_dim']}, "
              f"depths={cfg['depths']}, heads={cfg['num_heads']}, "
              f"window={cfg['window_size']}")
        print(f" Patch: {phw}x{phw}, batch={bs}")
        print(f" Schedule: warmup={cfg['warmup_iters']}, "
              f"n2v={cfg['n2v_iters']}")
        print(f" Precision: fp32")

    # Normalize
    norm_name = cfg.get("normalization", DEFAULT_NORMALIZATION)
    norm_strategy = _prep.resolve_normalization(norm_name)
    norm_params = norm_strategy.compute_params(stack)
    cfg["norm_params"] = norm_params
    cfg["__resolved_normalization"] = norm_strategy.name
    stack_norm = norm_strategy.forward(stack, norm_params)

    # Temporal target
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

    model = SwinIR2D(
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
        f = np.random.randint(0, F_total)
        y0 = np.random.randint(0, max(H - phw, 1))
        x0 = np.random.randint(0, max(W - phw, 1))
        h = min(phw, H); w = min(phw, W)
        return (stack_t[f, y0:y0+h, x0:x0+w],
                tmed_t[y0:y0+h, x0:x0+w])

    # Stage 0
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Per-frame warmup — {cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1)
        model.train()
        rl = 0.0
        for it in range(cfg["warmup_iters"]):
            patches_in, patches_tgt = [], []
            for _ in range(bs):
                fr, tgt = random_patch()
                aug = np.random.randint(0, 8)
                fr = _augment_2d(fr, aug)
                tgt = _augment_2d(tgt, aug)
                patches_in.append(fr.unsqueeze(0))
                patches_tgt.append(tgt.unsqueeze(0))
            inp = torch.stack(patches_in, dim=0).to(device)
            tgt = torch.stack(patches_tgt, dim=0).to(device)

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

    # Stage 1
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] Per-frame Noise2Void — "
                  f"{cfg['n2v_iters']} iters")
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
                fr, _ = random_patch()
                aug = np.random.randint(0, 8)
                fr = _augment_2d(fr, aug)
                masked, (my, mx), orig = n2v_mask_2d(
                    fr, mask_ratio=cfg["mask_ratio"],
                    radius=cfg["mask_radius"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((my, mx, orig))
            inp = torch.stack(patches, dim=0).to(device)

            opt.zero_grad()
            pred = model(inp)
            loss = torch.tensor(0.0, device=device)
            for b, (my, mx, orig) in enumerate(all_orig):
                loss = loss + F.l1_loss(pred[b, 0, my, mx], orig)
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
# PER-FRAME SLIDING-WINDOW INFERENCE
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(model: SwinIR2D, stack: np.ndarray, config: dict,
                   device: torch.device, verbose: bool = True) -> np.ndarray:
    model.eval()
    model = model.float()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    ws = config.get("window_size", 8)
    phw = min(config.get("patch_hw", 128), H, W)
    phw = max((phw // ws) * ws, ws)
    stride = max(phw // 2, ws)

    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)

    if verbose:
        print(f" Per-frame sliding window: patch={phw}x{phw}, stride={stride}")

    y_starts = list(range(0, max(H - phw, 0) + 1, stride))
    if not y_starts or y_starts[-1] + phw < H:
        y_starts.append(max(H - phw, 0))
    x_starts = list(range(0, max(W - phw, 0) + 1, stride))
    if not x_starts or x_starts[-1] + phw < W:
        x_starts.append(max(W - phw, 0))
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))
    gauss_win = _gaussian_window_2d((phw, phw), sigma_frac=0.3, device=device)

    output = np.empty_like(stack_norm, dtype=np.float32)
    t0 = time.time()
    for t in range(F_total):
        frame_t = torch.from_numpy(stack_norm[t]).float().to(device)
        out_sum = torch.zeros_like(frame_t)
        w_sum = torch.zeros_like(frame_t)
        for y0 in y_starts:
            y1 = min(y0 + phw, H); ah = y1 - y0
            for x0 in x_starts:
                x1 = min(x0 + phw, W); aw = x1 - x0
                patch = frame_t[y0:y1, x0:x1]
                if (ah < phw) or (aw < phw):
                    mode = "reflect" if (ah > phw - ah and aw > phw - aw) \
                        else "replicate"
                    patch = F.pad(patch, (0, phw - aw, 0, phw - ah),
                                   mode=mode)
                inp = patch.unsqueeze(0).unsqueeze(0)
                pred = model(inp).squeeze(0).squeeze(0).float()
                pred = pred[:ah, :aw]
                win = gauss_win[:ah, :aw]
                out_sum[y0:y1, x0:x1] += pred * win
                w_sum[y0:y1, x0:x1] += win
        output[t] = (out_sum / w_sum.clamp(min=1e-8)).cpu().numpy()
        if verbose and (t + 1) % 100 == 0:
            print(f"   {t+1}/{F_total} frames  "
                  f"{time.time()-t0:.1f}s", end="\r")
    if verbose:
        print(f"\n Inference: {F_total} frames in {time.time()-t0:.1f}s")

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
                "config": config, "arch": "SwinIR2D"}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SwinIR2D(
        in_chans=1,
        embed_dim=cfg.get("embed_dim", 60),
        depths=tuple(cfg.get("depths", (4, 4, 4, 4))),
        num_heads=tuple(cfg.get("num_heads", (4, 4, 4, 4))),
        window_size=cfg.get("window_size", 8),
        mlp_ratio=cfg.get("mlp_ratio", 2.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
