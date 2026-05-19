"""
3D Swin-Unet + GSC + FUE + Restormer-style Decoder for Calcium Imaging
======================================================================

Self-supervised zero-shot denoiser combining:

  * Swin-Unet backbone   — Cao et al., ECCV 2022 (arXiv:2105.05537).
    Pure-transformer U-shaped encoder-decoder built from shifted-window
    multi-head self-attention (SW-MSA) blocks. We lift to 3D: windows
    are cubes [Mt, Mh, Mw], attention is windowed within each cube,
    and every other block shifts by (Mt//2, Mh//2, Mw//2).

  * GSC (Gated Spatial Convolution) — SegMamba, Xing et al., MICCAI 2024
    (arXiv:2401.13560). Inserted before each window-attention block:
        GSC(z) = z + Conv3x3x3( Conv3x3x3(z) * Conv1x1x1(z) )
    Restores 3D spatial structure that the windowed attention discards
    when it reshapes features into per-window 1D sequences.

  * FUE (Feature-level Uncertainty Estimation) — SegMamba, §3.3.
    Applied to each encoder skip before decoder concat:
        tilde_z = z + z * (1 - softmax_entropy(z) / log(C))
    Reweights skip features by how confidently the network has decided
    their channel direction. Helps high-confidence features dominate
    the reconstruction.

  * Restormer-style decoder — Zamir et al., CVPR 2022 (arXiv:2111.09881).
    Each up-stage uses transposed-conv upsample, concat-with-skip,
    1x1x1 channel reduction, and a GDFN block (gated depth-wise FFN)
    in addition to the windowed attention. The GDFN provides
    multiplicative gating that helps suppress residual noise features
    at high resolutions.

The whole network is bias-free (Mohan et al., ICLR 2019) so it cannot
learn the "predict the local mean" shortcut that breaks Noise2Void.
Output is residual: denoised = noisy - predicted_noise.

Training:
  Zero-shot 3D Noise2Void with a short temporal-median warmup. L1 loss
  at masked voxels only (no gradient-consistency: pushing output
  gradients to match noisy-input gradients teaches the network to keep
  noise, which we learned the hard way). Mixed precision (fp16) on CUDA.

Budget: targets ~8 min/stack on T4 for 7 stacks/hour.

API parity:
  compute_norm_params, normalize, denormalize
  train_self_supervised(stack, device, config) -> (model, cfg)
  denoise_stack(model, stack, config, device)  -> np.ndarray
  save_checkpoint, load_checkpoint
  UNet3D alias -> SwinUnet3D
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


# ══════════════════════════════════════════════════════════════
# NORMALIZATION
# ══════════════════════════════════════════════════════════════

def compute_norm_params(stack: np.ndarray) -> dict:
    """Robust p0.5-p99.5 percentile normalization."""
    n = min(300, stack.shape[0])
    idx = np.linspace(0, stack.shape[0] - 1, n, dtype=int)
    sampled = stack[idx].astype(np.float64)
    p_lo = float(np.percentile(sampled, 0.5))
    p_hi = float(np.percentile(sampled, 99.5))
    scale = max(p_hi - p_lo, 1e-6)
    return {"shift": p_lo, "scale": scale}


def normalize(data, params):
    return (data.astype(np.float32) - params["shift"]) / params["scale"]


def denormalize(data, params):
    return data.astype(np.float32) * params["scale"] + params["shift"]


# ══════════════════════════════════════════════════════════════
# GSC — Gated Spatial Convolution (SegMamba)
# ══════════════════════════════════════════════════════════════

class GroupNormBF(nn.Module):
    """GroupNorm with bias frozen at 0 (bias-free) — paper Restormer §4.4."""
    def __init__(self, channels, num_groups=8):
        super().__init__()
        ng = min(num_groups, channels)
        while channels % ng != 0:
            ng -= 1
        self.gn = nn.GroupNorm(ng, channels, affine=True)
        self.gn.bias.requires_grad_(False)
        with torch.no_grad():
            self.gn.bias.zero_()

    def forward(self, x):
        return self.gn(x)


class GSC3D(nn.Module):
    """
    GSC(z) = z + Conv3x3x3( Conv3x3x3(z) * Conv1x1x1(z) )

    Each inner Conv -> GroupNorm -> GELU. The pixel-wise product of the
    two pre-activated paths acts as a multiplicative gate; the final
    Conv3x3x3 fuses, and a residual connection preserves the input.
    """
    def __init__(self, channels):
        super().__init__()
        self.b3 = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            GroupNormBF(channels), nn.GELU(),
        )
        self.b1 = nn.Sequential(
            nn.Conv3d(channels, channels, 1, bias=False),
            GroupNormBF(channels), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, bias=False),
            GroupNormBF(channels), nn.GELU(),
        )

    def forward(self, x):
        a = self.b3(x)
        b = self.b1(x)
        return x + self.fuse(a * b)


# ══════════════════════════════════════════════════════════════
# FUE — Feature-level Uncertainty Estimation (SegMamba)
# ══════════════════════════════════════════════════════════════

class FUE3D(nn.Module):
    """
    Per-voxel softmax entropy over the channel axis, normalized by
    log(C). Skips with HIGH entropy (uncertain channel direction) get
    suppressed; skips with LOW entropy get reinforced.

        H_norm = -sum_c p_c log p_c  /  log C        in [0, 1]
        out    = z + z * (1 - H_norm)
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, z):
        C = z.shape[1]
        p = F.softmax(z, dim=1).clamp(min=self.eps)
        H = -(p * p.log()).sum(dim=1, keepdim=True)
        H_norm = H / math.log(max(C, 2))
        return z + z * (1.0 - H_norm)


# ══════════════════════════════════════════════════════════════
# WINDOW ATTENTION (3D, with shift and relative position bias)
# ══════════════════════════════════════════════════════════════

def _window_partition(x, ws):
    """[B, D, H, W, C] -> [B*nW, Mt*Mh*Mw, C], windows ws=(Mt, Mh, Mw)."""
    B, D, H, W, C = x.shape
    Mt, Mh, Mw = ws
    x = x.view(B, D // Mt, Mt, H // Mh, Mh, W // Mw, Mw, C)
    # bring window dims together, then batch them with B
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    windows = x.view(-1, Mt * Mh * Mw, C)
    return windows


def _window_reverse(windows, ws, D, H, W):
    """Inverse of _window_partition."""
    Mt, Mh, Mw = ws
    B = int(windows.shape[0] / ((D // Mt) * (H // Mh) * (W // Mw)))
    x = windows.view(B, D // Mt, H // Mh, W // Mw, Mt, Mh, Mw, -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    x = x.view(B, D, H, W, -1)
    return x


class WindowAttention3D(nn.Module):
    """
    Windowed multi-head self-attention with learnable relative position
    bias (Swin §3.2). 3D version: positions span (Mt, Mh, Mw).
    Bias-free linear projections.
    """
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size       # (Mt, Mh, Mw)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        Mt, Mh, Mw = window_size
        # Relative position bias table: (2Mt-1)*(2Mh-1)*(2Mw-1) x heads
        self.rel_bias = nn.Parameter(
            torch.zeros((2*Mt-1) * (2*Mh-1) * (2*Mw-1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        # Precompute relative-position index inside one window
        coords = torch.stack(torch.meshgrid(
            torch.arange(Mt), torch.arange(Mh), torch.arange(Mw),
            indexing="ij",
        ))                                          # [3, Mt, Mh, Mw]
        coords = coords.flatten(1)                  # [3, N]
        rel = coords[:, :, None] - coords[:, None, :]     # [3, N, N]
        rel = rel.permute(1, 2, 0).contiguous()           # [N, N, 3]
        rel[:, :, 0] += Mt - 1
        rel[:, :, 1] += Mh - 1
        rel[:, :, 2] += Mw - 1
        rel[:, :, 0] *= (2*Mh - 1) * (2*Mw - 1)
        rel[:, :, 1] *= (2*Mw - 1)
        index = rel.sum(-1)                              # [N, N]
        self.register_buffer("rel_index", index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, mask=None):
        """
        x: [B*nW, N, C]   where N = Mt*Mh*Mw
        mask: [nW, N, N] optional attn mask (for shifted windows)
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]      # each [B_, heads, N, head_dim]
        attn = (q @ k.transpose(-2, -1)) * self.scale     # [B_, heads, N, N]

        # add relative position bias
        bias = self.rel_bias[self.rel_index.view(-1)].view(N, N, -1)
        bias = bias.permute(2, 0, 1).contiguous()         # [heads, N, N]
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


def _compute_shift_mask(D, H, W, ws, shift, device):
    """
    Build the attention mask used inside SW-MSA so that voxels that are
    cyclically wrapped don't attend across the wrap boundary.
    Returns mask shaped [nW, Mt*Mh*Mw, Mt*Mh*Mw] with -inf where attn
    is disallowed.
    """
    Mt, Mh, Mw = ws
    St, Sh, Sw = shift
    img = torch.zeros((1, D, H, W, 1), device=device)
    # Label every spatial region by a unique integer so windows that
    # straddle the cyclic shift boundary get different labels.
    cnt = 0
    for slc_d in (slice(0, -Mt), slice(-Mt, -St), slice(-St, None)):
        for slc_h in (slice(0, -Mh), slice(-Mh, -Sh), slice(-Sh, None)):
            for slc_w in (slice(0, -Mw), slice(-Mw, -Sw), slice(-Sw, None)):
                img[:, slc_d, slc_h, slc_w, :] = cnt
                cnt += 1
    win = _window_partition(img, ws)              # [nW, N, 1]
    win = win.squeeze(-1)                          # [nW, N]
    mask = win.unsqueeze(2) - win.unsqueeze(1)     # [nW, N, N]
    mask = mask.masked_fill(mask != 0, float("-inf"))
    mask = mask.masked_fill(mask == 0, 0.0)
    return mask


class GDFN3D(nn.Module):
    """Gated-Dconv FFN from Restormer (3D version), bias-free."""
    def __init__(self, dim, expand=2.0):
        super().__init__()
        hidden = int(dim * expand)
        self.proj_in = nn.Conv3d(dim, hidden * 2, 1, bias=False)
        self.dwconv = nn.Conv3d(hidden * 2, hidden * 2, 3, padding=1,
                                 groups=hidden * 2, bias=False)
        self.proj_out = nn.Conv3d(hidden, dim, 1, bias=False)

    def forward(self, x):
        x = self.dwconv(self.proj_in(x))
        a, b = x.chunk(2, dim=1)
        return self.proj_out(F.gelu(a) * b)


# ══════════════════════════════════════════════════════════════
# SWIN BLOCK 3D — GSC -> (S)W-MSA -> GDFN
# ══════════════════════════════════════════════════════════════

class LayerNormChannelLast(nn.LayerNorm):
    """Same as nn.LayerNorm; only here for clarity inside the block."""
    pass


class SwinBlock3D(nn.Module):
    """
    One 3D Swin block:
        x -> GSC (3D)
          -> Norm + W-MSA or SW-MSA  (residual)
          -> Norm + GDFN (residual)
    """
    def __init__(self, dim, num_heads, window_size, shift=(0, 0, 0)):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift = shift

        self.gsc = GSC3D(dim)
        self.norm1 = LayerNormChannelLast(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.Identity()              # GDFN already has GroupNorm-like behavior
        self.ffn = GDFN3D(dim, expand=2.0)

    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # ── GSC (channels-first) ────────────────────────────
        x = self.gsc(x)

        # ── (S)W-MSA ────────────────────────────────────────
        Mt, Mh, Mw = self.window_size
        St, Sh, Sw = self.shift

        # Pad to multiples of window_size. Use replicate (not reflect)
        # because at the bottleneck the spatial dims can be as small as
        # 2, and reflect-pad requires pad < dim.
        ppd = (Mt - D % Mt) % Mt
        pph = (Mh - H % Mh) % Mh
        ppw = (Mw - W % Mw) % Mw
        if ppd or pph or ppw:
            x = F.pad(x, (0, ppw, 0, pph, 0, ppd), mode="replicate")
        Dp, Hp, Wp = x.shape[2], x.shape[3], x.shape[4]

        # to channels-last
        x_attn = x.permute(0, 2, 3, 4, 1).contiguous()       # [B, D, H, W, C]
        residual = x_attn

        # shift if needed
        if St or Sh or Sw:
            x_attn = torch.roll(
                x_attn, shifts=(-St, -Sh, -Sw), dims=(1, 2, 3),
            )
            attn_mask = _compute_shift_mask(
                Dp, Hp, Wp, self.window_size, self.shift, x_attn.device,
            )
        else:
            attn_mask = None

        x_attn = self.norm1(x_attn)
        # partition into windows
        windows = _window_partition(x_attn, self.window_size)  # [B*nW, N, C]
        attended = self.attn(windows, mask=attn_mask)
        # reverse
        x_attn = _window_reverse(attended, self.window_size, Dp, Hp, Wp)

        # un-shift
        if St or Sh or Sw:
            x_attn = torch.roll(
                x_attn, shifts=(St, Sh, Sw), dims=(1, 2, 3),
            )

        x_attn = residual + x_attn
        # back to channels-first
        x_attn = x_attn.permute(0, 4, 1, 2, 3).contiguous()    # [B, C, Dp, Hp, Wp]

        # ── GDFN ────────────────────────────────────────────
        x_attn = x_attn + self.ffn(self.norm2(x_attn))

        # strip pad
        if ppd or pph or ppw:
            x_attn = x_attn[:, :, :D, :H, :W]
        return x_attn


# ══════════════════════════════════════════════════════════════
# DOWN / UP SAMPLE — strided conv (down), transpose conv (up)
# ══════════════════════════════════════════════════════════════

class Down3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.Conv3d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Up3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.body = nn.ConvTranspose3d(in_ch, out_ch, 2, stride=2, bias=False)

    def forward(self, x):
        return self.body(x)


def _match_cat(up, skip):
    dd = skip.shape[2] - up.shape[2]
    dh = skip.shape[3] - up.shape[3]
    dw = skip.shape[4] - up.shape[4]
    if dd or dh or dw:
        if dd < 0 or dh < 0 or dw < 0:
            md = min(up.shape[2], skip.shape[2])
            mh = min(up.shape[3], skip.shape[3])
            mw = min(up.shape[4], skip.shape[4])
            up = up[:, :, :md, :mh, :mw]
            skip = skip[:, :, :md, :mh, :mw]
        else:
            up = F.pad(up, (0, dw, 0, dh, 0, dd))
    return torch.cat([up, skip], dim=1)


# ══════════════════════════════════════════════════════════════
# MAIN MODEL — 3D Swin-Unet + GSC + FUE + Restormer-style decoder
# ══════════════════════════════════════════════════════════════

def _make_stage(dim, num_heads, window_size, depth):
    """Alternating W-MSA and SW-MSA Swin blocks (Swin §3.2)."""
    Mt, Mh, Mw = window_size
    shift = (Mt // 2, Mh // 2, Mw // 2)
    blocks = []
    for i in range(depth):
        sh = (0, 0, 0) if (i % 2 == 0) else shift
        blocks.append(SwinBlock3D(dim, num_heads, window_size, sh))
    return nn.Sequential(*blocks)


class SwinUnet3D(nn.Module):
    """
    3D Swin-Unet with GSC, FUE, and a Restormer-style transposed-conv
    decoder. Designed to fit ~8 min/stack on T4 at the default config.

    Topology:
        Stem 3x3x3 conv -> dim channels
        Enc-1: SwinBlocks (depth d0, dim c1)       -> skip1 -> FUE
        Down1 (stride 2)
        Enc-2: SwinBlocks (depth d1, dim c2)       -> skip2 -> FUE
        Down2 (stride 2)
        Enc-3: SwinBlocks (depth d2, dim c3)       -> skip3 -> FUE
        Down3 (stride 2)
        Bot:   SwinBlocks (depth d3, dim c4)
        Up3 -> cat(skip3) -> 1x1 reduce -> SwinBlocks
        Up2 -> cat(skip2) -> 1x1 reduce -> SwinBlocks
        Up1 -> cat(skip1) -> 1x1 reduce -> SwinBlocks
        Out 3x3x3 conv -> noise_pred
    """
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        dim: int = 24,
        num_blocks: tuple = (1, 1, 1, 2),
        num_heads: tuple = (2, 2, 4, 4),
        window_size: tuple = (4, 4, 4),
    ):
        super().__init__()
        c1, c2, c3, c4 = dim, dim * 2, dim * 4, dim * 8
        self.window_size = window_size

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, c1, 3, padding=1, bias=False),
            GroupNormBF(c1), nn.GELU(),
        )

        # Encoder
        self.enc1 = _make_stage(c1, num_heads[0], window_size, num_blocks[0])
        self.down1 = Down3D(c1, c2)
        self.enc2 = _make_stage(c2, num_heads[1], window_size, num_blocks[1])
        self.down2 = Down3D(c2, c3)
        self.enc3 = _make_stage(c3, num_heads[2], window_size, num_blocks[2])
        self.down3 = Down3D(c3, c4)
        self.bot = _make_stage(c4, num_heads[3], window_size, num_blocks[3])

        # FUE on encoder skips
        self.fue1 = FUE3D()
        self.fue2 = FUE3D()
        self.fue3 = FUE3D()

        # Decoder
        self.up3 = Up3D(c4, c3)
        self.reduce3 = nn.Conv3d(c3 * 2, c3, 1, bias=False)
        self.dec3 = _make_stage(c3, num_heads[2], window_size, num_blocks[2])

        self.up2 = Up3D(c3, c2)
        self.reduce2 = nn.Conv3d(c2 * 2, c2, 1, bias=False)
        self.dec2 = _make_stage(c2, num_heads[1], window_size, num_blocks[1])

        self.up1 = Up3D(c2, c1)
        self.reduce1 = nn.Conv3d(c1 * 2, c1, 1, bias=False)
        self.dec1 = _make_stage(c1, num_heads[0], window_size, num_blocks[0])

        # Predict noise residual
        self.out_conv = nn.Conv3d(c1, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        identity = x
        # Pad to multiples of 8 (3 downsamples by 2) AND of window_size
        _, _, D, H, W = x.shape
        Mt, Mh, Mw = self.window_size
        # Pad so that after 3 downsamples (factor 8) the deepest spatial
        # dim is at least window_size. Need D, H, W % (8 * Mt/Mh/Mw)??
        # Actually after 3 downsamples we have D/8 — for windowing at
        # the bottleneck we need D/8 % Mt == 0, i.e. D % (8*Mt) == 0.
        # But the Swin block already auto-pads to window_size *internally*,
        # so we only need D, H, W divisible by 8.
        pd = (8 - D % 8) % 8
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")
            identity = F.pad(identity, (0, pw, 0, ph, 0, pd), mode="reflect")

        x = self.stem(x)
        s1 = self.enc1(x)
        s2 = self.enc2(self.down1(s1))
        s3 = self.enc3(self.down2(s2))
        b = self.bot(self.down3(s3))

        # FUE on the encoder skips
        s1 = self.fue1(s1)
        s2 = self.fue2(s2)
        s3 = self.fue3(s3)

        # Restormer-style decoder
        d3 = self.up3(b)
        d3 = _match_cat(d3, s3)
        d3 = self.reduce3(d3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = _match_cat(d2, s2)
        d2 = self.reduce2(d2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_cat(d1, s1)
        d1 = self.reduce1(d1)
        d1 = self.dec1(d1)

        noise = self.out_conv(d1)
        out = identity - noise

        if pd or ph or pw:
            out = out[:, :, :D, :H, :W]
        return out


# Drop-in alias for anything that imports UNet3D
UNet3D = SwinUnet3D


# ══════════════════════════════════════════════════════════════
# N2V MASKING + WINDOW + AUGMENTATION
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume, mask_ratio=0.020, radius_s=2, radius_t=0):
    """Spatial-only neighbor masking by default (radius_t=0)."""
    D, H, W = volume.shape
    n_vox = D * H * W
    n_mask = max(int(n_vox * mask_ratio), 1)

    flat_idx = torch.randperm(n_vox, device=volume.device)[:n_mask]
    mz = flat_idx // (H * W)
    my = (flat_idx % (H * W)) // W
    mx = flat_idx % W
    original = volume[mz, my, mx].clone()

    if radius_t > 0:
        dz = torch.randint(-radius_t, radius_t + 1, (n_mask,), device=volume.device)
    else:
        dz = torch.zeros(n_mask, dtype=torch.long, device=volume.device)
    dy = torch.randint(-radius_s, radius_s + 1, (n_mask,), device=volume.device)
    dx = torch.randint(-radius_s, radius_s + 1, (n_mask,), device=volume.device)
    same = (dz == 0) & (dy == 0) & (dx == 0)
    dy[same] = 1

    nz = (mz + dz).clamp(0, D - 1)
    ny = (my + dy).clamp(0, H - 1)
    nx = (mx + dx).clamp(0, W - 1)

    masked = volume.clone()
    masked[mz, my, mx] = volume[nz, ny, nx]
    return masked, (mz, my, mx), original


def _gaussian_window_3d(shape, sigma_frac=0.3, device="cpu"):
    windows = []
    for s in shape:
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = max(s * sigma_frac, 1.0)
        windows.append(torch.exp(-0.5 * ((coords - center) / sigma) ** 2))
    w3d = (windows[0][:, None, None]
           * windows[1][None, :, None]
           * windows[2][None, None, :])
    return w3d.clamp(min=1e-6)


def _augment_3d(vol, aug_id):
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


# ══════════════════════════════════════════════════════════════
# TRAINING — fp16, two-stage, L1 at masked voxels only
# ══════════════════════════════════════════════════════════════

def train_self_supervised(stack, device, config=None, verbose=True):
    t0 = time.time()
    cfg = {
        # backbone
        "dim":              24,
        "num_blocks":       (1, 1, 1, 2),
        "num_heads":        (2, 2, 4, 4),
        "window_size":      (4, 4, 4),
        # patch sampling
        "patch_d":          32,
        "patch_hw":         64,
        "batch_size":       2,
        # schedule (fits ~8 min/stack on T4)
        "warmup_iters":     100,
        "n2v_iters":        1200,
        "lr":               4e-4,
        # masking
        "mask_ratio":       0.020,
        "mask_radius_s":    2,
        "mask_radius_t":    0,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]
    use_amp = (device.type == "cuda")

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" SwinUnet3D: dim={cfg['dim']}, blocks={cfg['num_blocks']}, "
              f"heads={cfg['num_heads']}, window={cfg['window_size']}")
        print(f" Patch: {pd}x{phw}x{phw}, batch={bs}")
        print(f" Schedule: warmup={cfg['warmup_iters']}, n2v={cfg['n2v_iters']}")
        print(f" Masking: ratio={cfg['mask_ratio']}, "
              f"radius_s={cfg['mask_radius_s']}, radius_t={cfg['mask_radius_t']}")
        print(f" Mixed precision (fp16): {use_amp}")

    norm_params = compute_norm_params(stack)
    cfg["norm_params"] = norm_params
    stack_norm = normalize(stack, norm_params)
    if verbose:
        print(f" Norm: shift={norm_params['shift']:.2f}, scale={norm_params['scale']:.2f}")

    # Temporal median for warmup
    n_med = min(500, F_total)
    med_idx = np.linspace(0, F_total - 1, n_med, dtype=int)
    temporal_med = np.median(stack_norm[med_idx], axis=0).astype(np.float32)

    stack_t = torch.from_numpy(stack_norm).float().to(device)
    tmed_t  = torch.from_numpy(temporal_med).float().to(device)

    model = SwinUnet3D(
        in_channels=1, out_channels=1,
        dim=cfg["dim"], num_blocks=cfg["num_blocks"],
        num_heads=cfg["num_heads"], window_size=cfg["window_size"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if verbose:
        print(f" Model params: {n_params:,}")

    def random_patch():
        d = min(pd, F_total)
        h = min(phw, H)
        w = min(phw, W)
        t0_ = np.random.randint(0, max(F_total - d, 1))
        y0 = np.random.randint(0, max(H - h, 1))
        x0 = np.random.randint(0, max(W - w, 1))
        return stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w], tmed_t[y0:y0+h, x0:x0+w]

    # ─── Stage 0: warmup ──────────────────────────────────────
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] warmup — {cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=1e-4, betas=(0.9, 0.999))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1)
        scaler = GradScaler(enabled=use_amp)
        model.train()
        rl = 0.0
        for it in range(cfg["warmup_iters"]):
            patches, targets = [], []
            for _ in range(bs):
                vol, tmed_crop = random_patch()
                aug = np.random.randint(0, 8)
                vol = _augment_3d(vol, aug)
                tmed_crop = _augment_3d(
                    tmed_crop.unsqueeze(0).expand(vol.shape[0], -1, -1), aug)
                patches.append(vol.unsqueeze(0))
                targets.append(tmed_crop.unsqueeze(0))
            inp = torch.stack(patches, dim=0).to(device)
            tgt = torch.stack(targets, dim=0).to(device)

            opt.zero_grad()
            with autocast(enabled=use_amp, dtype=torch.float16):
                pred = model(inp)
                loss = F.l1_loss(pred, tgt)
            if not torch.isfinite(loss):
                if verbose:
                    print(f"   WARN non-finite at iter {it}, skip")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            rl += loss.item()
            if verbose and (it + 1) % 50 == 0:
                print(f"   {it+1:>4}/{cfg['warmup_iters']} "
                      f"loss={rl/50:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # ─── Stage 1: N2V (pure L1 at masked voxels) ──────────────
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] 3D Noise2Void — {cfg['n2v_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg["lr"] * 0.5, weight_decay=1e-4,
                                betas=(0.9, 0.999))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6)
        scaler = GradScaler(enabled=use_amp)
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
                    radius_s=cfg["mask_radius_s"],
                    radius_t=cfg["mask_radius_t"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((mz, my, mx, orig))
            inp = torch.stack(patches, dim=0).to(device)

            opt.zero_grad()
            with autocast(enabled=use_amp, dtype=torch.float16):
                pred = model(inp)
                loss = torch.tensor(0.0, device=device)
                for b, (mz, my, mx, orig) in enumerate(all_orig):
                    pred_at_mask = pred[b, 0, mz, my, mx]
                    loss = loss + F.l1_loss(pred_at_mask, orig)
                loss = loss / bs

            if not torch.isfinite(loss):
                if verbose:
                    print(f"   WARN non-finite at iter {it}, skip")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            rl += loss.item()
            if verbose and (it + 1) % 200 == 0:
                lr_now = sch.get_last_lr()[0]
                print(f"   {it+1:>5}/{cfg['n2v_iters']} "
                      f"loss={rl/200:.5f} lr={lr_now:.2e}  "
                      f"{time.time()-t0:.1f}s")
                rl = 0.0

    elapsed = time.time() - t0
    if verbose:
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return model, cfg


# ══════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE (no time overlap, fp32, flat-frame guard)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(model, stack, config, device, verbose=True):
    """
    fp32 inference, no time overlap (each frame denoised by one patch),
    50% spatial overlap with Gaussian blending. Includes flat-frame
    safety net.
    """
    model.eval()
    model = model.float()

    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    pd  = min(config.get("patch_d", 32), F_total)
    phw = min(config.get("patch_hw", 64), H, W)
    pd  = max((pd  // 8) * 8, 8)
    phw = max((phw // 8) * 8, 8)
    stride_d  = pd                       # no time overlap (faster + cleaner)
    stride_hw = max(phw // 2, 8)         # 50% spatial overlap

    if verbose:
        print(f" Sliding window: patch={pd}x{phw}x{phw}, "
              f"stride={stride_d}x{stride_hw}x{stride_hw}")

    stack_norm = normalize(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    output_sum = torch.zeros(F_total, H, W, device=device)
    weight_sum = torch.zeros(F_total, H, W, device=device)

    # Flat on time, gaussian on space (since no time overlap)
    g_t = torch.ones(pd, device=device)
    sp = _gaussian_window_3d((1, phw, phw), sigma_frac=0.3, device=device).squeeze(0)
    gauss_win = (g_t[:, None, None] * sp[None, :, :]).clamp(min=1e-6)

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
                    patch = F.pad(patch,
                        (0, phw - aw, 0, phw - ah, 0, pd - ad), mode="reflect")
                inp = patch.unsqueeze(0).unsqueeze(0).float()
                pred = model(inp).squeeze(0).squeeze(0).float()
                pred = pred[:ad, :ah, :aw]
                win  = gauss_win[:ad, :ah, :aw]
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
    output = denormalize(output, norm_params)

    # Safety: NaN/Inf clean up + clip to input range with headroom
    output = np.nan_to_num(
        output, nan=float(stack.mean()),
        posinf=float(stack.max()), neginf=float(stack.min()),
    )
    in_lo, in_hi = float(stack.min()), float(stack.max())
    in_hi_padded = in_hi + 0.1 * max(in_hi - in_lo, 1.0)
    output = np.clip(output, in_lo, in_hi_padded)

    # Flat-frame guard
    per_frame_std = output.std(axis=(1, 2))
    flat = per_frame_std < 1.0
    if flat.any():
        n_flat = int(flat.sum())
        if verbose:
            print(f" WARN: {n_flat} flat frame(s) — replacing with input")
        output[flat] = stack[flat].astype(np.float32)

    return output


# ══════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "arch": "SwinUnet3D-GSC-FUE",
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = SwinUnet3D(
        in_channels=1, out_channels=1,
        dim=cfg.get("dim", 24),
        num_blocks=cfg.get("num_blocks", (1, 1, 1, 2)),
        num_heads=cfg.get("num_heads", (2, 2, 4, 4)),
        window_size=cfg.get("window_size", (4, 4, 4)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
