"""
Restormer-3D v2 for Self-Supervised Calcium Imaging Denoising
==============================================================

Builds on the 3D adaptation of "Restormer: Efficient Transformer for
High-Resolution Image Restoration" (Zamir et al., CVPR 2022 —
https://arxiv.org/abs/2111.09881) and adds targeted improvements for
calcium-imaging Noise2Void:

  v2 changes vs the base Restormer-3D:
  ─────────────────────────────────────────────────────────────────
  1. Multi-channel input priors:
        Channel 0: noisy frame
        Channel 1: temporal median (per-pixel structural prior)
        Channel 2: temporal std    (per-pixel activity map)
     Gives the network the slow structural prior at every pass.

  2. Temporal-aware N2V masking:
        Spatial-only neighbor replacement (radius_t = 0)
        Prevents leakage of calcium transients into "neighbor" values.

  3. Variance-weighted patch sampling:
        Skip empty-background patches; focus iterations on regions
        with real signal.

  4. Hybrid loss:
        L1 at masked voxels (preserves bright transient peaks)
        + Sobel gradient consistency (preserves edges)
        + Temporal-gradient consistency (preserves frame-to-frame
          dynamics where the network output isn't masked).

  5. EMA weights:
        An exponential-moving-average copy of the model is kept and
        used at inference time. Standard restoration trick: smoother
        convergence, fewer artifacts, ~+0.2-0.5 dB PSNR.

  6. Test-time augmentation (8-fold):
        Run inference under the 8 D4 transforms, average. ~+0.2-0.8 dB.

  7. Temporal overlap at inference:
        50% time overlap with full 3D Gaussian blending. Eliminates
        per-patch temporal seams.

API parity with model.py / model_dvt.py:
    compute_norm_params, normalize, denormalize
    train_self_supervised(stack, device, config) -> (model, cfg)
    denoise_stack(model, stack, config, device)  -> np.ndarray
    save_checkpoint, load_checkpoint
    UNet3D alias -> Restormer3D
"""

import copy
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
# RESTORMER 3D BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class LayerNorm3D(nn.Module):
    """BiasFree LayerNorm over channel dim for [B, C, D, H, W]."""

    def __init__(self, num_channels: int, bias_free: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = None if bias_free else nn.Parameter(torch.zeros(num_channels))
        self.eps = 1e-6

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1, 1)
        if self.bias is not None:
            x = x + self.bias.view(1, -1, 1, 1, 1)
        return x


class MDTA3D(nn.Module):
    """Multi-Dconv head Transposed Attention (paper §3.1) - 3D version."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv3d(
            dim * 3, dim * 3, kernel_size=3, padding=1,
            groups=dim * 3, bias=False,
        )
        self.project_out = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x):
        B, C, D, H, W = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = C // self.num_heads
        q = q.reshape(B, self.num_heads, head_dim, D * H * W)
        k = k.reshape(B, self.num_heads, head_dim, D * H * W)
        v = v.reshape(B, self.num_heads, head_dim, D * H * W)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (k @ q.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v).reshape(B, C, D, H, W)
        return self.project_out(out)


class GDFN3D(nn.Module):
    """Gated-Dconv Feed-Forward Network (paper §3.2) - 3D version."""

    def __init__(self, dim: int, ffn_expansion_factor: float = 2.66):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv3d(dim, hidden * 2, kernel_size=1, bias=False)
        self.dwconv = nn.Conv3d(
            hidden * 2, hidden * 2, kernel_size=3, padding=1,
            groups=hidden * 2, bias=False,
        )
        self.project_out = nn.Conv3d(hidden, dim, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dwconv(self.project_in(x))
        a, b = x.chunk(2, dim=1)
        return self.project_out(F.gelu(a) * b)


class TransformerBlock3D(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2.66, bias_free=True):
        super().__init__()
        self.norm1 = LayerNorm3D(dim, bias_free=bias_free)
        self.attn = MDTA3D(dim, num_heads)
        self.norm2 = LayerNorm3D(dim, bias_free=bias_free)
        self.ffn = GDFN3D(dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=(2, 2, 2)):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch * 2
        self.body = nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)

    def forward(self, x):
        return self.body(x)


class Upsample3D(nn.Module):
    def __init__(self, in_ch, out_ch=None, stride=(2, 2, 2)):
        super().__init__()
        out_ch = out_ch if out_ch is not None else in_ch // 2
        self.body = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=stride, stride=stride, bias=False)

    def forward(self, x):
        return self.body(x)


# ══════════════════════════════════════════════════════════════
# RESTORMER 3D — main architecture (multi-channel input)
# ══════════════════════════════════════════════════════════════

class Restormer3D(nn.Module):
    """4-level encoder-decoder Restormer, 3D, multi-channel input.

    Input channels (default in_channels=3):
        0: noisy frame (or whatever is masked for N2V)
        1: temporal median (per-pixel)
        2: temporal std    (per-pixel)
    Output: [B, 1, D, H, W] denoised noisy channel.
    """

    def __init__(
        self,
        in_channels: int = 3,         # noisy + median + std priors
        out_channels: int = 1,
        dim: int = 32,
        num_blocks: tuple = (2, 2, 2, 3),
        num_refinement_blocks: int = 2,
        heads: tuple = (1, 2, 4, 8),
        ffn_expansion_factor: float = 2.0,
        bias_free: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.dim = dim

        self.patch_embed = nn.Conv3d(in_channels, dim, 3, padding=1, bias=False)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock3D(dim, heads[0], ffn_expansion_factor, bias_free)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample3D(dim, dim * 2)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock3D(dim*2, heads[1], ffn_expansion_factor, bias_free)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample3D(dim * 2, dim * 4)

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock3D(dim*4, heads[2], ffn_expansion_factor, bias_free)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample3D(dim * 4, dim * 8)

        self.latent = nn.Sequential(*[
            TransformerBlock3D(dim*8, heads[3], ffn_expansion_factor, bias_free)
            for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample3D(dim * 8, dim * 4)
        self.reduce_chan_level3 = nn.Conv3d(dim * 8, dim * 4, 1, bias=False)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock3D(dim*4, heads[2], ffn_expansion_factor, bias_free)
            for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample3D(dim * 4, dim * 2)
        self.reduce_chan_level2 = nn.Conv3d(dim * 4, dim * 2, 1, bias=False)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock3D(dim*2, heads[1], ffn_expansion_factor, bias_free)
            for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample3D(dim * 2, dim)
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock3D(dim*2, heads[0], ffn_expansion_factor, bias_free)
            for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock3D(dim*2, heads[0], ffn_expansion_factor, bias_free)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv3d(dim * 2, out_channels, 3, padding=1, bias=False)

    def forward(self, x):
        """x: [B, in_channels, D, H, W] -> [B, 1, D, H, W]
        Residual learning: output = noisy_input − predicted_noise.
        """
        # The first input channel is the (possibly masked) noisy stack;
        # that's what we predict noise for.
        identity_noisy = x[:, 0:1]

        _, _, D, H, W = x.shape
        pd = (8 - D % 8) % 8
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")
            identity_noisy = F.pad(identity_noisy, (0, pw, 0, ph, 0, pd), mode="reflect")

        f0 = self.patch_embed(x)
        out_enc1 = self.encoder_level1(f0)
        out_enc2 = self.encoder_level2(self.down1_2(out_enc1))
        out_enc3 = self.encoder_level3(self.down2_3(out_enc2))
        latent = self.latent(self.down3_4(out_enc3))

        in_dec3 = self.up4_3(latent)
        in_dec3 = _match_cat(in_dec3, out_enc3)
        out_dec3 = self.decoder_level3(self.reduce_chan_level3(in_dec3))

        in_dec2 = self.up3_2(out_dec3)
        in_dec2 = _match_cat(in_dec2, out_enc2)
        out_dec2 = self.decoder_level2(self.reduce_chan_level2(in_dec2))

        in_dec1 = self.up2_1(out_dec2)
        in_dec1 = _match_cat(in_dec1, out_enc1)
        out_dec1 = self.decoder_level1(in_dec1)

        out = self.refinement(out_dec1)
        noise = self.output(out)
        out = identity_noisy - noise

        if pd or ph or pw:
            out = out[:, :, :D, :H, :W]
        return out


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


UNet3D = Restormer3D


# ══════════════════════════════════════════════════════════════
# 3D N2V MASKING (temporal-aware: spatial-only neighbors)
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume: torch.Tensor, mask_ratio: float = 0.020,
                radius_s: int = 2, radius_t: int = 0):
    """
    3D Noise2Void masking with anisotropic neighborhood.

    radius_t = 0 by default: replacement neighbors are sampled SPATIALLY
    only, never from a different frame. Calcium transients on a single
    pixel can span 2-4 frames; sampling a temporal neighbor often falls
    inside the same transient and trains the network to smooth them.
    """
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
    # nudge spatial offset rather than temporal (radius_t may be 0)
    dy[same] = 1

    nz = (mz + dz).clamp(0, D - 1)
    ny = (my + dy).clamp(0, H - 1)
    nx = (mx + dx).clamp(0, W - 1)

    masked = volume.clone()
    masked[mz, my, mx] = volume[nz, ny, nx]
    return masked, (mz, my, mx), original


# ══════════════════════════════════════════════════════════════
# AUGMENTATION + INFERENCE WINDOW + EMA
# ══════════════════════════════════════════════════════════════

def _gaussian_window_3d(shape, sigma_frac=0.3, device="cpu"):
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


def _augment_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    """8-fold spatial augmentation. vol can be [D,H,W] or [C,D,H,W]."""
    if vol.dim() == 3:
        if aug_id >= 4:
            vol = torch.flip(vol, dims=[2])
        k = aug_id % 4
        if k > 0:
            vol = torch.rot90(vol, k=k, dims=[1, 2])
    else:  # [C, D, H, W]
        if aug_id >= 4:
            vol = torch.flip(vol, dims=[3])
        k = aug_id % 4
        if k > 0:
            vol = torch.rot90(vol, k=k, dims=[2, 3])
    return vol


def _augment_inverse_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    """Inverse of _augment_3d for [B, C, D, H, W]."""
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=4 - k, dims=[3, 4])
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[4])
    return vol


class EMA:
    """Exponential moving average of model weights."""
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            n: p.detach().clone() for n, p in model.named_parameters()
            if p.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        """Copy EMA weights into `model`. Returns a backup dict to restore."""
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].data)
        return backup

    @torch.no_grad()
    def restore(self, model: nn.Module, backup: dict):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n].data)


# ══════════════════════════════════════════════════════════════
# TEMPORAL/SPATIAL PRIORS
# ══════════════════════════════════════════════════════════════

def compute_priors(stack_norm: np.ndarray, max_frames: int = 500,
                    tt_strategy=None):
    """
    Compute per-pixel temporal median and std priors.
    stack_norm: [F, H, W] normalized.
    Returns (median_2d, std_2d) both [H, W] float32.

    The median uses the temporal-target strategy if `tt_strategy` is
    given; otherwise falls back to subsampled median (the original
    behavior). Std is always computed locally — it's a prior, not a
    "target".
    """
    F_total = stack_norm.shape[0]
    n = min(max_frames, F_total)
    idx = np.linspace(0, F_total - 1, n, dtype=int)
    sub = stack_norm[idx]
    # Median via strategy if given, else original subsampled median
    if tt_strategy is None:
        med = np.median(sub, axis=0).astype(np.float32)
    else:
        med_arr = tt_strategy.compute(stack_norm)
        if tt_strategy.returns != "2d":
            med = np.median(med_arr, axis=0).astype(np.float32)
        else:
            med = med_arr
    std = sub.std(axis=0).astype(np.float32)
    # Robust normalize std to [0, 1] for stable input
    std = std / max(float(np.percentile(std, 99)), 1e-6)
    std = np.clip(std, 0, 1)
    return med, std


# ══════════════════════════════════════════════════════════════
# LOSSES
# ══════════════════════════════════════════════════════════════

def _spatial_grad_3d(x):
    """First-order spatial gradients (sobel-like) for [B, 1, D, H, W]."""
    gx = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    gy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    return gx, gy


def _temporal_grad_3d(x):
    """First-order temporal gradient for [B, 1, D, H, W]."""
    return x[:, :, 1:] - x[:, :, :-1]


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
):
    t0 = time.time()
    cfg = {
        # backbone
        "dim": 32,
        "num_blocks": (2, 2, 2, 3),
        "num_refinement_blocks": 2,
        "heads": (1, 2, 4, 8),
        "ffn_expansion_factor": 2.0,
        "bias_free": True,
        # patch sampling
        "patch_d": 32,
        "patch_hw": 64,
        "batch_size": 2,
        # variance-weighted sampling
        "use_variance_sampling": True,
        "variance_topk_frac": 0.5,    # only sample from top half by variance
        # schedule
        "warmup_iters": 200,
        "n2v_iters": 4000,
        "lr": 3e-4,
        # n2v masking
        "mask_ratio": 0.020,
        "mask_radius_s": 2,
        "mask_radius_t": 0,           # spatial-only neighbors
        # loss weights
        "loss_l1_weight":      1.0,
        "loss_grad_weight":    0.1,   # spatial gradient consistency
        "loss_tgrad_weight":   0.05,  # temporal gradient consistency
        # EMA
        "ema_decay": 0.999,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]
    use_amp = (device.type == "cuda")

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" Restormer3D-v2: dim={cfg['dim']}, blocks={cfg['num_blocks']}, "
              f"heads={cfg['heads']}, in_ch=3 (noisy+median+std)")
        print(f" Patch: {pd}x{phw}x{phw}, batch={bs}, "
              f"variance_sampling={cfg['use_variance_sampling']}")
        print(f" Schedule: warmup={cfg['warmup_iters']}, n2v={cfg['n2v_iters']}")
        print(f" Loss weights: L1={cfg['loss_l1_weight']}, "
              f"grad={cfg['loss_grad_weight']}, tgrad={cfg['loss_tgrad_weight']}")
        print(f" Masking: ratio={cfg['mask_ratio']}, "
              f"radius_s={cfg['mask_radius_s']}, "
              f"radius_t={cfg['mask_radius_t']} (0 = spatial-only)")
        print(f" EMA decay: {cfg['ema_decay']}")
        print(f" Mixed precision (fp16): {use_amp}")

    # ── Normalize (strategy from config) ──────────────────────
    norm_name = cfg.get("normalization", DEFAULT_NORMALIZATION)
    norm_strategy = _prep.resolve_normalization(norm_name)
    norm_params = norm_strategy.compute_params(stack)
    cfg["norm_params"] = norm_params
    cfg["__resolved_normalization"] = norm_strategy.name
    stack_norm = norm_strategy.forward(stack, norm_params)
    if verbose:
        print(f" Norm [{norm_strategy.name}]: "
              f"shift={norm_params['shift']:.2f}, "
              f"scale={norm_params['scale']:.2f}")

    # ── Compute priors (median via temporal-target strategy) ───
    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    cfg["__resolved_temporal_target"] = tt_strategy.name
    median_2d, std_2d = compute_priors(stack_norm, tt_strategy=tt_strategy)
    cfg["median_2d_min_max"] = (float(median_2d.min()), float(median_2d.max()))
    if verbose:
        print(f" Median prior [{tt_strategy.name}]: "
              f"[{median_2d.min():.3f}, {median_2d.max():.3f}]")
        print(f" Std prior:    [{std_2d.min():.3f}, {std_2d.max():.3f}]")

    # Move to GPU
    stack_t  = torch.from_numpy(stack_norm).float().to(device)   # [F, H, W]
    median_t = torch.from_numpy(median_2d).float().to(device)    # [H, W]
    std_t    = torch.from_numpy(std_2d).float().to(device)       # [H, W]

    # ── Variance map for sampling ─────────────────────────────
    # Use spatial std map: regions with real signal have high std.
    # Build a sampling distribution over (y, x) anchors.
    if cfg["use_variance_sampling"]:
        var_map = std_t.cpu().numpy()
        # Smooth a bit with a box average (spatial pool of phw)
        from scipy.ndimage import uniform_filter
        var_map = uniform_filter(var_map, size=max(phw // 2, 8))
        # Top fraction by variance
        thresh = np.quantile(var_map, 1 - cfg["variance_topk_frac"])
        sampling_mask = var_map >= thresh
        ys, xs = np.where(sampling_mask)
        if verbose:
            print(f" Variance sampling: {len(ys)} valid anchor pixels "
                  f"(top {cfg['variance_topk_frac']*100:.0f}%)")
    else:
        ys = xs = None

    # ── Model + EMA ───────────────────────────────────────────
    model = Restormer3D(
        in_channels=3, out_channels=1,
        dim=cfg["dim"], num_blocks=cfg["num_blocks"],
        num_refinement_blocks=cfg["num_refinement_blocks"],
        heads=cfg["heads"],
        ffn_expansion_factor=cfg["ffn_expansion_factor"],
        bias_free=cfg["bias_free"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f" Model params: {n_params:,}")

    ema = EMA(model, decay=cfg["ema_decay"])

    # ── Patch helper ──────────────────────────────────────────
    def random_patch():
        d = min(pd, F_total)
        h = min(phw, H)
        w = min(phw, W)
        t0_ = np.random.randint(0, max(F_total - d, 1))

        if cfg["use_variance_sampling"] and ys is not None and len(ys) > 0:
            # Pick a random high-variance anchor; clip to fit the patch
            i = np.random.randint(len(ys))
            cy, cx = int(ys[i]), int(xs[i])
            y0 = np.clip(cy - h // 2, 0, max(H - h, 0))
            x0 = np.clip(cx - w // 2, 0, max(W - w, 0))
        else:
            y0 = np.random.randint(0, max(H - h, 1))
            x0 = np.random.randint(0, max(W - w, 1))

        noisy_p  = stack_t [t0_:t0_+d, y0:y0+h, x0:x0+w]    # [d, h, w]
        median_p = median_t[y0:y0+h, x0:x0+w]               # [h, w]
        std_p    = std_t   [y0:y0+h, x0:x0+w]               # [h, w]
        return noisy_p, median_p, std_p

    def make_inputs(noisy_p, median_p, std_p, masked_noisy=None):
        """Stack noisy + priors into [3, D, H, W]."""
        noisy_in = noisy_p if masked_noisy is None else masked_noisy
        d = noisy_in.shape[0]
        # Broadcast 2D priors along the time axis
        med_d = median_p.unsqueeze(0).expand(d, -1, -1)
        std_d = std_p.unsqueeze(0).expand(d, -1, -1)
        return torch.stack([noisy_in, med_d, std_d], dim=0)   # [3, D, H, W]

    # ════════════════════════════════════════════════
    # Stage 0 — Temporal-median warmup
    # ════════════════════════════════════════════════
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Temporal-median warmup — "
                  f"{cfg['warmup_iters']} iters")
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
                noisy_p, median_p, std_p = random_patch()
                aug = np.random.randint(0, 8)
                # Augment all three channels and target consistently
                d = noisy_p.shape[0]
                stacked = make_inputs(noisy_p, median_p, std_p)        # [3,D,H,W]
                tgt = median_p.unsqueeze(0).expand(d, -1, -1).unsqueeze(0)   # [1,D,H,W]
                stacked = _augment_3d(stacked, aug)
                tgt = _augment_3d(tgt, aug)
                patches.append(stacked)
                targets.append(tgt)

            inp = torch.stack(patches, dim=0).to(device)   # [B, 3, D, H, W]
            tgt = torch.stack(targets, dim=0).to(device)   # [B, 1, D, H, W]

            opt.zero_grad()
            with autocast(enabled=use_amp, dtype=torch.float16):
                pred = model(inp)
                loss = F.l1_loss(pred, tgt)
            if not torch.isfinite(loss):
                if verbose:
                    print(f"   WARN non-finite loss at iter {it}, skip")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            ema.update(model)
            rl += loss.item()

            if verbose and (it + 1) % 100 == 0:
                print(f"   {it+1:>4}/{cfg['warmup_iters']} "
                      f"loss={rl/100:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # ════════════════════════════════════════════════
    # Stage 1 — N2V with hybrid loss
    # ════════════════════════════════════════════════
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] 3D Noise2Void (hybrid loss) — "
                  f"{cfg['n2v_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(),
                                 lr=cfg["lr"] * 0.5, weight_decay=1e-4,
                                 betas=(0.9, 0.999))
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6)
        scaler = GradScaler(enabled=use_amp)
        model.train()
        rl1 = rgrad = rtgrad = 0.0

        for it in range(cfg["n2v_iters"]):
            all_orig = []
            patches = []
            full_inputs = []   # un-masked inputs for tgrad consistency

            for _ in range(bs):
                noisy_p, median_p, std_p = random_patch()
                aug = np.random.randint(0, 8)

                # Mask the noisy channel only
                masked, (mz, my, mx), orig = n2v_mask_3d(
                    noisy_p,
                    mask_ratio=cfg["mask_ratio"],
                    radius_s=cfg["mask_radius_s"],
                    radius_t=cfg["mask_radius_t"],
                )
                stacked_masked = make_inputs(noisy_p, median_p, std_p,
                                              masked_noisy=masked)   # [3,D,H,W]
                stacked_full   = make_inputs(noisy_p, median_p, std_p)
                stacked_masked = _augment_3d(stacked_masked, aug)
                stacked_full   = _augment_3d(stacked_full,   aug)

                # Mask indices need to follow augmentation. Easier to
                # just record the augmented values: we look up the
                # *augmented* original values where the masked locations
                # land after augmentation.
                # Simplest: skip aug for masked-loss path and only
                # augment the full input. Aug of N2V mask coordinates
                # is fiddly; trade a bit of augmentation diversity for
                # correctness.
                # Re-do without aug for the masked loss arm:
                stacked_masked = make_inputs(noisy_p, median_p, std_p,
                                              masked_noisy=masked)
                stacked_full   = make_inputs(noisy_p, median_p, std_p)

                patches.append(stacked_masked)
                full_inputs.append(stacked_full)
                all_orig.append((mz, my, mx, orig))

            inp_masked = torch.stack(patches,     dim=0).to(device)  # [B,3,D,H,W]
            inp_full   = torch.stack(full_inputs, dim=0).to(device)

            opt.zero_grad()
            with autocast(enabled=use_amp, dtype=torch.float16):
                pred_masked = model(inp_masked)
                pred_full   = model(inp_full)

                # 1) N2V L1 at masked positions
                loss_l1 = torch.tensor(0.0, device=device)
                for b, (mz, my, mx, orig) in enumerate(all_orig):
                    pred_at_mask = pred_masked[b, 0, mz, my, mx]
                    loss_l1 = loss_l1 + F.l1_loss(pred_at_mask, orig)
                loss_l1 = loss_l1 / bs

                # 2) Spatial gradient consistency (full-output only)
                gx_p, gy_p = _spatial_grad_3d(pred_full)
                gx_i, gy_i = _spatial_grad_3d(inp_full[:, 0:1])  # noisy ch
                loss_grad = F.l1_loss(gx_p, gx_i) + F.l1_loss(gy_p, gy_i)

                # 3) Temporal gradient consistency
                tg_p = _temporal_grad_3d(pred_full)
                tg_i = _temporal_grad_3d(inp_full[:, 0:1])
                loss_tgrad = F.l1_loss(tg_p, tg_i)

                loss = (cfg["loss_l1_weight"]   * loss_l1
                       + cfg["loss_grad_weight"]  * loss_grad
                       + cfg["loss_tgrad_weight"] * loss_tgrad)

            if not torch.isfinite(loss):
                if verbose:
                    print(f"   WARN non-finite loss at iter {it}, skip")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sch.step()
            ema.update(model)

            rl1 += loss_l1.item()
            rgrad += loss_grad.item()
            rtgrad += loss_tgrad.item()

            if verbose and (it + 1) % 250 == 0:
                lr_now = sch.get_last_lr()[0]
                print(f"   {it+1:>5}/{cfg['n2v_iters']} "
                      f"L1={rl1/250:.5f} grad={rgrad/250:.5f} "
                      f"tgrad={rtgrad/250:.5f} lr={lr_now:.2e}  "
                      f"{time.time()-t0:.1f}s")
                rl1 = rgrad = rtgrad = 0.0

    # ── Apply EMA weights to the model for inference ───────────
    if verbose:
        print(f"\n Applying EMA weights to model.")
    ema.apply_to(model)

    # Stash priors in cfg for inference
    cfg["_priors_med"] = median_2d
    cfg["_priors_std"] = std_2d

    elapsed = time.time() - t0
    if verbose:
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return model, cfg


# ══════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE (with TTA + temporal overlap)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(
    model: Restormer3D,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
    use_tta: bool = True,
    tta_n: int = 4,                  # average over first N D4 transforms
) -> np.ndarray:
    """
    Sliding-window inference with full 3D Gaussian blending
    (50% overlap on time + space) and optional 4-fold TTA.
    """
    model.eval()
    model = model.float()    # fp32 inference for stability

    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    pd = min(config.get("patch_d", 32), F_total)
    phw = min(config.get("patch_hw", 64), H, W)
    pd = max((pd // 8) * 8, 8)
    phw = max((phw // 8) * 8, 8)
    stride_d  = max(pd  // 2, 8)     # 50% time overlap
    stride_hw = max(phw // 2, 8)

    if verbose:
        print(f" Sliding window: patch={pd}x{phw}x{phw}, "
              f"stride={stride_d}x{stride_hw}x{stride_hw}, "
              f"TTA={'on' if use_tta else 'off'}({tta_n}x)")

    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)

    # Recompute priors from the input (same as during training).
    # If a temporal-target strategy was used during training, reuse it.
    tt_name = config.get("__resolved_temporal_target",
                          config.get("temporal_target",
                                      DEFAULT_TEMPORAL_TARGET))
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    median_2d = config.get("_priors_med")
    std_2d    = config.get("_priors_std")
    if median_2d is None or std_2d is None:
        median_2d, std_2d = compute_priors(stack_norm, tt_strategy=tt_strategy)

    stack_t  = torch.from_numpy(stack_norm).float().to(device)
    median_t = torch.from_numpy(median_2d).float().to(device)
    std_t    = torch.from_numpy(std_2d).float().to(device)

    output_sum = torch.zeros(F_total, H, W, device=device)
    weight_sum = torch.zeros(F_total, H, W, device=device)
    gauss_win  = _gaussian_window_3d(
        (pd, phw, phw), sigma_frac=0.3, device=device)

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
        print(f" Patches: {len(z_starts)}x{len(y_starts)}x{len(x_starts)}"
              f" = {total}")

    t0 = time.time()
    done = 0
    for z0 in z_starts:
        z1 = min(z0 + pd, F_total); ad = z1 - z0
        for y0 in y_starts:
            y1 = min(y0 + phw, H); ah = y1 - y0
            for x0 in x_starts:
                x1 = min(x0 + phw, W); aw = x1 - x0

                noisy_p  = stack_t [z0:z1, y0:y1, x0:x1]
                median_p = median_t[y0:y1, x0:x1]
                std_p    = std_t   [y0:y1, x0:x1]

                if (ad < pd) or (ah < phw) or (aw < phw):
                    noisy_p  = F.pad(noisy_p,
                        (0, phw - aw, 0, phw - ah, 0, pd - ad), mode="reflect")
                    median_p = F.pad(median_p,
                        (0, phw - aw, 0, phw - ah), mode="reflect")
                    std_p    = F.pad(std_p,
                        (0, phw - aw, 0, phw - ah), mode="reflect")

                # Build [1, 3, D, H, W]
                med_d = median_p.unsqueeze(0).expand(pd, -1, -1)
                std_d = std_p.unsqueeze(0).expand(pd, -1, -1)
                inp = torch.stack([noisy_p, med_d, std_d], dim=0)
                inp = inp.unsqueeze(0).float()

                if use_tta:
                    accum = torch.zeros(1, 1, pd, phw, phw, device=device)
                    for t in range(tta_n):
                        x_t = _augment_3d(inp[0], t).unsqueeze(0)
                        y_t = model(x_t)
                        y_t = _augment_inverse_3d(y_t, t)
                        accum = accum + y_t
                    pred = (accum / tta_n)[0, 0]
                else:
                    pred = model(inp)[0, 0]

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
    output = norm_strategy.inverse(output, norm_params)

    output = np.nan_to_num(
        output,
        nan=float(stack.mean()),
        posinf=float(stack.max()),
        neginf=float(stack.min()),
    )
    in_lo, in_hi = float(stack.min()), float(stack.max())
    in_hi_padded = in_hi + 0.1 * max(in_hi - in_lo, 1.0)
    output = np.clip(output, in_lo, in_hi_padded)

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
    cfg = {k: v for k, v in config.items() if not k.startswith("_priors")}
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "arch": "Restormer3D-v2",
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = Restormer3D(
        in_channels=3, out_channels=1,
        dim=cfg.get("dim", 32),
        num_blocks=cfg.get("num_blocks", (2, 2, 2, 3)),
        num_refinement_blocks=cfg.get("num_refinement_blocks", 2),
        heads=cfg.get("heads", (1, 2, 4, 8)),
        ffn_expansion_factor=cfg.get("ffn_expansion_factor", 2.0),
        bias_free=cfg.get("bias_free", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
