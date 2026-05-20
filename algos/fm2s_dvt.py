"""
FM2S + DVT-Temporal-ViT Denoiser for Calcium Imaging
=====================================================

Combines:
  * FM2S spatial CNN — Wang et al. (https://arxiv.org/abs/2412.10031)
    "Fluorescence Micrograph to Self": a tiny 3-layer 2D CNN trained
    per-image on synthesized Poisson-Gaussian noise pairs whose
    parameters are estimated from the noisy input itself. Strong
    per-frame baseline; preserves bright transients well.

  * DVT-style Temporal ViT — Yang et al. (https://arxiv.org/abs/2401.02957)
    "Denoising Vision Transformers". Three-way feature decomposition
        ViT(x) ≈ f(x) + g(E_pos) + h(x, E_pos)
    realized by a learnable artifact field G over temporal positions,
    a 3-layer residual MLP h_ψ, and a single-Transformer-block denoiser
    with new learnable positional embeddings (DVT Tab. 6 row d).
    Applied as a corrective ADDED to the FM2S spatial output.

Architecture:
    For each frame t:
        spatial_pred = FM2S_CNN(noisy[t])
        # Temporal window around t (T frames, e.g. T=11)
        ctx = noisy[t-k : t+k+1]
        # Per-pixel temporal attention with DVT decomposition
        temp_correction = TemporalDVT(ctx)[center]
        final[t] = spatial_pred + alpha * temp_correction
    where alpha is a learnable scalar (init ~0).

The per-pixel temporal attention is what makes this efficient: each
spatial location (y, x) runs its own T-token sequence through the ViT,
sharing weights but not state. Complexity is O(H·W · T²·d) instead of
the O((H·W·T)²·d) of full 3D attention. With H=W=64, T=11, d=24 this
is roughly 50 MFLOPs per patch.

Training (per-stack, zero-shot):
    Stage 0: Train FM2S CNN alone (synthesized noise → temporal median)
    Stage 1: Freeze FM2S. Train Temporal ViT to minimize
                 L1( spatial_pred + α·temp_correction,  temporal_median )
             + temporal-N2V mask loss on the noisy input.

API parity with model.py / model_dvt.py:
    compute_norm_params, normalize, denormalize
    train_self_supervised(stack, device, config) -> (model, cfg)
    denoise_stack(model, stack, config, device)  -> np.ndarray
    save_checkpoint, load_checkpoint
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
# COMPONENT 1 — FM2S spatial CNN (paper architecture)
# ══════════════════════════════════════════════════════════════

class FM2SNet(nn.Module):
    """
    The 3-layer 2D CNN from the FM2S paper (§3.4.1):
        Conv 5x5 (n_chan -> 24) + LeakyReLU
        Conv 3x3 (24 -> 12)     + LeakyReLU
        Conv 5x5 (12 -> n_chan)
    Channel-multiplexing: input is replicated to n_chan channels, output
    is averaged back to 1 channel.
    """
    def __init__(self, n_chan: int = 5, leak: float = 1e-3):
        super().__init__()
        self.n_chan = n_chan
        self.conv1 = nn.Conv2d(n_chan, 24, kernel_size=5, padding=2, bias=True)
        self.conv2 = nn.Conv2d(24, 12, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(12, n_chan, kernel_size=5, padding=2, bias=True)
        self.act = nn.LeakyReLU(negative_slope=leak, inplace=True)

    def forward(self, x):
        """
        x: [B, 1, H, W] -> [B, 1, H, W]
        Internally replicates to n_chan channels, averages output.
        """
        # Replicate the single channel
        x_rep = x.repeat(1, self.n_chan, 1, 1)
        h = self.act(self.conv1(x_rep))
        h = self.act(self.conv2(h))
        y = self.conv3(h)
        # Average back to 1 channel
        return y.mean(dim=1, keepdim=True)


def fm2s_noise_injection(clean: torch.Tensor, lam_p: float,
                          g_map: torch.Tensor, p_map: torch.Tensor,
                          stride: int = 5) -> torch.Tensor:
    """
    FM2S §3.4.2 synthesized noise. For each `stride x stride` region we
    sample a Poisson rate from a per-image map and additive Gaussian
    noise with per-image sigma. We use simplified scalar maps adapted
    from the noise level estimated by MAD on the noisy input.

    clean: [B, 1, H, W]
    lam_p: scalar Poisson scale
    g_map, p_map: [H, W] tensors of Gaussian sigma and Poisson scale
    """
    B, _, H, W = clean.shape
    # Poisson component: scale-based shot noise
    rate = (clean.clamp(min=0) * lam_p + 1e-6)
    poisson = torch.poisson(rate) / lam_p - clean.clamp(min=0)
    # Gaussian component: per-pixel sigma from g_map
    gauss = torch.randn_like(clean) * g_map.unsqueeze(0).unsqueeze(0)
    return clean + poisson + gauss


def fm2s_estimate_noise(noisy_2d: np.ndarray):
    """Cheap MAD-based sigma estimate on a 2D image."""
    flat = noisy_2d.ravel().astype(np.float32)
    med = float(np.median(flat))
    mad = float(np.median(np.abs(flat - med))) + 1e-6
    sigma = 1.4826 * mad
    return sigma


# ══════════════════════════════════════════════════════════════
# COMPONENT 2 — DVT-style Temporal ViT
# ══════════════════════════════════════════════════════════════

class _MultiHeadAttn1D(nn.Module):
    """Pre-LN multi-head self-attention over a 1D sequence (no bias)."""
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, D)
        return self.proj(out)


class TransformerBlock1D(nn.Module):
    """Pre-LN transformer block with MHA + GELU MLP (bias-free)."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.attn = _MultiHeadAttn1D(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim, bias=False)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualMLP3(nn.Module):
    """3-layer MLP h_ψ from DVT §A.1 (hidden = dim/4)."""
    def __init__(self, dim: int):
        super().__init__()
        h = max(dim // 4, 8)
        self.net = nn.Sequential(
            nn.Linear(dim, h, bias=False), nn.ReLU(inplace=True),
            nn.Linear(h, h, bias=False), nn.ReLU(inplace=True),
            nn.Linear(h, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class TemporalDVT(nn.Module):
    """
    Per-pixel temporal ViT with DVT decomposition.

    Input:  [B, 1, T, H, W]  — T temporal samples per (y, x) pixel
    Output: [B, 1, H, W]      — temporal correction at center frame

    For each spatial position (y, x), the T-step time series of pixel
    values is:
      1. Lifted to dim D via a small projector (Linear from 1 to D after
         input-pixel + global-stats features).
      2. Passed through `n_vit_blocks` Transformer blocks (the "ViT
         output" y in DVT Eq. 10).
      3. Decomposed:
              clean_tokens = D_ζ(y - G + post_PE)        # DVT Tab. 6d
              residual     = h_ψ(y)                       # 3-layer MLP
              out_tokens   = clean_tokens + 0.1 * residual
      4. Projected back to scalar with a small head.
      5. The center-frame scalar becomes the output correction.

    Weights are SHARED across all (y, x) — only T matters, not H·W.
    """
    def __init__(
        self,
        n_temporal: int = 11,
        dim: int = 24,
        n_heads: int = 4,
        n_vit_blocks: int = 2,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.T = n_temporal
        self.dim = dim
        self.center = n_temporal // 2

        # Token embed: each "token" represents a (frame_t, pixel) value.
        # We give it: the pixel value at this frame, the offset from
        # the temporal median (per-frame stat), and the temporal index.
        # Three scalars -> Linear -> dim.
        self.token_proj = nn.Linear(3, dim, bias=False)

        # Pre-denoiser ViT blocks
        self.vit_blocks = nn.ModuleList([
            TransformerBlock1D(dim, n_heads, mlp_ratio)
            for _ in range(n_vit_blocks)
        ])

        # DVT artifact field G: input-independent, position-keyed.
        # Shape [1, T, dim]; truncated-normal init per the paper.
        self.artifact_field = nn.Parameter(torch.zeros(1, n_temporal, dim))
        nn.init.trunc_normal_(self.artifact_field, std=0.02)

        # Residual predictor h_ψ
        self.residual_predictor = ResidualMLP3(dim)

        # Single-Transformer-block denoiser with new learnable PE
        # (DVT Table 6 row d)
        self.denoiser_pe = nn.Parameter(torch.zeros(1, n_temporal, dim))
        nn.init.trunc_normal_(self.denoiser_pe, std=0.02)
        self.denoiser_block = TransformerBlock1D(dim, n_heads, mlp_ratio)

        # Output head: dim -> 1
        self.output_head = nn.Linear(dim, 1, bias=False)

        # Learnable mixing scalar α (init small so the correction starts
        # near zero — model degrades gracefully to "just spatial").
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, ctx, spatial_pred):
        """
        ctx:          [B, 1, T, H, W]  noisy context window
        spatial_pred: [B, 1, H, W]      FM2S output at center frame

        Returns: [B, 1, H, W] final denoised image.
        """
        B, _, T, H, W = ctx.shape
        assert T == self.T, f"Expected T={self.T} temporal frames, got {T}"

        # Build token features. For each (y, x) we have T values:
        #   - the noisy value at frame t
        #   - the value minus the temporal median (per-pixel-along-T)
        #   - the temporal index normalized to [-1, 1]
        x_flat = ctx[:, 0]                                    # [B, T, H, W]
        med_per_pixel = x_flat.median(dim=1, keepdim=True).values   # [B, 1, H, W]
        delta = x_flat - med_per_pixel                              # [B, T, H, W]

        # Temporal index feature: [T] -> [1, T, 1, 1]
        t_idx = torch.linspace(-1.0, 1.0, T, device=ctx.device)
        t_idx = t_idx.view(1, T, 1, 1).expand(B, T, H, W)

        # Stack the three feature channels: [B, T, 3, H, W]
        feats = torch.stack([x_flat, delta, t_idx], dim=2)

        # Reshape so each (y, x) is its own sequence of T tokens:
        # [B, T, 3, H, W] -> [B*H*W, T, 3]
        feats = feats.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, 3)

        # Token projection: [B*HW, T, 3] -> [B*HW, T, D]
        y = self.token_proj(feats)

        # Pre-denoiser ViT
        for blk in self.vit_blocks:
            y = blk(y)

        # DVT decomposition (Eq. 10):
        #   y_minus_g = y − G              (input-independent artifact)
        #   y_for_dec = y_minus_g + PE_d   (post-PE for denoiser)
        #   F_clean   = D_ζ(y_for_dec)
        #   ĥ         = h_ψ(y)
        #   tokens    = F_clean + 0.1 · ĥ
        y_minus_g = y - self.artifact_field
        y_for_dec = y_minus_g + self.denoiser_pe
        F_clean = self.denoiser_block(y_for_dec)
        h_res = self.residual_predictor(y)
        tokens = F_clean + 0.1 * h_res

        # Extract the center-frame token and project to scalar correction
        center_token = tokens[:, self.center]                  # [B*HW, D]
        correction_flat = self.output_head(center_token)        # [B*HW, 1]

        # Reshape back to [B, 1, H, W]
        correction = correction_flat.reshape(B, H, W, 1).permute(0, 3, 1, 2)

        # Combine: spatial + α · temporal_correction
        return spatial_pred + self.alpha * correction


# ══════════════════════════════════════════════════════════════
# COMBINED MODEL
# ══════════════════════════════════════════════════════════════

class FM2S_DVT_Denoiser(nn.Module):
    """
    Wraps FM2S CNN and Temporal DVT. Forward expects:
        noisy_ctx:  [B, 1, T, H, W]  temporal window
    and returns:
        [B, 1, H, W]  denoised center frame
    """
    def __init__(
        self,
        n_temporal: int = 11,
        fm2s_chan: int = 5,
        vit_dim: int = 24,
        vit_heads: int = 4,
        vit_blocks: int = 2,
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.fm2s = FM2SNet(n_chan=fm2s_chan)
        self.tdvt = TemporalDVT(
            n_temporal=n_temporal, dim=vit_dim,
            n_heads=vit_heads, n_vit_blocks=vit_blocks,
            mlp_ratio=mlp_ratio,
        )
        self.T = n_temporal
        self.center = n_temporal // 2

    def forward(self, ctx, fm2s_only: bool = False):
        """
        ctx:        [B, 1, T, H, W]
        fm2s_only:  if True, skip temporal path (used in Stage 0 training)
        """
        center_frame = ctx[:, :, self.center]                  # [B, 1, H, W]
        spatial = self.fm2s(center_frame)
        if fm2s_only:
            return spatial
        return self.tdvt(ctx, spatial)


# Alias for drop-in compatibility
UNet3D = FM2S_DVT_Denoiser


# ══════════════════════════════════════════════════════════════
# DATA HELPERS — temporal patches, N2V masking
# ══════════════════════════════════════════════════════════════

def _sample_temporal_patch(stack_t, T_window, patch_hw):
    """
    Sample a [T_window, h, w] crop from stack_t, centered at a random
    (y, x) location and a random center frame t_c.

    Returns: (ctx [T, h, w], center_t int, y0 int, x0 int)
    """
    F_, H, W = stack_t.shape
    k = T_window // 2
    h = min(patch_hw, H)
    w = min(patch_hw, W)
    t_c = np.random.randint(k, F_ - k - 1)
    y0 = np.random.randint(0, max(H - h, 1))
    x0 = np.random.randint(0, max(W - w, 1))
    ctx = stack_t[t_c - k : t_c + k + 1, y0:y0+h, x0:x0+w]
    return ctx, t_c, y0, x0


def _n2v_mask_2d(frame: torch.Tensor, mask_ratio: float = 0.020,
                  radius_s: int = 2):
    """
    2D Noise2Void mask for a [H, W] frame. Used to supervise the FM2S
    CNN in Stage 0 (so it can train on the raw noisy frame without a
    clean target).
    """
    H, W = frame.shape
    n_vox = H * W
    n_mask = max(int(n_vox * mask_ratio), 1)
    flat_idx = torch.randperm(n_vox, device=frame.device)[:n_mask]
    my = flat_idx // W
    mx = flat_idx % W
    original = frame[my, mx].clone()
    dy = torch.randint(-radius_s, radius_s + 1, (n_mask,),
                        device=frame.device)
    dx = torch.randint(-radius_s, radius_s + 1, (n_mask,),
                        device=frame.device)
    same = (dy == 0) & (dx == 0)
    dy[same] = 1
    ny = (my + dy).clamp(0, H - 1)
    nx = (mx + dx).clamp(0, W - 1)
    masked = frame.clone()
    masked[my, mx] = frame[ny, nx]
    return masked, (my, mx), original


def _n2v_mask_time(ctx: torch.Tensor, mask_ratio: float = 0.05):
    """
    Mask the CENTER FRAME of a [T, H, W] context by replacing some
    pixels with their TEMPORAL neighbor (one frame earlier). The
    network must predict the original from temporal context.
    """
    T, H, W = ctx.shape
    center = T // 2
    if center == 0:
        return ctx, None, None
    n_vox = H * W
    n_mask = max(int(n_vox * mask_ratio), 1)
    flat_idx = torch.randperm(n_vox, device=ctx.device)[:n_mask]
    my = flat_idx // W
    mx = flat_idx % W
    original = ctx[center, my, mx].clone()
    # Replace each masked pixel with the value from frame center-1
    masked = ctx.clone()
    masked[center, my, mx] = ctx[center - 1, my, mx]
    return masked, (my, mx), original


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_self_supervised(stack, device, config=None, verbose=True):
    t0 = time.time()
    cfg = {
        # FM2S
        "fm2s_chan":      5,
        # Temporal ViT
        "T_window":       11,
        "vit_dim":        24,
        "vit_heads":      4,
        "vit_blocks":     2,
        "mlp_ratio":      2.0,
        # patches
        "patch_hw":       64,
        "batch_size":     2,
        # schedule
        "fm2s_iters":     800,
        "vit_iters":      1200,
        "lr_fm2s":        1e-3,
        "lr_vit":         3e-4,
        # n2v / aux
        "mask_ratio":     0.020,
        "mask_radius_s":  2,
        "vit_mask_ratio": 0.05,
        # loss mixing in Stage 1
        "loss_median_weight": 1.0,    # match temporal median
        "loss_n2v_weight":    0.5,    # temporal-N2V auxiliary
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    T_window = cfg["T_window"]
    phw = cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" FM2S CNN: n_chan={cfg['fm2s_chan']}")
        print(f" Temporal DVT: T={T_window}, dim={cfg['vit_dim']}, "
              f"heads={cfg['vit_heads']}, blocks={cfg['vit_blocks']}")
        print(f" Patch (H,W)={phw}, batch={bs}")
        print(f" Schedule: fm2s={cfg['fm2s_iters']}, vit={cfg['vit_iters']}")
        print(f" Precision: fp32")

    # ── Normalize (strategy from config) ──────────────────────
    norm_name = cfg.get("normalization", DEFAULT_NORMALIZATION)
    norm_strategy = _prep.resolve_normalization(norm_name)
    norm_params = norm_strategy.compute_params(stack)
    cfg["norm_params"] = norm_params
    cfg["__resolved_normalization"] = norm_strategy.name
    stack_norm = norm_strategy.forward(stack, norm_params)

    # ── Temporal target (strategy from config) ────────────────
    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    cfg["__resolved_temporal_target"] = tt_strategy.name
    if tt_strategy.returns != "2d":
        _tt_3d = tt_strategy.compute(stack_norm)
        temporal_med = np.median(_tt_3d, axis=0).astype(np.float32)
    else:
        temporal_med = tt_strategy.compute(stack_norm)
    if verbose:
        print(f" Norm [{norm_strategy.name}], "
              f"temporal target [{tt_strategy.name}]: "
              f"[{temporal_med.min():.3f}, {temporal_med.max():.3f}]")

    stack_t = torch.from_numpy(stack_norm).float().to(device)
    tmed_t  = torch.from_numpy(temporal_med).float().to(device)

    # ── Build model ───────────────────────────────────────────
    model = FM2S_DVT_Denoiser(
        n_temporal=T_window, fm2s_chan=cfg["fm2s_chan"],
        vit_dim=cfg["vit_dim"], vit_heads=cfg["vit_heads"],
        vit_blocks=cfg["vit_blocks"], mlp_ratio=cfg["mlp_ratio"],
    ).to(device)
    n_fm2s = sum(p.numel() for p in model.fm2s.parameters())
    n_tdvt = sum(p.numel() for p in model.tdvt.parameters())
    if verbose:
        print(f" Params: FM2S={n_fm2s:,}, TemporalDVT={n_tdvt:,}, "
              f"total={n_fm2s+n_tdvt:,}")

    # ────────────────────────────────────────────────
    # Stage 0 — Train FM2S CNN (per the FM2S paper)
    # ────────────────────────────────────────────────
    if cfg["fm2s_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] FM2S CNN — {cfg['fm2s_iters']} iters")

        # Estimate per-pixel Gaussian sigma from MAD on a sample frame
        sigma = fm2s_estimate_noise(stack_norm[F_total // 2])
        if verbose:
            print(f"   estimated σ (MAD): {sigma:.4f}")
        g_map = torch.full((H, W), sigma, device=device)
        p_map = torch.full((H, W), 1.0, device=device)
        lam_p = 5.0     # Poisson scaling; gives mild shot noise

        opt = torch.optim.Adam(model.fm2s.parameters(),
                                lr=cfg["lr_fm2s"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["fm2s_iters"], eta_min=cfg["lr_fm2s"] * 0.1)
        model.fm2s.train()
        rl = 0.0

        for it in range(cfg["fm2s_iters"]):
            patches_in, patches_tgt = [], []
            for _ in range(bs):
                y0 = np.random.randint(0, max(H - phw, 1))
                x0 = np.random.randint(0, max(W - phw, 1))
                # Use the temporal median as the *clean* target (this is
                # the FM2S "supervisor"). Inject synth noise on top.
                clean_crop = tmed_t[y0:y0+phw, x0:x0+phw]    # [phw, phw]
                clean_in = clean_crop.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
                noisy_in = fm2s_noise_injection(
                    clean_in, lam_p, g_map[:phw, :phw], p_map[:phw, :phw],
                )
                patches_in.append(noisy_in.squeeze(0))
                patches_tgt.append(clean_in.squeeze(0))
            inp = torch.stack(patches_in, dim=0).to(device)
            tgt = torch.stack(patches_tgt, dim=0).to(device)

            opt.zero_grad()
            pred = model.fm2s(inp)
            loss = F.l1_loss(pred, tgt)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.fm2s.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()
            if verbose and (it + 1) % 200 == 0:
                print(f"   {it+1:>4}/{cfg['fm2s_iters']} "
                      f"loss={rl/200:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # ────────────────────────────────────────────────
    # Stage 1 — Freeze FM2S, train Temporal DVT
    # ────────────────────────────────────────────────
    if cfg["vit_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] Temporal DVT — {cfg['vit_iters']} iters")

        # Freeze FM2S
        for p in model.fm2s.parameters():
            p.requires_grad_(False)
        model.fm2s.eval()

        opt = torch.optim.AdamW(
            model.tdvt.parameters(), lr=cfg["lr_vit"],
            weight_decay=1e-4, betas=(0.9, 0.999),
        )
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["vit_iters"], eta_min=1e-6)
        model.tdvt.train()
        rm = rn = 0.0

        k = T_window // 2

        for it in range(cfg["vit_iters"]):
            patches_ctx, patches_tgt, n2v_targets = [], [], []
            for _ in range(bs):
                ctx, t_c, y0, x0 = _sample_temporal_patch(
                    stack_t, T_window, phw,
                )                                            # [T, h, w]
                # Target: temporal median at the SAME spatial crop
                tgt_2d = tmed_t[y0:y0+ctx.shape[1], x0:x0+ctx.shape[2]]

                # Optionally apply temporal-N2V mask
                ctx_in, mask_idx, orig = _n2v_mask_time(
                    ctx, mask_ratio=cfg["vit_mask_ratio"],
                )
                patches_ctx.append(ctx_in.unsqueeze(0))      # [1, T, h, w]
                patches_tgt.append(tgt_2d.unsqueeze(0))      # [1, h, w]
                n2v_targets.append((mask_idx, orig))

            inp = torch.stack(patches_ctx, dim=0).to(device)  # [B,1,T,h,w]
            inp = inp.unsqueeze(1) if inp.dim() == 4 else inp
            # Sanity: ensure [B, 1, T, H, W]
            if inp.dim() == 4:
                inp = inp.unsqueeze(1)
            tgt = torch.stack(patches_tgt, dim=0).to(device)  # [B, 1, h, w]

            opt.zero_grad()
            pred = model(inp)                             # [B, 1, h, w]
            # Loss 1: match temporal median
            loss_med = F.l1_loss(pred, tgt)

            # Loss 2: temporal-N2V at masked positions
            loss_n2v = torch.tensor(0.0, device=device)
            for b, (mask_idx, orig) in enumerate(n2v_targets):
                if mask_idx is None:
                    continue
                my, mx = mask_idx
                pred_at_mask = pred[b, 0, my, mx]
                loss_n2v = loss_n2v + F.l1_loss(pred_at_mask, orig)
            loss_n2v = loss_n2v / max(bs, 1)

            loss = (cfg["loss_median_weight"] * loss_med
                   + cfg["loss_n2v_weight"]    * loss_n2v)

            if not torch.isfinite(loss):
                continue
            loss.backward()
            nn.utils.clip_grad_norm_(model.tdvt.parameters(), 1.0)
            opt.step()
            sch.step()
            rm += loss_med.item()
            rn += loss_n2v.item()
            if verbose and (it + 1) % 200 == 0:
                lr_now = sch.get_last_lr()[0]
                alpha_now = float(model.tdvt.alpha.detach())
                print(f"   {it+1:>5}/{cfg['vit_iters']} "
                      f"med={rm/200:.5f} n2v={rn/200:.5f} "
                      f"α={alpha_now:.3f} lr={lr_now:.2e}  "
                      f"{time.time()-t0:.1f}s")
                rm = rn = 0.0

    elapsed = time.time() - t0
    if verbose:
        final_alpha = float(model.tdvt.alpha.detach())
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f" Final α (temporal mixing weight): {final_alpha:.4f}")
    return model, cfg


# ══════════════════════════════════════════════════════════════
# INFERENCE — frame-by-frame with temporal window
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(model, stack, config, device, verbose=True):
    """
    Frame-by-frame denoising. For each frame t, gather a T-frame window
    centered at t (mirror-pad at boundaries), spatially tile the frame
    into overlapping windows, run the model, blend with Gaussian
    weights, and stitch.
    """
    model.eval()
    model = model.float()
    norm_params = config["norm_params"]
    T_window = config["T_window"]
    k = T_window // 2
    phw = config.get("patch_hw", 64)
    F_total, H, W = stack.shape

    stride = max(phw // 2, 8)
    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    # Per-spatial-patch gaussian window
    def _gauss_2d(s):
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = max(s * 0.3, 1.0)
        return torch.exp(-0.5 * ((coords - center) / sigma) ** 2).clamp(min=1e-6)

    g = _gauss_2d(phw)
    gauss_win = (g[:, None] * g[None, :]).clamp(min=1e-6)     # [phw, phw]

    output = torch.zeros(F_total, H, W, device=device)
    if verbose:
        print(f" Inference: {F_total} frames, "
              f"patch={phw}x{phw} stride={stride}, "
              f"temporal window={T_window}")

    y_starts = list(range(0, max(H - phw, 0) + 1, stride))
    if not y_starts or y_starts[-1] + phw < H:
        y_starts.append(max(H - phw, 0))
    x_starts = list(range(0, max(W - phw, 0) + 1, stride))
    if not x_starts or x_starts[-1] + phw < W:
        x_starts.append(max(W - phw, 0))
    y_starts = sorted(set(y_starts)); x_starts = sorted(set(x_starts))

    t_inf = time.time()
    for t in range(F_total):
        # Build the T-frame temporal context with mirror padding at edges
        idxs = []
        for offset in range(-k, k + 1):
            ti = t + offset
            if ti < 0:
                ti = -ti
            elif ti >= F_total:
                ti = 2 * F_total - ti - 2
            idxs.append(max(0, min(F_total - 1, ti)))
        ctx_full = stack_t[idxs]                            # [T, H, W]

        out_sum = torch.zeros(H, W, device=device)
        wsum    = torch.zeros(H, W, device=device)

        for y0 in y_starts:
            y1 = min(y0 + phw, H); ah = y1 - y0
            for x0 in x_starts:
                x1 = min(x0 + phw, W); aw = x1 - x0

                ctx_p = ctx_full[:, y0:y1, x0:x1]            # [T, ah, aw]
                if ah < phw or aw < phw:
                    can_reflect = (phw - aw < aw) and (phw - ah < ah)
                    pad_mode = "reflect" if can_reflect else "replicate"
                    # F.pad with [T, H, W]: treat as [N=T, C=1, H, W]
                    ctx_p = F.pad(
                        ctx_p.unsqueeze(0),
                        (0, phw - aw, 0, phw - ah), mode=pad_mode,
                    ).squeeze(0)
                inp = ctx_p.unsqueeze(0).unsqueeze(0)        # [1, 1, T, phw, phw]
                pred = model(inp).squeeze(0).squeeze(0)      # [phw, phw]
                pred = pred[:ah, :aw]
                w = gauss_win[:ah, :aw]
                out_sum[y0:y1, x0:x1] += pred * w
                wsum[y0:y1, x0:x1]    += w

        frame_out = out_sum / wsum.clamp(min=1e-8)
        output[t] = frame_out

        if verbose and (t + 1) % 100 == 0:
            print(f"   frame {t+1}/{F_total}  "
                  f"{time.time()-t_inf:.1f}s elapsed", end="\r")

    if verbose:
        print(f"\n Inference: {F_total} frames in {time.time()-t_inf:.1f}s")

    output = output.cpu().numpy()
    output = norm_strategy.inverse(output, norm_params)

    # ── Safety nets ───────────────────────────────────────────
    # 1) NaN/Inf -> input value
    bad = ~np.isfinite(output)
    if bad.any():
        n_bad = int(bad.sum())
        output[bad] = stack[bad].astype(np.float32)
        if verbose:
            print(f" Replaced {n_bad} non-finite voxels")

    # 2) Clip to input range with 10% headroom on the bright tail
    in_lo, in_hi = float(stack.min()), float(stack.max())
    in_hi_padded = in_hi + 0.1 * max(in_hi - in_lo, 1.0)
    output = np.clip(output, in_lo, in_hi_padded)

    # 3) Salt-and-pepper guard (3D median)
    try:
        from scipy.ndimage import median_filter
        smooth = median_filter(output, size=(3, 3, 3))
        dev = np.abs(output - smooth)
        local_dev = median_filter(dev, size=(5, 5, 5))
        speck_mask = dev > 5.0 * np.clip(local_dev, 1.0, None)
        n_specks = int(speck_mask.sum())
        if speck_mask.any():
            output[speck_mask] = smooth[speck_mask]
            if verbose:
                print(f" Replaced {n_specks} salt-and-pepper specks")
    except ImportError:
        if verbose:
            print(" (scipy not available; skipping speck remover)")

    # 4) Flat-frame guard
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
        "arch": "FM2S_DVT_Denoiser",
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FM2S_DVT_Denoiser(
        n_temporal=cfg.get("T_window", 11),
        fm2s_chan=cfg.get("fm2s_chan", 5),
        vit_dim=cfg.get("vit_dim", 24),
        vit_heads=cfg.get("vit_heads", 4),
        vit_blocks=cfg.get("vit_blocks", 2),
        mlp_ratio=cfg.get("mlp_ratio", 2.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
