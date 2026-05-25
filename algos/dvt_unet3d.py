"""
DVT-Inspired 3D Denoising Network for Calcium Imaging
======================================================

Adapts the architectural ideas of "Denoising Vision Transformers"
(Yang et al., 2024 — https://arxiv.org/abs/2401.02957) to volumetric
image denoising for the AI4Life-CIDC25 challenge.

What DVT actually does (in the paper):
    Removes grid-like positional-embedding artifacts from a *frozen*
    ViT's feature maps. The denoiser does not produce images — only
    cleaner features for downstream tasks (segmentation, depth, etc.).

What we adapt here:
    DVT's core feature decomposition (Eq. 5 of the paper)

        ViT(x) ≈ f(x) + g(E_pos) + h(x, E_pos)
                 └──┬──┘ └──┬───┘ └────┬─────┘
                  clean    pos.-       input-
                  feature  artifact    dep. noise

    fits calcium imaging surprisingly well:
        • f(x)        : the clean fluorescence signal we want to recover
        • g(E_pos)    : sensor-fixed pattern noise / dark-current offsets
        • h(x, E_pos) : Poisson shot noise + read noise (signal-dependent)

Architecture (mirrors §4 of the paper, adapted for 3D image denoising):
    Conv encoder (2× downsample)
        ↓ produces bottleneck features
    Patchify → tokens
        ↓
    ViT block(s)             ← noisy bottleneck features y
        ↓
    DVT denoiser block:
        clean_tokens = TransformerBlock(y − G + post_PE)
        residual     = MLP_h(y)               (3-layer MLP, paper §A.1)
        ↓ uses clean_tokens as F (semantics field analogue)
    Unpatchify
        ↓
    Conv decoder with skip connections (2× upsample)
        ↓ residual: clean = noisy − predicted_noise
    Denoised image

Where G is a learnable spatial artifact field (the paper's "C × K × K
learnable feature map", §A.1 "Artifact field G"). The denoiser is a
single Transformer block with new learnable positional embeddings
applied after the encoder — the variant the paper found best (Tab. 6,
row d): "Single Transformer Block + PE."

Training:
    Self-supervised Noise2Void (3D blind-spot) — same as model.py, since
    no clean targets exist. The DVT decomposition is what changes; the
    training signal is unchanged.

API parity with model.py:
    compute_norm_params, normalize, denormalize
    train_self_supervised(stack, device, config) → (model, cfg)
    denoise_stack(model, stack, config, device)  → np.ndarray
    save_checkpoint, load_checkpoint
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ══════════════════════════════════════════════════════════════
# NORMALIZATION (p3–p97 robust scaling — kept identical to model.py)
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
# CONVOLUTIONAL BLOCKS (encoder / decoder, identical style to model.py)
# ══════════════════════════════════════════════════════════════

class ConvBlock3d(nn.Module):
    """Two 3×3×3 convolutions + GroupNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        gn = min(8, out_ch)
        while out_ch % gn != 0:
            gn -= 1
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn, out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# TRANSFORMER BUILDING BLOCKS (DVT bottleneck)
# ══════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Standard pre-norm Transformer block — the building block of every
    ViT in the DVT paper, and the unit used for the generalizable
    denoiser (paper §4.3).
    """

    def __init__(self, dim: int, num_heads: int = 4, mlp_ratio: float = 2.0,
                 dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim, num_heads=num_heads, dropout=dropout, batch_first=True,
        )
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
        # x: [B, N, D]
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


class ResidualMLP(nn.Module):
    """
    3-layer MLP residual predictor h_ψ — paper §A.1 ("Residual predictor h"):
    "structured as a 3-layer MLP with ReLU activation after the hidden
    layers. The hidden dimension is set to be one-quarter of the channel
    dimension of the ViT being studied."
    """

    def __init__(self, dim: int):
        super().__init__()
        hidden = max(dim // 4, 8)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )

    def forward(self, x):
        return self.net(x)


class DVTBottleneck(nn.Module):
    """
    DVT-style bottleneck: takes a 3D feature map [B, C, D, H, W] and
    returns a *denoised* feature map of the same shape.

    Implementation of the paper's decomposition (Eqs. 5–10):

        y         = ViT(x)                       (raw bottleneck features)
        ĥ         = h_ψ(y)                       (residual predictor)
        clean_tok = D_ζ(y - G + post_PE)         (denoiser block — paper Tab. 6d)

    The final feature returned to the decoder is `clean_tok`, which plays
    the role of the semantics field F in the paper. The artifact field G
    is a learnable parameter shared across all inputs, exactly as the
    paper describes ("a 2D learnable feature map of size C × K × K", §A.1).

    Args:
        in_ch:        bottleneck channel count (from conv encoder)
        token_dim:    dimensionality used inside the transformer
        grid_shape:   (D', H', W') — the target token-grid shape inside the
                      bottleneck. Token count = D' * H' * W'.
        n_vit_blocks: number of "ViT" blocks before the denoiser (paper
                      uses a frozen pre-trained ViT; we learn a small one)
        n_heads:      attention heads
    """

    def __init__(
        self,
        in_ch: int,
        token_dim: int = 192,
        grid_shape: tuple = (4, 8, 8),
        n_vit_blocks: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        self.in_ch = in_ch
        self.token_dim = token_dim
        self.grid_shape = grid_shape
        Dp, Hp, Wp = grid_shape
        self.n_tokens = Dp * Hp * Wp

        # Patch embed: 1×1×1 conv to lift channels to token_dim.
        # The actual spatial→token reduction is handled by adaptive pooling
        # to (Dp, Hp, Wp) so the bottleneck handles variable input sizes.
        self.patch_embed = nn.Conv3d(in_ch, token_dim, kernel_size=1, bias=False)
        self.patch_unembed = nn.Conv3d(token_dim, in_ch, kernel_size=1, bias=False)

        # Pre-denoiser ViT blocks (analogous to the frozen ViT in the paper).
        self.vit_blocks = nn.ModuleList([
            TransformerBlock(token_dim, num_heads=n_heads, mlp_ratio=2.0)
            for _ in range(n_vit_blocks)
        ])

        # Artifact field G — learnable, input-independent, position-keyed.
        # Shape [1, N, D] mirrors the paper's "C × K × K" map flattened to
        # token order.
        self.artifact_field = nn.Parameter(
            torch.zeros(1, self.n_tokens, token_dim)
        )
        nn.init.trunc_normal_(self.artifact_field, std=0.02)

        # Residual predictor h_ψ — 3-layer MLP.
        self.residual_predictor = ResidualMLP(token_dim)

        # Generalizable denoiser block — single Transformer + new PE
        # (paper §4.3, Tab. 6 row d).
        self.denoiser_pe = nn.Parameter(
            torch.zeros(1, self.n_tokens, token_dim)
        )
        nn.init.trunc_normal_(self.denoiser_pe, std=0.02)
        self.denoiser_block = TransformerBlock(
            token_dim, num_heads=n_heads, mlp_ratio=2.0,
        )

    def forward(self, x):
        """
        x: [B, in_ch, D, H, W]   (any sizes; we adapt-pool to grid_shape)
        returns: [B, in_ch, D, H, W]  (denoised, same shape as input)
        """
        B, C, D_in, H_in, W_in = x.shape
        Dp, Hp, Wp = self.grid_shape

        # ── Patchify into a fixed grid via adaptive pooling ────────
        # This gives the transformer a fixed-size input regardless of
        # the patch size used for training/inference.
        x_pool = F.adaptive_avg_pool3d(x, output_size=(Dp, Hp, Wp))   # [B,C,Dp,Hp,Wp]
        tokens = self.patch_embed(x_pool)                              # [B,Td,Dp,Hp,Wp]
        tokens = tokens.flatten(2).transpose(1, 2)                     # [B,N,Td]

        # ── Pre-denoiser ViT + DVT decomposition ────────────────────
        y = tokens
        for blk in self.vit_blocks:
            y = blk(y)

        # ── DVT decomposition (Eq. 10 in the paper) ────────────────
        # Subtract artifact field G (input-independent) ...
        y_minus_g = y - self.artifact_field
        # ... add learnable post-PE for the denoiser (Tab. 6 row d) ...
        y_for_denoiser = y_minus_g + self.denoiser_pe
        # ... and run the single-block denoiser to get the clean
        # semantics F.
        clean_tokens = self.denoiser_block(y_for_denoiser)

        # Residual term ĥ = h_ψ(y). We mix a small fraction back so
        # that signal-dependent fluctuations the denoiser shouldn't
        # kill (e.g. genuine calcium transients) survive — the paper
        # uses this term to *re-explain* parts of the noisy ViT output
        # during training (Eqs. 8–10), and it acts as a controlled
        # bypass.
        residual = self.residual_predictor(y)
        clean_tokens = clean_tokens + 0.05 * residual

        # ── Unpatchify ─────────────────────────────────────────────
        out = clean_tokens.transpose(1, 2).reshape(B, self.token_dim, Dp, Hp, Wp)
        out = self.patch_unembed(out)                                  # [B,C,Dp,Hp,Wp]
        # Resample back to the original bottleneck spatial size.
        out = F.interpolate(out, size=(D_in, H_in, W_in),
                            mode="trilinear", align_corners=False)
        return out


# ══════════════════════════════════════════════════════════════
# DVT U-NET 3D — main architecture
# ══════════════════════════════════════════════════════════════

class DVTUNet3D(nn.Module):
    """
    3D U-Net with a DVT-inspired transformer bottleneck.

    Input/output:  [B, 1, D, H, W] → [B, 1, D, H, W]
    Residual learning: output = input − predicted_noise.
    Spatial dims are auto-padded to multiples of 4.

    This is a drop-in replacement for `UNet3D` in model.py.
    """

    def __init__(
        self,
        base_ch: int = 32,
        token_dim: int = 192,
        grid_shape: tuple = (4, 8, 8),
        n_vit_blocks: int = 2,
        n_heads: int = 4,
    ):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4    # 32, 64, 128

        # ── Encoder ────────────────────────────────────────────────
        self.enc1 = ConvBlock3d(1, c1)
        self.pool1 = nn.MaxPool3d((2, 2, 2))
        self.enc2 = ConvBlock3d(c1, c2)
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        # ── Conv bottleneck ────────────────────────────────────────
        self.bottleneck_conv = ConvBlock3d(c2, c3)

        # ── DVT denoising bottleneck (the paper's contribution) ────
        self.dvt = DVTBottleneck(
            in_ch=c3,
            token_dim=token_dim,
            grid_shape=grid_shape,
            n_vit_blocks=n_vit_blocks,
            n_heads=n_heads,
        )

        # ── Decoder ────────────────────────────────────────────────
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2, bias=False)
        self.dec2 = ConvBlock3d(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2, bias=False)
        self.dec1 = ConvBlock3d(c1 + c1, c1)

        # ── Noise predictor ────────────────────────────────────────
        self.out_conv = nn.Conv3d(c1, 1, 1, bias=False)

    def forward(self, x):
        """x: [B, 1, D, H, W] → [B, 1, D, H, W] (denoised, residual)."""
        identity = x

        # Pad spatial dims to multiples of 4
        _, _, D, H, W = x.shape
        pd = (4 - D % 4) % 4
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")
            identity = F.pad(identity, (0, pw, 0, ph, 0, pd), mode="reflect")

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck_conv(self.pool2(e2))

        # DVT bottleneck (the heart of the paper's contribution)
        b_clean = self.dvt(b)

        # Decoder with skip connections
        d2 = self.up2(b_clean)
        d2 = _match_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_cat(d1, e1)
        d1 = self.dec1(d1)

        # Residual: clean = noisy − predicted_noise
        noise_pred = self.out_conv(d1)
        out = identity - noise_pred

        # Strip padding
        if pd or ph or pw:
            out = out[:, :, :D, :H, :W]
        return out


def _match_cat(up, skip):
    """Pad the upsampled tensor to match skip dims, then concatenate."""
    dd = skip.shape[2] - up.shape[2]
    dh = skip.shape[3] - up.shape[3]
    dw = skip.shape[4] - up.shape[4]
    if dd or dh or dw:
        up = F.pad(up, (0, dw, 0, dh, 0, dd))
    return torch.cat([up, skip], dim=1)


# Alias so existing code that imports `UNet3D` keeps working.
UNet3D = DVTUNet3D


# ══════════════════════════════════════════════════════════════
# 3D BLIND-SPOT MASKING (Noise2Void) — same as model.py
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume: torch.Tensor, mask_ratio: float = 0.008,
                radius: int = 2):
    """
    Noise2Void 3D masking. Picks random voxels, replaces each with a
    random neighbour's value. Returns (masked_volume, indices, originals).
    """
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


def _gaussian_window_3d(shape, sigma_frac=0.25, device="cpu"):
    """3D Gaussian blending window for sliding-window inference."""
    windows = []
    for s in shape:
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = s * sigma_frac
        w = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
        windows.append(w)
    w3d = (windows[0][:, None, None]
           * windows[1][None, :, None]
           * windows[2][None, None, :])
    return w3d.clamp(min=1e-6)


def _random_augment_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    """8 spatial augmentations (4 rot × 2 flip) for [D, H, W]."""
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


# ══════════════════════════════════════════════════════════════
# TRAINING (self-supervised, two-stage — same flow as model.py)
# ══════════════════════════════════════════════════════════════

def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
    init_state_dict: dict = None,
):
    """
    Stage 0 — Temporal-median warmup
    Stage 1 — 3D Blind-spot (Noise2Void)

    Trains in full fp32 (no mixed precision) for numerical stability —
    the attention softmax in the DVT bottleneck is fragile under fp16
    on high-magnitude inputs.

    Args:
        stack:  [F, H, W] numpy array (raw, original values).
        device: torch device.
        config: optional overrides for any default key below.
        init_state_dict: optional pre-trained weights to load into the
            model BEFORE training starts. Used for fine-tuning from a
            pretrained checkpoint. The architecture in `config` must match
            the architecture the state_dict was saved with; loading uses
            `strict=True` and will fail loud on mismatch.
    Returns:
        (model, cfg)  — the trained DVTUNet3D and the full config used.
    """
    t0 = time.time()
    cfg = {
        # backbone
        "base_ch": 32,
        # DVT bottleneck
        "token_dim": 192,
        "grid_shape": (4, 8, 8),
        "n_vit_blocks": 2,
        "n_heads": 4,
        # patch sampling
        "patch_d": 32,
        "patch_hw": 128,
        "batch_size": 2,
        # schedule
        "warmup_iters": 500,
        "n2v_iters": 3000,
        "lr": 3e-4,
        # n2v masking
        "mask_ratio": 0.008,
        "mask_radius": 2,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        n_tok = int(np.prod(cfg["grid_shape"]))
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" DVT-UNet3D: base_ch={cfg['base_ch']}, "
              f"token_dim={cfg['token_dim']}, "
              f"grid={cfg['grid_shape']} ({n_tok} tokens), "
              f"vit_blocks={cfg['n_vit_blocks']}")
        print(f" Patch: {pd}×{phw}×{phw}, batch={bs}")
        print(f" Stages: warmup={cfg['warmup_iters']}, "
              f"n2v={cfg['n2v_iters']}")
        print(f" Precision: fp32")

    # ── Normalize (strategy from config) ────────────────
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

    # ── Temporal target (strategy from config) ──────────
    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    cfg["__resolved_temporal_target"] = tt_strategy.name
    if tt_strategy.returns != "2d":
        _tt_3d = tt_strategy.compute(stack_norm)
        temporal_med = np.median(_tt_3d, axis=0).astype(np.float32)
    else:
        temporal_med = tt_strategy.compute(stack_norm)
    if verbose:
        print(f" Temporal target [{tt_strategy.name}]: "
              f"[{temporal_med.min():.3f}, {temporal_med.max():.3f}]")

    stack_t = torch.from_numpy(stack_norm).float().to(device)   # [F, H, W]
    tmed_t = torch.from_numpy(temporal_med).float().to(device)  # [H, W]

    # Model
    model = DVTUNet3D(
        base_ch=cfg["base_ch"],
        token_dim=cfg["token_dim"],
        grid_shape=cfg["grid_shape"],
        n_vit_blocks=cfg["n_vit_blocks"],
        n_heads=cfg["n_heads"],
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        n_dvt = sum(p.numel() for p in model.dvt.parameters())
        print(f" Model params: {n_params:,}  "
              f"(DVT bottleneck: {n_dvt:,} = {100*n_dvt/n_params:.1f}%)")

    # Optionally initialize from pretrained weights (fine-tuning).
    if init_state_dict is not None:
        # strict=True will raise on any size or key mismatch — this is
        # intentional. Silently loading partial weights leads to subtle
        # quality regressions that are hard to diagnose later.
        missing, unexpected = model.load_state_dict(
            init_state_dict, strict=False
        )
        if missing or unexpected:
            raise RuntimeError(
                f"Pretrained state_dict does not match current model:\n"
                f"  missing keys:    {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
                f"  unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}\n"
                f"This usually means the architecture (base_ch, token_dim, "
                f"grid_shape, n_vit_blocks, n_heads) differs between "
                f"pretraining and fine-tuning. They must match exactly."
            )
        if verbose:
            print(f" Initialized from pretrained state_dict "
                  f"({len(init_state_dict)} tensors loaded)")

    # Random patch helper
    def random_patch():
        t0_ = np.random.randint(0, max(F_total - pd, 1))
        y0  = np.random.randint(0, max(H - phw, 1))
        x0  = np.random.randint(0, max(W - phw, 1))
        d = min(pd, F_total)
        h = min(phw, H)
        w = min(phw, W)
        return stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w], tmed_t[y0:y0+h, x0:x0+w]

    # ────────────────────────────────────────────────
    # Stage 0: temporal-median warmup
    # ────────────────────────────────────────────────
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Temporal-median warmup — "
                  f"{cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1,
        )
        crit = nn.MSELoss()
        model.train()
        rl = 0.0

        for it in range(cfg["warmup_iters"]):
            patches, targets = [], []
            for _ in range(bs):
                vol, tmed_crop = random_patch()
                aug = np.random.randint(0, 8)
                vol = _random_augment_3d(vol, aug)
                tmed_crop = _random_augment_3d(
                    tmed_crop.unsqueeze(0).expand(vol.shape[0], -1, -1), aug,
                )
                patches.append(vol.unsqueeze(0))
                targets.append(tmed_crop.unsqueeze(0))

            inp = torch.stack(patches, dim=0).to(device)
            tgt = torch.stack(targets, dim=0).to(device)

            opt.zero_grad()
            pred = model(inp)
            loss = crit(pred, tgt)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()

            if verbose and (it + 1) % 100 == 0:
                print(f"   {it+1:>5}/{cfg['warmup_iters']} "
                      f"loss={rl/100:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # ────────────────────────────────────────────────
    # Stage 1: 3D Blind-spot (Noise2Void)
    # ────────────────────────────────────────────────
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] 3D Blind-spot — {cfg['n2v_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(),
                                 lr=cfg["lr"] * 0.5, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6,
        )
        model.train()
        rl = 0.0
        # NaN guard: snapshot weights periodically. With fp32 training
        # this rarely fires, but keeps the framework safe against any
        # divergence cause (extreme outlier patches, bad lr/schedule,
        # custom user-supplied normalization that produces very large
        # inputs, etc.). Cheap to keep, important when it matters.
        nan_consecutive = 0
        nan_total = 0
        last_good_state = None
        snapshot_every = 100
        max_consecutive_nan = 50

        for it in range(cfg["n2v_iters"]):
            all_orig = []
            patches = []
            for _ in range(bs):
                vol, _ = random_patch()
                aug = np.random.randint(0, 8)
                vol = _random_augment_3d(vol, aug)
                masked, (mz, my, mx), orig = n2v_mask_3d(
                    vol,
                    mask_ratio=cfg["mask_ratio"],
                    radius=cfg["mask_radius"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((mz, my, mx, orig))

            inp = torch.stack(patches, dim=0).to(device)

            opt.zero_grad()
            pred = model(inp)
            loss = torch.tensor(0.0, device=device)
            for b, (mz, my, mx, orig) in enumerate(all_orig):
                pred_at_mask = pred[b, 0, mz, my, mx]
                loss = loss + F.mse_loss(pred_at_mask, orig)
            loss = loss / bs

            # ── NaN guard ────────────────────────────────────────
            loss_is_bad = (not torch.isfinite(loss)) or torch.isnan(loss)
            if loss_is_bad:
                nan_consecutive += 1
                nan_total += 1
                opt.zero_grad(set_to_none=True)
                if nan_consecutive == 1 and verbose:
                    print(f"   NaN at iter {it+1} — skipping update "
                          f"(may roll back weights if persistent)")
                # If we've snapshotted weights and NaN persists, restore
                if (nan_consecutive >= 5 and last_good_state is not None):
                    model.load_state_dict(last_good_state)
                    if verbose and nan_consecutive == 5:
                        print(f"   Rolled back weights to last "
                              f"snapshot at iter {it+1}")
                if nan_consecutive >= max_consecutive_nan:
                    if verbose:
                        print(f"   ABORTING N2V — {nan_consecutive} "
                              f"consecutive NaN steps. Try a tighter "
                              f"normalization (e.g. 'p0.5_p99.5') or "
                              f"a lower lr.")
                    break
                continue

            loss.backward()
            # Skip step if gradients themselves are non-finite
            grad_ok = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_ok = False
                    break
            if not grad_ok:
                nan_consecutive += 1
                nan_total += 1
                opt.zero_grad(set_to_none=True)
                continue

            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()
            nan_consecutive = 0

            # Periodically snapshot weights as a rollback point
            if (it + 1) % snapshot_every == 0:
                last_good_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }

            if verbose and (it + 1) % 250 == 0:
                lr_now = sch.get_last_lr()[0]
                avg_loss = rl / 250
                extra = f"  (NaN skipped: {nan_total})" if nan_total else ""
                print(f"   {it+1:>5}/{cfg['n2v_iters']} "
                      f"loss={avg_loss:.6f} lr={lr_now:.2e} "
                      f"{time.time()-t0:.1f}s{extra}")
                rl = 0.0

        if nan_total > 0 and verbose:
            print(f"   Total NaN steps skipped: {nan_total} "
                  f"of {cfg['n2v_iters']} "
                  f"({100*nan_total/cfg['n2v_iters']:.1f}%)")
        # Final safety: if model state is corrupted with NaN, restore
        # from last snapshot (or abort cleanly).
        any_nan_in_weights = any(
            (not torch.isfinite(p).all()) for p in model.parameters()
        )
        if any_nan_in_weights:
            if last_good_state is not None:
                model.load_state_dict(last_good_state)
                if verbose:
                    print(f"   Final model had NaN weights — restored "
                          f"from snapshot.")
            else:
                if verbose:
                    print(f"   WARN: model has NaN weights and no "
                          f"snapshot available. Inference will likely "
                          f"produce garbage.")

    elapsed = time.time() - t0
    if verbose:
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return model, cfg


# ══════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE (same flow as model.py)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(
    model: DVTUNet3D,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    """Sliding-window denoising with Gaussian blending."""
    model.eval()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    pd  = min(config.get("patch_d", 32), F_total)
    phw = min(config.get("patch_hw", 128), H, W)
    pd  = max((pd  // 4) * 4, 4)
    phw = max((phw // 4) * 4, 4)
    stride_d  = max(pd  // 2, 4)
    stride_hw = max(phw // 2, 4)
    # stride_d  = pd 
    # stride_hw = max(phw // 2, 4)

    if verbose:
        print(f" Sliding window: patch={pd}×{phw}×{phw}, "
              f"stride={stride_d}×{stride_hw}×{stride_hw}")

    # Resolve trained strategy
    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    output_sum = torch.zeros(F_total, H, W, device=device)
    weight_sum = torch.zeros(F_total, H, W, device=device)
    gauss_win  = _gaussian_window_3d((pd, phw, phw), sigma_frac=0.3, device=device)

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
        print(f" Patches: {len(z_starts)}×{len(y_starts)}×{len(x_starts)} = {total}")

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
                    patch = F.pad(
                        patch,
                        (0, phw - aw, 0, phw - ah, 0, pd - ad),
                        mode="reflect",
                    )
                inp = patch.unsqueeze(0).unsqueeze(0)
                pred = model(inp).squeeze(0).squeeze(0)

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

    # ── Catastrophic-failure guard ───────────────────────────
    # If the model was wiped to NaN during training (e.g. divergence on
    # outlier patches with an unsafe normalization choice), the inference
    # output will be all-NaN. Detect this and fall back to the noisy
    # input rather than silently writing a black TIFF that the user
    # then has to diagnose. This will hurt metrics but produces a
    # usable file and a clear warning instead of zeros.
    n_bad = int((~np.isfinite(output)).sum())
    total_voxels = output.size
    bad_frac = n_bad / max(total_voxels, 1)
    if bad_frac > 0.5:
        if verbose:
            print(f"\n WARNING: {100*bad_frac:.1f}% of output voxels are "
                  f"NaN/Inf — model likely diverged during training. "
                  f"Falling back to the noisy input. "
                  f"Try config['normalization']='p0.5_p99.5' (algo default) "
                  f"and/or a lower lr.")
        return stack.astype(np.float32)
    if n_bad > 0:
        if verbose:
            print(f" Replacing {n_bad} non-finite voxels with input")
        bad_mask = ~np.isfinite(output)
        output[bad_mask] = stack.astype(np.float32)[bad_mask]

    output = norm_strategy.inverse(output, norm_params)

    # Clip only to the input's actual range — the original safe_hi was
    # 1.5×scale above shift, which truncates calcium-transient peaks
    # that legitimately exceed the 97th percentile.
    in_lo, in_hi = float(stack.min()), float(stack.max())
    output = np.clip(output, in_lo, in_hi)
    return output


# ══════════════════════════════════════════════════════════════
# CHECKPOINTS (same names as model.py for drop-in compatibility)
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({"model_state_dict": model.state_dict(),
                "config": config,
                "arch": "DVTUNet3D"}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DVTUNet3D(
        base_ch=cfg["base_ch"],
        token_dim=cfg.get("token_dim", 192),
        grid_shape=cfg.get("grid_shape", (4, 8, 8)),
        n_vit_blocks=cfg.get("n_vit_blocks", 2),
        n_heads=cfg.get("n_heads", 4),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
