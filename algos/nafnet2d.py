"""
NAFNet 2D — Nonlinear Activation Free Network for image denoising
=================================================================

Implements the architecture from Chen et al., "Simple Baselines for Image
Restoration" (ECCV 2022, https://arxiv.org/abs/2204.04676) — original
NAFNet repo: https://github.com/megvii-research/NAFNet.

This is a 2D (per-frame) denoiser. The 3D variant (`nafnet3d.py`)
processes whole stacks; this one processes each frame independently
during inference. Calcium-imaging-specific inspirations were drawn from
https://github.com/GolpedeRemo37/Calcium_Imaging_Denoising which used
the same architecture for the AI4Life-CIDC25 challenge.

Architecture summary
--------------------

A NAFBlock has:

    Block 1:
      x' = x + β · drop( Conv1x1( SCA( SimpleGate( DWConv3x3(
                                Conv1x1(LayerNorm(x)) )))) )

    Block 2:
      x  = x' + γ · drop( Conv1x1( SimpleGate( Conv1x1(LayerNorm(x')) )) )

Where:
    SimpleGate(z)  : z = a, b = chunk(z, 2);  return a * b
    SCA(z)         : z * Conv1x1( GlobalAvgPool(z) )
    β, γ           : learnable scalar (initialised to 0)

The U-Net wraps these in encoder/middle/decoder stages with PixelUnshuffle
downsamples and PixelShuffle upsamples — no transposed convs.

API parity with the rest of our algos:
    compute_norm_params, normalize, denormalize
    train_self_supervised(stack, device, config) -> (model, cfg)
    denoise_stack(model, stack, config, device)  -> np.ndarray
    save_checkpoint, load_checkpoint
    DEFAULT_NORMALIZATION, DEFAULT_TEMPORAL_TARGET
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runner import preprocessing as _prep

# ══════════════════════════════════════════════════════════════
# NORMALIZATION (strategy-driven; see runner.preprocessing)
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
# NAFNet building blocks
# ══════════════════════════════════════════════════════════════

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for [B, C, H, W] tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # Channel-norm: mean/var over channel dim only
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """SimpleGate: split channels in half, return elementwise product.
    Equivalent to a learned gating without any activation function."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class SimpleChannelAttention(nn.Module):
    """Single-conv channel attention — the "SCA" of NAFNet."""

    def __init__(self, channels: int):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x):
        attn = self.fc(self.gap(x))
        return x * attn


class NAFBlock(nn.Module):
    """One NAFBlock: two sub-blocks with norm, simple-gate, SCA."""

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2,
                 drop_out_rate: float = 0.0):
        super().__init__()
        c = channels
        dw_c = c * dw_expand
        ffn_c = c * ffn_expand

        # ── Block 1 (DWConv + SimpleGate + SCA) ─────────────
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1, bias=True)
        self.dwconv = nn.Conv2d(dw_c, dw_c, 3, padding=1, groups=dw_c, bias=True)
        self.gate1 = SimpleGate()   # halves channels
        self.sca = SimpleChannelAttention(dw_c // 2)
        self.conv2 = nn.Conv2d(dw_c // 2, c, 1, bias=True)

        # ── Block 2 (FFN-style with SimpleGate) ─────────────
        self.norm2 = LayerNorm2d(c)
        self.conv3 = nn.Conv2d(c, ffn_c, 1, bias=True)
        self.gate2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_c // 2, c, 1, bias=True)

        self.drop1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 \
            else nn.Identity()
        self.drop2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0 \
            else nn.Identity()

        # Learnable residual scales β, γ — initialised to 0 so the
        # block starts as identity (paper §3, "skip-init").
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)))
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)))

    def forward(self, x):
        # Sub-block 1
        h = self.norm1(x)
        h = self.conv1(h)
        h = self.dwconv(h)
        h = self.gate1(h)
        h = self.sca(h)
        h = self.conv2(h)
        x = x + self.beta * self.drop1(h)

        # Sub-block 2
        h = self.norm2(x)
        h = self.conv3(h)
        h = self.gate2(h)
        h = self.conv4(h)
        x = x + self.gamma * self.drop2(h)
        return x


# ══════════════════════════════════════════════════════════════
# NAFNet — full U-Net topology
# ══════════════════════════════════════════════════════════════

class NAFNet(nn.Module):
    """
    NAFNet U-Net backbone. Takes [B, 1, H, W] noisy input and predicts
    [B, 1, H, W] denoised output via residual learning (output = input -
    predicted_noise). Spatial dims auto-padded to multiples of 2**depth.
    """

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 32,
        enc_blocks: tuple = (2, 2, 4, 8),
        middle_blocks: int = 12,
        dec_blocks: tuple = (2, 2, 2, 2),
        dw_expand: int = 2,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ):
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks), \
            "enc_blocks and dec_blocks must have same length"
        self.depth = len(enc_blocks)
        self.divisor = 2 ** self.depth     # input dims must be multiples of this

        # Stem 3×3 conv
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1, bias=True)
        c = width

        # Encoder stages
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        enc_channels = []
        for n in enc_blocks:
            enc_channels.append(c)
            blocks = nn.Sequential(*[
                NAFBlock(c, dw_expand, ffn_expand, drop_out_rate)
                for _ in range(n)
            ])
            self.encoders.append(blocks)
            # Downsample by 2 via stride-2 conv (cheaper than PixelUnshuffle
            # + projection); doubles channels.
            self.downs.append(nn.Conv2d(c, c * 2, 2, stride=2, bias=True))
            c *= 2

        # Middle stage at the bottleneck
        self.middle = nn.Sequential(*[
            NAFBlock(c, dw_expand, ffn_expand, drop_out_rate)
            for _ in range(middle_blocks)
        ])

        # Decoder stages
        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i, n in enumerate(dec_blocks):
            # Upsample by 2 via PixelShuffle; halves channels.
            self.ups.append(nn.Sequential(
                nn.Conv2d(c, c * 2, 1, bias=True),
                nn.PixelShuffle(2),
            ))
            c //= 2
            blocks = nn.Sequential(*[
                NAFBlock(c, dw_expand, ffn_expand, drop_out_rate)
                for _ in range(n)
            ])
            self.decoders.append(blocks)

        # Final 3×3 conv to predict the residual
        self.ending = nn.Conv2d(width, in_channels, 3, padding=1, bias=True)

    def forward(self, x):
        """x: [B, C_in, H, W]  →  [B, C_in, H, W] (denoised)."""
        identity = x

        # Pad spatial dims to multiples of 2**depth
        _, _, H, W = x.shape
        pad_h = (self.divisor - H % self.divisor) % self.divisor
        pad_w = (self.divisor - W % self.divisor) % self.divisor
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
            identity_pad = F.pad(identity, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            identity_pad = identity

        # Encoder
        h = self.intro(x)
        skips = []
        for blocks, down in zip(self.encoders, self.downs):
            h = blocks(h)
            skips.append(h)
            h = down(h)

        # Middle
        h = self.middle(h)

        # Decoder with skip connections (sum, not concat — paper §3)
        for blocks, up, skip in zip(self.decoders, self.ups,
                                      reversed(skips)):
            h = up(h)
            h = h + skip
            h = blocks(h)

        # Residual prediction
        noise_pred = self.ending(h)
        out = identity_pad - noise_pred

        # Strip padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]
        return out


# ══════════════════════════════════════════════════════════════
# 2D BLIND-SPOT MASKING (Noise2Void per-frame)
# ══════════════════════════════════════════════════════════════

def n2v_mask_2d(frame: torch.Tensor, mask_ratio: float = 0.015,
                 radius: int = 2):
    """
    Noise2Void 2D masking on a single frame [H, W]. Returns
    (masked_frame, indices, originals).
    """
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
    """8 spatial augmentations (4 rot × 2 flip) for [H, W]."""
    if aug_id >= 4:
        frame = torch.flip(frame, dims=[1])
    k = aug_id % 4
    if k > 0:
        frame = torch.rot90(frame, k=k, dims=[0, 1])
    return frame


def _gaussian_window_2d(shape, sigma_frac=0.25, device="cpu"):
    """2D Gaussian blending window for sliding-window inference."""
    windows = []
    for s in shape:
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = max(s * sigma_frac, 1.0)
        w = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
        windows.append(w)
    w2d = windows[0][:, None] * windows[1][None, :]
    return w2d.clamp(min=1e-6)


# ══════════════════════════════════════════════════════════════
# TRAINING (self-supervised, two-stage)
# ══════════════════════════════════════════════════════════════

def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
):
    """
    Stage 0 — Per-frame supervised warmup against the temporal median
    Stage 1 — Per-frame Noise2Void (blind-spot)

    Full fp32 training.
    """
    t0 = time.time()
    cfg = {
        # backbone
        "width":          32,
        "enc_blocks":     (2, 2, 4, 8),
        "middle_blocks":  12,
        "dec_blocks":     (2, 2, 2, 2),
        "dw_expand":      2,
        "ffn_expand":     2,
        "drop_out_rate":  0.0,
        # patch sampling
        "patch_hw":       128,
        "batch_size":     4,
        # schedule
        "warmup_iters":   300,
        "n2v_iters":      3000,
        "lr":             3e-4,
        # n2v masking
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
        print(f" NAFNet2D: width={cfg['width']}, "
              f"enc={cfg['enc_blocks']}, mid={cfg['middle_blocks']}, "
              f"dec={cfg['dec_blocks']}")
        print(f" Patch: {phw}x{phw}, batch={bs}")
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

    # ── Temporal target (used by warmup; collapse if 3D) ──
    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    tt_strategy = _prep.resolve_temporal_target(tt_name)
    cfg["__resolved_temporal_target"] = tt_strategy.name
    if tt_strategy.returns != "2d":
        _tt_3d = tt_strategy.compute(stack_norm)
        temporal_med = np.median(_tt_3d, axis=0).astype(np.float32)
    else:
        temporal_med = tt_strategy.compute(stack_norm)

    stack_t = torch.from_numpy(stack_norm).float().to(device)   # [F, H, W]
    tmed_t = torch.from_numpy(temporal_med).float().to(device)  # [H, W]

    # ── Build model ────────────────────────────────────────
    model = NAFNet(
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

    def random_patch_frame():
        f = np.random.randint(0, F_total)
        y0 = np.random.randint(0, max(H - phw, 1))
        x0 = np.random.randint(0, max(W - phw, 1))
        h = min(phw, H); w = min(phw, W)
        return (stack_t[f, y0:y0+h, x0:x0+w],
                tmed_t[y0:y0+h, x0:x0+w])

    # ── Stage 0 — warmup ──────────────────────────────────
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Per-frame warmup vs temporal median — "
                  f"{cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                 weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1)
        model.train()
        rl = 0.0
        for it in range(cfg["warmup_iters"]):
            patches_in, patches_tgt = [], []
            for _ in range(bs):
                fr, tgt = random_patch_frame()
                aug = np.random.randint(0, 8)
                fr = _augment_2d(fr, aug)
                tgt = _augment_2d(tgt, aug)
                patches_in.append(fr.unsqueeze(0))     # [1, h, w]
                patches_tgt.append(tgt.unsqueeze(0))
            inp = torch.stack(patches_in, dim=0).to(device)  # [B,1,h,w]
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

    # ── Stage 1 — Noise2Void ──────────────────────────────
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
                fr, _ = random_patch_frame()
                aug = np.random.randint(0, 8)
                fr = _augment_2d(fr, aug)
                masked, (my, mx), orig = n2v_mask_2d(
                    fr, mask_ratio=cfg["mask_ratio"],
                    radius=cfg["mask_radius"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((my, mx, orig))
            inp = torch.stack(patches, dim=0).to(device)   # [B,1,h,w]

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
# SLIDING-WINDOW INFERENCE (per-frame)
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(
    model: NAFNet,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    """
    Per-frame sliding-window denoising. Frames are processed independently
    (no temporal context — that's NAFNet3D's job).
    """
    model.eval()
    model = model.float()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    phw = min(config.get("patch_hw", 128), H, W)
    phw = max((phw // 8) * 8, 8)
    stride = max(phw // 2, 8)

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
        frame_t = torch.from_numpy(stack_norm[t]).float().to(device)  # [H, W]
        out_sum = torch.zeros_like(frame_t)
        w_sum = torch.zeros_like(frame_t)
        for y0 in y_starts:
            y1 = min(y0 + phw, H); ah = y1 - y0
            for x0 in x_starts:
                x1 = min(x0 + phw, W); aw = x1 - x0
                patch = frame_t[y0:y1, x0:x1]
                if (ah < phw) or (aw < phw):
                    patch = F.pad(patch,
                                   (0, phw - aw, 0, phw - ah),
                                   mode="reflect")
                inp = patch.unsqueeze(0).unsqueeze(0)   # [1, 1, h, w]
                pred = model(inp).squeeze(0).squeeze(0).float()
                pred = pred[:ah, :aw]
                win = gauss_win[:ah, :aw]
                out_sum[y0:y1, x0:x1] += pred * win
                w_sum[y0:y1, x0:x1] += win
        out_frame = out_sum / w_sum.clamp(min=1e-8)
        output[t] = out_frame.cpu().numpy()
        if verbose and (t + 1) % 100 == 0:
            print(f"   {t+1}/{F_total} frames  "
                  f"{time.time()-t0:.1f}s", end="\r")
    if verbose:
        print(f"\n Inference: {F_total} frames in {time.time()-t0:.1f}s")

    # Catastrophic-failure guard
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
                "arch": "NAFNet"}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = NAFNet(
        in_channels=1,
        width=cfg.get("width", 32),
        enc_blocks=tuple(cfg.get("enc_blocks", (2, 2, 4, 8))),
        middle_blocks=cfg.get("middle_blocks", 12),
        dec_blocks=tuple(cfg.get("dec_blocks", (2, 2, 2, 2))),
        dw_expand=cfg.get("dw_expand", 2),
        ffn_expand=cfg.get("ffn_expand", 2),
        drop_out_rate=cfg.get("drop_out_rate", 0.0),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
