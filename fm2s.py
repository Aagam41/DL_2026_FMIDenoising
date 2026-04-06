"""
FM2S: Fluorescence Micrograph to Self — Maximum Quality Zero-Shot

Based on: https://arxiv.org/abs/2412.10031

Optimizations for best possible output:
  1. Temporal median as pre-denoise target (exploits video temporal redundancy)
  2. Automatic noise-level estimation → adaptive noise injection parameters
  3. Self-ensemble (8-fold geometric augmentation) at inference → +0.1-0.3 dB
  4. Full paper training budget: 450 samples × 5 epochs
  5. Per-frame zero-shot mode available (paper-faithful)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy import ndimage as nd
from einops import rearrange
import numpy as np
import time
import copy

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 3407
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ══════════════════════════════════════════════
#  NETWORK  (paper Section 3.4.1)
# ══════════════════════════════════════════════
class FM2SNetwork(nn.Module):
    """3-layer CNN with ~3.5k parameters (paper architecture)."""

    def __init__(self, n_chan: int):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=1e-3)
        self.conv1 = nn.Conv2d(n_chan, 24, 5, padding=2)
        self.conv2 = nn.Conv2d(24, 12, 3, padding=1)
        self.conv3 = nn.Conv2d(12, n_chan, 5, padding=2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)
        return x


# ══════════════════════════════════════════════
#  NOISE INJECTION  (paper Section 3.3)
# ══════════════════════════════════════════════
def noise_injection(img: torch.Tensor, config: dict) -> torch.Tensor:
    """
    Region-Wise (Eq.5) + Overall (Eq.6) noise injection.

    Args:
        img:    Pre-denoised image [B, C, H, W] in [0, 1].
        config: Must contain stride, g_map, p_map, lam_p.
    Returns:
        Synthetically noisy image [B, C, H, W] in [0, 1].
    """
    B, C, H, W = img.shape
    stride = config["stride"]

    new_H = ((H + stride - 1) // stride) * stride
    new_W = ((W + stride - 1) // stride) * stride
    img_padded = F.pad(img, (0, new_W - W, 0, new_H - H), mode="constant", value=0)

    patches = img_padded.unfold(2, stride, stride).unfold(3, stride, stride)

    # Region mask: mean intensity per patch → noise factors (Eq.4)
    noise_idx = patches.mean(dim=(1, 4, 5)).clamp(1e-5, 0.15)
    gaussian_level = config["g_map"] * noise_idx           # σ_g[k] = kg × M[k]
    poisson_level  = config["p_map"] / noise_idx            # λ_p[k] = kp / M[k]

    g_exp = rearrange(gaussian_level, "b nh nw -> b 1 nh nw 1 1")
    p_exp = rearrange(poisson_level,  "b nh nw -> b 1 nh nw 1 1")

    # Region-Wise: Poisson + Gaussian per region (Eq.5)
    # clamp to ensure non-negative rates (required by torch.poisson)
    poisson_rate = torch.clamp(patches * p_exp, min=0.0)
    patches_noisy = torch.poisson(poisson_rate) / p_exp
    gauss = torch.normal(mean=torch.zeros_like(patches_noisy), std=(g_exp / 255.0))
    patches_noisy = torch.clamp(patches_noisy + gauss, 0, 1)

    noisy_img = rearrange(patches_noisy, "b c nh nw ph pw -> b c (nh ph) (nw pw)")
    noisy_img = noisy_img[:, :, :H, :W]

    # Overall Noise Injection: global Poisson (Eq.6)
    # clamp to ensure non-negative rates
    overall_rate = torch.clamp(noisy_img * config["lam_p"], min=0.0)
    noisy_img = torch.poisson(overall_rate) / config["lam_p"]
    noisy_img = torch.clamp(noisy_img, 0, 1)

    return noisy_img


# ══════════════════════════════════════════════
#  NOISE ESTIMATION
# ══════════════════════════════════════════════
def estimate_noise_sigma(stack_float: np.ndarray, clean: np.ndarray) -> float:
    """
    Estimate noise σ from residuals using the MAD (median absolute deviation)
    estimator, which is robust to outliers (calcium spikes).
    """
    n_sample = min(100, stack_float.shape[0])
    idx = np.linspace(0, stack_float.shape[0] - 1, n_sample, dtype=int)

    residuals = []
    for i in idx:
        diff = stack_float[i].astype(np.float64) - clean.astype(np.float64)
        residuals.append(diff.ravel())
    residuals = np.concatenate(residuals)

    mad = np.median(np.abs(residuals - np.median(residuals)))
    return float(1.4826 * mad)  # MAD → Gaussian σ conversion


def adaptive_noise_params(noise_sigma: float, val_range: float) -> dict:
    """
    Map estimated noise σ to FM2S noise injection hyper-parameters.

    Scaling follows Table 11 of the paper:
      Higher noise → larger kg (more Gaussian), smaller kp & lam_p
      Lower  noise → smaller kg, larger kp & lam_p
    """
    noise_frac = noise_sigma / val_range
    # Reference: SRDTrans moderate noise ≈ 0.03 in [0,1] range
    ratio = max(noise_frac / 0.03, 0.1)

    return {
        "g_map": float(np.clip(60 * ratio,       10,  300)),
        "p_map": float(np.clip(30 / ratio,        5,  500)),
        "lam_p": float(np.clip(150 / ratio,      20, 2000)),
    }


# ══════════════════════════════════════════════
#  TEMPORAL PRE-DENOISE (video-specific)
# ══════════════════════════════════════════════
def compute_temporal_clean_target(
    stack_float: np.ndarray,
    max_frames: int = 800,
) -> np.ndarray:
    """
    Compute a clean target from temporal median.

    The temporal median is ideal for calcium imaging:
      - Noise is independent across frames → cancelled by median
      - Calcium transients are brief → rejected by median
      - Static structures (neurons, background) are preserved
    """
    F_total = stack_float.shape[0]
    if F_total > max_frames:
        idx = np.linspace(0, F_total - 1, max_frames, dtype=int)
        sub = stack_float[idx]
    else:
        sub = stack_float

    return np.median(sub, axis=0).astype(np.float32)


# ══════════════════════════════════════════════
#  CORE TRAINING LOOP
# ══════════════════════════════════════════════
def _train_model(
    model: FM2SNetwork,
    clean_t: torch.Tensor,
    noisy_t: torch.Tensor,
    config: dict,
    verbose: bool = True,
    label: str = "",
) -> FM2SNetwork:
    """
    Two-stage FM2S training (paper Sec 3.4.2).

    Stage 1 (Eq.7): noisy → clean_target  (coarse, few steps)
    Stage 2 (Eq.8): synth_noisy(clean_target) → clean_target  (fine-grained)
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    model.train()

    t0 = time.time()

    # ── Stage 1: Coarse Feature Learning ──────────────
    for _ in range(config["stage1_steps"]):
        pred = model(noisy_t)
        loss = criterion(pred, clean_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if verbose:
        print(f"  {label}Stage 1 done | loss={loss.item():.6f}")

    # ── Stage 2: Fine-Grained Feature Learning ───────
    train_num = config["train_num"]
    max_epoch = config["max_epoch"]
    for i in range(train_num):
        synth_noisy = noise_injection(clean_t, config)
        for _ in range(max_epoch):
            pred = model(synth_noisy)
            loss = criterion(pred, clean_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {label}Stage 2 [{i+1:>4}/{train_num}]  "
                  f"loss={loss.item():.6f}  {elapsed:.1f}s")

    if verbose:
        print(f"  {label}Training complete: {time.time()-t0:.1f}s")

    return model


# ══════════════════════════════════════════════
#  SELF-ENSEMBLE (test-time geometric augmentation)
# ══════════════════════════════════════════════
def _apply_transform(x: torch.Tensor, t: int) -> torch.Tensor:
    """Apply one of 8 geometric transforms (4 rotations × 2 flips)."""
    if t >= 4:
        x = torch.flip(x, dims=[3])   # horizontal flip
    k = t % 4
    if k > 0:
        x = torch.rot90(x, k=k, dims=[2, 3])
    return x


def _apply_inverse_transform(x: torch.Tensor, t: int) -> torch.Tensor:
    """Inverse of _apply_transform."""
    k = t % 4
    if k > 0:
        x = torch.rot90(x, k=4 - k, dims=[2, 3])
    if t >= 4:
        x = torch.flip(x, dims=[3])
    return x


@torch.no_grad()
def _inference_self_ensemble(
    model: FM2SNetwork,
    inp: torch.Tensor,
    n_transforms: int = 8,
) -> torch.Tensor:
    """
    Self-ensemble: apply model under 8 geometric transforms, average.
    Typical improvement: +0.1–0.3 dB PSNR.
    """
    model.eval()
    accum = torch.zeros_like(inp)

    for t in range(n_transforms):
        x_t = _apply_transform(inp, t)
        y_t = model(x_t)
        accum += _apply_inverse_transform(y_t, t)

    return accum / n_transforms


# ══════════════════════════════════════════════
#  PUBLIC API: TRAIN ON VIDEO STACK
# ══════════════════════════════════════════════
def fm2s_train_on_stack(
    stack: np.ndarray,
    device: torch.device,
    config_overrides: dict = None,
    verbose: bool = True,
) -> tuple:
    """
    Zero-shot FM2S training exploiting temporal redundancy of a video stack.

    1. Computes temporal median → clean target
    2. Estimates noise σ → adapts injection parameters
    3. Trains with full paper budget (450 samples × 5 epochs)

    Args:
        stack:  3D numpy array [F, H, W] (any dtype/range)
        device: torch device
        config_overrides: override any default params
        verbose: print progress

    Returns:
        (model, config)
    """
    t0 = time.time()

    # ── Value range (handle negative values) ──────
    # Calcium imaging data can have negative values from background
    # subtraction.  Shift so that min → 0, then normalize to [0, 1].
    stack_float = stack.astype(np.float32)

    # Robust min/max (ignore extreme outliers)
    n_sample = min(200, stack_float.shape[0])
    sample_idx = np.linspace(0, stack_float.shape[0] - 1, n_sample, dtype=int)
    sampled = stack_float[sample_idx]
    robust_min = float(np.percentile(sampled, 0.1))
    robust_max = float(np.percentile(sampled, 99.9))

    # val_min: offset to shift data so minimum → 0
    # val_range: span from shifted-min (0) to shifted-max
    val_min = min(robust_min, 0.0)            # only shift if there are negatives
    val_range = max(robust_max - val_min, 1.0)
    val_range *= 1.05                          # 5% headroom

    if verbose:
        print(f"  Input: {stack.shape}, "
              f"raw=[{float(stack_float.min()):.1f}, {float(stack_float.max()):.1f}], "
              f"val_min={val_min:.1f}, val_range={val_range:.1f}")

    # ── Temporal median clean target ──────────────
    if verbose:
        print("  Computing temporal median clean target...")
    temporal_med = compute_temporal_clean_target(stack_float, max_frames=800)

    # Shift by val_min then normalize to [0, 1]
    clean_norm = np.clip((temporal_med - val_min) / val_range, 0, 1).astype(np.float32)

    # Light spatial median to suppress any remaining hot/dead pixels
    spatial_smooth = nd.median_filter(clean_norm, size=3).astype(np.float32)
    # Blend: mostly temporal (very clean) + a touch of spatial for hot pixels
    clean_target = np.clip(0.85 * clean_norm + 0.15 * spatial_smooth, 0, 1)

    if verbose:
        print(f"  Clean target range: [{clean_target.min():.4f}, {clean_target.max():.4f}]")

    # ── Noise estimation ──────────────────────────
    noise_sigma = estimate_noise_sigma(stack_float, temporal_med)
    adaptive = adaptive_noise_params(noise_sigma, val_range)

    if verbose:
        print(f"  Noise σ = {noise_sigma:.2f}  (normalized: {noise_sigma/val_range:.4f})")
        print(f"  Adaptive params: g_map={adaptive['g_map']:.1f}, "
              f"p_map={adaptive['p_map']:.1f}, lam_p={adaptive['lam_p']:.1f}")

    # ── Config ────────────────────────────────────
    config = {
        "stride": 5,            # Paper Table 10 (SRDTrans)
        **adaptive,             # Adaptive noise injection
        "n_chan": 5,            # Paper Table 10 (λ=5 for SRDTrans)
        "train_num": 450,       # Paper default (Fig.6 optimal)
        "max_epoch": 10,         # Paper default (Fig.6 optimal)
        "stage1_steps": 10,      # Paper default
        "lr": 1e-3,             # Paper default
        "val_range": val_range,
        "val_min": val_min,     # Offset for negative-value data
        "self_ensemble": True,  # 8-fold test-time augmentation
    }
    if config_overrides:
        config.update(config_overrides)

    n_chan = config["n_chan"]

    # ── Tensors ───────────────────────────────────
    clean_t = (
        torch.tensor(clean_target, dtype=torch.float32, device=device)
        .unsqueeze(0).repeat(1, n_chan, 1, 1)
    )

    # Stage 1 input: middle frame (shifted + normalized)
    mid = stack_float[stack.shape[0] // 2]
    mid_norm = np.clip((mid - val_min) / val_range, 0, 1).astype(np.float32)
    noisy_t = (
        torch.tensor(mid_norm, dtype=torch.float32, device=device)
        .unsqueeze(0).repeat(1, n_chan, 1, 1)
    )

    # ── Train ─────────────────────────────────────
    model = FM2SNetwork(n_chan).to(device)
    model = _train_model(model, clean_t, noisy_t, config, verbose=verbose)

    if verbose:
        print(f"  Total training wall time: {time.time()-t0:.1f}s")

    return model, config


# ══════════════════════════════════════════════
#  PUBLIC API: TRAIN ON SINGLE IMAGE (paper-faithful)
# ══════════════════════════════════════════════
def fm2s_train_single_image(
    noisy_img: np.ndarray,
    device: torch.device,
    config_overrides: dict = None,
    verbose: bool = True,
) -> tuple:
    """
    Paper-faithful zero-shot: train on ONE image using spatial median filter.
    This is the original FM2S algorithm (Eq.3, Eq.7, Eq.8).

    Args:
        noisy_img: 2D numpy array [H, W] (any dtype/range)
    Returns:
        (model, config)
    """
    img_float = noisy_img.astype(np.float32)
    # Robust range (handle negative values from background subtraction)
    robust_min = float(np.percentile(img_float, 0.1))
    robust_max = float(np.percentile(img_float, 99.9))
    val_min = min(robust_min, 0.0)
    val_range = max(robust_max - val_min, 1.0) * 1.05

    img_norm = np.clip((img_float - val_min) / val_range, 0, 1).astype(np.float32)
    # Paper Sec 3.3.1: median filter for Pre-Denoise
    clean_target = nd.median_filter(img_norm, size=3).astype(np.float32)

    config = {
        "stride": 5, "g_map": 60, "p_map": 30, "lam_p": 150,
        "n_chan": 5, "train_num": 450, "max_epoch": 5,
        "stage1_steps": 5, "lr": 1e-3,
        "val_range": val_range, "val_min": val_min,
        "self_ensemble": True,
    }
    if config_overrides:
        config.update(config_overrides)

    n_chan = config["n_chan"]

    clean_t = (
        torch.tensor(clean_target, dtype=torch.float32, device=device)
        .unsqueeze(0).repeat(1, n_chan, 1, 1)
    )
    noisy_t = (
        torch.tensor(img_norm, dtype=torch.float32, device=device)
        .unsqueeze(0).repeat(1, n_chan, 1, 1)
    )

    model = FM2SNetwork(n_chan).to(device)
    model = _train_model(model, clean_t, noisy_t, config, verbose=verbose)

    return model, config


# ══════════════════════════════════════════════
#  PUBLIC API: INFERENCE
# ══════════════════════════════════════════════
@torch.no_grad()
def fm2s_denoise_frame(
    model: FM2SNetwork,
    frame: np.ndarray,
    config: dict,
    device: torch.device,
) -> np.ndarray:
    """
    Denoise one 2D frame.  Returns float32 in original value range.
    Uses self-ensemble if config["self_ensemble"] is True.
    """
    n_chan     = config["n_chan"]
    val_range  = config["val_range"]
    val_min    = config.get("val_min", 0.0)
    use_ens    = config.get("self_ensemble", False)

    # Shift by val_min (handles negative values), then normalize to [0, 1]
    frame_norm = np.clip((frame.astype(np.float32) - val_min) / val_range, 0, 1)
    inp = (
        torch.tensor(frame_norm, dtype=torch.float32, device=device)
        .unsqueeze(0).repeat(1, n_chan, 1, 1)
    )

    if use_ens:
        out = _inference_self_ensemble(model, inp, n_transforms=8)
    else:
        model.eval()
        out = model(inp)

    # Unshift: map [0, 1] back to [val_min, val_min + val_range]
    out = torch.clamp(out, 0, 1) * val_range + val_min
    out = torch.mean(out, dim=1).squeeze()  # average amplified channels
    return out.cpu().numpy()


@torch.no_grad()
def fm2s_denoise_stack(
    model: FM2SNetwork,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    """
    Denoise a full 3D video stack [F, H, W] frame-by-frame.
    Returns float32 array in original value range.
    """
    F_total, H, W = stack.shape
    output = np.empty((F_total, H, W), dtype=np.float32)
    model.eval()

    ens_str = "ON (8x)" if config.get("self_ensemble", False) else "OFF"
    if verbose:
        print(f"  Self-ensemble: {ens_str}")

    t0 = time.time()
    for i in range(F_total):
        output[i] = fm2s_denoise_frame(
            model, stack[i].astype(np.float32), config, device
        )
        if verbose and (i + 1) % 300 == 0:
            fps = (i + 1) / (time.time() - t0)
            eta = (F_total - i - 1) / fps
            print(f"    [{i+1:>5}/{F_total}]  {fps:.1f} fps  ETA {eta:.0f}s")

    elapsed = time.time() - t0
    if verbose:
        print(f"  Inference done: {F_total} frames in {elapsed:.1f}s "
              f"({F_total/elapsed:.1f} fps)")

    return output


@torch.no_grad()
def fm2s_denoise_stack_perframe_zeroshot(
    stack: np.ndarray,
    device: torch.device,
    config_overrides: dict = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    TRUE per-frame zero-shot (paper-faithful):
    Train a fresh model on EACH frame, then denoise it.

    Highest quality but slow (~1-2s/frame with reduced budget).
    Budget is reduced to fit within 60 min for 1500 frames.
    """
    F_total, H, W = stack.shape
    output = np.empty((F_total, H, W), dtype=np.float32)

    # Reduced budget for feasibility within 60 min
    overrides = {
        "train_num": 80,
        "max_epoch": 3,
        "stage1_steps": 3,
        "self_ensemble": True,
    }
    if config_overrides:
        overrides.update(config_overrides)

    t0 = time.time()
    for i in range(F_total):
        frame = stack[i].astype(np.float32)

        model, cfg = fm2s_train_single_image(
            frame, device, config_overrides=overrides, verbose=False
        )

        output[i] = fm2s_denoise_frame(model, frame, cfg, device)

        if verbose and (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            fps = (i + 1) / elapsed
            eta = (F_total - i - 1) / fps
            print(f"    [{i+1:>5}/{F_total}]  {fps:.2f} fps  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.time() - t0
    if verbose:
        print(f"  Per-frame zero-shot: {elapsed:.1f}s ({F_total/elapsed:.2f} fps)")

    return output


# ══════════════════════════════════════════════
#  CHECKPOINT SAVE / LOAD
# ══════════════════════════════════════════════
def save_checkpoint(model: FM2SNetwork, config: dict, path: str):
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path: str, device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = FM2SNetwork(cfg["n_chan"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
