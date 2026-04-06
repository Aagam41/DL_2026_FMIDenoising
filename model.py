"""
3D U-Net Blind-Spot Denoiser for Calcium Imaging
=================================================

Architecture:
  - 3D U-Net processes video as a volume (D×H×W) so temporal
    consistency is learned jointly with spatial structure.
  - Two encoder levels with 3D convolutions, skip connections,
    GroupNorm, and residual (noise-prediction) output.

Training (self-supervised, no clean data):
  Stage 0 — Temporal-median warmup:
    Supervised initialization against the temporal median.
    Gives the network a strong structural prior quickly.

  Stage 1 — 3D Blind-spot (Noise2Void):
    Randomly mask ~1% of voxels, replace with a random 3D neighbor.
    Network predicts the original value from surrounding spatial AND
    temporal context.  Because noise is independent per voxel, the
    optimal prediction converges to the clean signal.

Inference:
  Sliding-window over overlapping 3D patches with Gaussian blending
  to eliminate boundary artifacts.

Normalization:
  Percentile-based (p3–p97) to handle intensity outliers and negative
  values common in microscopy data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math


# ══════════════════════════════════════════════════════════════
#  NORMALIZATION (p3–p97 robust scaling)
# ══════════════════════════════════════════════════════════════

def compute_norm_params(stack: np.ndarray) -> dict:
    """Robust percentile-based normalization parameters."""
    n = min(300, stack.shape[0])
    idx = np.linspace(0, stack.shape[0] - 1, n, dtype=int)
    sampled = stack[idx].astype(np.float64)
    p3  = float(np.percentile(sampled, 3))
    p97 = float(np.percentile(sampled, 97))
    scale = max(p97 - p3, 1e-6)
    return {"shift": p3, "scale": scale}


def normalize(data, params):
    return (data.astype(np.float32) - params["shift"]) / params["scale"]


def denormalize(data, params):
    return data.astype(np.float32) * params["scale"] + params["shift"]


# ══════════════════════════════════════════════════════════════
#  3D U-NET
# ══════════════════════════════════════════════════════════════

class ConvBlock3d(nn.Module):
    """Two 3×3×3 convolutions + GroupNorm + LeakyReLU."""
    def __init__(self, in_ch, out_ch):
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


class UNet3D(nn.Module):
    """
    3D U-Net for volumetric (D×H×W) denoising.

    Input:  [B, 1, D, H, W]
    Output: [B, 1, D, H, W]

    Uses residual learning: output = input - predicted_noise.
    Spatial dims must be divisible by 4, temporal by 4.
    """
    def __init__(self, base_ch: int = 32):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4  # 32, 64, 128

        self.enc1 = ConvBlock3d(1, c1)
        self.pool1 = nn.MaxPool3d((2, 2, 2))

        self.enc2 = ConvBlock3d(c1, c2)
        self.pool2 = nn.MaxPool3d((2, 2, 2))

        self.bottleneck = ConvBlock3d(c2, c3)

        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock3d(c2 + c2, c2)

        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock3d(c1 + c1, c1)

        self.out_conv = nn.Conv3d(c1, 1, 1)

    def forward(self, x):
        """x: [B, 1, D, H, W]  →  [B, 1, D, H, W]"""
        identity = x

        # Pad to multiples of 4
        _, _, D, H, W = x.shape
        pd = (4 - D % 4) % 4
        ph = (4 - H % 4) % 4
        pw = (4 - W % 4) % 4
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")
            identity = F.pad(identity, (0, pw, 0, ph, 0, pd), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b  = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = _match_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_cat(d1, e1)
        d1 = self.dec1(d1)

        noise = self.out_conv(d1)
        out = identity - noise   # residual: clean = noisy − predicted noise

        # Remove padding
        if pd or ph or pw:
            out = out[:, :, :D, :H, :W]
        return out


def _match_cat(up, skip):
    """Pad the upsampled tensor to match the skip tensor, then concatenate."""
    dd = skip.shape[2] - up.shape[2]
    dh = skip.shape[3] - up.shape[3]
    dw = skip.shape[4] - up.shape[4]
    if dd or dh or dw:
        up = F.pad(up, (0, dw, 0, dh, 0, dd))
    return torch.cat([up, skip], dim=1)


# ══════════════════════════════════════════════════════════════
#  3D BLIND-SPOT MASKING (Noise2Void)
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(
    volume: torch.Tensor,
    mask_ratio: float = 0.008,
    radius: int = 2,
):
    """
    Noise2Void 3D masking.

    Selects random voxels, replaces each with a random neighbour's value.
    Returns the masked volume, mask indices, and the original values.

    Args:
        volume: [D, H, W] tensor
        mask_ratio: fraction of voxels to mask
        radius: neighbourhood radius for replacement sampling
    Returns:
        (masked_volume, flat_indices, original_values)
    """
    D, H, W = volume.shape
    n_vox = D * H * W
    n_mask = max(int(n_vox * mask_ratio), 1)

    flat_idx = torch.randperm(n_vox, device=volume.device)[:n_mask]
    # Convert flat → 3D indices
    mz = flat_idx // (H * W)
    my = (flat_idx % (H * W)) // W
    mx = flat_idx % W

    original = volume[mz, my, mx].clone()

    # Random 3D neighbour offset (avoid self)
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


# ══════════════════════════════════════════════════════════════
#  GAUSSIAN BLENDING WINDOW (for sliding-window inference)
# ══════════════════════════════════════════════════════════════

def _gaussian_window_3d(shape, sigma_frac=0.25, device="cpu"):
    """
    3D Gaussian blending window.
    Higher weight at center, tapers to near-zero at edges.
    """
    windows = []
    for s in shape:
        coords = torch.arange(s, dtype=torch.float32, device=device)
        center = (s - 1) / 2.0
        sigma = s * sigma_frac
        w = torch.exp(-0.5 * ((coords - center) / sigma) ** 2)
        windows.append(w)
    # Outer product: D × H × W
    w3d = windows[0][:, None, None] * windows[1][None, :, None] * windows[2][None, None, :]
    return w3d.clamp(min=1e-6)


# ══════════════════════════════════════════════════════════════
#  TRAINING
# ══════════════════════════════════════════════════════════════

def _random_augment_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    """Apply one of 8 spatial augmentations (4 rot × 2 flip) to [D, H, W]."""
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])   # flip W
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])  # rotate H×W plane
    return vol


def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
) -> tuple:
    """
    Self-supervised training of the 3D U-Net on a noisy video stack.

    Stage 0 — Temporal-median warmup (fast structural initialization).
    Stage 1 — 3D Blind-spot (Noise2Void) training.

    Args:
        stack:  [F, H, W] numpy array
        device: torch device
        config: override defaults
        verbose: progress printing

    Returns:
        (model, config_dict)
    """
    t0 = time.time()

    cfg = {
        "base_ch": 32,
        "patch_d": 32,          # temporal patch depth
        "patch_hw": 128,        # spatial patch size
        "batch_size": 2,
        "warmup_iters": 500,    # Stage 0
        "n2v_iters": 3000,      # Stage 1
        "lr": 3e-4,
        "mask_ratio": 0.008,    # ~0.8% of voxels masked
        "mask_radius": 2,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        print(f"  Stack: {stack.shape}, device: {device}")
        print(f"  3D U-Net: base_ch={cfg['base_ch']}")
        print(f"  Patch: {pd}×{phw}×{phw}, batch={bs}")
        print(f"  Stages: warmup={cfg['warmup_iters']}, n2v={cfg['n2v_iters']}")

    # ── Normalize (p3–p97) ────────────────────────────
    norm_params = compute_norm_params(stack)
    cfg["norm_params"] = norm_params
    stack_norm = normalize(stack, norm_params)

    if verbose:
        print(f"  Norm: shift={norm_params['shift']:.2f}, "
              f"scale={norm_params['scale']:.2f}, "
              f"range=[{stack_norm.min():.3f}, {stack_norm.max():.3f}]")

    # ── Temporal median (for warmup target) ───────────
    n_med = min(500, F_total)
    med_idx = np.linspace(0, F_total - 1, n_med, dtype=int)
    temporal_med = np.median(stack_norm[med_idx], axis=0).astype(np.float32)
    if verbose:
        print(f"  Temporal median: [{temporal_med.min():.3f}, {temporal_med.max():.3f}]")

    # Move to GPU
    stack_t = torch.from_numpy(stack_norm).float().to(device)      # [F, H, W]
    tmed_t  = torch.from_numpy(temporal_med).float().to(device)    # [H, W]

    # ── Model ─────────────────────────────────────────
    model = UNet3D(base_ch=cfg["base_ch"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"  Model params: {n_params:,}")

    # Helper: extract random 3D patch
    def random_patch():
        t0_ = np.random.randint(0, max(F_total - pd, 1))
        y0  = np.random.randint(0, max(H - phw, 1))
        x0  = np.random.randint(0, max(W - phw, 1))
        d  = min(pd, F_total)
        h  = min(phw, H)
        w  = min(phw, W)
        return stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w], tmed_t[y0:y0+h, x0:x0+w]

    # ════════════════════════════════════════════════
    #  Stage 0: Temporal-Median Warmup
    # ════════════════════════════════════════════════
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n  [Stage 0] Temporal-median warmup — {cfg['warmup_iters']} iters")

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1,
        )
        criterion = nn.MSELoss()
        model.train()
        rl = 0.0

        for it in range(cfg["warmup_iters"]):
            patches = []
            targets = []
            for _ in range(bs):
                vol, tmed_crop = random_patch()
                aug = np.random.randint(0, 8)
                vol = _random_augment_3d(vol, aug)
                tmed_crop = _random_augment_3d(
                    tmed_crop.unsqueeze(0).expand(vol.shape[0], -1, -1), aug
                )
                patches.append(vol.unsqueeze(0))             # [1, D, H, W]
                targets.append(tmed_crop.unsqueeze(0))        # [1, D, H, W]

            inp = torch.stack(patches, dim=0).to(device)      # [B,1,D,H,W]
            tgt = torch.stack(targets, dim=0).to(device)      # [B,1,D,H,W]

            pred = model(inp)
            loss = criterion(pred, tgt)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            rl += loss.item()
            if verbose and (it + 1) % 200 == 0:
                print(f"    {it+1:>5}/{cfg['warmup_iters']}  "
                      f"loss={rl/200:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # ════════════════════════════════════════════════
    #  Stage 1: 3D Blind-Spot (Noise2Void)
    # ════════════════════════════════════════════════
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n  [Stage 1] 3D Blind-spot — {cfg['n2v_iters']} iters")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg["lr"] * 0.5, weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg["n2v_iters"], eta_min=1e-6,
        )
        model.train()
        rl = 0.0

        for it in range(cfg["n2v_iters"]):
            all_pred_vals = []
            all_orig_vals = []

            patches = []
            for _ in range(bs):
                vol, _ = random_patch()
                aug = np.random.randint(0, 8)
                vol = _random_augment_3d(vol, aug)

                # Apply 3D blind-spot mask
                masked, (mz, my, mx), orig = n2v_mask_3d(
                    vol, mask_ratio=cfg["mask_ratio"], radius=cfg["mask_radius"]
                )
                patches.append(masked.unsqueeze(0))  # [1, D, H, W]
                all_orig_vals.append((mz, my, mx, orig))

            inp = torch.stack(patches, dim=0).to(device)  # [B,1,D,H,W]
            pred = model(inp)                               # [B,1,D,H,W]

            # Loss only at masked voxels
            loss = torch.tensor(0.0, device=device)
            for b, (mz, my, mx, orig) in enumerate(all_orig_vals):
                pred_at_mask = pred[b, 0, mz, my, mx]
                loss = loss + F.mse_loss(pred_at_mask, orig)
            loss = loss / bs

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            rl += loss.item()
            if verbose and (it + 1) % 500 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"    {it+1:>5}/{cfg['n2v_iters']}  "
                      f"loss={rl/500:.6f}  lr={lr_now:.2e}  "
                      f"{time.time()-t0:.1f}s")
                rl = 0.0

    elapsed = time.time() - t0
    if verbose:
        print(f"\n  Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return model, cfg


# ══════════════════════════════════════════════════════════════
#  SLIDING-WINDOW INFERENCE
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(
    model: UNet3D,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    """
    Denoise the full stack using overlapping 3D sliding windows
    with Gaussian blending to prevent edge artifacts.

    Args:
        model:  Trained UNet3D
        stack:  [F, H, W] numpy array (original values)
        config: Training config (contains norm_params etc.)
        device: Torch device

    Returns:
        Denoised [F, H, W] numpy array in original value range.
    """
    model.eval()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    # Patch / stride configuration
    pd  = min(config.get("patch_d", 32), F_total)
    phw = min(config.get("patch_hw", 128), H, W)

    # Make patch dims divisible by 4 (U-Net constraint)
    pd  = max((pd  // 4) * 4, 4)
    phw = max((phw // 4) * 4, 4)

    stride_d  = max(pd  // 2, 4)
    stride_hw = max(phw // 2, 4)

    if verbose:
        print(f"  Sliding window: patch={pd}×{phw}×{phw}, "
              f"stride={stride_d}×{stride_hw}×{stride_hw}")

    # Normalize
    stack_norm = normalize(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    # Output buffers (accumulate weighted predictions)
    output_sum    = torch.zeros(F_total, H, W, device=device)
    weight_sum    = torch.zeros(F_total, H, W, device=device)

    # Gaussian blending window
    gauss_win = _gaussian_window_3d((pd, phw, phw), sigma_frac=0.3, device=device)

    # Generate patch positions
    z_starts = list(range(0, max(F_total - pd, 0) + 1, stride_d))
    if len(z_starts) == 0 or z_starts[-1] + pd < F_total:
        z_starts.append(max(F_total - pd, 0))
    y_starts = list(range(0, max(H - phw, 0) + 1, stride_hw))
    if len(y_starts) == 0 or y_starts[-1] + phw < H:
        y_starts.append(max(H - phw, 0))
    x_starts = list(range(0, max(W - phw, 0) + 1, stride_hw))
    if len(x_starts) == 0 or x_starts[-1] + phw < W:
        x_starts.append(max(W - phw, 0))

    # Deduplicate
    z_starts = sorted(set(z_starts))
    y_starts = sorted(set(y_starts))
    x_starts = sorted(set(x_starts))

    total_patches = len(z_starts) * len(y_starts) * len(x_starts)
    if verbose:
        print(f"  Patches: {len(z_starts)}×{len(y_starts)}×{len(x_starts)} "
              f"= {total_patches}")

    t0 = time.time()
    done = 0
    for z0 in z_starts:
        z1 = min(z0 + pd, F_total)
        actual_d = z1 - z0
        for y0 in y_starts:
            y1 = min(y0 + phw, H)
            actual_h = y1 - y0
            for x0 in x_starts:
                x1 = min(x0 + phw, W)
                actual_w = x1 - x0

                patch = stack_t[z0:z1, y0:y1, x0:x1]

                # Pad if patch is smaller than expected
                need_pad = (actual_d < pd) or (actual_h < phw) or (actual_w < phw)
                if need_pad:
                    patch = F.pad(
                        patch,
                        (0, phw - actual_w, 0, phw - actual_h, 0, pd - actual_d),
                        mode="reflect",
                    )

                inp = patch.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
                pred = model(inp).squeeze(0).squeeze(0) # [D, H, W]

                # Trim padding
                pred = pred[:actual_d, :actual_h, :actual_w]
                win  = gauss_win[:actual_d, :actual_h, :actual_w]

                output_sum[z0:z1, y0:y1, x0:x1] += pred * win
                weight_sum[z0:z1, y0:y1, x0:x1] += win

                done += 1

        if verbose:
            elapsed = time.time() - t0
            pct = done / total_patches * 100
            print(f"    {done}/{total_patches} patches ({pct:.0f}%)  "
                  f"{elapsed:.1f}s", end="\r")

    if verbose:
        print(f"\n  Inference: {total_patches} patches in "
              f"{time.time()-t0:.1f}s")

    # Divide by weight and denormalize
    output = output_sum / weight_sum.clamp(min=1e-8)
    output = output.cpu().numpy()
    output = denormalize(output, norm_params)

    # Clamp to safe range
    safe_lo = norm_params["shift"] - 0.5 * norm_params["scale"]
    safe_hi = norm_params["shift"] + 1.5 * norm_params["scale"]
    output = np.clip(output, safe_lo, safe_hi)

    return output


# ══════════════════════════════════════════════════════════════
#  CHECKPOINT
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({"model_state_dict": model.state_dict(), "config": config}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = UNet3D(base_ch=cfg["base_ch"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
