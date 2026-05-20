"""
n2v_3d_chhayansh — 3D U-Net Noise2Void denoiser, ported from

    https://github.com/chhayanshporwal/3d-n2v-calcium-denoising
    (AI4Life CIDC 2025 submission)

The upstream repo ships only inference + pre-trained weights. This module
reproduces the architecture exactly (so the shipped checkpoint loads
cleanly) and adds a 3D Noise2Void training loop that can train a fresh
model from scratch on your own stack.

Architecture (matches process.py + best_model_n2v.pth, ~298k params):
    enc1: Conv3d(1→32, 3³) → BN → ReLU → Conv3d(32→32, 3³) → BN → ReLU
    pool1: MaxPool3d(2)
    enc2: Conv3d(32→64, 3³) → BN → ReLU → Conv3d(64→64, 3³) → BN → ReLU
    pool2: MaxPool3d(2)            (defined but UNUSED — matches repo)
    up1:   ConvTranspose3d(64→32, kernel=2, stride=2)
    dec1:  Conv3d(64→32, 3³) → BN → ReLU → Conv3d(32→32, 3³) → BN → ReLU
    final: Conv3d(32→1, 1³)

There is ONE level of downsampling (the second pool is defined but the
forward pass does not use it — verified against the upstream code). The
network is NOT residual: it outputs the denoised image directly, not a
noise residual.

Inference uses sliding-window tiling with 12.5%/12.5% overlap and a
simple count-based average (no Gaussian blending — matches upstream).
Normalization is p3-p97 percentile scaling applied per-tile (the
upstream uses per-volume; we offer both).

Self-supervised training is 3D Noise2Void: random voxels are masked and
replaced by spatial neighbors; the network is supervised at the masked
positions only. This matches the standard N2V recipe used elsewhere in
this project, just on this specific architecture.

API (uniform with all other algos in this package):
    compute_norm_params, normalize, denormalize
    train_self_supervised(stack, device, config) -> (model, cfg)
    denoise_stack(model, stack, config, device)  -> np.ndarray
    save_checkpoint, load_checkpoint
    load_pretrained_inference(device)  -> (model, cfg)   # extra helper
"""

import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Default path to the pre-trained weights shipped with the upstream repo
PRETRAINED_WEIGHTS = Path(__file__).resolve().parent / "_weights" / "n2v_3d_chhayansh.pth"


# ══════════════════════════════════════════════════════════════
# NORMALIZATION  (p3-p97 percentile scaling — matches upstream)
# ══════════════════════════════════════════════════════════════

from runner import preprocessing as _prep

# IMPORTANT: the shipped pre-trained weights were trained with p3-p97
# percentile scaling on the FULL volume. Changing the default would
# break inference with those weights. Override at your own risk via
# config["normalization"].
DEFAULT_NORMALIZATION = "chhayansh"
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
# ARCHITECTURE — verbatim port from upstream process.py
# ══════════════════════════════════════════════════════════════

class UNet3D(nn.Module):
    """
    Exact match to upstream UNet3D so best_model_n2v.pth loads cleanly.

    Note: enc1 / enc2 / dec1 are nn.Sequential blocks whose internal
    layer indices are 0=Conv, 1=BN, 2=ReLU, 3=Conv, 4=BN, 5=ReLU.
    The checkpoint key names (enc1.0.weight, enc1.1.weight, ...) depend
    on this exact ordering — do not change.
    """

    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_c, out_c, 3, padding=1),
                nn.BatchNorm3d(out_c),
                nn.ReLU(inplace=True),
            )

        self.enc1  = conv_block(1, 32)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2  = conv_block(32, 64)
        # `pool2` exists in upstream's __init__ but is not used in
        # forward(). We keep it for checkpoint-key parity (though the
        # upstream checkpoint does not contain pool2 weights, so this is
        # benign).
        self.pool2 = nn.MaxPool3d(2)
        self.up1   = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec1  = conv_block(64, 32)
        self.final = nn.Conv3d(32, 1, 1)

    def forward(self, x):
        """x: [B, 1, D, H, W] -> [B, 1, D, H, W]
        Pads spatial dims to multiples of 2 (one downsample by /2)."""
        _, _, D, H, W = x.shape
        pd = (2 - D % 2) % 2
        ph = (2 - H % 2) % 2
        pw = (2 - W % 2) % 2
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        d1 = self.dec1(torch.cat([self.up1(e2), e1], dim=1))
        out = self.final(d1)

        if pd or ph or pw:
            out = out[:, :, :D, :H, :W]
        return out


# ══════════════════════════════════════════════════════════════
# 3D N2V MASKING — same recipe as the rest of the framework
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume: torch.Tensor, mask_ratio: float = 0.015,
                radius: int = 2):
    """Mask a fraction of voxels by replacing each with a random
    neighbor within `radius` (cube)."""
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
    """8-fold spatial augmentation for a [D, H, W] tensor."""
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


# ══════════════════════════════════════════════════════════════
# TRAINING — 3D Noise2Void on this architecture (zero-shot)
# ══════════════════════════════════════════════════════════════

def train_self_supervised(stack, device, config=None, verbose=True):
    """
    Self-supervised 3D Noise2Void training on the upstream UNet3D.

    Set `load_pretrained=True` in the config to skip training entirely
    and just load the shipped weights for inference-only evaluation.
    """
    t0 = time.time()
    cfg = {
        # Sampling
        "patch_d":          32,
        "patch_hw":         128,
        "batch_size":       2,
        # Schedule
        "warmup_iters":     0,
        "n2v_iters":        2000,
        "lr":               1e-3,
        # N2V mask
        "mask_ratio":       0.015,
        "mask_radius":      2,
        # Special: skip training, use upstream pre-trained weights
        "load_pretrained":  False,
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    pd, phw = cfg["patch_d"], cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        mode = ("inference-only (loading shipped weights)"
                if cfg["load_pretrained"] else "training from scratch")
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" UNet3D (upstream port): ~298k params")
        print(f" Mode: {mode}")
        if not cfg["load_pretrained"]:
            print(f"   Patch: {pd}x{phw}x{phw}, batch={bs}")
            print(f"   Schedule: warmup={cfg['warmup_iters']}, "
                  f"n2v={cfg['n2v_iters']}")
            print(f"   N2V: ratio={cfg['mask_ratio']}, "
                  f"radius={cfg['mask_radius']}")

    # ── Build the model ───────────────────────────────────────
    model = UNet3D().to(device)

    # Resolve normalization strategy. Default = "chhayansh" (p3-p97
    # full-volume) which is what the shipped weights expect.
    norm_name = cfg.get("normalization", DEFAULT_NORMALIZATION)
    norm_strategy = _prep.resolve_normalization(norm_name)
    norm_params = norm_strategy.compute_params(stack)
    cfg["norm_params"] = norm_params
    cfg["__resolved_normalization"] = norm_strategy.name
    # We don't use a temporal-target in chhayansh (no warmup stage), but
    # record the resolution anyway for traceability.
    tt_name = cfg.get("temporal_target", DEFAULT_TEMPORAL_TARGET)
    cfg["__resolved_temporal_target"] = _prep.resolve_temporal_target(tt_name).name

    if verbose:
        print(f"   Normalization: {norm_strategy.name}")
        if norm_strategy.name != "chhayansh":
            print(f"   WARNING: using non-default normalization with "
                  f"shipped pre-trained weights may produce poor output. "
                  f"The upstream model was trained on 'chhayansh' "
                  f"(p3-p97 full-volume).")

    # ── Inference-only path ───────────────────────────────────
    if cfg["load_pretrained"]:
        if not PRETRAINED_WEIGHTS.exists():
            raise FileNotFoundError(
                f"Pre-trained weights not found at {PRETRAINED_WEIGHTS}. "
                f"Place the upstream `best_model_n2v.pth` there or set "
                f"load_pretrained=False to train from scratch."
            )
        if verbose:
            print(f"   Loading weights from {PRETRAINED_WEIGHTS}")
        sd = torch.load(
            str(PRETRAINED_WEIGHTS), map_location=device, weights_only=False,
        )
        if isinstance(sd, dict) and "model_state_dict" in sd:
            sd = sd["model_state_dict"]
        elif isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if verbose:
            print(f"   Loaded ({len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys)")
        cfg["pretrained_path"] = str(PRETRAINED_WEIGHTS)
        elapsed = time.time() - t0
        if verbose:
            print(f"\n Setup complete: {elapsed:.1f}s")
        return model, cfg

    # ── Training path ─────────────────────────────────────────
    stack_norm = norm_strategy.forward(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    def random_patch():
        d = min(pd, F_total); h = min(phw, H); w = min(phw, W)
        t0_ = np.random.randint(0, max(F_total - d, 1))
        y0  = np.random.randint(0, max(H - h, 1))
        x0  = np.random.randint(0, max(W - w, 1))
        return stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w]

    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [N2V] training — {cfg['n2v_iters']} iters")
        opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6)
        model.train()
        rl = 0.0
        for it in range(cfg["n2v_iters"]):
            all_orig = []
            patches = []
            for _ in range(bs):
                vol = random_patch()
                aug = np.random.randint(0, 8)
                vol = _augment_3d(vol, aug)
                masked, (mz, my, mx), orig = n2v_mask_3d(
                    vol,
                    mask_ratio=cfg["mask_ratio"],
                    radius=cfg["mask_radius"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((mz, my, mx, orig))
            inp = torch.stack(patches, dim=0).to(device)
            pred = model(inp)
            loss = torch.tensor(0.0, device=device)
            for b, (mz, my, mx, orig) in enumerate(all_orig):
                pred_at_mask = pred[b, 0, mz, my, mx]
                # Predict the original-scale pixel value (denoised image
                # directly — matches upstream architecture which does NOT
                # use residual output).
                loss = loss + F.mse_loss(pred_at_mask, orig)
            loss = loss / bs

            if not torch.isfinite(loss):
                if verbose:
                    print(f"   WARN non-finite loss at iter {it}, skip")
                continue
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()
            if verbose and (it + 1) % 200 == 0:
                lr_now = sch.get_last_lr()[0]
                print(f"   {it+1:>5}/{cfg['n2v_iters']} "
                      f"loss={rl/200:.6f} lr={lr_now:.2e}  "
                      f"{time.time()-t0:.1f}s")
                rl = 0.0

    elapsed = time.time() - t0
    if verbose:
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return model, cfg


# ══════════════════════════════════════════════════════════════
# INFERENCE — sliding window matching upstream process.py
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(model, stack, config, device, verbose=True):
    """
    Sliding-window inference matching the upstream recipe:
        TILE_SIZE = (32, 128, 128)
        OVERLAP   = (4, 16, 16)
        Per-tile p3/p97 percentile normalization (upstream uses per-
        VOLUME; we use per-volume for consistency with our framework).
        Simple count-based averaging at overlaps (no Gaussian blend).
    """
    model.eval()
    model = model.float()

    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    td = min(config.get("tile_d", 32), F_total)
    thw = min(config.get("tile_hw", 128), H, W)
    od  = config.get("overlap_d",  4)
    ohw = config.get("overlap_hw", 16)
    sd  = max(td  - od,  1)
    shw = max(thw - ohw, 1)

    if verbose:
        print(f" Sliding window: tile={td}x{thw}x{thw}, "
              f"overlap={od}/{ohw}, stride={sd}/{shw}")

    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)
    stack_t = torch.from_numpy(stack_norm).float().to(device)

    prediction = torch.zeros(F_total, H, W, device=device)
    counts     = torch.zeros(F_total, H, W, device=device)

    z_starts = list(range(0, F_total, sd))
    y_starts = list(range(0, H, shw))
    x_starts = list(range(0, W, shw))

    total = len(z_starts) * len(y_starts) * len(x_starts)
    if verbose:
        print(f" Patches: {len(z_starts)}x{len(y_starts)}x{len(x_starts)} = {total}")

    t0 = time.time()
    done = 0
    for z in z_starts:
        z_end = min(z + td, F_total); z_start = max(0, z_end - td)
        for y in y_starts:
            y_end = min(y + thw, H); y_start = max(0, y_end - thw)
            for x in x_starts:
                x_end = min(x + thw, W); x_start = max(0, x_end - thw)
                crop = stack_t[z_start:z_end, y_start:y_end, x_start:x_end]
                inp = crop.unsqueeze(0).unsqueeze(0).float()
                out = model(inp).squeeze(0).squeeze(0)
                prediction[z_start:z_end, y_start:y_end, x_start:x_end] += out
                counts[z_start:z_end, y_start:y_end, x_start:x_end] += 1.0
                done += 1
                if verbose and (done % 50 == 0 or done == total):
                    print(f"   {done}/{total} ({100*done/total:.0f}%)  "
                          f"{time.time()-t0:.1f}s", end="\r")
    if verbose:
        print(f"\n Inference: {total} patches in {time.time()-t0:.1f}s")

    prediction = prediction / counts.clamp(min=1e-8)
    output = prediction.cpu().numpy()
    output = norm_strategy.inverse(output, norm_params)

    # ── Safety net (cheap, matches the rest of the framework) ──
    output = np.nan_to_num(
        output,
        nan=float(stack.mean()),
        posinf=float(stack.max()),
        neginf=float(stack.min()),
    )
    in_lo = max(float(stack.min()), 0.0)
    in_hi = float(stack.max())
    in_hi_padded = in_hi + 0.1 * max(in_hi - in_lo, 1.0)
    output = np.clip(output, in_lo, in_hi_padded)
    return output


# ══════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
        "arch": "UNet3D-N2V-chhayansh",
    }, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        cfg = ckpt.get("config", {})
        sd = ckpt["model_state_dict"]
    else:
        # Raw state_dict (e.g. upstream's best_model_n2v.pth)
        cfg = {}
        sd = ckpt
    model = UNet3D().to(device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, cfg


# ══════════════════════════════════════════════════════════════
# CONVENIENCE: load the shipped pre-trained weights directly
# ══════════════════════════════════════════════════════════════

def load_pretrained_inference(device=None):
    """
    One-call helper to load the upstream `best_model_n2v.pth` for
    inference-only use. The config dict has no `norm_params` — caller
    must call compute_norm_params(stack) and store the result before
    calling denoise_stack().
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not PRETRAINED_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Pre-trained weights not found at {PRETRAINED_WEIGHTS}."
        )
    model, _ = load_checkpoint(str(PRETRAINED_WEIGHTS), device=device)
    return model, {"pretrained_path": str(PRETRAINED_WEIGHTS)}
