"""
Bias-Free 3D U-Net for Self-Supervised Calcium Imaging Denoising
================================================================

A clean, minimal Noise2Void implementation following the recipe of the
top performer on AI4Life-CIDC25 ("N2V_3DNB_V1 No-ConvBias 64³").

KEY DESIGN CHOICES (all matter, in order of importance):

1. NO BIAS in any Conv3d / ConvTranspose3d / GroupNorm / Linear layer.
   This is the single most important change. In N2V, bias terms let the
   network learn the local mean as a degenerate shortcut that satisfies
   the blind-spot loss without using spatial input. A bias-free network
   is mathematically incapable of this shortcut: scaling the input by
   any positive constant scales the output by the same constant
   (positive homogeneity), so the network MUST use spatial structure.

2. CUBIC 64x64x64 PATCHES. Balanced temporal/spatial context. Calcium
   imaging signals have strong temporal continuity (a neuron firing
   spans several frames) which a 32x128x128 patch underweights.

3. SMALL N2V MASK RADIUS (1 voxel). Calcium transients are 2-4 frames
   wide. A radius-2 neighborhood on the time axis often samples within
   the same transient, teaching the network to smooth across spikes.

4. RESIDUAL OUTPUT (predict noise, subtract from input). Easier
   optimization and prevents the network from having to reconstruct
   pixels it could trivially copy.

5. SHORT WARMUP. Just enough to give the network a structural prior;
   any longer and it anchors to the temporal median and loses
   transients.

API matches model.py / model_dvt.py exactly — drop-in replacement.
"""

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# NORMALIZATION (robust p3-p97)
# ══════════════════════════════════════════════════════════════

def compute_norm_params(stack: np.ndarray) -> dict:
    n = min(300, stack.shape[0])
    idx = np.linspace(0, stack.shape[0] - 1, n, dtype=int)
    sampled = stack[idx].astype(np.float64)
    p3 = float(np.percentile(sampled, 3))
    p97 = float(np.percentile(sampled, 97))
    scale = max(p97 - p3, 1e-6)
    return {"shift": p3, "scale": scale}


def normalize(data, params):
    return (data.astype(np.float32) - params["shift"]) / params["scale"]


def denormalize(data, params):
    return data.astype(np.float32) * params["scale"] + params["shift"]


# ══════════════════════════════════════════════════════════════
# BIAS-FREE BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class BFConvBlock3d(nn.Module):
    """
    Two bias-free 3x3x3 convolutions with GroupNorm (bias=False) and
    LeakyReLU.

    Why bias=False everywhere:
        f(alpha * x) = alpha * f(x)  for any alpha > 0.
    This positive-homogeneity property prevents the network from
    learning a constant offset that would let it predict the local mean
    in N2V's blind-spot loss without using neighborhood structure.
    GroupNorm's affine=False keeps this invariant intact.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        gn = min(8, out_ch)
        while out_ch % gn != 0:
            gn -= 1
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn, out_ch, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(gn, out_ch, affine=False),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ══════════════════════════════════════════════════════════════
# 3D U-NET — main architecture
# ══════════════════════════════════════════════════════════════

class BiasFreeUNet3D(nn.Module):
    """
    Standard 3D U-Net with three down/up stages — entirely bias-free.

    Encoder:  1 -> 32 -> 64 -> 128
    Decoder:  128 -> 64 -> 32 -> 1
    Output:   residual (predicted noise), subtracted from input.

    Spatial dims are auto-padded to multiples of 8 (three 2x downsamples).
    Drop-in replacement for the DVTUNet3D class — same name `UNet3D`.
    """

    def __init__(self, base_ch: int = 32):
        super().__init__()
        c1, c2, c3 = base_ch, base_ch * 2, base_ch * 4
        self.c1, self.c2, self.c3 = c1, c2, c3

        # Encoder
        self.enc1 = BFConvBlock3d(1, c1)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = BFConvBlock3d(c1, c2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = BFConvBlock3d(c2, c3)
        self.pool3 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = BFConvBlock3d(c3, c3)

        # Decoder (transpose convs are bias-free too)
        self.up3 = nn.ConvTranspose3d(c3, c3, 2, stride=2, bias=False)
        self.dec3 = BFConvBlock3d(c3 + c3, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, stride=2, bias=False)
        self.dec2 = BFConvBlock3d(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, stride=2, bias=False)
        self.dec1 = BFConvBlock3d(c1 + c1, c1)

        # Noise predictor — bias-free 1x1x1 conv
        self.out_conv = nn.Conv3d(c1, 1, 1, bias=False)

    def forward(self, x):
        identity = x

        # Pad spatial dims to multiples of 8
        _, _, D, H, W = x.shape
        pd = (8 - D % 8) % 8
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if pd or ph or pw:
            x = F.pad(x, (0, pw, 0, ph, 0, pd), mode="reflect")
            identity = F.pad(identity, (0, pw, 0, ph, 0, pd), mode="reflect")

        # Encoder with skip connections
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        # Decoder
        d3 = self.up3(b)
        d3 = _match_cat(d3, e3)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = _match_cat(d2, e2)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = _match_cat(d1, e1)
        d1 = self.dec1(d1)

        # Residual: clean = noisy - predicted_noise
        noise_pred = self.out_conv(d1)
        out = identity - noise_pred

        # Strip padding
        if pd or ph or pw:
            out = out[:, :, :D, :H, :W]
        return out


def _match_cat(up, skip):
    """Pad upsampled tensor to match skip shape, then concatenate."""
    dd = skip.shape[2] - up.shape[2]
    dh = skip.shape[3] - up.shape[3]
    dw = skip.shape[4] - up.shape[4]
    if dd or dh or dw:
        up = F.pad(up, (0, dw, 0, dh, 0, dd))
    return torch.cat([up, skip], dim=1)


# Alias for drop-in compatibility with anything importing `UNet3D`
UNet3D = BiasFreeUNet3D


# ══════════════════════════════════════════════════════════════
# 3D N2V MASKING — anisotropic to protect calcium transients
# ══════════════════════════════════════════════════════════════

def n2v_mask_3d(volume: torch.Tensor, mask_ratio: float = 0.015,
                radius_t: int = 1, radius_s: int = 2):
    """
    3D Noise2Void masking with anisotropic neighborhood.

    radius_t < radius_s on purpose: calcium transients are sharp on the
    time axis (2-4 frames). Sampling neighbors within radius 2 on time
    often falls inside the same transient, teaching the network to
    smooth them away. radius_t=1 keeps them sharp.
    """
    D, H, W = volume.shape
    n_vox = D * H * W
    n_mask = max(int(n_vox * mask_ratio), 1)

    flat_idx = torch.randperm(n_vox, device=volume.device)[:n_mask]
    mz = flat_idx // (H * W)
    my = (flat_idx % (H * W)) // W
    mx = flat_idx % W

    original = volume[mz, my, mx].clone()

    dz = torch.randint(-radius_t, radius_t + 1, (n_mask,), device=volume.device)
    dy = torch.randint(-radius_s, radius_s + 1, (n_mask,), device=volume.device)
    dx = torch.randint(-radius_s, radius_s + 1, (n_mask,), device=volume.device)
    same = (dz == 0) & (dy == 0) & (dx == 0)
    dz[same] = 1

    nz = (mz + dz).clamp(0, D - 1)
    ny = (my + dy).clamp(0, H - 1)
    nx = (mx + dx).clamp(0, W - 1)

    masked = volume.clone()
    masked[mz, my, mx] = volume[nz, ny, nx]
    return masked, (mz, my, mx), original


def _gaussian_window_3d(shape, sigma_frac=0.3, device="cpu"):
    """3D Gaussian window for sliding-window inference blending."""
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
    """8-fold spatial augmentation for [D, H, W] volumes."""
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════

def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
):
    """
    Two-stage zero-shot self-supervised training:
        Stage 0: short warmup against temporal median (structural prior)
        Stage 1: 3D Noise2Void blind-spot training (the main objective)
    """
    t0 = time.time()
    cfg = {
        # backbone
        "base_ch": 32,
        # patch sampling — cubic, leaderboard-winner style
        "patch_size": 64,
        "batch_size": 2,
        # schedule
        "warmup_iters": 150,
        "n2v_iters": 6000,
        "lr": 3e-4,
        # n2v masking
        "mask_ratio": 0.015,
        "mask_radius_t": 1,    # keep transients sharp
        "mask_radius_s": 2,    # standard spatial radius
    }
    if config:
        cfg.update(config)

    F_total, H, W = stack.shape
    ps = cfg["patch_size"]
    bs = cfg["batch_size"]

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" Bias-free U-Net3D, base_ch={cfg['base_ch']}")
        print(f" Patch: {ps}^3, batch={bs}")
        print(f" Stages: warmup={cfg['warmup_iters']}, "
              f"n2v={cfg['n2v_iters']}")
        print(f" N2V: mask_ratio={cfg['mask_ratio']}, "
              f"radius_t={cfg['mask_radius_t']}, "
              f"radius_s={cfg['mask_radius_s']}")

    # Normalize
    norm_params = compute_norm_params(stack)
    cfg["norm_params"] = norm_params
    stack_norm = normalize(stack, norm_params)
    if verbose:
        print(f" Norm: shift={norm_params['shift']:.2f}, "
              f"scale={norm_params['scale']:.2f}, "
              f"range=[{stack_norm.min():.3f}, {stack_norm.max():.3f}]")

    # Temporal median (warmup target)
    n_med = min(500, F_total)
    med_idx = np.linspace(0, F_total - 1, n_med, dtype=int)
    temporal_med = np.median(stack_norm[med_idx], axis=0).astype(np.float32)

    stack_t = torch.from_numpy(stack_norm).float().to(device)
    tmed_t = torch.from_numpy(temporal_med).float().to(device)

    # Model
    model = BiasFreeUNet3D(base_ch=cfg["base_ch"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f" Model params: {n_params:,}")

    # Verify the bias-free property
    n_bias = sum(1 for n, p in model.named_parameters() if "bias" in n)
    if verbose:
        print(f" Bias parameters: {n_bias} (must be 0 for bias-free)")
    assert n_bias == 0, "Network has bias parameters — not bias-free!"

    def random_patch():
        d = min(ps, F_total)
        h = min(ps, H)
        w = min(ps, W)
        t0_ = np.random.randint(0, max(F_total - d, 1))
        y0 = np.random.randint(0, max(H - h, 1))
        x0 = np.random.randint(0, max(W - w, 1))
        return stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w], tmed_t[y0:y0+h, x0:x0+w]

    # ─── Stage 0: temporal-median warmup ───────────────────
    if cfg["warmup_iters"] > 0:
        if verbose:
            print(f"\n [Stage 0] Warmup — {cfg['warmup_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["warmup_iters"], eta_min=cfg["lr"] * 0.1)
        crit = nn.MSELoss()
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
            pred = model(inp)
            loss = crit(pred, tgt)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()
            if verbose and (it + 1) % 50 == 0:
                print(f"   {it+1:>4}/{cfg['warmup_iters']} "
                      f"loss={rl/50:.6f}  {time.time()-t0:.1f}s")
                rl = 0.0

    # ─── Stage 1: 3D Noise2Void ────────────────────────────
    if cfg["n2v_iters"] > 0:
        if verbose:
            print(f"\n [Stage 1] 3D Noise2Void — {cfg['n2v_iters']} iters")
        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg["lr"] * 0.5, weight_decay=1e-5)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, cfg["n2v_iters"], eta_min=1e-6)
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
                    vol,
                    mask_ratio=cfg["mask_ratio"],
                    radius_t=cfg["mask_radius_t"],
                    radius_s=cfg["mask_radius_s"],
                )
                patches.append(masked.unsqueeze(0))
                all_orig.append((mz, my, mx, orig))
            inp = torch.stack(patches, dim=0).to(device)
            pred = model(inp)
            loss = torch.tensor(0.0, device=device)
            for b, (mz, my, mx, orig) in enumerate(all_orig):
                pred_at_mask = pred[b, 0, mz, my, mx]
                loss = loss + F.mse_loss(pred_at_mask, orig)
            loss = loss / bs
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sch.step()
            rl += loss.item()
            if verbose and (it + 1) % 500 == 0:
                lr_now = sch.get_last_lr()[0]
                print(f"   {it+1:>5}/{cfg['n2v_iters']} "
                      f"loss={rl/500:.6f} lr={lr_now:.2e} "
                      f"{time.time()-t0:.1f}s")
                rl = 0.0

    elapsed = time.time() - t0
    if verbose:
        print(f"\n Training complete: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    return model, cfg


# ══════════════════════════════════════════════════════════════
# SLIDING-WINDOW INFERENCE
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def denoise_stack(
    model: BiasFreeUNet3D,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    """Sliding-window inference with Gaussian blending."""
    model.eval()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    ps = config.get("patch_size", 64)
    pd = min(ps, F_total)
    phw = min(ps, min(H, W))
    pd = max((pd // 8) * 8, 8)
    phw = max((phw // 8) * 8, 8)
    stride = max(ps // 2, 8)
    stride_d = min(stride, max(pd // 2, 8))
    stride_hw = min(stride, max(phw // 2, 8))

    if verbose:
        print(f" Sliding window: patch={pd}x{phw}x{phw}, "
              f"stride={stride_d}x{stride_hw}x{stride_hw}")

    stack_norm = normalize(stack, norm_params)
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
                    patch = F.pad(
                        patch,
                        (0, phw - aw, 0, phw - ah, 0, pd - ad),
                        mode="reflect",
                    )
                inp = patch.unsqueeze(0).unsqueeze(0)
                pred = model(inp).squeeze(0).squeeze(0)
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
    output = denormalize(output, norm_params)

    safe_lo = norm_params["shift"] - 0.5 * norm_params["scale"]
    safe_hi = norm_params["shift"] + 1.5 * norm_params["scale"]
    output = np.clip(output, safe_lo, safe_hi)
    return output


# ══════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({"model_state_dict": model.state_dict(),
                "config": config,
                "arch": "BiasFreeUNet3D"}, path)
    print(f"Checkpoint saved -> {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BiasFreeUNet3D(base_ch=cfg["base_ch"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
