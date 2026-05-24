"""
DeepCAD — Deep self-supervised learning for Calcium imaging Denoising
======================================================================

Implements the network from Li et al., "Reinforcing neuron extraction and
spike inference in calcium imaging using deep self-supervised denoising"
(Nature Methods, 2021).  Architecture summary from the paper's Supplementary
Fig. 1:

    Encoder (3 blocks):  each = 2 × Conv3D(3×3×3) → LeakyReLU → GroupNorm
                                   → MaxPool3D(2×2×2, stride=2)
    Bottleneck:          2 × Conv3D(3×3×3) → LeakyReLU → GroupNorm
    Decoder (3 blocks):  3D-nearest upsample → 2 × Conv3D(3×3×3) →
                                   LeakyReLU → GroupNorm
    Skip connections from each encoder block to the matching decoder block.

DeepCAD-RT (Li et al., Nature Biotech 2022) is the same architecture with
fewer channels ("compressed model parameters by 94%"). We ship that as a
config preset (configs/deepcad_rt.py) on the same module rather than as a
separate algo.

Training methodology — IMPORTANT NOTE
-------------------------------------
The official DeepCAD/DeepCAD-RT training pipeline samples *pairs* of
sub-volumes from interleaved (odd/even) frames and trains the network to
predict one sub-volume from the other (a Noise2Noise-style scheme). Our
framework uses self-supervised Noise2Void blind-spot loss for all algos for
consistency. This means:

  * The ARCHITECTURE is paper-faithful.
  * The TRAINING is our standard self-supervised N2V flow, NOT the official
    interleaved-frame N2N training.

If you want to compare to published DeepCAD numbers directly, that requires
running the official repo (https://github.com/cabooster/DeepCAD-RT) — our
implementation here is wired into the framework's uniform training so it
can be benchmarked apples-to-apples against other algos in our suite.

API parity with the rest of our algos.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from runner import preprocessing as _prep


# ══════════════════════════════════════════════════════════════
# NORMALIZATION
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
# 3D U-NET BUILDING BLOCKS (matching the paper's description)
# ══════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two 3×3×3 convolutions, each followed by LeakyReLU and GroupNorm.

    Order = Conv → LeakyReLU → GroupNorm (matching the paper text "two
    3×3×3 convolutional layers followed by a leaky rectified linear unit
    (LeakyReLU), a group normalization layer"). Some implementations use
    Conv → GN → ReLU; we follow the paper.
    """

    def __init__(self, in_ch: int, out_ch: int, n_groups: int = 4):
        super().__init__()
        # GroupNorm requires out_ch % n_groups == 0; auto-shrink if not
        g = n_groups
        while g > 1 and out_ch % g != 0:
            g -= 1
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.GroupNorm(g, out_ch),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.GroupNorm(g, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class DeepCADUNet(nn.Module):
    """
    The DeepCAD 3D U-Net.

    Defaults match the paper: 3 encoder + 3 decoder blocks, base_ch=16,
    doubling at each downsample. DeepCAD-RT uses base_ch=8 (or smaller) —
    set via config.

    Input:  [B, 1, D, H, W]   (D = temporal dim, H/W = spatial)
    Output: [B, 1, D, H, W]   (residual: out = noisy - predicted_noise)
    """

    def __init__(self, base_ch: int = 16, depth: int = 3,
                 n_groups: int = 4):
        super().__init__()
        self.depth = depth
        self.divisor = 2 ** depth   # input dims must be multiples of this

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        ch = base_ch
        for _ in range(depth):
            self.encoders.append(ConvBlock(in_ch, ch, n_groups))
            self.pools.append(nn.MaxPool3d(2))
            in_ch, ch = ch, ch * 2

        # Bottleneck
        self.bottleneck = ConvBlock(in_ch, ch, n_groups)

        # Decoder
        self.decoders = nn.ModuleList()
        for _ in range(depth):
            # 3D nearest upsample → conv block with skip concat
            #    concat is along channel dim, so decoder input has
            #    (ch + skip_ch) channels.
            skip_ch = in_ch              # this loop's "in_ch" before update
            self.decoders.append(ConvBlock(ch + skip_ch, in_ch, n_groups))
            ch, in_ch = in_ch, in_ch // 2

        # Final 1×1×1 conv to predict the residual
        # (after the loop, `ch` is now base_ch, the last decoder output)
        self.out_conv = nn.Conv3d(ch, 1, 1, bias=True)

    def forward(self, x):
        identity = x

        # Pad spatial+temporal dims to multiples of 2**depth
        _, _, D, H, W = x.shape
        pd_ = (self.divisor - D % self.divisor) % self.divisor
        ph_ = (self.divisor - H % self.divisor) % self.divisor
        pw_ = (self.divisor - W % self.divisor) % self.divisor
        if pd_ or ph_ or pw_:
            # reflect pad fails when pad >= dim; fall back to replicate
            mode = ("reflect" if (pd_ < D and ph_ < H and pw_ < W)
                    else "replicate")
            x = F.pad(x, (0, pw_, 0, ph_, 0, pd_), mode=mode)
            identity_p = F.pad(identity, (0, pw_, 0, ph_, 0, pd_),
                                mode=mode)
        else:
            identity_p = identity

        # Encoder pass — store skips
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder pass with skips (in reverse order)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        noise_pred = self.out_conv(x)
        out = identity_p - noise_pred

        # Strip padding
        if pd_ or ph_ or pw_:
            out = out[:, :, :D, :H, :W]
        return out


# ══════════════════════════════════════════════════════════════
# AUGMENTATIONS + INFERENCE WINDOW
# ══════════════════════════════════════════════════════════════

def _augment_3d(vol: torch.Tensor, aug_id: int) -> torch.Tensor:
    if aug_id >= 4:
        vol = torch.flip(vol, dims=[2])
    k = aug_id % 4
    if k > 0:
        vol = torch.rot90(vol, k=k, dims=[1, 2])
    return vol


def _gaussian_window_3d(shape, sigma_frac=0.25, device="cpu"):
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


# ══════════════════════════════════════════════════════════════
# TRAINING — Noise2Noise on interleaved odd/even frames
# (paper-faithful per Li et al. Nat Methods 2021)
# ══════════════════════════════════════════════════════════════
#
# The DeepCAD self-supervised scheme:
#   1. Crop a random 3D sub-volume of T frames (T even).
#   2. Split into TWO interleaved sub-stacks:
#         INPUT  = frames at indices 0,2,4,...  (every other frame)
#         TARGET = frames at indices 1,3,5,...  (the OTHER set)
#      Each has T/2 frames.
#   3. The two share approximately the same calcium signal (assuming the
#      stack is imaged near video rate, which is typical) but have
#      INDEPENDENT noise. So predicting one from the other is a valid
#      Noise2Noise (N2N) setup.
#   4. Loss is direct L1 / L2 on the full prediction — no blind-spot
#      masking, no temporal-median warmup.
#   5. At inference, the model is applied to the FULL stack (T frames,
#      not T/2). DeepCAD's 3D U-Net is fully convolutional on the time
#      axis so the architecture handles arbitrary T.
#
# Architecture note: input shape goes from [B,1,T/2,H,W] (train) to
# [B,1,T_infer,H,W] (infer). The U-Net's pad-to-multiple-of-2^depth
# logic handles both.

def train_self_supervised(
    stack: np.ndarray,
    device: torch.device,
    config: dict = None,
    verbose: bool = True,
):
    """Noise2Noise training on interleaved odd/even frames (paper-faithful)."""
    t0 = time.time()
    cfg = {
        # Backbone
        "base_ch":        16,        # DeepCAD default; RT preset uses 4
        "depth":          3,
        # Patch sampling — patch_d is the TRAINING sub-volume size (even);
        # the network sees patch_d // 2 frames per iteration since we split
        # into odd/even halves.
        "patch_d":        32,        # must be even
        "patch_hw":       64,
        "batch_size":     2,
        # Schedule (DeepCAD paper trains for a few thousand iters per stack)
        "n2n_iters":      3000,
        "lr":             3e-4,
        # Loss — paper uses L1 (mean absolute error)
        "loss":           "l1",      # "l1" or "l2"
    }
    if config:
        cfg.update(config)

    # Ensure patch_d is even — N2N split requires it
    if cfg["patch_d"] % 2 != 0:
        cfg["patch_d"] += 1
        if verbose:
            print(f" patch_d was odd; bumped to {cfg['patch_d']} for N2N split")

    F_total, H, W = stack.shape
    pd = min(cfg["patch_d"], F_total - (F_total % 2))  # ensure even
    if pd < 2:
        raise ValueError(f"Stack too short for N2N: only {F_total} frames")
    phw = cfg["patch_hw"]
    bs = cfg["batch_size"]

    if verbose:
        print(f" Stack: {stack.shape}, device: {device}")
        print(f" DeepCAD: base_ch={cfg['base_ch']}, depth={cfg['depth']}")
        print(f" Patch: {pd}x{phw}x{phw} → split into 2 interleaved "
              f"{pd//2}-frame substacks (N2N), batch={bs}")
        print(f" Schedule: n2n_iters={cfg['n2n_iters']}, lr={cfg['lr']}")
        print(f" Loss: {cfg['loss']}")
        print(f" Precision: fp32")
        print(f" Self-supervision: Noise2Noise (paper-faithful, "
              f"NO temporal-median warmup)")

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

    # NOTE: DeepCAD does NOT use temporal-median warmup. The temporal_target
    # config key is ignored here. We still record it so other framework
    # components don't error out.
    cfg["__resolved_temporal_target"] = "noop_deepcad_n2n_does_not_use_it"

    stack_t = torch.from_numpy(stack_norm).float().to(device)

    model = DeepCADUNet(
        base_ch=cfg["base_ch"],
        depth=cfg["depth"],
    ).to(device)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f" Model params: {n_params:,}")

    def random_n2n_pair():
        """Sample a sub-volume, split into odd/even halves, return both."""
        # Pick a random spatiotemporal anchor that fits a pd-frame sub-volume
        t0_ = np.random.randint(0, max(F_total - pd + 1, 1))
        y0 = np.random.randint(0, max(H - phw + 1, 1))
        x0 = np.random.randint(0, max(W - phw + 1, 1))
        d = min(pd, F_total); h = min(phw, H); w = min(phw, W)
        sub = stack_t[t0_:t0_+d, y0:y0+h, x0:x0+w]   # [d, h, w]
        # Split into interleaved halves
        even = sub[0::2]   # frames 0, 2, 4, ...
        odd  = sub[1::2]   # frames 1, 3, 5, ...
        # Randomly assign which is input and which is target — this
        # gives the network both directions and is a standard N2N trick.
        if np.random.rand() < 0.5:
            return even, odd
        else:
            return odd, even

    loss_fn = F.l1_loss if cfg["loss"] == "l1" else F.mse_loss

    # ── Single-stage N2N training ────────────────────────
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                             weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, cfg["n2n_iters"], eta_min=cfg["lr"] * 0.05)
    model.train()
    rl = 0.0
    print_every = max(50, cfg["n2n_iters"] // 20)
    if verbose:
        print(f"\n [N2N training] {cfg['n2n_iters']} iters")
    for it in range(cfg["n2n_iters"]):
        inputs, targets = [], []
        for _ in range(bs):
            inp_vol, tgt_vol = random_n2n_pair()
            aug = np.random.randint(0, 8)
            inp_vol = _augment_3d(inp_vol, aug)
            tgt_vol = _augment_3d(tgt_vol, aug)
            inputs.append(inp_vol.unsqueeze(0))
            targets.append(tgt_vol.unsqueeze(0))
        inp = torch.stack(inputs, dim=0).to(device)    # [B, 1, d/2, h, w]
        tgt = torch.stack(targets, dim=0).to(device)   # [B, 1, d/2, h, w]

        opt.zero_grad()
        pred = model(inp)
        loss = loss_fn(pred, tgt)
        if not torch.isfinite(loss):
            continue
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sch.step()
        rl += loss.item()
        if verbose and (it + 1) % print_every == 0:
            lr_now = sch.get_last_lr()[0]
            print(f"   {it+1:>5}/{cfg['n2n_iters']} "
                  f"loss={rl/print_every:.6f} lr={lr_now:.2e} "
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
    model: DeepCADUNet,
    stack: np.ndarray,
    config: dict,
    device: torch.device,
    verbose: bool = True,
) -> np.ndarray:
    model.eval()
    model = model.float()
    norm_params = config["norm_params"]
    F_total, H, W = stack.shape

    divisor = 2 ** config.get("depth", 3)
    pd = min(config.get("patch_d", 32), F_total)
    phw = min(config.get("patch_hw", 64), H, W)
    pd = max((pd // divisor) * divisor, divisor)
    phw = max((phw // divisor) * divisor, divisor)
    stride_d = max(pd // 2, divisor)
    stride_hw = max(phw // 2, divisor)

    if verbose:
        print(f" Sliding window: patch={pd}x{phw}x{phw}, "
              f"stride={stride_d}x{stride_hw}x{stride_hw}")

    norm_strategy = _prep.resolve_normalization(
        config.get("__resolved_normalization",
                    config.get("normalization", DEFAULT_NORMALIZATION))
    )
    stack_norm = norm_strategy.forward(stack, norm_params)
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
                    mode = ("reflect" if (pd - ad < ad and
                                          phw - ah < ah and
                                          phw - aw < aw)
                            else "replicate")
                    patch = F.pad(patch,
                                   (0, phw - aw, 0, phw - ah, 0, pd - ad),
                                   mode=mode)
                inp = patch.unsqueeze(0).unsqueeze(0).float()
                pred = model(inp).squeeze(0).squeeze(0).float()
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


# ══════════════════════════════════════════════════════════════
# CHECKPOINTS
# ══════════════════════════════════════════════════════════════

def save_checkpoint(model, config, path):
    torch.save({"model_state_dict": model.state_dict(),
                "config": config,
                "arch": "DeepCADUNet"}, path)
    print(f"Checkpoint saved → {path}")


def load_checkpoint(path, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DeepCADUNet(
        base_ch=cfg.get("base_ch", 16),
        depth=cfg.get("depth", 3),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg
