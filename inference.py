"""
FM2S + DVT-Temporal-ViT Zero-Shot Denoiser for AI4Life-CIDC25.

For each input stack:
    1. Robust p0.5-p99.5 normalization.
    2. Stage 0: Train FM2S 2D CNN (paper-faithful Poisson-Gaussian
       noise injection) against the temporal median target.
    3. Stage 1: Freeze FM2S. Train Temporal DVT (per-pixel temporal
       ViT with the f + g(E_pos) + h decomposition) to produce a
       corrective additive applied to FM2S's per-frame output.
    4. Frame-by-frame inference: for each frame, gather a T-frame
       temporal window (mirror-padded at edges), tile spatially with
       50% overlap, blend with Gaussian weights.

Targets ~10 min/stack on T4 — fits 7 stacks within the 1-hour budget.
"""

from pathlib import Path
import json
from glob import glob
import time

import SimpleITK
import numpy as np

from model_fm2s_dvt import (
    train_self_supervised,
    denoise_stack,
    load_checkpoint,
)


# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
INPUT_PATH = Path("/home/aagamsheth/Documents/DL_2026_FMIDenoising/test/input/interf0")
OUTPUT_PATH = Path("/home/aagamsheth/Documents/DL_2026_FMIDenoising/test/output/interf0")
PRETRAINED_PATH = Path("/opt/ml/model/fm2s_dvt_weights.pth")

# Local-test paths (uncomment for debugging)
# INPUT_PATH = Path("./test/input/interf0")
# OUTPUT_PATH = Path("./test/output/interf0")


# ─────────────────────────────────────────────────────────────
# T4-tuned config — ~10 min/stack
# ─────────────────────────────────────────────────────────────
BEST_CONFIG = {
    # FM2S spatial CNN (paper §3.4.1)
    "fm2s_chan":         5,
    # Temporal DVT (per-pixel temporal ViT)
    "T_window":          11,
    "vit_dim":           24,
    "vit_heads":         4,
    "vit_blocks":        2,
    "mlp_ratio":         2.0,
    # Patch sampling
    "patch_hw":          64,
    "batch_size":        2,
    # Schedule
    "fm2s_iters":        800,
    "vit_iters":         1200,
    "lr_fm2s":           1e-3,
    "lr_vit":            3e-4,
    # Masking
    "mask_ratio":        0.020,
    "mask_radius_s":     2,
    "vit_mask_ratio":    0.05,
    # Loss weights
    "loss_median_weight": 1.0,
    "loss_n2v_weight":    0.5,
}

# Higher-quality (~18 min/stack on T4)
HIGH_QUALITY_CONFIG = {
    **BEST_CONFIG,
    "T_window":          15,
    "vit_dim":           32,
    "vit_blocks":        3,
    "fm2s_iters":        1500,
    "vit_iters":         2500,
    "patch_hw":          96,
}

# Fast smoke (~3 min/stack on T4)
FAST_CONFIG = {
    **BEST_CONFIG,
    "fm2s_iters":        100,
    "vit_iters":         200,
    "patch_hw":          48,
}


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def run():
    return interf0_handler()


def interf0_handler():
    import torch

    _show_torch_cuda_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}\n")

    pretrained_state = None
    if PRETRAINED_PATH.exists():
        print(f"Found pre-trained weights at {PRETRAINED_PATH}")
        try:
            pre_model, pre_cfg = load_checkpoint(
                str(PRETRAINED_PATH), device=device,
            )
            pretrained_state = pre_model.state_dict()
            print(f"  Pre-trained: T={pre_cfg.get('T_window')}, "
                  f"dim={pre_cfg.get('vit_dim')}")
        except Exception as e:
            print(f"  Failed to load: {e}. Training from scratch.")
            pretrained_state = None

    print("[1/4] Loading input stack…")
    t_total = time.time()
    input_files = load_image_file_paths(
        location=INPUT_PATH / "images/stacked-neuron-images-with-noise",
    )
    print(f"  Found {len(input_files)} input file(s).")

    for input_tif in input_files:
        print(f"\n══ Processing {Path(input_tif).name} ══")
        input_tif_result = SimpleITK.ReadImage(input_tif)
        input_stack = SimpleITK.GetArrayFromImage(input_tif_result)
        print(f"  Shape: {input_stack.shape}  dtype: {input_stack.dtype}")
        print(f"  Range: [{input_stack.min()}, {input_stack.max()}]")

        # cfg = dict(BEST_CONFIG)
        cfg = dict(HIGH_QUALITY_CONFIG)
        if pretrained_state is not None:
            cfg["fm2s_iters"] = 300
            cfg["vit_iters"] = 500
            cfg["lr_fm2s"] = 3e-4
            cfg["lr_vit"] = 1e-4
            print(f"  (Fine-tune: fm2s={cfg['fm2s_iters']}, "
                  f"vit={cfg['vit_iters']})")

        print("\n[2/4] FM2S + Temporal-DVT training…")
        model, config = train_self_supervised(
            stack=input_stack, device=device, config=cfg, verbose=True,
        )

        if pretrained_state is not None:
            try:
                missing, unexpected = model.load_state_dict(
                    pretrained_state, strict=False,
                )
                if missing or unexpected:
                    print(f"  Pre-trained loaded with "
                          f"{len(missing)} missing / "
                          f"{len(unexpected)} unexpected keys.")
            except Exception as e:
                print(f"  Could not apply pre-trained weights: {e}")

        print("\n  Final config:")
        for k in ("fm2s_chan", "T_window", "vit_dim", "vit_heads",
                  "vit_blocks", "patch_hw", "fm2s_iters", "vit_iters",
                  "loss_median_weight", "loss_n2v_weight"):
            print(f"    {k}: {config.get(k)}")

        print(f"\n[3/4] Denoising {input_stack.shape[0]} frames…")
        denoised = denoise_stack(
            model=model,
            stack=input_stack.astype(np.float32),
            config=config, device=device, verbose=True,
        )

        print("\n[4/4] Saving output…")
        if np.issubdtype(input_stack.dtype, np.integer):
            info = np.iinfo(input_stack.dtype)
            denoised = np.clip(denoised, info.min, info.max)
            denoised = np.round(denoised).astype(input_stack.dtype)
        else:
            denoised = denoised.astype(np.float32)
        print(f"  Output shape: {denoised.shape}  dtype: {denoised.dtype}")
        print(f"  Output range: [{denoised.min()}, {denoised.max()}]")

        write_array_as_image_file(
            location=OUTPUT_PATH
            / "images/stacked-neuron-images-with-reduced-noise",
            array=denoised, name=Path(input_tif).name,
        )

        del model, denoised
        try:
            import torch as _t
            _t.cuda.empty_cache()
        except Exception:
            pass

    total_time = time.time() - t_total
    print(f"\n{'=' * 50}")
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"{'=' * 50}")
    return 0


# ─────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────
def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_paths(*, location):
    return (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )


def write_array_as_image_file(*, location, array, name):
    location.mkdir(parents=True, exist_ok=True)
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(image, location / f"{name}", useCompression=True)


def _show_torch_cuda_info():
    import torch
    print("=+=" * 10)
    print("Torch CUDA info")
    print(f"  CUDA available: {(available := torch.cuda.is_available())}")
    if available:
        cur = torch.cuda.current_device()
        print(f"  device count : {torch.cuda.device_count()}")
        print(f"  current      : {cur}")
        print(f"  properties   : {torch.cuda.get_device_properties(cur)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
