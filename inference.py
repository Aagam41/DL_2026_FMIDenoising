"""
FM2S Zero-Shot Denoiser for AI4Life-CIDC25.

Purely zero-shot: no pre-trained weights needed.
For each input video stack, the method:
  1. Computes temporal median as a clean target (superior to spatial median)
  2. Estimates noise level and adapts injection parameters
  3. Trains a 3.5k-param CNN from scratch (~30s on T4)
  4. Applies the trained model to every frame with 8-fold self-ensemble

This preserves both spatial detail and temporal dynamics (calcium transients),
because the model learns to remove noise — not signal.

Reference: https://arxiv.org/abs/2412.10031
"""

from pathlib import Path
import json
from glob import glob
import time

import SimpleITK
import numpy as np

from fm2s import (
    fm2s_train_on_stack,
    fm2s_denoise_stack,
)

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
#INPUT_PATH = Path("/home/aagamsheth/Documents/Education/Ahmedabad University/MTECH CSE 2025 - 2027/SEM 2/CSE 602/Project/AI4Life-CIDC25/test/input/interf0")
#OUTPUT_PATH = Path("/home/aagamsheth/Documents/Education/Ahmedabad University/MTECH CSE 2025 - 2027/SEM 2/CSE 602/Project/AI4Life-CIDC25/test/output/interf0")


def run():
    interface_key = get_interface_key()

    # handler = {
    #     ("stacked-neuron-images-with-noise",): interf0_handler,
    # }[interface_key]
    handler = interf0_handler

    return handler()


def interf0_handler():
    import torch

    _show_torch_cuda_info()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # ── Load input ────────────────────────────────────────
    print("\n[1/4] Loading input stack...")
    t_total = time.time()

    input_stack_array = load_image_file_as_array(
        location=INPUT_PATH / "images/stacked-neuron-images-with-noise",
    )

    for input_tif in input_stack_array:
        input_tif_result = SimpleITK.ReadImage(input_tif)
        input_stack = SimpleITK.GetArrayFromImage(input_tif_result)

        print(f"  Shape: {input_stack.shape}  dtype: {input_stack.dtype}")
        print(f"  Range: [{input_stack.min()}, {input_stack.max()}]")

        # ── Train zero-shot ───────────────────────────────────
        print("\n[2/4] Zero-shot FM2S training...")
        print("  Using temporal median as clean target (exploits multi-frame info)")
        print("  Estimating noise level for adaptive parameters...")

        model, config = fm2s_train_on_stack(
            stack=input_stack,
            device=device,
            config_overrides=None,  # Use adaptive defaults
            verbose=True,
        )

        print(f"\n  Final config:")
        for k in ["g_map", "p_map", "lam_p", "n_chan", "train_num",
                  "max_epoch", "val_range", "val_min", "self_ensemble"]:
            print(f"    {k}: {config.get(k)}")

        # ── Inference ─────────────────────────────────────────
        print(f"\n[3/4] Denoising {input_stack.shape[0]} frames...")

        denoised = fm2s_denoise_stack(
            model=model,
            stack=input_stack.astype(np.float32),
            config=config,
            device=device,
            verbose=True,
        )

        # ── Save output ───────────────────────────────────────
        print("\n[4/4] Saving output...")

        # The denoised data is already in the original value range
        # (fm2s_denoise_frame handles the val_min shift/unshift internally).
        # Clip to the original data's plausible range.
        val_min = config.get("val_min", 0.0)
        val_range = config["val_range"]
        denoised = np.clip(denoised, val_min, val_min + val_range)

        # Match input dtype
        if np.issubdtype(input_stack.dtype, np.integer):
            denoised = np.round(denoised).astype(input_stack.dtype)
        else:
            denoised = denoised.astype(np.float32)

        print(f"  Output shape: {denoised.shape}  dtype: {denoised.dtype}")
        print(f"  Output range: [{denoised.min()}, {denoised.max()}]")

        write_array_as_image_file(
            location=OUTPUT_PATH / "images/stacked-neuron-images-with-reduced-noise",
            array=denoised,
            name=Path(input_tif).name,
        )

        total_time = time.time() - t_total
        print(f"\n{'='*50}")
        print(f"Total time: {total_time:.1f}s  ({total_time/60:.1f} min)")
        print(f"{'='*50}")

    return 0


# ─────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────

def get_interface_key():
    inputs = load_json_file(location=INPUT_PATH / "inputs.json")
    socket_slugs = [sv["interface"]["slug"] for sv in inputs]
    return tuple(sorted(socket_slugs))


def load_json_file(*, location):
    with open(location, "r") as f:
        return json.loads(f.read())


def load_image_file_as_array(*, location):
    input_files = (
        glob(str(location / "*.tif"))
        + glob(str(location / "*.tiff"))
        + glob(str(location / "*.mha"))
    )
    return input_files


def write_array_as_image_file(*, location, array, name):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".tif"
    image = SimpleITK.GetImageFromArray(array)
    SimpleITK.WriteImage(
        image,
        location / f"{name}",
        useCompression=True,
    )


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: {(current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
