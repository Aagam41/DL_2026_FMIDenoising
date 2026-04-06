"""
Offline training script for FM2S on the CIDC25 training data.

Run this LOCALLY before submission to pre-train weights.
The saved checkpoint can then be uploaded as a model tarball to Grand Challenge.

Usage:
    python train.py --data_dir /path/to/training_tiffs --output fm2s_weights.pth

    Then create the model tarball:
        mkdir -p model_tarball
        cp fm2s_weights.pth model_tarball/
        tar -czf model_tarball.tar.gz -C model_tarball .

    Upload model_tarball.tar.gz to your Algorithm > Models page on Grand Challenge.
    It will be extracted to /opt/ml/model/ at runtime.

Training data format:
    Four TIFF stacks of shape [1500, 490, 490] (F×H×W) from Zenodo.
"""

import argparse
import time
from pathlib import Path
from glob import glob

import numpy as np
import torch

try:
    import tifffile
except ImportError:
    import SimpleITK

from fm2s import fm2s_train, save_checkpoint, DEFAULT_CONFIG


def load_tiff(path: str) -> np.ndarray:
    """Load a TIFF stack, trying tifffile first, then SimpleITK."""
    try:
        return tifffile.imread(path)
    except NameError:
        img = SimpleITK.ReadImage(path)
        return SimpleITK.GetArrayFromImage(img)


def get_representative_frame(stack: np.ndarray, n_subsample: int = 200) -> np.ndarray:
    """Compute temporal median from a subsampled set of frames."""
    F = stack.shape[0]
    if F > n_subsample:
        indices = np.linspace(0, F - 1, n_subsample, dtype=int)
        return np.median(stack[indices].astype(np.float32), axis=0)
    else:
        return np.median(stack.astype(np.float32), axis=0)


def main():
    parser = argparse.ArgumentParser(description="Pre-train FM2S on CIDC25 data")
    parser.add_argument(
        "--data_dir", required=True,
        help="Directory containing training TIFF files"
    )
    parser.add_argument(
        "--output", default="fm2s_weights.pth",
        help="Output checkpoint path"
    )
    parser.add_argument("--train_num", type=int, default=450, help="Samples per training")
    parser.add_argument("--max_epoch", type=int, default=5, help="Epochs per sample")
    parser.add_argument("--n_chan", type=int, default=5, help="Channel amplification")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--g_map", type=float, default=60)
    parser.add_argument("--p_map", type=float, default=30)
    parser.add_argument("--lam_p", type=float, default=150)
    parser.add_argument("--w_size", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Find all training TIFFs
    tiff_files = sorted(
        glob(str(Path(args.data_dir) / "*.tif"))
        + glob(str(Path(args.data_dir) / "*.tiff"))
    )
    print(f"Found {len(tiff_files)} TIFF files")

    # Build config
    cfg = {
        **DEFAULT_CONFIG,
        "stride": args.stride,
        "g_map": args.g_map,
        "p_map": args.p_map,
        "lam_p": args.lam_p,
        "w_size": args.w_size,
        "n_chan": args.n_chan,
        "train_num": args.train_num,
        "max_epoch": args.max_epoch,
    }

    # Strategy: Train on representative frames from ALL training stacks.
    # We pick the temporal median from each stack, then train on each one
    # sequentially (continuing to refine the same model).
    # This gives the model exposure to different content and noise levels.

    model = None
    for idx, tiff_path in enumerate(tiff_files):
        print(f"\n{'='*60}")
        print(f"Processing [{idx+1}/{len(tiff_files)}]: {Path(tiff_path).name}")
        print(f"{'='*60}")

        t0 = time.time()
        stack = load_tiff(tiff_path)
        print(f"  Shape: {stack.shape}, dtype: {stack.dtype}")
        print(f"  Range: [{stack.min()}, {stack.max()}]")

        # Get representative frame
        rep_frame = get_representative_frame(stack)

        # Determine value range
        val_max = float(stack.max())
        if val_max > 255:
            val_range = 65535.0
        elif val_max > 1.0:
            val_range = 255.0
        else:
            val_range = 1.0

        # Normalize to [0, 255] for FM2S
        rep_255 = (rep_frame / val_range) * 255.0

        print(f"  Representative frame range: [{rep_255.min():.2f}, {rep_255.max():.2f}]")

        # Train (or continue training)
        model, cfg = fm2s_train(rep_255, config=cfg, device=device, verbose=True)
        print(f"  Stack processed in {time.time() - t0:.1f}s")

    # Save final checkpoint
    if model is not None:
        save_checkpoint(model, cfg, args.output)
        print(f"\nDone! Upload '{args.output}' as part of your model tarball.")
    else:
        print("No TIFF files found. Nothing trained.")


if __name__ == "__main__":
    main()