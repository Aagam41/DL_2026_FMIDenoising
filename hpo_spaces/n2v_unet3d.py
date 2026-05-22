"""HPO search space for n2v_unet3d.

Standard 3D U-Net trained with Noise2Void. The dominant knobs are lr,
masking, schedule, and model width.
"""

BASE_CONFIG = {
    "algo":            "n2v_unet3d",
    "paper_frame":     750,
    "patch_d":         32,
    "patch_hw":        128,
    "batch_size":      2,
    "normalization":   "p3_p97",   # algo default
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "Vanilla N2V-UNet3D — search over lr, masking, iters, and channel width."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 200, 800, step=100)
    n2v_iters = trial.suggest_int("n2v_iters", 2000, 5000, step=500)
    base_ch = trial.suggest_categorical("base_ch", [16, 32, 48])
    return {
        "lr":            lr,
        "mask_ratio":    mask_ratio,
        "mask_radius":   mask_radius,
        "warmup_iters":  warmup_iters,
        "n2v_iters":     n2v_iters,
        "base_ch":       base_ch,
    }
