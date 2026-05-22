"""HPO search space for swin_unet3d (3D Swin-Unet with GSC + FUE).

Window size is a major lever — too large and memory blows up; too small
and the model loses long-range context. We keep a discrete set of safe
window choices.
"""

BASE_CONFIG = {
    "algo":            "swin_unet3d",
    "paper_frame":     750,
    "patch_d":         32,
    "patch_hw":        64,
    "batch_size":      2,
    "num_heads":       (2, 2, 4, 4),
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "SwinUnet3D — search over lr, masking (s/t), iters, channel width, "
    "and discrete window-size."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.01, 0.05, log=True)
    mask_radius_s = trial.suggest_int("mask_radius_s", 1, 3)
    mask_radius_t = trial.suggest_int("mask_radius_t", 0, 2)
    warmup_iters = trial.suggest_int("warmup_iters", 50, 300, step=50)
    n2v_iters = trial.suggest_int("n2v_iters", 800, 2500, step=200)
    dim = trial.suggest_categorical("dim", [16, 24, 32])
    nb_choice = trial.suggest_categorical(
        "num_blocks_choice", ["small", "medium", "large"]
    )
    num_blocks = {
        "small":  (1, 1, 1, 1),
        "medium": (1, 1, 1, 2),
        "large":  (1, 2, 2, 2),
    }[nb_choice]
    win_choice = trial.suggest_categorical(
        "window_choice", ["w4", "w8x4x4", "w4x8x8"]
    )
    window_size = {
        "w4":      (4, 4, 4),
        "w8x4x4":  (8, 4, 4),
        "w4x8x8":  (4, 8, 8),
    }[win_choice]
    return {
        "lr":             lr,
        "mask_ratio":     mask_ratio,
        "mask_radius_s":  mask_radius_s,
        "mask_radius_t":  mask_radius_t,
        "warmup_iters":   warmup_iters,
        "n2v_iters":      n2v_iters,
        "dim":            dim,
        "num_blocks":     num_blocks,
        "window_size":    window_size,
    }
