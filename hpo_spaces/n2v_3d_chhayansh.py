"""HPO search space for n2v_3d_chhayansh.

This algo has two modes — pretrained (load weights, no training) and
scratch (train from scratch via N2V). The HPO space only makes sense
in the scratch path; pretrained is fixed weights with no learnable knobs.
"""

BASE_CONFIG = {
    "algo":             "n2v_3d_chhayansh",
    "paper_frame":      750,
    "load_pretrained":  False,    # HPO requires scratch training
    "patch_d":          32,
    "patch_hw":         128,
    "batch_size":       2,
    "tile_d":           32,
    "tile_hw":          128,
    "overlap_d":        4,
    "overlap_hw":       16,
    "normalization":    "chhayansh",
    "temporal_target":  "temporal_median_2d",
}

DESCRIPTION = (
    "chhayansh 3D-N2V (scratch mode) — search over lr, masking, iters."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 0, 300, step=50)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 4000, step=500)
    return {
        "lr":            lr,
        "mask_ratio":    mask_ratio,
        "mask_radius":   mask_radius,
        "warmup_iters":  warmup_iters,
        "n2v_iters":     n2v_iters,
    }
