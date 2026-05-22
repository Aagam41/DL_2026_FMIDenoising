"""HPO search space for deepcad."""

BASE_CONFIG = {
    "algo":            "deepcad",
    "paper_frame":     750,
    "patch_d":         32,
    "patch_hw":        64,
    "batch_size":      2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "DeepCAD — search over lr, masking, iters, base channels, depth."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 400, step=50)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 4500, step=500)
    base_ch = trial.suggest_categorical("base_ch", [8, 16, 24])
    depth = trial.suggest_categorical("depth", [2, 3])
    return {
        "lr":            lr,
        "mask_ratio":    mask_ratio,
        "mask_radius":   mask_radius,
        "warmup_iters":  warmup_iters,
        "n2v_iters":     n2v_iters,
        "base_ch":       base_ch,
        "depth":         depth,
    }
