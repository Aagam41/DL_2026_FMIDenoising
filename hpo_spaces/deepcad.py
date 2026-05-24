"""HPO search space for deepcad (paper-faithful Noise2Noise)."""

BASE_CONFIG = {
    "algo":            "deepcad",
    "paper_frame":     750,
    "patch_d":         32,
    "patch_hw":        64,
    "batch_size":      2,
    "loss":            "l1",
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",   # ignored by deepcad N2N
}

DESCRIPTION = (
    "DeepCAD (N2N) — search over lr, n2n_iters, base channels, depth, patch_d."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    n2n_iters = trial.suggest_int("n2n_iters", 1500, 5000, step=500)
    base_ch = trial.suggest_categorical("base_ch", [4, 8, 16, 24])
    depth = trial.suggest_categorical("depth", [2, 3])
    # patch_d must be even for the N2N split
    patch_d = trial.suggest_categorical("patch_d", [16, 32, 48])
    loss = trial.suggest_categorical("loss", ["l1", "l2"])
    return {
        "lr":         lr,
        "n2n_iters":  n2n_iters,
        "base_ch":    base_ch,
        "depth":      depth,
        "patch_d":    patch_d,
        "loss":       loss,
    }
