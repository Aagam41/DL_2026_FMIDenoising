"""HPO search space for swinir2d (paper architecture)."""

BASE_CONFIG = {
    "algo":            "swinir2d",
    "paper_frame":     750,
    "patch_hw":        128,
    "batch_size":      2,
    "mlp_ratio":       2.0,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "SwinIR2D — search over embed_dim, depths/heads, window, lr, masking, iters."
)


def suggest(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [32, 60, 96])
    n_rstb = trial.suggest_int("n_rstb", 3, 6)
    depth_per_rstb = trial.suggest_categorical("depth_per_rstb", [2, 4, 6])
    heads = trial.suggest_categorical("heads", [4, 6])
    window_size = trial.suggest_categorical("window_size", [4, 8])
    depths = tuple([depth_per_rstb] * n_rstb)
    num_heads = tuple([heads] * n_rstb)
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 200, 600, step=100)
    n2v_iters = trial.suggest_int("n2v_iters", 2000, 5000, step=500)
    return {
        "embed_dim":    embed_dim,
        "depths":       depths,
        "num_heads":    num_heads,
        "window_size":  window_size,
        "lr":           lr,
        "mask_ratio":   mask_ratio,
        "mask_radius":  mask_radius,
        "warmup_iters": warmup_iters,
        "n2v_iters":    n2v_iters,
    }
