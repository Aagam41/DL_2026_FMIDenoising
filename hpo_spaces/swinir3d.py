"""HPO search space for swinir3d (research 3D extension).

3D windowed attention is heavy; ranges are conservative.
"""

BASE_CONFIG = {
    "algo":            "swinir3d",
    "paper_frame":     750,
    "patch_d":         16,
    "patch_hw":        48,
    "batch_size":      1,
    "mlp_ratio":       2.0,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "SwinIR3D — search over embed_dim, depths/heads, window, lr, "
    "masking, iters. 3D attention is O(window^6) per window so "
    "window choices are kept small."
)


def suggest(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 48])
    n_rstb = trial.suggest_int("n_rstb", 2, 4)
    depth_per_rstb = trial.suggest_categorical("depth_per_rstb", [1, 2, 3])
    heads = trial.suggest_categorical("heads", [2, 4])
    # window=4 keeps things tractable; window=8 in 3D is 8^3=512 tokens
    # per window which is heavy. Make this categorical with two safe
    # choices.
    window_size = trial.suggest_categorical("window_size", [4])
    depths = tuple([depth_per_rstb] * n_rstb)
    num_heads = tuple([heads] * n_rstb)
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 400, step=100)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 4000, step=500)
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
