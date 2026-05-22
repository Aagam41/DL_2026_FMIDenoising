"""HPO search space for srdtrans (paper-interpretation, our N2V training)."""

BASE_CONFIG = {
    "algo":            "srdtrans",
    "paper_frame":     750,
    "patch_d":         16,
    "patch_hw":        48,
    "batch_size":      1,
    "mlp_ratio":       2.0,
    "time_compress_r": 2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "SRDTrans — search over embed_dim, time-encoder depth, STB stacks, "
    "lr, masking, iters."
)


def suggest(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 48])
    # Number of temporal compression levels (each level compresses T by r=2)
    n_time_levels = trial.suggest_int("n_time_levels", 1, 3)
    n_stb_blocks = trial.suggest_int("n_stb_blocks", 1, 3)
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 400, step=100)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 4000, step=500)
    return {
        "embed_dim":     embed_dim,
        "n_time_levels": n_time_levels,
        "n_stb_blocks":  n_stb_blocks,
        "num_heads":     num_heads,
        "lr":            lr,
        "mask_ratio":    mask_ratio,
        "mask_radius":   mask_radius,
        "warmup_iters":  warmup_iters,
        "n2v_iters":     n2v_iters,
    }
