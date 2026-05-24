"""HPO search space for srdtrans (paper-faithful spatial-redundancy training)."""

BASE_CONFIG = {
    "algo":            "srdtrans",
    "paper_frame":     750,
    "patch_d":         16,
    "patch_hw":        48,
    "batch_size":      1,
    "mlp_ratio":       2.0,
    "time_compress_r": 2,
    "loss":            "l1",
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",   # ignored
}

DESCRIPTION = (
    "SRDTrans (SRD) — search over embed_dim, time-encoder depth, STB stacks, "
    "lr, iters, loss."
)


def suggest(trial):
    embed_dim = trial.suggest_categorical("embed_dim", [16, 32, 48])
    n_time_levels = trial.suggest_int("n_time_levels", 1, 3)
    n_stb_blocks = trial.suggest_int("n_stb_blocks", 1, 3)
    num_heads = trial.suggest_categorical("num_heads", [2, 4])
    lr = trial.suggest_float("lr", 5e-5, 5e-4, log=True)
    srd_iters = trial.suggest_int("srd_iters", 1500, 4000, step=500)
    loss = trial.suggest_categorical("loss", ["l1", "l2"])
    # patch_hw must be even
    patch_hw = trial.suggest_categorical("patch_hw", [32, 48, 64])
    return {
        "embed_dim":     embed_dim,
        "n_time_levels": n_time_levels,
        "n_stb_blocks":  n_stb_blocks,
        "num_heads":     num_heads,
        "lr":            lr,
        "srd_iters":     srd_iters,
        "loss":          loss,
        "patch_hw":      patch_hw,
    }
