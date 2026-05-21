"""HPO search space for restormer3d. See dvt_unet3d.py for format docs."""

BASE_CONFIG = {
    "algo":            "restormer3d",
    "paper_frame":     750,
    "patch_d":         32,
    "patch_hw":        64,
    "batch_size":      2,
    "bias_free":       True,
    "ffn_expansion_factor": 2.0,
    "heads":           (1, 2, 4, 8),
    "num_refinement_blocks": 2,
    "normalization":   "p3_p97",
    "temporal_target": "temporal_median_2d",
    "temporal_overlap": 0.5,
}

DESCRIPTION = (
    "Restormer3D — search over lr, masking, iters, and channel width."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.05, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 4)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 600, step=50)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 5000, step=500)
    dim = trial.suggest_categorical("dim", [24, 32, 40])
    nb_choice = trial.suggest_categorical(
        "num_blocks_choice", ["small", "medium", "large"]
    )
    num_blocks = {
        "small":  (1, 1, 1, 2),
        "medium": (2, 2, 2, 3),
        "large":  (2, 3, 3, 4),
    }[nb_choice]
    return {
        "lr":           lr,
        "mask_ratio":   mask_ratio,
        "mask_radius":  mask_radius,
        "warmup_iters": warmup_iters,
        "n2v_iters":    n2v_iters,
        "dim":          dim,
        "num_blocks":   num_blocks,
    }
