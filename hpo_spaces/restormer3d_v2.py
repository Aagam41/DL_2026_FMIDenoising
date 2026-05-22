"""HPO search space for restormer3d_v2 (multi-channel priors + EMA + TTA).

The v2 algo has more knobs than v1 — auxiliary loss weights and EMA
decay are the new searchable things.
"""

BASE_CONFIG = {
    "algo":                    "restormer3d_v2",
    "paper_frame":             750,
    "patch_d":                 32,
    "patch_hw":                64,
    "batch_size":              2,
    "bias_free":               True,
    "ffn_expansion_factor":    2.0,
    "heads":                   (1, 2, 4, 8),
    "num_refinement_blocks":   2,
    "normalization":           "p0.5_p99.5",
    "temporal_target":         "temporal_median_2d",
    "temporal_overlap":        0.5,
}

DESCRIPTION = (
    "Restormer3D v2 — search over lr, masking (s/t), iters, channel width, "
    "and the gradient/temporal-gradient auxiliary loss weights."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.05, log=True)
    mask_radius_s = trial.suggest_int("mask_radius_s", 1, 3)
    mask_radius_t = trial.suggest_int("mask_radius_t", 0, 2)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 600, step=50)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 6000, step=500)
    dim = trial.suggest_categorical("dim", [24, 32, 40, 64])
    nb_choice = trial.suggest_categorical(
        "num_blocks_choice", ["small", "medium", "large"]
    )
    num_blocks = {
        "small":  (1, 1, 1, 2),
        "medium": (2, 2, 2, 3),
        "large":  (2, 3, 3, 4),
    }[nb_choice]
    # Auxiliary loss weights — these are the "v2" differentiator
    loss_grad_weight = trial.suggest_float("loss_grad_weight", 0.0, 0.3)
    loss_tgrad_weight = trial.suggest_float("loss_tgrad_weight", 0.0, 0.2)
    ema_decay = trial.suggest_float("ema_decay", 0.99, 0.9999, log=False)
    use_variance_sampling = trial.suggest_categorical(
        "use_variance_sampling", [True, False]
    )
    return {
        "lr":                    lr,
        "mask_ratio":            mask_ratio,
        "mask_radius_s":         mask_radius_s,
        "mask_radius_t":         mask_radius_t,
        "warmup_iters":          warmup_iters,
        "n2v_iters":             n2v_iters,
        "dim":                   dim,
        "num_blocks":            num_blocks,
        "loss_grad_weight":      loss_grad_weight,
        "loss_tgrad_weight":     loss_tgrad_weight,
        "ema_decay":             ema_decay,
        "use_variance_sampling": use_variance_sampling,
    }
