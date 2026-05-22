"""Ablation space for restormer3d_v2 (multi-channel priors + hybrid loss + EMA + TTA).

This v2 has more knobs than v1 — especially the auxiliary loss weights
and EMA decay. Ablating those tells you whether the v2-specific
improvements are actually helping.
"""

BASE_CONFIG = {
    "algo":                    "restormer3d_v2",
    "paper_frame":             750,
    "dim":                     32,
    "num_blocks":              (2, 2, 2, 3),
    "num_refinement_blocks":   2,
    "heads":                   (1, 2, 4, 8),
    "ffn_expansion_factor":    2.0,
    "bias_free":               True,
    "patch_d":                 32,
    "patch_hw":                64,
    "batch_size":              2,
    "use_variance_sampling":   True,
    "variance_topk_frac":      0.5,
    "warmup_iters":            200,
    "n2v_iters":               3000,
    "lr":                      3e-4,
    "mask_ratio":              0.020,
    "mask_radius_s":           2,
    "mask_radius_t":           0,
    "loss_l1_weight":          1.0,
    "loss_grad_weight":        0.10,
    "loss_tgrad_weight":       0.05,
    "ema_decay":               0.999,
    "normalization":           "p0.5_p99.5",
    "temporal_target":         "temporal_median_2d",
    "temporal_overlap":        0.5,
}

DESCRIPTION = (
    "Restormer3D v2 — ablate variance sampling, auxiliary loss weights, "
    "EMA, masking strategy."
)

ABLATIONS = {
    "baseline":              {},

    # v2-specific feature ablations
    "no_variance_sampling":  {"use_variance_sampling": False},
    "no_grad_loss":          {"loss_grad_weight": 0.0},
    "no_tgrad_loss":         {"loss_tgrad_weight": 0.0},
    "no_aux_losses":         {"loss_grad_weight": 0.0,
                                "loss_tgrad_weight": 0.0},
    "no_ema":                {"ema_decay": 0.0},

    # Masking
    "mask_radius_s_1":       {"mask_radius_s": 1},
    "mask_radius_s_3":       {"mask_radius_s": 3},
    "with_temporal_mask":    {"mask_radius_t": 1},

    # Schedule
    "no_warmup":             {"warmup_iters": 0},
    "short_n2v":             {"n2v_iters": 1500},

    # Architecture
    "biased":                {"bias_free": False},
    "small_dim":             {"dim": 24},
}
