"""Ablation space for swinir3d (research 3D extension).

3D-adjusted ranges — smaller everything than the 2D variant.
"""

BASE_CONFIG = {
    "algo":            "swinir3d",
    "paper_frame":     750,
    "embed_dim":       32,
    "depths":          (2, 2, 2),
    "num_heads":       (2, 2, 2),
    "window_size":     4,
    "mlp_ratio":       2.0,
    "patch_d":         16,
    "patch_hw":        48,
    "batch_size":      1,
    "warmup_iters":    200,
    "n2v_iters":       2500,
    "lr":              2e-4,
    "mask_ratio":      0.015,
    "mask_radius":     2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "SwinIR3D — ablate embed_dim, depth/heads, window, schedule."

ABLATIONS = {
    "baseline":            {},

    # Schedule
    "no_warmup":           {"warmup_iters": 0},
    "short_n2v":           {"n2v_iters": 1500},

    # Capacity
    "small_embed_dim":     {"embed_dim": 16},
    "large_embed_dim":     {"embed_dim": 48},
    "shallow_rstbs":       {"depths": (1, 1, 1),
                              "num_heads": (2, 2, 2)},
    "deep_rstbs":          {"depths": (3, 3, 3),
                              "num_heads": (2, 2, 2)},
    "more_heads":          {"num_heads": (4, 4, 4)},

    # Masking
    "mask_radius_1":       {"mask_radius": 1},
    "mask_radius_3":       {"mask_radius": 3},
    "high_mask_ratio":     {"mask_ratio": 0.03},

    # Patch shape
    "deeper_patch":        {"patch_d": 32},

    # LR
    "low_lr":              {"lr": 5e-5},
}
