"""Ablation space for swinir2d (paper architecture)."""

BASE_CONFIG = {
    "algo":            "swinir2d",
    "paper_frame":     750,
    "embed_dim":       60,
    "depths":          (4, 4, 4, 4),
    "num_heads":       (4, 4, 4, 4),
    "window_size":     8,
    "mlp_ratio":       2.0,
    "patch_hw":        128,
    "batch_size":      2,
    "warmup_iters":    300,
    "n2v_iters":       3000,
    "lr":              2e-4,
    "mask_ratio":      0.015,
    "mask_radius":     2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "SwinIR2D — ablate embed_dim, depth/heads, window size, schedule."

ABLATIONS = {
    "baseline":            {},

    # Schedule
    "no_warmup":           {"warmup_iters": 0},
    "short_n2v":           {"n2v_iters": 1500},

    # Capacity
    "small_embed_dim":     {"embed_dim": 32},
    "large_embed_dim":     {"embed_dim": 96},
    "shallow_rstbs":       {"depths": (2, 2, 2), "num_heads": (4, 4, 4)},
    "deep_rstbs":          {"depths": (6, 6, 6, 6),
                              "num_heads": (4, 4, 4, 4)},
    "shallow_per_rstb":    {"depths": (2, 2, 2, 2)},

    # Window size — Swin-specific
    "window_4":            {"window_size": 4},
    "window_16":           {"window_size": 16, "patch_hw": 128},

    # Masking
    "mask_radius_1":       {"mask_radius": 1},
    "high_mask_ratio":     {"mask_ratio": 0.03},

    # LR
    "low_lr":              {"lr": 5e-5},
}
