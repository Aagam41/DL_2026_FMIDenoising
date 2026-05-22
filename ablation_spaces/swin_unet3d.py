"""Ablation space for swin_unet3d (3D Swin-Unet with GSC + FUE)."""

BASE_CONFIG = {
    "algo":            "swin_unet3d",
    "paper_frame":     750,
    "dim":             24,
    "num_blocks":      (1, 1, 1, 2),
    "num_heads":       (2, 2, 4, 4),
    "window_size":     (4, 4, 4),
    "patch_d":         32,
    "patch_hw":        64,
    "batch_size":      2,
    "warmup_iters":    100,
    "n2v_iters":       1200,
    "lr":              4e-4,
    "mask_ratio":      0.020,
    "mask_radius_s":   2,
    "mask_radius_t":   0,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "SwinUnet3D — ablate window size, depth, masking strategy."

ABLATIONS = {
    "baseline":           {},

    # Schedule
    "no_warmup":          {"warmup_iters": 0},
    "long_n2v":           {"n2v_iters": 2500},

    # Window size — Swin-specific
    "window_small":       {"window_size": (2, 4, 4)},
    "window_large":       {"window_size": (4, 8, 8)},

    # Masking
    "with_temporal_mask": {"mask_radius_t": 1},
    "mask_radius_s_1":    {"mask_radius_s": 1},
    "mask_radius_s_3":    {"mask_radius_s": 3},

    # Capacity
    "small_dim":          {"dim": 16},
    "shallow_blocks":     {"num_blocks": (1, 1, 1, 1)},
    "deep_blocks":        {"num_blocks": (1, 2, 2, 2)},

    # LR
    "low_lr":             {"lr": 1e-4},
    "high_lr":            {"lr": 1e-3},
}
