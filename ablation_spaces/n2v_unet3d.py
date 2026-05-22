"""Ablation space for n2v_unet3d (vanilla 3D U-Net + N2V)."""

BASE_CONFIG = {
    "algo":            "n2v_unet3d",
    "paper_frame":     750,
    "base_ch":         32,
    "patch_d":         32,
    "patch_hw":        128,
    "batch_size":      2,
    "warmup_iters":    500,
    "n2v_iters":       3000,
    "lr":              3e-4,
    "mask_ratio":      0.008,
    "mask_radius":     2,
    "normalization":   "p3_p97",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "n2v_unet3d — ablate masking, schedule, normalization, capacity."

ABLATIONS = {
    "baseline":          {},

    # Schedule
    "no_warmup":         {"warmup_iters": 0},
    "short_n2v":         {"n2v_iters": 1500},
    "long_n2v":          {"n2v_iters": 5000},

    # Masking
    "mask_radius_1":     {"mask_radius": 1},
    "mask_radius_3":     {"mask_radius": 3},
    "high_mask_ratio":   {"mask_ratio": 0.03},

    # Normalization
    "norm_p0_5_p99_5":   {"normalization": "p0.5_p99.5"},
    "norm_p1_p99":       {"normalization": "p1_p99"},

    # Temporal target
    "warmup_full_median": {"temporal_target": "full_stack_median_2d"},

    # Capacity
    "small_base_ch":     {"base_ch": 16},
    "large_base_ch":     {"base_ch": 48},
}
