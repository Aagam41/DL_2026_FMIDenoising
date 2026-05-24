"""Ablation space for deepcad (paper-faithful Noise2Noise)."""

BASE_CONFIG = {
    "algo":            "deepcad",
    "paper_frame":     750,
    "base_ch":         16,
    "depth":           3,
    "patch_d":         32,
    "patch_hw":        64,
    "batch_size":      2,
    "n2n_iters":       3000,
    "lr":              3e-4,
    "loss":            "l1",
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "DeepCAD (N2N) — ablate channels, depth, loss, schedule, patch."

ABLATIONS = {
    "baseline":         {},

    # Schedule
    "short_n2n":        {"n2n_iters": 1500},
    "long_n2n":         {"n2n_iters": 5000},

    # Loss
    "l2_loss":          {"loss": "l2"},

    # Capacity — base_ch=4 = DeepCAD-RT preset
    "small_base_ch":    {"base_ch": 8},
    "tiny_base_ch_rt":  {"base_ch": 4},      # = DeepCAD-RT
    "large_base_ch":    {"base_ch": 24},
    "shallow_depth_2":  {"depth": 2},

    # Patch shape
    "small_patch_d":    {"patch_d": 16},
    "large_patch_d":    {"patch_d": 48},
    "small_patch_hw":   {"patch_hw": 32},

    # Normalization
    "norm_p3_p97":      {"normalization": "p3_p97"},

    # LR
    "low_lr":           {"lr": 1e-4},
    "high_lr":          {"lr": 1e-3},
}
