"""Ablation space for deepcad (Li et al. Nat Methods 2021).

Config-toggle ablations only.

Notable: `small_base_ch` (base_ch=8) effectively reproduces the DeepCAD-RT
preset — so this ablation answers "how much does the RT compression cost?".
"""

BASE_CONFIG = {
    "algo":            "deepcad",
    "paper_frame":     750,
    "base_ch":         16,
    "depth":           3,
    "patch_d":         32,
    "patch_hw":        64,
    "batch_size":      2,
    "warmup_iters":    200,
    "n2v_iters":       3000,
    "lr":              3e-4,
    "mask_ratio":      0.015,
    "mask_radius":     2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "DeepCAD — ablate masking, schedule, channels, depth, normalization."

ABLATIONS = {
    "baseline":         {},

    # Schedule
    "no_warmup":        {"warmup_iters": 0},
    "short_n2v":        {"n2v_iters": 1500},
    "long_n2v":         {"n2v_iters": 5000},

    # Masking
    "mask_radius_1":    {"mask_radius": 1},
    "mask_radius_3":    {"mask_radius": 3},
    "high_mask_ratio":  {"mask_ratio": 0.03},
    "low_mask_ratio":   {"mask_ratio": 0.005},

    # Normalization
    "norm_p3_p97":      {"normalization": "p3_p97"},

    # Capacity — base_ch=4 reproduces the DeepCAD-RT preset
    "small_base_ch":    {"base_ch": 8},
    "tiny_base_ch_rt":  {"base_ch": 4},   # = DeepCAD-RT
    "shallow_depth_2":  {"depth": 2},

    # LR
    "low_lr":           {"lr": 1e-4},
    "high_lr":          {"lr": 1e-3},
}
