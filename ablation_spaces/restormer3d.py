"""Ablation space for restormer3d.

Config-toggle ablations only. The temporal-overlap fix (added in a
prior session) is exposed as a knob via the `temporal_overlap` config
key — set to 0.0 to disable it (back to the pre-fix periodic-noise
behavior).
"""

BASE_CONFIG = {
    "algo":                    "restormer3d",
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
    "warmup_iters":            200,
    "n2v_iters":               3000,
    "lr":                      3e-4,
    "mask_ratio":              0.015,
    "mask_radius":             1,
    "temporal_overlap":        0.5,
    "normalization":           "p0.5_p99.5",
    "temporal_target":         "temporal_median_2d",
}

DESCRIPTION = "Restormer3D config-toggle ablations."

ABLATIONS = {
    "baseline":           {},

    # Schedule
    "no_warmup":          {"warmup_iters": 0},
    "short_n2v":          {"n2v_iters": 1500},

    # Masking
    "mask_radius_2":      {"mask_radius": 2},
    "high_mask_ratio":    {"mask_ratio": 0.03},

    # Normalization
    "norm_p3_p97":        {"normalization": "p3_p97"},

    # Capacity
    "small_dim":          {"dim": 24},
    "shallow_blocks":     {"num_blocks": (1, 1, 1, 2)},
    "biased":             {"bias_free": False},

    # Architecture / inference
    # temporal_overlap=0 → no temporal overlap in sliding window inference,
    # reproduces the period-32 artifact that we fixed.
    "no_temporal_overlap": {"temporal_overlap": 0.0},

    # LR
    "low_lr":             {"lr": 1e-4},
    "high_lr":            {"lr": 1e-3},
}
