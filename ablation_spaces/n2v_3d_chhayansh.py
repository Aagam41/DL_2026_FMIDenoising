"""Ablation space for n2v_3d_chhayansh.

Forces scratch training (load_pretrained=False) so ablations affect
actual learned behaviour. If you want to ablate "pretrained vs
scratch" run two configs explicitly through the benchmark.
"""

BASE_CONFIG = {
    "algo":             "n2v_3d_chhayansh",
    "paper_frame":      750,
    "load_pretrained":  False,
    "patch_d":          32,
    "patch_hw":         128,
    "batch_size":       2,
    "tile_d":           32,
    "tile_hw":          128,
    "overlap_d":        4,
    "overlap_hw":       16,
    "warmup_iters":     150,
    "n2v_iters":        3000,
    "lr":               1e-3,
    "mask_ratio":       0.015,
    "mask_radius":      2,
    "normalization":    "chhayansh",
    "temporal_target":  "temporal_median_2d",
}

DESCRIPTION = "chhayansh 3D-N2V (scratch) — ablate masking, tiling, schedule, lr."

ABLATIONS = {
    "baseline":          {},

    # Schedule
    "no_warmup":         {"warmup_iters": 0},
    "short_n2v":         {"n2v_iters": 1500},

    # Masking
    "mask_radius_1":     {"mask_radius": 1},
    "mask_radius_3":     {"mask_radius": 3},
    "high_mask_ratio":   {"mask_ratio": 0.03},

    # Tiling at inference
    "small_overlap_hw":  {"overlap_hw": 8},
    "large_overlap_hw":  {"overlap_hw": 32},
    "small_tile_hw":     {"tile_hw": 64},

    # Normalization
    "norm_p0_5_p99_5":   {"normalization": "p0.5_p99.5"},
    "norm_p3_p97":       {"normalization": "p3_p97"},

    # LR
    "low_lr":            {"lr": 3e-4},
}
