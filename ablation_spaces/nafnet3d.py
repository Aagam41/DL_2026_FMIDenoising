"""Ablation space for nafnet3d (volumetric NAFNet)."""

BASE_CONFIG = {
    "algo":            "nafnet3d",
    "paper_frame":     750,
    "width":           16,
    "enc_blocks":      (1, 1, 2, 2),
    "middle_blocks":   4,
    "dec_blocks":      (1, 1, 1, 1),
    "dw_expand":       2,
    "ffn_expand":      2,
    "drop_out_rate":   0.0,
    "patch_d":         16,
    "patch_hw":        64,
    "batch_size":      1,
    "warmup_iters":    300,
    "n2v_iters":       2500,
    "lr":              3e-4,
    "mask_ratio":      0.015,
    "mask_radius":     2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "NAFNet3D — ablate width, depth, patch sizes, masking."

ABLATIONS = {
    "baseline":          {},

    # Capacity
    "narrow_width":      {"width": 8},
    "wide_width":        {"width": 24},
    "shallow_topo":      {"enc_blocks": (1, 1, 1, 1),
                            "middle_blocks": 2,
                            "dec_blocks": (1, 1, 1, 1)},

    # Patch shape
    "deeper_patch":      {"patch_d": 32},
    "smaller_spatial":   {"patch_hw": 32},

    # Regularisation
    "with_dropout":      {"drop_out_rate": 0.1},

    # Schedule
    "no_warmup":         {"warmup_iters": 0},
    "short_n2v":         {"n2v_iters": 1500},

    # Masking
    "mask_radius_1":     {"mask_radius": 1},
    "mask_radius_3":     {"mask_radius": 3},
    "high_mask_ratio":   {"mask_ratio": 0.03},
}
