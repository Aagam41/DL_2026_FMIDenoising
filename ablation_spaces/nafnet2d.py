"""Ablation space for nafnet2d (per-frame ECCV 2022 NAFNet)."""

BASE_CONFIG = {
    "algo":            "nafnet2d",
    "paper_frame":     750,
    "width":           32,
    "enc_blocks":      (2, 2, 4, 8),
    "middle_blocks":   12,
    "dec_blocks":      (2, 2, 2, 2),
    "dw_expand":       2,
    "ffn_expand":      2,
    "drop_out_rate":   0.0,
    "patch_hw":        128,
    "batch_size":      4,
    "warmup_iters":    300,
    "n2v_iters":       3000,
    "lr":              3e-4,
    "mask_ratio":      0.015,
    "mask_radius":     2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "NAFNet2D — ablate width, depth, dropout, masking, schedule."

ABLATIONS = {
    "baseline":          {},

    # Capacity
    "narrow_width":      {"width": 16},
    "wide_width":        {"width": 48},
    "shallow_topo":      {"enc_blocks": (1, 1, 2, 4),
                            "middle_blocks": 6,
                            "dec_blocks": (1, 1, 1, 1)},
    "deep_topo":         {"enc_blocks": (2, 4, 8, 16),
                            "middle_blocks": 16,
                            "dec_blocks": (2, 2, 2, 2)},

    # Regularisation
    "with_dropout":      {"drop_out_rate": 0.1},
    "narrow_ffn":        {"ffn_expand": 1},
    "narrow_dw":         {"dw_expand": 1},

    # Schedule
    "no_warmup":         {"warmup_iters": 0},
    "short_n2v":         {"n2v_iters": 1500},

    # Masking
    "mask_radius_1":     {"mask_radius": 1},
    "high_mask_ratio":   {"mask_ratio": 0.03},
}
