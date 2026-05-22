"""Ablation space for srdtrans.

Key SRDTrans-specific ablations: number of temporal compression levels,
number of STB blocks, attention head count.
"""

BASE_CONFIG = {
    "algo":            "srdtrans",
    "paper_frame":     750,
    "embed_dim":       32,
    "n_time_levels":   2,
    "n_stb_blocks":    2,
    "num_heads":       4,
    "mlp_ratio":       2.0,
    "time_compress_r": 2,
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

DESCRIPTION = "SRDTrans — ablate temporal-encoder depth, STB count, heads."

ABLATIONS = {
    "baseline":             {},

    # Schedule
    "no_warmup":            {"warmup_iters": 0},
    "short_n2v":            {"n2v_iters": 1500},

    # SRDTrans-specific architecture
    # n_time_levels=0 is invalid (loop runs zero times) → minimum is 1
    "shallow_time_encoder": {"n_time_levels": 1},
    "deep_time_encoder":    {"n_time_levels": 3, "patch_d": 32},
    "more_stb_blocks":      {"n_stb_blocks": 4},
    "fewer_stb_blocks":     {"n_stb_blocks": 1},
    "fewer_heads":          {"num_heads": 2},

    # Capacity
    "small_embed_dim":      {"embed_dim": 16},
    "large_embed_dim":      {"embed_dim": 48},

    # Masking
    "mask_radius_1":        {"mask_radius": 1},
    "mask_radius_3":        {"mask_radius": 3},
    "high_mask_ratio":      {"mask_ratio": 0.03},

    # LR
    "low_lr":               {"lr": 5e-5},
}
