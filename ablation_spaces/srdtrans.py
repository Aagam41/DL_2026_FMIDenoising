"""Ablation space for srdtrans (paper-faithful spatial-redundancy training)."""

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
    "srd_iters":       2500,
    "lr":              2e-4,
    "loss":            "l1",
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "SRDTrans (SRD) — ablate temporal-encoder depth, STB count, heads, loss."

ABLATIONS = {
    "baseline":             {},

    # Schedule
    "short_srd":            {"srd_iters": 1500},
    "long_srd":             {"srd_iters": 4000},

    # Loss
    "l2_loss":              {"loss": "l2"},

    # SRDTrans-specific architecture
    "shallow_time_encoder": {"n_time_levels": 1},
    "deep_time_encoder":    {"n_time_levels": 3, "patch_d": 32},
    "more_stb_blocks":      {"n_stb_blocks": 4},
    "fewer_stb_blocks":     {"n_stb_blocks": 1},
    "fewer_heads":          {"num_heads": 2},

    # Capacity
    "small_embed_dim":      {"embed_dim": 16},
    "large_embed_dim":      {"embed_dim": 48},

    # Patch
    "small_patch_hw":       {"patch_hw": 32},
    "large_patch_hw":       {"patch_hw": 64},

    # LR
    "low_lr":               {"lr": 5e-5},
}
