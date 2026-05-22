"""Ablation space for fm2s_dvt (FM2S CNN + Temporal-DVT refiner)."""

BASE_CONFIG = {
    "algo":               "fm2s_dvt",
    "paper_frame":        750,
    "fm2s_chan":          5,
    "T_window":           11,
    "vit_dim":            24,
    "vit_heads":          4,
    "vit_blocks":         2,
    "mlp_ratio":          2.0,
    "patch_hw":           64,
    "batch_size":         2,
    "fm2s_iters":         800,
    "vit_iters":          1200,
    "lr_fm2s":            1e-3,
    "lr_vit":             3e-4,
    "mask_ratio":         0.020,
    "mask_radius_s":      2,
    "vit_mask_ratio":     0.05,
    "loss_median_weight": 1.0,
    "loss_n2v_weight":    0.5,
    "normalization":      "p0.5_p99.5",
    "temporal_target":    "temporal_median_2d",
}

DESCRIPTION = "fm2s_dvt — ablate two-stage pieces, loss weights, ViT capacity."

ABLATIONS = {
    "baseline":            {},

    # Stage ablations
    "fm2s_only":           {"vit_iters": 0},     # skip ViT refinement
    "no_vit_n2v_loss":     {"loss_n2v_weight": 0.0},
    "no_median_loss":      {"loss_median_weight": 0.0},

    # ViT capacity
    "small_vit":           {"vit_dim": 16, "vit_blocks": 1},
    "large_vit":           {"vit_dim": 32, "vit_blocks": 3},
    "narrow_T":            {"T_window": 5},
    "wide_T":              {"T_window": 17},

    # Masking
    "low_vit_mask_ratio":  {"vit_mask_ratio": 0.02},
    "high_vit_mask_ratio": {"vit_mask_ratio": 0.10},

    # Schedule
    "short_fm2s":          {"fm2s_iters": 300},
    "short_vit":           {"vit_iters": 600},
}
