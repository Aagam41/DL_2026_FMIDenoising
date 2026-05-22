"""Ablation space for n2v_unet3d_biasfree (Mohan et al. bias-free CNN)."""

BASE_CONFIG = {
    "algo":            "n2v_unet3d_biasfree",
    "paper_frame":     750,
    "base_ch":         32,
    "patch_size":      64,
    "batch_size":      2,
    "warmup_iters":    150,
    "n2v_iters":       3000,
    "lr":              3e-4,
    "mask_ratio":      0.015,
    "mask_radius_s":   2,
    "mask_radius_t":   1,
    "normalization":   "p3_p97",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = "Bias-free N2V-UNet3D — same axes as n2v_unet3d, separate s/t masking."

ABLATIONS = {
    "baseline":           {},

    # Schedule
    "no_warmup":          {"warmup_iters": 0},
    "short_n2v":          {"n2v_iters": 1500},

    # Masking (spatial vs temporal)
    "mask_radius_s_1":    {"mask_radius_s": 1},
    "mask_radius_s_3":    {"mask_radius_s": 3},
    "no_temporal_mask":   {"mask_radius_t": 0},
    "mask_radius_t_2":    {"mask_radius_t": 2},
    "high_mask_ratio":    {"mask_ratio": 0.03},

    # Normalization
    "norm_p0_5_p99_5":    {"normalization": "p0.5_p99.5"},

    # Capacity
    "small_base_ch":      {"base_ch": 16},
    "large_base_ch":      {"base_ch": 48},
}
