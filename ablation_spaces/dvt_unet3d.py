"""Ablation space for dvt_unet3d.

Each ablation toggles ONE config knob relative to the baseline. These
are config-toggle ablations only — architectural ablations (e.g.
removing the DVT artifact field) would require a code-level flag that
doesn't currently exist in DVTUNet3D's forward pass.

If you want architectural ablations later, add a config flag to
algos/dvt_unet3d.py (e.g. `disable_artifact_field`) and read it in
DVTBottleneck.forward, then add the corresponding ablation here.
"""

BASE_CONFIG = {
    "algo":            "dvt_unet3d",
    "paper_frame":     750,
    "base_ch":         64,
    "token_dim":       192,
    "grid_shape":      (4, 8, 8),
    "n_vit_blocks":    2,
    "n_heads":         4,
    "patch_d":         32,
    "patch_hw":        128,
    "batch_size":      2,
    "warmup_iters":    400,
    "n2v_iters":       4000,
    "lr":              5e-4,
    "mask_ratio":      0.025,
    "mask_radius":     2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "DVT-UNet3D config-toggle ablations: schedule, masking, normalization, "
    "and capacity."
)

ABLATIONS = {
    "baseline":             {},

    # Schedule ablations
    "no_warmup":            {"warmup_iters": 0},
    "short_n2v":            {"n2v_iters": 1000},
    "long_n2v":             {"n2v_iters": 6000},

    # Masking ablations
    "mask_radius_1":        {"mask_radius": 1},
    "mask_radius_3":        {"mask_radius": 3},
    "low_mask_ratio":       {"mask_ratio": 0.005},
    "high_mask_ratio":      {"mask_ratio": 0.05},

    # Normalization ablations
    "norm_p3_p97":          {"normalization": "p3_p97"},
    "norm_p1_p99":          {"normalization": "p1_p99"},

    # Capacity ablations
    "small_base_ch":        {"base_ch": 32},
    "small_token_dim":      {"token_dim": 96},
    "no_vit_blocks":        {"n_vit_blocks": 1},

    # LR ablations
    "low_lr":               {"lr": 1e-4},
    "high_lr":              {"lr": 1e-3},
}
