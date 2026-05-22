"""NAFNet 3D — volumetric self-supervised denoiser."""

CONFIG = {
    "algo":         "nafnet3d",
    "name":         "nafnet3d_default",
    "description":  "3D NAFNet + 3D Noise2Void (volumetric).",
    "paper_frame":  750,

    # Backbone — smaller defaults than 2D (3D ops are much heavier)
    "width":          16,
    "enc_blocks":     (1, 1, 2, 2),
    "middle_blocks":  4,
    "dec_blocks":     (1, 1, 1, 1),
    "dw_expand":      2,
    "ffn_expand":     2,
    "drop_out_rate":  0.0,

    # Patch sampling (3D)
    "patch_d":        16,
    "patch_hw":       64,
    "batch_size":     1,

    # Schedule
    "warmup_iters":   300,
    "n2v_iters":      2500,
    "lr":             3e-4,

    # N2V masking (3D)
    "mask_ratio":     0.015,
    "mask_radius":    2,

    # Preprocessing
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
