"""NAFNet 2D — per-frame self-supervised denoiser (ECCV 2022 architecture)."""

CONFIG = {
    "algo":         "nafnet2d",
    "name":         "nafnet2d_default",
    "description":  "Per-frame NAFNet + 2D Noise2Void.",
    "paper_frame":  750,

    # Backbone
    "width":          32,
    "enc_blocks":     (2, 2, 4, 8),
    "middle_blocks":  12,
    "dec_blocks":     (2, 2, 2, 2),
    "dw_expand":      2,
    "ffn_expand":     2,
    "drop_out_rate":  0.0,

    # Patch sampling (2D per-frame)
    "patch_hw":       128,
    "batch_size":     4,

    # Schedule
    "warmup_iters":   300,
    "n2v_iters":      3000,
    "lr":             3e-4,

    # N2V masking (2D)
    "mask_ratio":     0.015,
    "mask_radius":    2,

    # Preprocessing
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
