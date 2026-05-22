"""SwinIR 3D — our research extension of SwinIR to volumetric input."""

CONFIG = {
    "algo":         "swinir3d",
    "name":         "swinir3d_default",
    "description":  "SwinIR 3D — 3D windowed attention (research extension).",
    "paper_frame":  750,

    # Backbone — much smaller than 2D since 3D windows are heavier
    "embed_dim":    32,
    "depths":       (2, 2, 2),
    "num_heads":    (2, 2, 2),
    "window_size":  4,
    "mlp_ratio":    2.0,

    # Patches (3D)
    "patch_d":      16,
    "patch_hw":     48,
    "batch_size":   1,

    # Schedule
    "warmup_iters": 200,
    "n2v_iters":    2500,
    "lr":           2e-4,

    "mask_ratio":   0.015,
    "mask_radius":  2,

    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
