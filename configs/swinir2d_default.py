"""SwinIR 2D — Liang et al. ICCVW 2021, denoising configuration."""

CONFIG = {
    "algo":         "swinir2d",
    "name":         "swinir2d_default",
    "description":  "SwinIR per-frame denoiser (no upscale).",
    "paper_frame":  750,

    # Backbone
    "embed_dim":    60,
    "depths":       (4, 4, 4, 4),
    "num_heads":    (4, 4, 4, 4),
    "window_size":  8,
    "mlp_ratio":    2.0,

    # Patch sampling (per-frame)
    "patch_hw":     128,
    "batch_size":   2,

    # Schedule
    "warmup_iters": 300,
    "n2v_iters":    3000,
    "lr":           2e-4,

    "mask_ratio":   0.015,
    "mask_radius":  2,

    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
