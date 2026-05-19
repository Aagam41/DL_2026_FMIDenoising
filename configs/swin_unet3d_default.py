"""swin_unet3d — Swin-Unet 3D with GSC, FUE, Restormer-style decoder."""

CONFIG = {
    "algo":         "swin_unet3d",
    "name":         "swin_unet3d_t4",
    "description":  "Swin-Unet 3D + GSC + FUE + Restormer-style decoder.",
    "paper_frame":  750,

    "dim":          24,
    "num_blocks":   (1, 1, 1, 2),
    "num_heads":    (2, 2, 4, 4),
    "window_size":  (4, 4, 4),

    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   2,

    "warmup_iters": 100,
    "n2v_iters":    1200,
    "lr":           4e-4,
    "mask_ratio":   0.020,
    "mask_radius_s": 2,
    "mask_radius_t": 0,
}
