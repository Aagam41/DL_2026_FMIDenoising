"""dvt_unet3d — DVT-style transformer bottleneck inside a 3D U-Net."""

CONFIG = {
    "algo":         "dvt_unet3d",
    "name":         "dvt_unet3d_t4",
    "description":  "DVT-style bottleneck UNet, T4-friendly schedule.",
    "paper_frame":  750,

    # Backbone
    "base_ch":      32,

    # DVT bottleneck
    "token_dim":    192,
    "grid_shape":   (4, 8, 8),
    "n_vit_blocks": 2,
    "n_heads":      4,

    # Patch sampling
    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   2,

    # Schedule
    "warmup_iters": 150,
    "n2v_iters":    1500,
    "lr":           3e-4,

    # N2V mask
    "mask_ratio":   0.015,
    "mask_radius":  1,
}
