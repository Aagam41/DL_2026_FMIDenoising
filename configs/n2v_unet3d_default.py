"""n2v_unet3d — vanilla 3D U-Net with Noise2Void, defaults from model.py."""

CONFIG = {
    "algo":         "n2v_unet3d",
    "name":         "n2v_unet3d_default",
    "description":  "Vanilla 3D U-Net + Noise2Void, original defaults.",
    "paper_frame":  750,

    # Backbone
    "base_ch":      32,

    # Patch sampling
    "patch_d":      32,
    "patch_hw":     128,
    "batch_size":   2,

    # Schedule
    "warmup_iters": 500,
    "n2v_iters":    3000,
    "lr":           3e-4,

    # N2V mask
    "mask_ratio":   0.008,
    "mask_radius":  2,
}
