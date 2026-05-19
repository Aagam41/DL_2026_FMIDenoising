"""n2v_unet3d_biasfree — bias-free 3D U-Net + N2V (Mohan et al. ICLR 2019)."""

CONFIG = {
    "algo":         "n2v_unet3d_biasfree",
    "name":         "biasfree_default",
    "description":  "Bias-free 3D U-Net + Noise2Void.",
    "paper_frame":  750,

    "base_ch":      32,
    "patch_size":   64,
    "batch_size":   2,
    "warmup_iters": 150,
    "n2v_iters":    6000,
    "lr":           3e-4,
    "mask_ratio":   0.015,
    "mask_radius_t": 1,
    "mask_radius_s": 2,
}
