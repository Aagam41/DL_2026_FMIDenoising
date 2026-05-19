"""fm2s_dvt — FM2S 2D CNN + DVT-style Temporal ViT refiner."""

CONFIG = {
    "algo":         "fm2s_dvt",
    "name":         "fm2s_dvt_default",
    "description":  "FM2S CNN + Temporal DVT refiner.",
    "paper_frame":  750,

    # FM2S spatial CNN
    "fm2s_chan":         5,

    # Temporal DVT
    "T_window":          11,
    "vit_dim":           24,
    "vit_heads":         4,
    "vit_blocks":        2,
    "mlp_ratio":         2.0,

    "patch_hw":          64,
    "batch_size":        2,

    "fm2s_iters":        800,
    "vit_iters":         1200,
    "lr_fm2s":           1e-3,
    "lr_vit":            3e-4,

    "mask_ratio":        0.020,
    "mask_radius_s":     2,
    "vit_mask_ratio":    0.05,

    "loss_median_weight": 1.0,
    "loss_n2v_weight":    0.5,
}
