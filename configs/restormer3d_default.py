"""restormer3d — 3D Restormer (MDTA + GDFN)."""

CONFIG = {
    "algo":         "restormer3d",
    "name":         "restormer3d_t4",
    "description":  "3D Restormer for denoising, fp32.",
    "paper_frame":  750,

    "dim":                    32,
    "num_blocks":             (1, 1, 1, 2),
    "num_refinement_blocks":  2,
    "heads":                  (1, 2, 4, 8),
    "ffn_expansion_factor":   2.0,
    "bias_free":              True,

    "patch_d":       32,
    "patch_hw":      64,
    "batch_size":    2,

    "warmup_iters":  600,
    "n2v_iters":     4500,
    "lr":            0.0009883029120086638,
    "mask_ratio":    0.027043927794832567,
    "mask_radius":   2,
}
