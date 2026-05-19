"""restormer3d_v2 — Restormer3D with multi-channel priors, hybrid loss, EMA, TTA."""

CONFIG = {
    "algo":         "restormer3d_v2",
    "name":         "restormer3d_v2_t4",
    "description":  "Restormer3D + priors + EMA + TTA.",
    "paper_frame":  750,

    "dim":                    32,
    "num_blocks":             (2, 2, 2, 3),
    "num_refinement_blocks":  2,
    "heads":                  (1, 2, 4, 8),
    "ffn_expansion_factor":   2.0,
    "bias_free":              True,

    "patch_d":                32,
    "patch_hw":               64,
    "batch_size":             2,

    "use_variance_sampling":  True,
    "variance_topk_frac":     0.5,
    "warmup_iters":           200,
    "n2v_iters":              4000,
    "lr":                     3e-4,

    "mask_ratio":             0.020,
    "mask_radius_s":          2,
    "mask_radius_t":          0,

    "loss_l1_weight":         1.0,
    "loss_grad_weight":       0.10,
    "loss_tgrad_weight":      0.05,
    "ema_decay":              0.999,
}
