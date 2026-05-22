"""SRDTrans — Li et al. Nature Computational Science 2023.

Architecture-faithful (per the paper sketch); trained with our framework's
N2V flow rather than the official spatial-redundancy sampling.
"""

CONFIG = {
    "algo":         "srdtrans",
    "name":         "srdtrans_default",
    "description":  "SRDTrans — spatiotemporal transformer with temporal up/down.",
    "paper_frame":  750,

    # Backbone
    "embed_dim":       32,
    "n_time_levels":   2,
    "n_stb_blocks":    2,
    "num_heads":       4,
    "mlp_ratio":       2.0,
    "time_compress_r": 2,

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
