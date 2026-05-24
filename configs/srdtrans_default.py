"""SRDTrans — Li et al. Nature Computational Science 2023.

Paper-faithful spatial-redundancy training:
  - Crop sub-volumes [T, H, W] with H, W even.
  - Sample (input, target) from orthogonal 2×2 spatial corners — both
    sub-stacks at H/2 × W/2 share signal but have independent noise.
  - L1 loss, NO temporal-median warmup, NO blind-spot masking.
"""

CONFIG = {
    "algo":         "srdtrans",
    "name":         "srdtrans_default",
    "description":  "SRDTrans with spatial-redundancy sampling (paper-faithful).",
    "paper_frame":  750,

    # Backbone
    "embed_dim":       32,
    "n_time_levels":   2,
    "n_stb_blocks":    2,
    "num_heads":       4,
    "mlp_ratio":       2.0,
    "time_compress_r": 2,

    # Patch sampling — patch_hw must be EVEN (2x2 spatial split)
    "patch_d":      16,
    "patch_hw":     48,
    "batch_size":   1,

    # Schedule — single-stage SRD (no warmup)
    "srd_iters":    2500,
    "lr":           2e-4,
    "loss":         "l1",

    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",   # ignored (no warmup)
}
