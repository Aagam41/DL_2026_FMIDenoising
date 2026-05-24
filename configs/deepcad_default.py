"""DeepCAD — Li et al. Nature Methods 2021 default.

Uses the paper-faithful Noise2Noise training scheme: interleaved odd/even
frame splits as input/target. NO temporal-median warmup, NO blind-spot
masking.

For the lighter DeepCAD-RT preset, see configs/deepcad_rt.py.
"""

CONFIG = {
    "algo":         "deepcad",
    "name":         "deepcad_default",
    "description":  "DeepCAD 3D U-Net + N2N (paper-faithful training).",
    "paper_frame":  750,

    # Backbone (paper default)
    "base_ch":      16,
    "depth":        3,

    # N2N sampling — patch_d must be EVEN (split into odd/even halves)
    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   2,

    # Schedule — single-stage N2N (no warmup)
    "n2n_iters":    3000,
    "lr":           3e-4,
    "loss":         "l1",        # paper uses L1

    "normalization":   "p0.5_p99.5",
    # temporal_target is IGNORED for DeepCAD (no warmup), retained for
    # framework compatibility.
    "temporal_target": "temporal_median_2d",
}
