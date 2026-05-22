"""DeepCAD — Li et al. Nature Methods 2021 default.

This is the full-size DeepCAD config. For the lighter DeepCAD-RT preset,
see configs/deepcad_rt.py.
"""

CONFIG = {
    "algo":         "deepcad",
    "name":         "deepcad_default",
    "description":  "DeepCAD 3D U-Net (full size).",
    "paper_frame":  750,

    # Backbone
    "base_ch":      16,
    "depth":        3,

    # Patch sampling
    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   2,

    # Schedule
    "warmup_iters": 200,
    "n2v_iters":    3000,
    "lr":           3e-4,

    # Masking
    "mask_ratio":   0.015,
    "mask_radius":  2,

    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
