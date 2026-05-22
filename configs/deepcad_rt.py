"""DeepCAD-RT — Li et al. Nature Biotechnology 2022.

Same DeepCAD 3D U-Net architecture, with the RT-specific reduction:
"pruning redundant features inside the network architecture, we
constructed a lightweight network and compressed the model parameters
by 94%".

Our approximation: base_ch=4 (vs DeepCAD's 16) gives ~93.7% parameter
reduction, matching the paper's claim. The official RT model achieves
this via non-uniform per-layer channel pruning (not just a smaller
base_ch); our uniform-shrink approximation hits the same gross param
count but the layer-wise distribution differs slightly. If you want
to match the official RT model exactly, the published weights from
the official repo are the right reference.
"""

CONFIG = {
    "algo":         "deepcad",          # same module as DeepCAD
    "name":         "deepcad_rt",
    "description":  "DeepCAD-RT preset: ~94% fewer params than DeepCAD.",
    "paper_frame":  750,

    # RT-specific: very small base_ch hits paper's ~94% param reduction
    "base_ch":      4,
    "depth":        3,

    # Smaller patches / batches — RT was about throughput
    "patch_d":      32,
    "patch_hw":     64,
    "batch_size":   2,

    # Shorter schedule — RT-class models converge faster
    "warmup_iters": 150,
    "n2v_iters":    2000,
    "lr":           3e-4,

    "mask_ratio":   0.015,
    "mask_radius":  2,

    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}
