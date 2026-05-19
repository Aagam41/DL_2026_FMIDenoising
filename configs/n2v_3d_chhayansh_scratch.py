"""
n2v_3d_chhayansh — train this architecture FROM SCRATCH on the input
stack using 3D Noise2Void.

For a head-to-head zero-shot comparison with the other algos in the
framework. The upstream model used external training data and ships
pretrained weights; this preset tells you what the SAME architecture
would score in a zero-shot regime.

Use `n2v_3d_chhayansh_pretrained.py` if you instead want to score the
shipped pre-trained model.
"""

CONFIG = {
    "algo":         "n2v_3d_chhayansh",
    "name":         "n2v_3d_chhayansh_scratch",
    "description":  "chhayanshporwal 3D U-Net trained from scratch via "
                    "N2V (zero-shot, one stack at a time).",
    "paper_frame":  750,

    # Train from scratch (do NOT load shipped weights)
    "load_pretrained": False,

    # Patch sampling
    "patch_d":      32,
    "patch_hw":     128,
    "batch_size":   2,

    # Schedule — short, tuned for ~6-8 min on T4
    "warmup_iters": 0,
    "n2v_iters":    2000,
    "lr":           1e-3,

    # Noise2Void mask
    "mask_ratio":   0.015,
    "mask_radius":  2,

    # Inference tiling (matches upstream)
    "tile_d":       32,
    "tile_hw":      128,
    "overlap_d":    4,
    "overlap_hw":   16,
}
