"""
n2v_3d_chhayansh — INFERENCE-ONLY using the shipped pre-trained weights.

This config tells `train_self_supervised` to SKIP training entirely and
just load `algos/_weights/n2v_3d_chhayansh.pth` (the original
`best_model_n2v.pth` from the upstream repo). Use this preset to score
the published model on your test set as a baseline.

If you want to train this architecture from scratch on your own stack,
use `n2v_3d_chhayansh_scratch.py` instead.
"""

CONFIG = {
    "algo":         "n2v_3d_chhayansh",
    "name":         "n2v_3d_chhayansh_pretrained",
    "description":  "Published 3D N2V (chhayanshporwal) loaded from "
                    "shipped weights — pure inference baseline.",
    "paper_frame":  750,

    # The flag that switches the algo into inference-only mode
    "load_pretrained": True,

    # Sliding-window tiling (matches upstream process.py)
    "tile_d":       32,
    "tile_hw":      128,
    "overlap_d":    4,
    "overlap_hw":   16,
}
