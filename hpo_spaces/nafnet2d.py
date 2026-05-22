"""HPO search space for nafnet2d (per-frame ECCV 2022 NAFNet).

Width is the dominant capacity knob. The encoder/middle/decoder depths
are searched as discrete topology choices.
"""

BASE_CONFIG = {
    "algo":            "nafnet2d",
    "paper_frame":     750,
    "patch_hw":        128,
    "batch_size":      4,
    "dw_expand":       2,
    "ffn_expand":      2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "NAFNet2D — search over width, depth topology, lr, masking, iters."
)


def suggest(trial):
    width = trial.suggest_categorical("width", [16, 24, 32, 48])
    topo = trial.suggest_categorical(
        "topo", ["light", "balanced", "deep"]
    )
    topology = {
        # (enc_blocks, middle_blocks, dec_blocks)
        "light":    ((1, 1, 2, 4),  6,  (1, 1, 1, 1)),
        "balanced": ((2, 2, 4, 8), 12,  (2, 2, 2, 2)),
        "deep":     ((2, 4, 8, 16), 16, (2, 2, 2, 2)),
    }[topo]
    enc_blocks, middle_blocks, dec_blocks = topology
    drop_out_rate = trial.suggest_float("drop_out_rate", 0.0, 0.2)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 200, 600, step=100)
    n2v_iters = trial.suggest_int("n2v_iters", 2000, 5000, step=500)
    return {
        "width":         width,
        "enc_blocks":    enc_blocks,
        "middle_blocks": middle_blocks,
        "dec_blocks":    dec_blocks,
        "drop_out_rate": drop_out_rate,
        "lr":            lr,
        "mask_ratio":    mask_ratio,
        "mask_radius":   mask_radius,
        "warmup_iters":  warmup_iters,
        "n2v_iters":     n2v_iters,
    }
