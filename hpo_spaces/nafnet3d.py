"""HPO search space for nafnet3d (volumetric NAFNet).

3D ops are much heavier than 2D, so width/depth ranges are more
conservative than nafnet2d. Also: patch sizes are searched because
they have huge memory implications in 3D.
"""

BASE_CONFIG = {
    "algo":            "nafnet3d",
    "paper_frame":     750,
    "batch_size":      1,           # 3D blocks fill memory fast
    "dw_expand":       2,
    "ffn_expand":      2,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "NAFNet3D — search over width, topology, patch, lr, masking, iters."
)


def suggest(trial):
    width = trial.suggest_categorical("width", [8, 16, 24])
    topo = trial.suggest_categorical("topo", ["light", "balanced"])
    topology = {
        "light":    ((1, 1, 1, 1),  2, (1, 1, 1, 1)),
        "balanced": ((1, 1, 2, 2),  4, (1, 1, 1, 1)),
    }[topo]
    enc_blocks, middle_blocks, dec_blocks = topology
    patch_d = trial.suggest_categorical("patch_d", [16, 32])
    patch_hw = trial.suggest_categorical("patch_hw", [32, 64])
    drop_out_rate = trial.suggest_float("drop_out_rate", 0.0, 0.1)
    lr = trial.suggest_float("lr", 1e-4, 1e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)
    warmup_iters = trial.suggest_int("warmup_iters", 200, 600, step=100)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 4000, step=500)
    return {
        "width":         width,
        "enc_blocks":    enc_blocks,
        "middle_blocks": middle_blocks,
        "dec_blocks":    dec_blocks,
        "patch_d":       patch_d,
        "patch_hw":      patch_hw,
        "drop_out_rate": drop_out_rate,
        "lr":            lr,
        "mask_ratio":    mask_ratio,
        "mask_radius":   mask_radius,
        "warmup_iters":  warmup_iters,
        "n2v_iters":     n2v_iters,
    }
