"""HPO search space for fm2s_dvt (FM2S CNN + Temporal-DVT refiner).

Two-stage: a 2D spatial CNN (FM2S) then a temporal ViT refiner. The
big knobs are the two loss weights and the ViT capacity.
"""

BASE_CONFIG = {
    "algo":            "fm2s_dvt",
    "paper_frame":     750,
    "patch_hw":        64,
    "batch_size":      2,
    "T_window":        11,
    "normalization":   "p0.5_p99.5",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "FM2S + Temporal-DVT — search over both stage learning rates, ViT "
    "capacity, masking, iter counts, and loss weights."
)


def suggest(trial):
    lr_fm2s = trial.suggest_float("lr_fm2s", 3e-4, 3e-3, log=True)
    lr_vit  = trial.suggest_float("lr_vit",  1e-4, 1e-3, log=True)
    fm2s_chan = trial.suggest_categorical("fm2s_chan", [4, 5, 8])
    vit_dim = trial.suggest_categorical("vit_dim", [16, 24, 32])
    vit_heads = trial.suggest_categorical("vit_heads", [2, 4])
    vit_blocks = trial.suggest_int("vit_blocks", 1, 3)
    fm2s_iters = trial.suggest_int("fm2s_iters", 400, 1500, step=100)
    vit_iters  = trial.suggest_int("vit_iters", 800, 2000, step=100)
    mask_ratio = trial.suggest_float("mask_ratio", 0.01, 0.05, log=True)
    mask_radius_s = trial.suggest_int("mask_radius_s", 1, 3)
    vit_mask_ratio = trial.suggest_float("vit_mask_ratio", 0.02, 0.10,
                                          log=True)
    loss_median_weight = trial.suggest_float("loss_median_weight", 0.5, 2.0)
    loss_n2v_weight    = trial.suggest_float("loss_n2v_weight",   0.1, 1.5)
    return {
        "lr_fm2s":            lr_fm2s,
        "lr_vit":             lr_vit,
        "fm2s_chan":          fm2s_chan,
        "vit_dim":            vit_dim,
        "vit_heads":          vit_heads,
        "vit_blocks":         vit_blocks,
        "fm2s_iters":         fm2s_iters,
        "vit_iters":          vit_iters,
        "mask_ratio":         mask_ratio,
        "mask_radius_s":      mask_radius_s,
        "vit_mask_ratio":     vit_mask_ratio,
        "loss_median_weight": loss_median_weight,
        "loss_n2v_weight":    loss_n2v_weight,
    }
