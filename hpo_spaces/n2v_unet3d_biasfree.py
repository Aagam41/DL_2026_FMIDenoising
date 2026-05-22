"""HPO search space for n2v_unet3d_biasfree (Mohan et al. bias-free CNN).

Bias-free CNN means no additive biases anywhere in the network — this
makes the model equivariant to additive noise scaling, which is good
theoretically but constrains the search a bit (some normalization
choices interact differently than with biased networks).
"""

BASE_CONFIG = {
    "algo":            "n2v_unet3d_biasfree",
    "paper_frame":     750,
    "patch_size":      64,
    "batch_size":      2,
    "normalization":   "p3_p97",
    "temporal_target": "temporal_median_2d",
}

DESCRIPTION = (
    "Bias-free N2V-UNet3D — search over lr, masking (sep. radii s/t), "
    "iters, and channel width."
)


def suggest(trial):
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.04, log=True)
    # spatial vs temporal radii are searched separately so the optimizer
    # can prefer spatial-only masking (radius_t=0) if temporal masking
    # blurs calcium transients.
    mask_radius_s = trial.suggest_int("mask_radius_s", 1, 3)
    mask_radius_t = trial.suggest_int("mask_radius_t", 0, 2)
    warmup_iters = trial.suggest_int("warmup_iters", 100, 400, step=50)
    n2v_iters = trial.suggest_int("n2v_iters", 1500, 4000, step=500)
    base_ch = trial.suggest_categorical("base_ch", [16, 32, 48])
    return {
        "lr":             lr,
        "mask_ratio":     mask_ratio,
        "mask_radius_s":  mask_radius_s,
        "mask_radius_t":  mask_radius_t,
        "warmup_iters":   warmup_iters,
        "n2v_iters":      n2v_iters,
        "base_ch":        base_ch,
    }
