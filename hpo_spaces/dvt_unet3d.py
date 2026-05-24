"""
HPO search space for dvt_unet3d.

A spec module exports:
  - BASE_CONFIG : dict — keys NOT being searched (fixed for every trial)
  - suggest(trial) : function returning the SEARCHED params as a dict
                     merged on top of BASE_CONFIG
  - DESCRIPTION : str — human-readable description for logs

The `trial` argument is an Optuna trial object. If Optuna isn't
available, the runner passes a RandomTrial shim with the same API
(suggest_int, suggest_float, suggest_categorical).

Notes:
  - Keep base iters MODERATE so each trial finishes in reasonable
    time. The `--reduce` flag on hpo.py multiplies warmup_iters and
    n2v_iters by a fraction (default 0.25) so HPO trials are ~4x
    faster than a production run. After finding the best config, you
    re-train at full budget — see scripts/hpo.py docstring.
  - Knobs that change MODEL SIZE (base_ch, token_dim, grid_shape)
    significantly impact memory + per-iter time. We expose a small
    set of safe choices rather than free continuous ranges.
"""

# Fixed across all trials for this algo
BASE_CONFIG = {
    "algo":            "dvt_unet3d",
    "paper_frame":     750,
    "patch_d":         32,
    "patch_hw":        128,
    "batch_size":      2,
    "n_vit_blocks":    2,
    "n_heads":         4,
    "grid_shape":      (4, 8, 8),
    # Preprocessing — keep p0.5_p99.5 (known stable for fp16-free training)
    "normalization":   "p3_p97",
    "temporal_target": "temporal_median_2d",
    # Schedule iters: searched
    # mask_ratio, mask_radius, lr: searched
    # base_ch, token_dim: searched
}


DESCRIPTION = (
    "DVT-UNet3D — search over learning rate, N2V masking ratio + radius, "
    "iterations, and a discrete model-capacity choice."
)


def suggest(trial):
    """Suggest one trial's hyperparameters.

    Returns a dict of params to merge over BASE_CONFIG.
    """
    # ── Learning rate (log scale, spans 2 orders of magnitude) ─────
    lr = trial.suggest_float("lr", 1e-4, 2e-3, log=True)

    # ── N2V masking ────────────────────────────────────────────────
    mask_ratio = trial.suggest_float("mask_ratio", 0.005, 0.05, log=True)
    mask_radius = trial.suggest_int("mask_radius", 1, 3)

    # ── Schedule (will be scaled down by --reduce in the optimizer) ─
    warmup_iters = trial.suggest_int("warmup_iters", 100, 300, step=50)
    n2v_iters = trial.suggest_int("n2v_iters", 1000, 2000, step=500)

    # ── Model capacity (discrete, small set) ───────────────────────
    base_ch = trial.suggest_categorical("base_ch", [32, 48, 64])
    token_dim = trial.suggest_categorical("token_dim", [128, 192, 256])

    return {
        "lr":           lr,
        "mask_ratio":   mask_ratio,
        "mask_radius":  mask_radius,
        "warmup_iters": warmup_iters,
        "n2v_iters":    n2v_iters,
        "base_ch":      base_ch,
        "token_dim":    token_dim,
    }
