"""Ablation space for fm2s_classic (paper recipe in video mode).

Fewer knobs to ablate — single-stage CNN. Mostly schedule and capacity.
"""

BASE_CONFIG = {
    "algo":            "fm2s_classic",
    "paper_frame":     750,
    "n_chan":          5,
    "train_num":       450,
    "max_epoch":       10,
    "stage1_steps":    10,
    "lr":              1e-3,
    "normalization":   "noop",
    "temporal_target": "framework_default",
}

DESCRIPTION = "fm2s_classic — ablate channel width, training-set size, lr."

ABLATIONS = {
    "baseline":         {},

    # Capacity
    "narrow_n_chan":    {"n_chan": 4},
    "wide_n_chan":      {"n_chan": 8},

    # Training set + schedule
    "small_train_num":  {"train_num": 200},
    "large_train_num":  {"train_num": 700},
    "few_epochs":       {"max_epoch": 5},
    "many_epochs":      {"max_epoch": 15},
    "short_stage1":     {"stage1_steps": 5},
    "long_stage1":      {"stage1_steps": 20},

    # LR
    "low_lr":           {"lr": 3e-4},
    "high_lr":          {"lr": 3e-3},
}
