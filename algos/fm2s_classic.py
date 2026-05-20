"""
fm2s_classic — adapter for the FM2S paper implementation.

Wraps the existing algos/_fm2s_paper.py (verbatim copy of fm2s.py) in the
uniform algorithm API used by every other algo file in this package:

    compute_norm_params(stack)         -> dict
    normalize(stack, params)           -> np.ndarray
    denormalize(stack, params)         -> np.ndarray
    train_self_supervised(stack, device, config) -> (model, cfg)
    denoise_stack(model, stack, config, device)  -> np.ndarray
    save_checkpoint(model, config, path)
    load_checkpoint(path, device)

The FM2S paper implementation has slightly different names — this file is
a thin shim that maps them to the uniform API so the runner can treat
all algos identically.
"""

import numpy as np
import torch

from . import _fm2s_paper as _fm2s
from runner import preprocessing as _prep

# FM2S handles its own scaling internally, so the default normalization
# is "noop" — but it can still be overridden via config if you want to
# apply an extra pre-scaling on top (rare).
DEFAULT_NORMALIZATION = "noop"
DEFAULT_TEMPORAL_TARGET = "temporal_median_2d"


def _default_norm():
    return _prep.resolve_normalization(DEFAULT_NORMALIZATION)


def compute_norm_params(stack: np.ndarray) -> dict:
    return _default_norm().compute_params(stack)


def normalize(stack, params):
    return _default_norm().forward(stack, params)


def denormalize(stack, params):
    return _default_norm().inverse(stack, params)


# ── Training ──────────────────────────────────────────────────

def train_self_supervised(stack, device, config=None, verbose=True):
    """
    Train FM2S on the given stack.

    Maps the framework's `config` dict to fm2s_train_on_stack's
    `config_overrides` argument (FM2S accepts a single override dict,
    not individual kwargs).
    """
    # Allowed knobs in the FM2S paper module's config_overrides.
    accepted = ("n_chan", "train_num", "max_epoch", "stage1_steps",
                "lr", "noise_inj_stride")
    overrides = {}
    if config:
        for k in accepted:
            if k in config:
                overrides[k] = config[k]

    # Resolve normalization / temporal-target for metadata logging.
    # FM2S handles its own scaling internally, so these don't change
    # behavior — they're recorded so the run is traceable.
    norm_name = (config or {}).get("normalization", DEFAULT_NORMALIZATION)
    tt_name   = (config or {}).get("temporal_target",
                                    DEFAULT_TEMPORAL_TARGET)
    norm_strategy = _prep.resolve_normalization(norm_name)
    tt_strategy   = _prep.resolve_temporal_target(tt_name)

    model, returned_cfg = _fm2s.fm2s_train_on_stack(
        stack=stack, device=device,
        config_overrides=overrides if overrides else None,
        verbose=verbose,
    )

    out_cfg = dict(returned_cfg)
    out_cfg["norm_params"] = norm_strategy.compute_params(stack)
    out_cfg["__resolved_normalization"] = norm_strategy.name
    out_cfg["__resolved_temporal_target"] = tt_strategy.name
    # Surface the FM2S overrides too for completeness
    for k, v in overrides.items():
        out_cfg.setdefault(k, v)
    return model, out_cfg


# ── Inference ─────────────────────────────────────────────────

def denoise_stack(model, stack, config, device, verbose=True):
    """Run FM2S inference (video mode, with 8-fold self-ensemble)."""
    return _fm2s.fm2s_denoise_stack(
        model=model, stack=stack, config=config, device=device,
        verbose=verbose,
    )


# ── Checkpointing ─────────────────────────────────────────────

def save_checkpoint(model, config, path):
    _fm2s.save_checkpoint(model, config, path)


def load_checkpoint(path, device=None):
    return _fm2s.load_checkpoint(path, device=device)
