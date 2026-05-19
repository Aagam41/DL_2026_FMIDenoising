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


# ── Normalization (FM2S normalizes internally; we expose no-ops here) ──

def compute_norm_params(stack: np.ndarray) -> dict:
    """No-op normalization params (FM2S handles scaling internally)."""
    # We still record the bounds in the params dict so other code that
    # reads them (e.g. plotting) doesn't crash.
    return {
        "shift": 0.0,
        "scale": 1.0,
        "in_min": float(stack.min()),
        "in_max": float(stack.max()),
    }


def normalize(stack, params):
    return stack.astype(np.float32)


def denormalize(stack, params):
    return stack.astype(np.float32)


# ── Training ──────────────────────────────────────────────────

def train_self_supervised(stack, device, config=None, verbose=True):
    """
    Train FM2S on the given stack.

    Maps the framework's `config` dict to fm2s_train_on_stack arguments.
    """
    cfg = {
        "n_chan":         5,
        "train_num":      450,
        "max_epoch":      10,
        "stage1_steps":   10,
        "lr":             1e-3,
        "verbose":        verbose,
    }
    if config:
        # Filter to the keys fm2s_train_on_stack actually accepts
        accepted = ("n_chan", "train_num", "max_epoch", "stage1_steps",
                    "lr", "noise_inj_stride", "verbose")
        for k in accepted:
            if k in config:
                cfg[k] = config[k]

    model, returned_cfg = _fm2s.fm2s_train_on_stack(
        stack=stack, device=device, **cfg,
    )

    # Stash the norm-params placeholder (the runner expects it)
    out_cfg = dict(returned_cfg)
    out_cfg["norm_params"] = compute_norm_params(stack)
    # Preserve framework knobs for downstream inspection
    out_cfg.update({k: v for k, v in cfg.items() if k not in out_cfg})
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
