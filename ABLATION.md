# Ablation Study

`scripts/ablation.py` runs a set of named configuration variants and produces a comparison table. Each ablation modifies a baseline config by toggling one knob — useful for quantifying which design choices matter on your data.

## Quick start

```bash
python scripts/ablation.py \
    --algo dvt_unet3d \
    --noisy-dir /data/noisy --clean-dir /data/clean \
    --stacks F0 F1 F2 \
    --reduce 0.5
```

Outputs end up in `ablation_results/abl_<algo>_<timestamp>/`:

| File | Contents |
|---|---|
| `ablations.csv` | One row per ablation: status, duration, per-stack metric, overrides JSON |
| `summary.csv` | Aggregated table: mean ± std across stacks, delta vs baseline |
| `runs.csv`, `metrics.csv`, … | Full framework logs (per-stack run details) |
| `outputs/…/` | Denoised .tif files for each ablation × stack |

## Key flags

| Flag | Default | What it does |
|---|---|---|
| `--algo` | required | Which algo. Needs a matching `ablation_spaces/<algo>.py`. |
| `--stacks` | all stacks | Subset to evaluate on. **Use 2-3 stacks** if you want any conclusion you'd defend. |
| `--names` | all ablations | Subset of ablation names. Pass `--names baseline no_warmup` to run only those. |
| `--metric` | `stSNR` | Which metric to compare. |
| `--reduce` | 1.0 | Multiply iter-style schedule keys (warmup_iters, n2v_iters, etc.). Default is full budget — `0.5` for half-time noisier conclusions. |
| `--baseline-name` | `baseline` | Which entry in ABLATIONS to treat as the reference for delta computation. |

## Reading the summary

Each row in `summary.csv` looks like:

```
ablation        | stSNR_mean | stSNR_std | delta_vs_baseline
baseline        | 19.81      | 0.62      | +0.0000
no_warmup       | 17.20      | 0.81      | -2.6100
mask_radius_3   | 19.55      | 0.55      | -0.2600
norm_p3_p97     | nan        | nan       |
```

The `delta_vs_baseline` column is the key result — negative deltas mean **the ablation hurt** (so the toggled feature was contributing positively in the baseline). Magnitude tells you how much. A flat 0.0 delta (within std) means that feature isn't doing much in the current configuration.

## What's actually ablated

These specs do **config-toggle ablations only** — changes you can express by overriding a config key. Examples:
- masking radius (1 vs 2 vs 3)
- normalization choice
- schedule (no_warmup, short_n2v, long_n2v)
- model capacity (width, depth, base_ch)
- auxiliary loss weights (where applicable, e.g. restormer3d_v2)

Truly **architectural ablations** (e.g. "remove the DVT artifact field") aren't implemented because they'd require adding config flags to each model and modifying the forward pass. If you want one, add a config flag to the relevant algo module (e.g. `disable_artifact_field` in `dvt_unet3d.py`) and add the corresponding entry to its ablation space.

## Adding a new ablation

Edit `ablation_spaces/<algo>.py`:

```python
ABLATIONS = {
    "baseline":         {},
    "my_new_ablation":  {"some_config_key": new_value},
    # ...
}
```

That's it. The runner picks it up automatically. Make sure the config key actually exists in the algo's config schema (compare to `configs/<algo>_default.py`) — silent no-ops are a hazard if you typo a key.

## Caveats

1. **Expensive.** Each ablation runs a full training, so the cost is `n_ablations × n_stacks × training_time`. For DVT with 12 ablations × 3 stacks × ~15 min on T4, that's ~9 hours per algo. Use `--reduce` to trade fidelity for speed.

2. **High variance on single stacks.** Ablation conclusions can flip on a different stack. Run on at least 2-3 stacks before drawing conclusions.

3. **Ablation importance is context-dependent.** "`no_warmup` hurts by 2.6 dB" only holds for the baseline you chose. With a different lr or n2v_iters, the warmup might matter less. The ablation result is a local sensitivity, not a global feature importance.

4. **The reduced-budget caveat from HPO applies here too.** A feature that matters at 4000 iters may look unimportant at 1000.
