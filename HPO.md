# Hyperparameter Optimization

`scripts/hpo.py` searches for the hyperparameter configuration that maximizes a chosen metric (default: stSNR) on the supplied noisy/clean stacks.

## Quick start

```bash
# 20 trials of DVT on F1 only, at 25% iter budget (~4x faster per trial)
python scripts/hpo.py \
    --algo dvt_unet3d \
    --noisy-dir /data/noisy --clean-dir /data/clean \
    --stacks F1 \
    --trials 20 \
    --reduce 0.25
```

Outputs end up in `hpo_results/hpo_<algo>_<timestamp>/`:
- `trials.csv` — every trial's params, per-stack metric, aggregated objective, duration
- `best_config.py` — best config found, ready to drop into `configs/` for a full-budget production run
- `study.db` — Optuna's database (if Optuna is installed; enables `--resume`)
- The usual `runs.csv`, `metrics.csv`, etc. — full framework logs for every trial

## Key flags

| Flag | Default | What it does |
|---|---|---|
| `--algo` | required | Which algorithm to optimize. Needs a matching `hpo_spaces/<algo>.py`. |
| `--stacks` | all in `--noisy-dir` | Subset of stacks to evaluate on. **Use a subset for HPO and hold out the rest as a check.** |
| `--trials` | 20 | Total trials to run. |
| `--metric` | `stSNR` | Metric to maximize. Any metric in the eval output works. |
| `--aggregate` | `mean` | How to combine across stacks. `mean`, `min` (conservative), `max`. |
| `--reduce` | 0.25 | Multiplies `warmup_iters` and `n2v_iters` for each trial. **Set to 1.0 for full-budget trials.** |
| `--method` | `optuna` if installed, else `random` | Search strategy. |
| `--timeout` | none | Wall-clock cap in seconds (Optuna only — cleanly stops mid-trial). |
| `--resume <db>` | none | Resume an existing Optuna study. |

## How to think about the search

**Don't over-trust the result.** HPO is a search over a defined space — what comes out is "the best config Optuna could find within `--trials` evaluations on the stacks you gave it." A few common pitfalls:

1. **Reduced-budget bias.** With `--reduce 0.25`, a config that converges quickly may look great but a slower-converging heavier model could be better at full budget. The standard mitigation: after HPO completes, re-train the top 3-5 configs at full budget and pick the best of those. The top configs are visible by sorting `trials.csv` by the objective column.

2. **Overfitting to your search stacks.** If you HPO on the same 7 stacks you submit, you might pick params that happen to win on those specific data. Either (a) hold out 1-2 stacks during HPO, or (b) accept that "best on these stacks" is what you want and don't expect it to generalize beyond them.

3. **Aggregating with mean vs min.** `mean` will accept a config that's amazing on F0 but mediocre on F5. `min` favors configs that are robust across the set. If you have wildly different stacks (some easy, some hard), `min` is the safer choice — it picks the config that does well on the *worst* of your eval stacks.

## Adding a new algo to HPO

Create `hpo_spaces/<algo>.py` exporting:

```python
BASE_CONFIG = {
    "algo": "your_algo",
    "paper_frame": 750,
    # ... non-searched knobs
}

def suggest(trial):
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        # ... whatever you want to search
    }
```

The `trial` argument is either an Optuna `Trial` or our `_RandomTrial` shim — they share the same API (`suggest_int`, `suggest_float`, `suggest_categorical`).

## Installing Optuna (recommended)

```bash
pip install optuna
```

Optuna's TPE sampler is roughly 3-10× more sample-efficient than random search for 20+ trials. With Optuna, the search learns from prior trials. Random search is a fine fallback if you don't want the dep.

## Workflow: HPO then production run

```bash
# 1. Search
python scripts/hpo.py \
    --algo dvt_unet3d \
    --noisy-dir /data/noisy --clean-dir /data/clean \
    --stacks F0 F1 F2 \
    --trials 30 \
    --reduce 0.25 \
    --aggregate min

# 2. Inspect: open hpo_results/hpo_dvt_unet3d_<timestamp>/trials.csv
#    Look at top configs, not just the winner.

# 3. Production run with the best config
cp hpo_results/hpo_dvt_unet3d_<timestamp>/best_config.py configs/dvt_unet3d_hpo.py
python scripts/benchmark.py \
    --noisy-dir /data/noisy --clean-dir /data/clean \
    --algos dvt_unet3d \
    --configs configs/dvt_unet3d_hpo.py
```
