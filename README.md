# DL_2026 — Calcium Imaging Denoising Framework

A modular benchmarking framework for self-supervised denoising of two-photon calcium-imaging videos (AI4Life-CIDC25 challenge). Each denoising **algorithm** is a plug-in; the **runner** drives them uniformly; results are written to a multi-table **CSV "database"** with full reproducibility logs.

## Layout

```
DL_2026/
├── algos/                    # Plug-in algorithms (existing files moved here)
│   ├── __init__.py          # REGISTRY mapping name → module
│   ├── n2v_unet3d.py
│   ├── n2v_unet3d_biasfree.py
│   ├── dvt_unet3d.py
│   ├── restormer3d.py
│   ├── restormer3d_v2.py
│   ├── swin_unet3d.py
│   ├── fm2s_dvt.py
│   ├── fm2s_classic.py      # adapter around _fm2s_paper.py
│   ├── n2v_3d_chhayansh.py
│   ├── _fm2s_paper.py       # original fm2s.py, verbatim
│   └── _weights/n2v_3d_chhayansh.pth   # shipped pretrained weights
│
├── configs/                  # One YAML-like .py per (algo, preset)
│   ├── n2v_unet3d_default.py
│   ├── dvt_unet3d_t4.py     # ⭐ proven best config
│   ├── dvt_unet3d_default.py # alias of t4 (single source of truth)
│   └── ...
│
├── runner/                   # The framework — algo-agnostic
│   ├── io.py                # TIFF + group-folder helpers, manifest writer
│   ├── csv_db.py            # Group-scoped multi-CSV "database"
│   ├── runtime_log.py       # GPU/CPU sampling, stage timer
│   ├── eval_runner.py       # Metric computation (calls existing eval.py)
│   ├── preprocessing.py     # Pluggable normalization + temporal-target
│   ├── plots.py             # Paper figures
│   ├── core.py              # run_one(...) — top-level entry
│   └── _eval_metrics.py     # original eval.py, verbatim
│
├── scripts/                  # CLI entry points
│   ├── run_one.py           # Run one (algo, stack) job
│   ├── benchmark.py         # Sweep across all algos × stacks
│   ├── eval_only.py         # Re-eval existing outputs
│   └── make_paper_figures.py
│
├── benchmark_results/        # Auto-created; never edit by hand
│   └── <group_id>/                   # ⭐ One folder per benchmark invocation
│       ├── group_manifest.json       # human-readable per-algo config snapshot
│       ├── runs.csv                  # master table (incl. group_id + config_name)
│       ├── metrics.csv
│       ├── config.csv
│       ├── timing.csv
│       ├── gpu_log.csv
│       ├── stacks.csv
│       ├── algos.csv
│       ├── outputs/<run_id>/<stack>.tif
│       ├── checkpoints/<run_id>/
│       └── errors/<run_id>.log
│
└── paper_figures/            # Auto-created
    └── <group_id>/                   # ⭐ Per-group figures
        ├── group_manifest.json       # mirror of the results-side manifest
        ├── <run_id>/<stack>_frame0750.png
        └── _leaderboard/<metric>.png
```

## Group IDs — what they are and why

Every `benchmark.py` invocation gets one **group_id** like `g_20260519-175306_a1b2c3`. All algos that ran together in that invocation share it. Their outputs all land under `benchmark_results/<group_id>/`. So when you write the paper later, you can look at `group_manifest.json` and see:

- which algos ran in that benchmark
- what config each one used (full dump)
- when it started and completed
- host / GPU / library versions
- success/failure counts

If you want to **add a single algo to an existing benchmark group** (e.g. someone asks you to also run X with config Y for comparison), pass `--group-id <existing>` to `run_one.py` and it'll append to that group's CSVs and manifest.

`scripts/benchmark.py` is safe to interrupt and re-run — completed (algo, stack) pairs are **skipped across all groups**, so reruns avoid duplicating work even if you started a new group.

`scripts/make_paper_figures.py` scans all groups by default (one leaderboard per group) but `--group-id <gid>` restricts it to one group.

## Quickstart

### Install

```bash
pip install -r requirements.txt
# Optional: pynvml + psutil for richer GPU/CPU sampling
pip install pynvml psutil matplotlib
```

### Run one job

```bash
python scripts/run_one.py \
    --config configs/dvt_unet3d_t4.py \
    --noisy  /path/to/noisy/F1.tif \
    --clean  /path/to/clean/F1.tif \
    --frame  750
```

This single command produces:
- A denoised `F1.tif` under `benchmark_results/outputs/<run_id>/`
- A `<stack>_frame0750.png` 1×4 comparison grid under `paper_figures/<run_id>/`
- New rows in `runs.csv` / `metrics.csv` / `config.csv` / `timing.csv`
- Continuous GPU/CPU samples in `gpu_log.csv`
- A model checkpoint under `benchmark_results/checkpoints/<run_id>/`

### Run the full benchmark

```bash
python scripts/benchmark.py \
    --noisy-dir /path/to/noisy \
    --clean-dir /path/to/clean \
    --algos all
```

Sweeps every algorithm in the registry × every stack in the noisy directory. **Already-completed `(algo, stack)` pairs are skipped** — safe to interrupt and rerun.

### Generate paper figures

```bash
python scripts/make_paper_figures.py \
    --clean-dir /path/to/clean \
    --frame 750
```

Regenerates all per-run 1×4 grids and a leaderboard bar chart per metric.

## How the CSV database works

Six tables sharing `run_id` as the join key:

| table       | grain                         | use it to ask                                          |
|-------------|-------------------------------|--------------------------------------------------------|
| runs.csv    | one row per run               | "what algos have I tried, and how did they do overall" |
| metrics.csv | one row per (run, metric)     | "give me the full metric set for runs X, Y, Z"         |
| config.csv  | one row per (run, key)        | "what config did this best run use"                    |
| timing.csv  | one row per (run, stage)      | "where did the 30 minutes go"                          |
| gpu_log.csv | ~30 rows per minute of run    | "what was peak GPU memory during inference"            |
| stacks.csv  | one row per loaded stack      | "frame counts, shapes, intensity ranges per stack"     |

To join in pandas:

```python
import pandas as pd
runs    = pd.read_csv("benchmark_results/runs.csv")
metrics = pd.read_csv("benchmark_results/metrics.csv")
joined  = runs.merge(metrics, on="run_id")
# Top-3 algos by stSNR on F1:
joined[(joined.metric == "stSNR") & (joined.stack_name == "F1")] \
      .nlargest(3, "value")[["algo", "value"]]
```

## Adding a new algorithm

1. Drop a Python module into `algos/your_algo.py` exposing this API:

   ```python
   # Declare default preprocessing strategies (REQUIRED for new algos)
   DEFAULT_NORMALIZATION   = "framework_default"   # see runner/preprocessing.py
   DEFAULT_TEMPORAL_TARGET = "temporal_median_2d"

   def compute_norm_params(stack): ...
   def normalize(stack, params): ...
   def denormalize(stack, params): ...
   def train_self_supervised(stack, device, config): -> (model, cfg)
   def denoise_stack(model, stack, config, device): -> np.ndarray
   def save_checkpoint(model, config, path): ...
   def load_checkpoint(path, device=None): -> (model, cfg)
   ```

2. Register it in `algos/__init__.py`:

   ```python
   "your_algo": AlgoSpec(
       name="your_algo",
       module="algos.your_algo",
       family="YourFamily",
       description="One-line summary.",
   ),
   ```

3. Drop a config in `configs/your_algo_default.py`:

   ```python
   CONFIG = {
       "algo": "your_algo",
       "name": "your_algo_default",
       "paper_frame": 750,
       # ... your algo's kwargs
   }
   ```

That's it — `scripts/benchmark.py` will pick it up automatically next time.

## Pluggable preprocessing (normalization + temporal target)

Both **normalization** and the **temporal-target / reference frame** computation are pluggable strategies, selectable per-run via the config dict.

### Selecting a strategy

```python
CONFIG = {
    "algo": "dvt_unet3d",
    # default per-algo if absent — override if you want different behavior
    "normalization":   "p0.5_p99.5",       # or "chhayansh", "p3_p97", "minmax", "noop", ...
    "temporal_target": "temporal_median_2d",  # or "temporal_mean_2d", "per_frame_median_3d", ...
    ...
}
```

### Available normalization strategies (`runner/preprocessing.py`)

| Name | What it does |
|---|---|
| `framework_default` | p0.5–p99.5 percentile on 300 sampled frames (most algos) |
| `p0.5_p99.5` | Synonym of `framework_default` |
| `p1_p99` | p1–p99 percentile |
| `p3_p97` | p3–p97 percentile on 300 sampled frames (n2v_unet3d default) |
| `chhayansh` | p3–p97 on FULL volume (matches upstream `chhayansh` repo) |
| `p3_p97_fullvol` | Synonym of `chhayansh` |
| `minmax` | Naive min/max scaling |
| `meanstd` | Zero-mean unit-std z-score |
| `noop` | Pass-through (no scaling; FM2S handles it internally) |

### Available temporal-target strategies

| Name | Returns | What it does |
|---|---|---|
| `framework_default` | 2D | Subsampled per-pixel temporal median (most algos) |
| `temporal_median_2d` | 2D | Synonym of `framework_default` |
| `temporal_mean_2d` | 2D | Subsampled per-pixel temporal mean — cheaper |
| `full_stack_median_2d` | 2D | Exact median over all frames (no subsampling) |
| `per_frame_median_3d` | 3D | Sliding-window median: each output frame has its OWN local median |

Strategies returning a 3D volume are automatically collapsed to 2D by the algos that expect a 2D target.

### Per-algo defaults

Each algo declares its default at module level. The framework's `run_one` reads these as the fallback when config doesn't specify an override:

| Algo | Default normalization | Default temporal target |
|---|---|---|
| `n2v_unet3d` | `p3_p97` | `temporal_median_2d` |
| `n2v_unet3d_biasfree` | `p3_p97` | `temporal_median_2d` |
| `dvt_unet3d` | `p0.5_p99.5` | `temporal_median_2d` |
| `restormer3d` | `p0.5_p99.5` | `temporal_median_2d` |
| `restormer3d_v2` | `p0.5_p99.5` | `temporal_median_2d` |
| `swin_unet3d` | `p0.5_p99.5` | `temporal_median_2d` |
| `fm2s_dvt` | `p0.5_p99.5` | `temporal_median_2d` |
| `fm2s_classic` | `noop` | `temporal_median_2d` |
| `n2v_3d_chhayansh` | `chhayansh` | `temporal_median_2d` |

### Adding a new strategy at runtime

```python
from runner import preprocessing as prep
class MyNorm(prep.NormalizationStrategy):
    name = "my_norm"
    def compute_params(self, stack):
        return {"shift": 0.0, "scale": float(stack.std()), "strategy": "my_norm"}
prep.register_normalization(MyNorm())

# Now use it from any config:
CONFIG["normalization"] = "my_norm"
```

The resolved strategy name is recorded in `runs.csv` (column `__resolved_normalization` / `__resolved_temporal_target`) and in `config.csv` for full traceability.

## Notes for the paper

- The 1×4 grid panels share intensity range across noisy/clean/denoised so over- or under-smoothing is visible at a glance.
- The residual panel uses a diverging colormap centered at zero — a healthy denoiser produces a residual that "looks like noise" (no visible structure).
- The leaderboard chart groups bars per stack, one per algo, so you can see both per-stack variability and the overall winner.
- Every figure has a footer with the frame number and a header with metrics — drop straight into the paper.
- All metric definitions match the CIDC25 evaluator exactly.
