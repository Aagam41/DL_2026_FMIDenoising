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
│   └── _fm2s_paper.py       # original fm2s.py, verbatim
│
├── configs/                  # One YAML-like .py per (algo, preset)
│   ├── n2v_unet3d_default.py
│   ├── dvt_unet3d_t4.py
│   ├── ...
│
├── runner/                   # The framework — algo-agnostic
│   ├── io.py                # TIFF loading, dataset discovery
│   ├── csv_db.py            # Multi-CSV "database" writes
│   ├── runtime_log.py       # GPU/CPU sampling, stage timer
│   ├── eval_runner.py       # Metric computation (calls existing eval.py)
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
│   ├── runs.csv             # ⭐ master table, one row per run
│   ├── metrics.csv          # one row per (run_id, metric)
│   ├── config.csv           # one row per (run_id, config_key)
│   ├── timing.csv           # per-stage durations
│   ├── gpu_log.csv          # ~1 sample / 2 seconds during the run
│   ├── stacks.csv           # one row per loaded stack
│   ├── algos.csv            # registry snapshot
│   ├── outputs/<run_id>/    # denoised .tif files
│   ├── checkpoints/<run_id>/
│   └── errors/<run_id>.log  # full traceback on failure
│
└── paper_figures/            # Auto-created
    ├── <run_id>/<stack>_frame0750.png   # 1×4 grids
    └── _leaderboard/<metric>.png        # bar charts
```

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
    --noisy-dir data/test \
    --clean-dir data/gt \
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

## Notes for the paper

- The 1×4 grid panels share intensity range across noisy/clean/denoised so over- or under-smoothing is visible at a glance.
- The residual panel uses a diverging colormap centered at zero — a healthy denoiser produces a residual that "looks like noise" (no visible structure).
- The leaderboard chart groups bars per stack, one per algo, so you can see both per-stack variability and the overall winner.
- Every figure has a footer with the frame number and a header with metrics — drop straight into the paper.
- All metric definitions match the CIDC25 evaluator exactly.
