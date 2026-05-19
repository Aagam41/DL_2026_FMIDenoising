#!/usr/bin/env python3
"""
make_paper_figures.py — generate paper-quality figures from existing runs.

Two outputs:

  1. Per-run 1x4 comparison grids (noisy | clean | denoised | residual)
     at a chosen frame. Saved under paper_figures/<run_id>/.

  2. A leaderboard bar chart across all algos for each metric.
     Saved under paper_figures/_leaderboard/<metric>.png.

This script is idempotent: re-running it overwrites the previous figures
without touching the underlying data.

Usage:
    python scripts/make_paper_figures.py \\
        --clean-dir /path/to/clean \\
        [--results-dir benchmark_results] \\
        [--figures-dir paper_figures] \\
        [--frame 750] \\
        [--metrics stSNR stPSNR stSI_PSNR sSNR tSNR]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import csv_db, io as io_, plots


def _read_metrics(results_dir: Path):
    """Return list of dicts joining runs.csv and metrics.csv."""
    runs_path = results_dir / "runs.csv"
    metrics_path = results_dir / "metrics.csv"
    if not runs_path.exists() or not metrics_path.exists():
        return []
    runs = {r["run_id"]: r for r in csv.DictReader(open(runs_path))}
    rows = []
    with open(metrics_path) as f:
        for m in csv.DictReader(f):
            r = runs.get(m["run_id"])
            if not r:
                continue
            rows.append({
                "run_id":     m["run_id"],
                "algo":       r["algo"],
                "stack_name": r["stack_name"],
                "metric":     m["metric"],
                "value":      m["value"],
            })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", default=None,
                    help="Where clean .tif stacks live (for the 1x4 grids).")
    p.add_argument("--results-dir",
                    default=str(ROOT / "benchmark_results"))
    p.add_argument("--figures-dir",
                    default=str(ROOT / "paper_figures"))
    p.add_argument("--frame", type=int, default=750,
                    help="Frame index for the 1x4 comparison grid.")
    p.add_argument("--metrics", nargs="+",
                    default=["stSNR", "stPSNR", "stSI_PSNR", "sSNR", "tSNR"],
                    help="Metrics to plot in the leaderboard bar chart.")
    p.add_argument("--runs", nargs="*", default=None,
                    help="Specific run_ids to regenerate. Default: all "
                         "successful runs.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)

    # ── 1) Per-run 1x4 grids ──────────────────────────────────
    runs = csv_db.read_runs(results_dir)
    if args.runs:
        wanted = set(args.runs)
        runs = [r for r in runs if r["run_id"] in wanted]
    else:
        runs = [r for r in runs if r.get("status") == "success"]

    clean_dir = Path(args.clean_dir) if args.clean_dir else None

    for i, r in enumerate(runs, 1):
        run_id = r["run_id"]; stack = r["stack_name"]
        out_path = Path(r.get("output_path", ""))
        noisy_path = Path(r.get("noisy_path", ""))
        if not out_path.exists() or not noisy_path.exists():
            print(f"[{i}/{len(runs)}] {run_id} — missing files, skip")
            continue

        clean_path = None
        if clean_dir is not None:
            cand = list(clean_dir.glob(f"{stack}.*"))
            if cand:
                clean_path = cand[0]

        try:
            noisy    = io_.load_stack(noisy_path)
            denoised = io_.load_stack(out_path)
            clean    = io_.load_stack(clean_path) if clean_path else None
        except Exception as e:
            print(f"[{i}/{len(runs)}] {run_id} — load failed: {e}")
            continue

        fig_dir = io_.run_figure_dir(figures_dir, run_id)
        fig_path = fig_dir / f"{stack}_frame{args.frame:04d}.png"

        metric_str = (
            f"stSNR={r.get('stSNR','')}  "
            f"stPSNR={r.get('stPSNR','')}  "
            f"stSI_PSNR={r.get('stSI_PSNR','')}"
        )
        try:
            plots.comparison_grid(
                noisy_stack=noisy, clean_stack=clean,
                denoised_stack=denoised, frame=args.frame,
                save_path=fig_path,
                title=f"{r['algo']}  /  {stack}",
                metric_str=metric_str,
            )
            print(f"[{i}/{len(runs)}] {run_id} -> {fig_path.name}")
        except Exception as e:
            print(f"[{i}/{len(runs)}] {run_id} figure failed: {e}")

    # ── 2) Leaderboard bar charts ─────────────────────────────
    rows = _read_metrics(results_dir)
    lb_dir = figures_dir / "_leaderboard"
    lb_dir.mkdir(parents=True, exist_ok=True)
    for metric in args.metrics:
        out = lb_dir / f"{metric}.png"
        plots.leaderboard_bar(
            rows, metric=metric, save_path=out,
            title=f"{metric} by algorithm",
        )
        print(f"Leaderboard {metric}  ->  {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
