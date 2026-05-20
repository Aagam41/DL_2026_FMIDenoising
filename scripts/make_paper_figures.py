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


def _read_metrics_for_group(results_dir: Path, group_id: str):
    """Return list of dicts joining runs.csv and metrics.csv for ONE group."""
    gp = results_dir / group_id
    runs_path = gp / "runs.csv"
    metrics_path = gp / "metrics.csv"
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
                "run_id":      m["run_id"],
                "group_id":    group_id,
                "algo":        r["algo"],
                "config_name": r.get("config_name", ""),
                "stack_name":  r["stack_name"],
                "metric":      m["metric"],
                "value":       m["value"],
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
    p.add_argument("--group-id", default=None,
                    help="Regenerate figures for a single group_id. Default: "
                         "ALL groups (one leaderboard per group).")
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

    # Decide which groups to process
    if args.group_id:
        groups = [args.group_id]
    else:
        groups = csv_db.list_groups(results_dir)
    if not groups:
        # Fall back to legacy top-level runs.csv (pre-grouping)
        groups = [None]
    print(f"Processing {len(groups)} group(s): "
          f"{[g if g else '(legacy top-level)' for g in groups]}")

    clean_dir = Path(args.clean_dir) if args.clean_dir else None

    for group_id in groups:
        # ── 1) Per-run 1x4 grids for this group ──────────────────
        runs = csv_db.read_runs(results_dir, group_id=group_id)
        if args.runs:
            wanted = set(args.runs)
            runs = [r for r in runs if r["run_id"] in wanted]
        else:
            runs = [r for r in runs if r.get("status") == "success"]

        if not runs:
            print(f"[group {group_id}] no successful runs, skip")
            continue

        print(f"\n[group {group_id}] {len(runs)} run(s)")

        for i, r in enumerate(runs, 1):
            run_id = r["run_id"]; stack = r["stack_name"]
            out_path = Path(r.get("output_path", ""))
            noisy_path = Path(r.get("noisy_path", ""))
            if not out_path.exists() or not noisy_path.exists():
                print(f"  [{i}/{len(runs)}] {run_id} — missing files, skip")
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
                print(f"  [{i}/{len(runs)}] {run_id} — load failed: {e}")
                continue

            if group_id:
                fig_dir = io_.run_figure_dir(figures_dir, group_id, run_id)
            else:
                # Legacy fallback (pre-grouping data)
                fig_dir = figures_dir / run_id
                fig_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fig_dir / f"{stack}_frame{args.frame:04d}.png"

            metric_str = (
                f"stSNR={r.get('stSNR','')}  "
                f"stPSNR={r.get('stPSNR','')}  "
                f"stSI_PSNR={r.get('stSI_PSNR','')}"
            )
            title = f"{r['algo']} ({r.get('config_name','')})  /  {stack}"
            try:
                plots.comparison_grid(
                    noisy_stack=noisy, clean_stack=clean,
                    denoised_stack=denoised, frame=args.frame,
                    save_path=fig_path,
                    title=title,
                    metric_str=metric_str,
                )
                print(f"  [{i}/{len(runs)}] {run_id} -> {fig_path.name}")
            except Exception as e:
                print(f"  [{i}/{len(runs)}] {run_id} figure failed: {e}")

        # ── 2) Leaderboard bar charts (per group) ───────────────
        if group_id:
            rows = _read_metrics_for_group(results_dir, group_id)
            lb_dir = figures_dir / group_id / "_leaderboard"
        else:
            # Legacy: read top-level files
            runs_path = results_dir / "runs.csv"
            metrics_path = results_dir / "metrics.csv"
            rows = []
            if runs_path.exists() and metrics_path.exists():
                runs_map = {r["run_id"]: r for r in
                              csv.DictReader(open(runs_path))}
                for m in csv.DictReader(open(metrics_path)):
                    rr = runs_map.get(m["run_id"])
                    if not rr:
                        continue
                    rows.append({
                        "run_id":     m["run_id"],
                        "algo":       rr["algo"],
                        "stack_name": rr["stack_name"],
                        "metric":     m["metric"],
                        "value":      m["value"],
                    })
            lb_dir = figures_dir / "_leaderboard"
        lb_dir.mkdir(parents=True, exist_ok=True)
        for metric in args.metrics:
            out = lb_dir / f"{metric}.png"
            plots.leaderboard_bar(
                rows, metric=metric, save_path=out,
                title=f"{metric} by algorithm"
                       + (f"  ({group_id})" if group_id else ""),
            )
            print(f"  Leaderboard {metric}  ->  {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
