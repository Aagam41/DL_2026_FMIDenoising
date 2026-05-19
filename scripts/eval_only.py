#!/usr/bin/env python3
"""
eval_only.py — re-evaluate existing denoised outputs against clean stacks.

Useful if:
  - You denoised before clean ground truth was available.
  - You want to recompute metrics after a metric-code change.

Reads:
    benchmark_results/runs.csv to find existing denoised outputs.

Writes:
    benchmark_results/metrics.csv — appends new rows tagged with the same
    run_id as the original denoise run.
    benchmark_results/eval_only.csv — record of which run_ids were re-eval'd.

Usage:
    python scripts/eval_only.py \\
        --clean-dir /path/to/clean \\
        [--results-dir benchmark_results] \\
        [--runs run_id_1 run_id_2 ...]      # default: all success runs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner import csv_db, io as io_, eval_runner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clean-dir", required=True,
                    help="Directory with clean GT .tif stacks.")
    p.add_argument("--results-dir",
                    default=str(ROOT / "benchmark_results"))
    p.add_argument("--runs", nargs="*", default=None,
                    help="Specific run_ids to re-evaluate. Default: all "
                         "successful runs in runs.csv.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    clean_dir   = Path(args.clean_dir)

    rows = csv_db.read_runs(results_dir)
    if args.runs:
        wanted = set(args.runs)
        rows = [r for r in rows if r["run_id"] in wanted]
    else:
        rows = [r for r in rows if r.get("status") == "success"]

    if not rows:
        print("No matching runs to evaluate.")
        return 0

    print(f"Re-evaluating {len(rows)} run(s) against clean dir: {clean_dir}")

    for i, row in enumerate(rows, 1):
        run_id = row["run_id"]; stack = row["stack_name"]
        out_path = Path(row.get("output_path", ""))
        if not out_path.exists():
            print(f"[{i}/{len(rows)}] {run_id} — output missing, skip")
            continue
        clean_candidates = list(clean_dir.glob(f"{stack}.*"))
        if not clean_candidates:
            print(f"[{i}/{len(rows)}] {run_id} — no clean for {stack}, skip")
            continue
        clean_path = clean_candidates[0]
        print(f"[{i}/{len(rows)}] {run_id}  vs  {clean_path.name}")

        try:
            denoised = io_.load_stack(out_path)
            clean    = io_.load_stack(clean_path)
            metrics  = eval_runner.evaluate_pair(denoised, clean)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        csv_db.write_metrics(results_dir, run_id, metrics)
        csv_db.append_row(results_dir / "eval_only.csv", {
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "run_id":       run_id,
            "stack_name":   stack,
            "clean_path":   str(clean_path),
            "stSNR":        metrics.get("stSNR", ""),
            "stPSNR":       metrics.get("stPSNR", ""),
            "stSI_PSNR":    metrics.get("stSI_PSNR", ""),
        })
        if "stSNR" in metrics:
            print(f"  stSNR={metrics['stSNR']:.4f}  "
                  f"stPSNR={metrics.get('stPSNR', 0):.4f}  "
                  f"stSI_PSNR={metrics.get('stSI_PSNR', 0):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
