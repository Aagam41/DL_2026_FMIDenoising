#!/usr/bin/env python3
"""
run_one.py — run a single (algo, stack) job.

Usage:
    python scripts/run_one.py \\
        --config  configs/dvt_unet3d_t4.py \\
        --noisy   /path/to/noisy/F1.tif \\
        --clean   /path/to/clean/F1.tif        # optional, for eval
        --results-dir benchmark_results        # default: ./benchmark_results
        --figures-dir paper_figures            # default: ./paper_figures
        --frame   750                          # paper-figure frame

Outputs:
    benchmark_results/runs.csv             — master table, one row per run
    benchmark_results/metrics.csv          — all metrics this run
    benchmark_results/config.csv           — flattened config
    benchmark_results/timing.csv           — per-stage durations
    benchmark_results/gpu_log.csv          — GPU samples during the run
    benchmark_results/outputs/<run_id>/<stack>.tif
    benchmark_results/checkpoints/<run_id>/<stack>.pth
    paper_figures/<run_id>/<stack>_frame0750.png
"""

import argparse
import sys
from pathlib import Path

# Make `algos` / `runner` / `configs` importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from runner.core import run_one
from configs import load_config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True,
                    help="Path to a configs/*.py file")
    p.add_argument("--noisy",      required=True,
                    help="Path to the noisy input .tif")
    p.add_argument("--clean",      default=None,
                    help="Optional path to the clean ground-truth .tif")
    p.add_argument("--results-dir", default=str(ROOT / "benchmark_results"),
                    help="Where CSVs, outputs, and checkpoints are written")
    p.add_argument("--figures-dir", default=str(ROOT / "paper_figures"),
                    help="Where paper figures are written")
    p.add_argument("--frame",      type=int, default=None,
                    help="Frame index for the 1x4 paper figure (default: "
                         "from config or 750)")
    p.add_argument("--no-checkpoint", action="store_true",
                    help="Don't save the trained model checkpoint")
    p.add_argument("--no-figures",   action="store_true",
                    help="Don't generate paper figures")
    p.add_argument("--quiet", action="store_true",
                    help="Less verbose output")
    args = p.parse_args()

    cfg = load_config(args.config)
    algo = cfg.pop("algo")     # mandatory
    # Pull out framework-level keys
    name = cfg.pop("name", None)
    desc = cfg.pop("description", None)
    paper_frame = args.frame or cfg.pop("paper_frame", 750)

    print(f"\nConfig: {Path(args.config).name}")
    if name:
        print(f"  name: {name}")
    if desc:
        print(f"  desc: {desc}")
    print(f"  algo: {algo}")
    print(f"  paper_frame: {paper_frame}")

    summary = run_one(
        algo=algo,
        config=cfg,
        noisy_path=args.noisy,
        clean_path=args.clean,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
        paper_frame=paper_frame,
        save_checkpoint=not args.no_checkpoint,
        save_figures=not args.no_figures,
        verbose=not args.quiet,
    )

    print(f"\n────  Run complete  ────")
    print(f"  status   : {summary['status']}")
    print(f"  run_id   : {summary['run_id']}")
    print(f"  output   : {summary.get('output_path', '(none)')}")
    print(f"  figure   : {summary.get('figure_path', '(none)')}")
    if summary.get("metrics"):
        m = summary["metrics"]
        print(f"  stSNR    : {m.get('stSNR', float('nan')):.4f}")
        print(f"  stPSNR   : {m.get('stPSNR', float('nan')):.4f}")
        print(f"  stSI_PSNR: {m.get('stSI_PSNR', float('nan')):.4f}")
    print(f"  total_sec: {summary.get('total_sec', 0):.1f}")

    return 0 if summary["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
