#!/usr/bin/env python3
"""
benchmark.py — run every (algo, stack) pair that hasn't been run yet.

Discovers noisy/clean pairs under a dataset directory and runs the
selected algos on each. Already-completed (algo, stack) pairs (per
runs.csv) are SKIPPED so the script is safe to rerun.

Usage:
    python scripts/benchmark.py \\
        --noisy-dir /path/to/noisy \\
        --clean-dir /path/to/clean \\
        --algos all                            # or: dvt_unet3d swin_unet3d
        --configs configs/dvt_unet3d_t4.py configs/swin_unet3d_default.py
        --results-dir benchmark_results
        --figures-dir paper_figures

If --configs is not given, the script uses ONE default config per algo
named configs/<algo>_default.py if it exists, else configs/<algo>_t4.py.

Outputs are exactly the same as scripts/run_one.py — one consolidated
benchmark_results/ tree with CSVs and per-run TIFF/PNG outputs.
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import algos as _algos_pkg
from configs import load_config
from runner.core import run_one
from runner.io import find_stacks
from runner.csv_db import (
    completed_pairs, write_algo_registry,
)


def _serializable(d):
    """Best-effort JSON-friendly version of a config dict."""
    if d is None:
        return None
    if isinstance(d, (str, int, float, bool)):
        return d
    if isinstance(d, (list, tuple)):
        return [_serializable(x) for x in d]
    if isinstance(d, dict):
        return {k: _serializable(v) for k, v in d.items()}
    return str(d)


def _resolve_config_for_algo(algo: str, config_paths) -> Path:
    """If user passed an explicit config for this algo, use it. Otherwise
    look up configs/<algo>_default.py or configs/<algo>_t4.py."""
    for cp in config_paths or []:
        cp = Path(cp)
        # Try matching by loading the file and checking its 'algo' key
        try:
            cfg = load_config(cp)
            if cfg.get("algo") == algo:
                return cp
        except Exception:
            continue
    # Fall back to defaults
    candidates = [
        ROOT / "configs" / f"{algo}_default.py",
        ROOT / "configs" / f"{algo}_t4.py",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No config found for algo '{algo}'. "
        f"Pass --configs explicitly or create configs/{algo}_default.py."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--noisy-dir",  required=True,
                    help="Directory with noisy stacks (*.tif)")
    p.add_argument("--clean-dir",  default=None,
                    help="Directory with clean GT stacks (*.tif). "
                         "If absent, metrics are not computed.")
    p.add_argument("--algos",      nargs="+", default=["all"],
                    help="Algo names from algos/__init__.py REGISTRY, "
                         "or 'all'.")
    p.add_argument("--configs",    nargs="*", default=None,
                    help="Explicit config files to use (matched to algos "
                         "via their 'algo' key). If omitted, looks up "
                         "configs/<algo>_default.py.")
    p.add_argument("--results-dir", default=str(ROOT / "benchmark_results"))
    p.add_argument("--figures-dir", default=str(ROOT / "paper_figures"))
    p.add_argument("--group-id", default=None,
                    help="Group ID for output scoping. If omitted, a fresh "
                         "one is generated. Pass an existing group_id to "
                         "append to a prior benchmark group.")
    p.add_argument("--frame", type=int, default=None,
                    help="Override paper_frame from config.")
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--no-figures",    action="store_true")
    p.add_argument("--dry-run", action="store_true",
                    help="List jobs that would run; don't execute.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Group ID: one per benchmark invocation ─────────────────
    # Every algo+stack run in this invocation gets the same group_id so
    # they land in <results_dir>/<group_id>/ together. If the user
    # passed --group-id, append to that group instead.
    from runner import io as io_
    group_id = args.group_id or io_.make_group_id()
    (results_dir / group_id).mkdir(parents=True, exist_ok=True)
    (figures_dir / group_id).mkdir(parents=True, exist_ok=True)
    print(f"\nGroup ID for this benchmark: {group_id}")

    # Resolve algos
    if "all" in args.algos:
        algo_names = sorted(_algos_pkg.REGISTRY.keys())
    else:
        algo_names = list(args.algos)
        for a in algo_names:
            if a not in _algos_pkg.REGISTRY:
                sys.exit(f"Unknown algo: {a}. "
                         f"Known: {sorted(_algos_pkg.REGISTRY.keys())}")

    # Write the registry snapshot to algos.csv (group-scoped)
    write_algo_registry(results_dir, _algos_pkg.list_algos(),
                         group_id=group_id)

    # Resolve configs per algo
    resolved_cfgs = {}
    for a in algo_names:
        try:
            resolved_cfgs[a] = _resolve_config_for_algo(a, args.configs)
        except FileNotFoundError as e:
            print(f"  [skip] {a}: {e}")

    # Discover stacks
    pairs = find_stacks(args.noisy_dir, args.clean_dir)
    if not pairs:
        sys.exit(f"No .tif files found under {args.noisy_dir}.")

    # Build the job list, filtering already-done (across ALL groups)
    done = completed_pairs(results_dir)
    jobs = []
    for stack_name, noisy_path, clean_path in pairs:
        for algo in resolved_cfgs:
            if (algo, stack_name) in done:
                continue
            jobs.append((algo, stack_name, noisy_path, clean_path,
                         resolved_cfgs[algo]))

    print(f"\nBenchmark plan:")
    print(f"  noisy dir : {args.noisy_dir}")
    print(f"  clean dir : {args.clean_dir or '(none)'}")
    print(f"  algos     : {len(resolved_cfgs)} ({list(resolved_cfgs.keys())})")
    print(f"  stacks    : {len(pairs)}")
    print(f"  jobs      : {len(jobs)} new "
          f"({len(done)} already completed across all groups)")
    print(f"  results   : {results_dir / group_id}")
    print(f"  figures   : {figures_dir / group_id}")

    if args.dry_run or len(jobs) == 0:
        for i, (algo, stack, noisy, clean, cfg_path) in enumerate(jobs, 1):
            print(f"  [{i:3d}] {algo:30s}  on  {stack:15s}  "
                  f"({cfg_path.name})")
        return 0

    # ── Build the group manifest BEFORE running any jobs ──────
    # We snapshot every algo's resolved config up front so the manifest
    # is useful even if a later job crashes. After all jobs complete we
    # update completed_at + summary fields.
    from runner import runtime_log as _rtl
    env_info = _rtl.env_info()
    manifest = {
        "group_id":      group_id,
        "started_at":    datetime.now(timezone.utc).isoformat(),
        "host":          env_info.get("host", ""),
        "python":        env_info.get("python", ""),
        "torch":         env_info.get("torch", ""),
        "gpu_name":      env_info.get("gpu_name", ""),
        "noisy_dir":     str(args.noisy_dir),
        "clean_dir":     str(args.clean_dir) if args.clean_dir else "",
        "stacks":        [name for name, _, _ in pairs],
        "configs":       {},
        "total_jobs":    len(jobs),
    }
    for algo, cfg_path in resolved_cfgs.items():
        try:
            cfg_snapshot = load_config(cfg_path)
        except Exception as e:
            cfg_snapshot = {"_load_error": str(e)}
        cn = Path(cfg_path).stem
        manifest["configs"][f"{algo}__{cn}"] = {
            "algo":        algo,
            "config_name": cn,
            "config_path": str(Path(cfg_path).resolve()),
            "full_config": _serializable(cfg_snapshot),
        }
    io_.write_group_manifest(
        results_dir=results_dir, group_id=group_id,
        manifest=manifest, figures_dir=figures_dir,
    )

    # Execute
    successes = 0; failures = 0
    for i, (algo, stack, noisy, clean, cfg_path) in enumerate(jobs, 1):
        print(f"\n[{i}/{len(jobs)}] {algo}  on  {stack}  (group={group_id})")
        cfg = load_config(cfg_path)
        cfg.pop("algo")
        paper_frame = args.frame or cfg.pop("paper_frame", 750)
        cfg.pop("name", None); cfg.pop("description", None)
        config_name = Path(cfg_path).stem
        try:
            summary = run_one(
                algo=algo, config=cfg,
                noisy_path=noisy, clean_path=clean,
                results_dir=results_dir, figures_dir=figures_dir,
                group_id=group_id, config_name=config_name,
                paper_frame=paper_frame,
                save_checkpoint=not args.no_checkpoint,
                save_figures=not args.no_figures,
                verbose=True,
            )
            if summary.get("status") == "success":
                successes += 1
            else:
                failures += 1
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
            return 130
        except Exception as e:
            print(f"  Job failed: {type(e).__name__}: {e}")
            # run_one already logs errors; continue to next job
            continue

    # ── Update manifest with completion summary ────────────────
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    manifest["successes"]    = successes
    manifest["failures"]     = failures
    io_.write_group_manifest(
        results_dir=results_dir, group_id=group_id,
        manifest=manifest, figures_dir=figures_dir,
    )

    print(f"\n{'='*70}\nBenchmark complete: {len(jobs)} job(s) executed "
          f"({successes} success, {failures} failed).")
    print(f"Group: {group_id}")
    print(f"Manifest: {results_dir / group_id / 'group_manifest.json'}")
    print(f"{'='*70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
