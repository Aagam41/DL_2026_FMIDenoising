#!/usr/bin/env python3
"""
Ablation study runner.

Each ablation is a named configuration that differs from a baseline
by toggling one feature. The script runs ALL ablations × ALL chosen
stacks, computes the chosen metric, and writes a comparison table.

Unlike HPO (Bayesian search over continuous params), ablations are an
exhaustive small grid of NAMED configs. The point isn't to find the
best — it's to quantify how much each feature contributes by comparing
its `*_off` variant to the baseline.

USAGE
─────
Run all ablations for one algo:

    python scripts/ablation.py \\
        --algo dvt_unet3d \\
        --noisy-dir /data/noisy --clean-dir /data/clean \\
        --stacks F0 F1 F2 \\
        --reduce 0.5

Limit to a subset of ablations:

    python scripts/ablation.py \\
        --algo dvt_unet3d \\
        --noisy-dir … --clean-dir … \\
        --stacks F1 \\
        --names baseline no_warmup mask_radius_1

OUTPUT
──────
ablation_results/<algo>_<timestamp>/
    ablations.csv      ← per-ablation × per-stack metric
    summary.csv        ← aggregated table (one row per ablation, mean ± std)
    runs.csv, ...      ← full framework logs for each ablation run

INTERPRETING
────────────
The `baseline` ablation IS the baseline. Every other ablation modifies
it by one feature. Look at the `delta_vs_baseline` column in summary.csv
to see the change in the chosen metric. Negative deltas mean the
feature being toggled HURT — i.e. it's contributing positively in
the baseline and removing it makes things worse.

CAVEATS
───────
1. Like HPO, ablation runs are EXPENSIVE. With 7+ ablations and 3
   stacks per algo, you'll spend hours per algo on T4. Use --reduce
   to speed up trials at the cost of conclusions being noisier.

2. The "importance" of a feature depends on the rest of the config.
   An ablation says "this feature matters by X" only in the context
   of the baseline you chose. If you change the baseline, the
   importance can change.

3. Single-stack ablations have high variance — the conclusions can
   flip on a different stack. Use --stacks with at least 2-3 stacks
   for any conclusion you'd put in a paper.
"""
import argparse
import csv
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Ablation space loader ───────────────────────────────────────

def load_ablation_space(algo: str):
    """Load ablation_spaces/<algo>.py and return (BASE_CONFIG, ABLATIONS, DESCRIPTION).

    ABLATIONS is a dict: name -> dict-of-overrides-to-merge-onto-BASE_CONFIG.
    The key 'baseline' is reserved for the unmodified baseline run.
    """
    spec_path = ROOT / "ablation_spaces" / f"{algo}.py"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"No ablation space spec for '{algo}'. Create "
            f"ablation_spaces/{algo}.py with BASE_CONFIG and "
            f"ABLATIONS dicts."
        )
    spec = importlib.util.spec_from_file_location(
        f"ablation_spaces.{algo}", str(spec_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "BASE_CONFIG"):
        raise AttributeError(
            f"ablation_spaces/{algo}.py must define BASE_CONFIG dict."
        )
    if not hasattr(mod, "ABLATIONS"):
        raise AttributeError(
            f"ablation_spaces/{algo}.py must define ABLATIONS dict "
            f"(name -> override dict)."
        )
    desc = getattr(mod, "DESCRIPTION", "")
    return mod.BASE_CONFIG, mod.ABLATIONS, desc


# ── Single ablation run ─────────────────────────────────────────

def run_ablation(
    *, ablation_name, overrides, base_config, algo,
    stack_pairs, results_dir, figures_dir, group_id,
    reduce_factor, metric, verbose,
):
    """
    Run one ablation across all chosen stacks. Returns
    {stack_name: metric_value_or_None}.
    """
    from runner.core import run_one

    cfg = dict(base_config)
    cfg.update(overrides)
    cfg.pop("algo", None)
    paper_frame = cfg.pop("paper_frame", 750)
    cfg.pop("name", None)
    cfg.pop("description", None)

    # Apply reduce_factor to all iter-style keys we know about
    for k in ("warmup_iters", "n2v_iters", "fm2s_iters", "vit_iters",
              "train_num", "max_epoch"):
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = max(int(round(cfg[k] * reduce_factor)), 1)

    per_stack = {}
    for stack_name, noisy_path, clean_path in stack_pairs:
        # config_name embeds the ablation name so the CSV row is
        # immediately readable. Each (algo, stack, ablation) is a
        # distinct job; the benchmark framework writes its standard
        # CSVs under <group>/.
        try:
            summary = run_one(
                algo=algo, config=dict(cfg),
                noisy_path=noisy_path, clean_path=clean_path,
                results_dir=results_dir, figures_dir=figures_dir,
                group_id=group_id,
                config_name=f"abl_{ablation_name}",
                paper_frame=paper_frame,
                save_checkpoint=False, save_figures=False,
                verbose=verbose,
            )
        except Exception:
            traceback.print_exc()
            per_stack[stack_name] = None
            continue

        if summary["status"] != "success":
            per_stack[stack_name] = None
            continue
        m = (summary.get("metrics") or {}).get(metric)
        try:
            mv = float(m)
            per_stack[stack_name] = (None if mv != mv else mv)
        except (TypeError, ValueError):
            per_stack[stack_name] = None

    return per_stack


# ── CSV writers ─────────────────────────────────────────────────

def init_ablations_csv(path: Path, stack_names, metric: str):
    """Header: ablation, status, duration_sec, then one column per stack."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["ablation", "started_at", "duration_sec", "status",
            *[f"{metric}__{s}" for s in stack_names], "overrides"]
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow(cols)


def append_ablations_row(path: Path, *, ablation, started_at, duration,
                          status, per_stack, stack_names, overrides):
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        row = [
            ablation, started_at, f"{duration:.1f}", status,
            *[(f"{per_stack.get(s):.6f}" if per_stack.get(s) is not None
               else "") for s in stack_names],
            json.dumps(overrides, default=str),
        ]
        w.writerow(row)


def write_summary(path: Path, *, results, baseline_name, metric,
                   stack_names):
    """
    Write one row per ablation:
        ablation, mean, std, min, max, delta_vs_baseline
    `results` is a dict: ablation_name -> {stack_name: value_or_None}.
    """
    import statistics

    def _stats(vals):
        v = [x for x in vals if x is not None]
        if not v:
            return None, None, None, None
        if len(v) == 1:
            return v[0], 0.0, v[0], v[0]
        return (statistics.mean(v), statistics.stdev(v),
                min(v), max(v))

    baseline_vals = list(results.get(baseline_name, {}).values())
    bm = sum(x for x in baseline_vals if x is not None) / max(
        1, sum(1 for x in baseline_vals if x is not None))
    if not any(x is not None for x in baseline_vals):
        bm = None

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ablation", f"{metric}_mean", f"{metric}_std",
                     f"{metric}_min", f"{metric}_max",
                     "delta_vs_baseline", "n_stacks_ok"])
        # baseline first, then the rest in their declared order
        ordered = list(results.keys())
        if baseline_name in ordered:
            ordered.remove(baseline_name)
            ordered = [baseline_name] + ordered
        for name in ordered:
            per_stack = results[name]
            mean, std, lo, hi = _stats(per_stack.values())
            n_ok = sum(1 for v in per_stack.values() if v is not None)
            delta = "" if (mean is None or bm is None) else f"{mean - bm:+.4f}"
            w.writerow([
                name,
                f"{mean:.4f}" if mean is not None else "",
                f"{std:.4f}" if std is not None else "",
                f"{lo:.4f}" if lo is not None else "",
                f"{hi:.4f}" if hi is not None else "",
                delta, n_ok,
            ])


# ── Main ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Ablation study runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--algo", required=True,
                    help="Algorithm name (must have ablation_spaces/<algo>.py).")
    p.add_argument("--noisy-dir", required=True)
    p.add_argument("--clean-dir", required=True)
    p.add_argument("--stacks", nargs="+", default=None,
                    help="Stack names. Default: all stacks in --noisy-dir.")
    p.add_argument("--names", nargs="+", default=None,
                    help="Subset of ablation names to run. Default: all "
                         "ablations declared in the spec.")
    p.add_argument("--metric", default="stSNR",
                    help="Metric to collect. Default: stSNR.")
    p.add_argument("--reduce", type=float, default=1.0,
                    help="Multiply iter-style schedule keys by this. "
                         "1.0 = full budget. 0.5 = half-iters (faster, "
                         "noisier). Default: 1.0 — ablations should "
                         "compare full-budget configs.")
    p.add_argument("--baseline-name", default="baseline",
                    help="Which entry in ABLATIONS to treat as the "
                         "baseline for delta computation. Default: "
                         "'baseline'.")
    p.add_argument("--results-dir", default=str(ROOT / "ablation_results"))
    p.add_argument("--figures-dir", default=str(ROOT / "paper_figures"))
    p.add_argument("--quiet", action="store_true",
                    help="Silence per-run training logs.")
    p.add_argument("--dry-run", action="store_true",
                    help="List the jobs that would run; don't execute.")
    args = p.parse_args()

    base_config, ablations, desc = load_ablation_space(args.algo)
    if not isinstance(ablations, dict) or not ablations:
        sys.exit(f"ABLATIONS in ablation_spaces/{args.algo}.py must be a "
                 f"non-empty dict.")

    # Pick the subset
    if args.names:
        unknown = set(args.names) - set(ablations.keys())
        if unknown:
            sys.exit(f"Unknown ablation names: {sorted(unknown)}. "
                     f"Available: {sorted(ablations.keys())}")
        ablations = {n: ablations[n] for n in args.names}

    # Discover stacks
    from runner import io as io_
    all_pairs = io_.find_stacks(args.noisy_dir, args.clean_dir)
    if args.stacks:
        wanted = set(args.stacks)
        pairs = [(n, np_, cp) for (n, np_, cp) in all_pairs if n in wanted]
        missing = wanted - {n for n, *_ in pairs}
        if missing:
            sys.exit(f"Stacks not found: {missing}")
    else:
        pairs = all_pairs
    if not pairs:
        sys.exit(f"No stacks under {args.noisy_dir}")
    if any(cp is None for n, np_, cp in pairs):
        sys.exit("Ablation requires clean GT for every chosen stack.")
    stack_names = [n for n, *_ in pairs]

    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    group_id = f"abl_{args.algo}_{ts}"
    results_dir = Path(args.results_dir)
    group_dir = results_dir / group_id
    group_dir.mkdir(parents=True, exist_ok=True)

    ablations_csv = group_dir / "ablations.csv"
    summary_csv = group_dir / "summary.csv"
    init_ablations_csv(ablations_csv, stack_names, args.metric)

    print("=" * 70)
    print(f"ABLATION — {args.algo}")
    print(f"  description: {desc}")
    print(f"  baseline:    {args.baseline_name}")
    print(f"  ablations:   {len(ablations)} ({list(ablations.keys())})")
    print(f"  stacks:      {stack_names}")
    print(f"  metric:      {args.metric}")
    print(f"  reduce:      {args.reduce}")
    print(f"  group:       {group_id}")
    print(f"  results dir: {results_dir}")
    total_jobs = len(ablations) * len(pairs)
    print(f"  total jobs:  {total_jobs}")
    print("=" * 70)

    if args.dry_run:
        for i, (name, overrides) in enumerate(ablations.items(), 1):
            print(f"  [{i:3d}] {name:30s}  on  {stack_names}  "
                  f"overrides={overrides}")
        return 0

    # Run
    overall_t0 = time.time()
    results = {}
    for ab_idx, (name, overrides) in enumerate(ablations.items(), 1):
        print(f"\n[{ab_idx}/{len(ablations)}] Ablation '{name}'")
        print(f"   overrides: {overrides}")
        t0 = time.time()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            per_stack = run_ablation(
                ablation_name=name, overrides=overrides,
                base_config=base_config, algo=args.algo,
                stack_pairs=pairs, results_dir=results_dir,
                figures_dir=Path(args.figures_dir), group_id=group_id,
                reduce_factor=args.reduce, metric=args.metric,
                verbose=not args.quiet,
            )
            status = "success" if any(v is not None for v in per_stack.values()) \
                else "all_failed"
        except Exception:
            traceback.print_exc()
            per_stack = {s: None for s in stack_names}
            status = "exception"
        duration = time.time() - t0
        append_ablations_row(
            ablations_csv, ablation=name, started_at=started_at,
            duration=duration, status=status, per_stack=per_stack,
            stack_names=stack_names, overrides=overrides,
        )
        results[name] = per_stack
        print(f"   per-stack: {per_stack}")
        print(f"   took {duration:.0f}s  "
              f"(elapsed {time.time()-overall_t0:.0f}s of total)")
        # Update summary incrementally so partial results are usable
        write_summary(summary_csv, results=results,
                       baseline_name=args.baseline_name, metric=args.metric,
                       stack_names=stack_names)

    # Final summary
    print("\n" + "=" * 70)
    print(f"ABLATION COMPLETE — {args.algo}")
    print(f"  per-run csv: {ablations_csv}")
    print(f"  summary csv: {summary_csv}")
    print()
    # Pretty-print the summary
    if summary_csv.exists():
        with open(summary_csv) as f:
            rows = list(csv.reader(f))
        widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
        for row in rows:
            print("  " + " | ".join(c.ljust(w) for c, w in zip(row, widths)))
    print("=" * 70)


if __name__ == "__main__":
    main()
