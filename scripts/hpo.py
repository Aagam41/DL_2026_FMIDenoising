#!/usr/bin/env python3
"""
Hyperparameter optimization for one algorithm.

Picks the hyperparameter configuration that maximizes stSNR (or any
listed metric) on the supplied noisy/clean stacks.

USAGE
─────
Quick start (DVT, 20 trials, on F1 only, with reduced iter budget):

    python scripts/hpo.py \\
        --algo dvt_unet3d \\
        --noisy-dir /path/to/noisy --clean-dir /path/to/clean \\
        --stacks F1 \\
        --trials 20 \\
        --reduce 0.25

Search across multiple stacks (mean stSNR — robust against overfit
to a single stack):

    python scripts/hpo.py \\
        --algo dvt_unet3d \\
        --noisy-dir … --clean-dir … \\
        --stacks F0 F1 F2 \\
        --aggregate mean \\
        --trials 30

Resume an interrupted study (Optuna only):

    python scripts/hpo.py \\
        --algo dvt_unet3d \\
        --noisy-dir … --clean-dir … \\
        --resume hpo_results/dvt_unet3d_20260520-120000/study.db \\
        --trials 50

HOW IT WORKS
────────────
For each trial:
    1. Sample a config from hpo_spaces/<algo>.py:suggest(trial)
    2. Merge over BASE_CONFIG from the same file
    3. Scale `warmup_iters` and `n2v_iters` by `--reduce` to keep
       each trial cheap (default 0.25 = 4x faster than production)
    4. Run train+infer on each stack in --stacks
    5. Compute stSNR (or chosen metric) per stack
    6. Aggregate across stacks (mean or min) — this is the OBJECTIVE
    7. Return objective to Optuna

After all trials, the best config (in the OBJECTIVE sense) is written
to <results_dir>/<group>/best_config.py — copy that to configs/ and
run a single, full-budget, non-reduced training for the actual
submission output.

CAVEATS
───────
1. EXPENSIVE EVALS: Even reduced-budget trials take real time
   (~5 min/stack for DVT in fp32). Plan for 5-50 trials, not 500.

2. OVERFIT RISK: If you optimize on all 7 stacks, you may overfit
   to that specific set. For competition use, hold out at least one
   stack as a check (run HPO on a subset, then verify the best
   config on the held-out stack before submitting).

3. THE REDUCED BUDGET MIGHT FAVOR THE WRONG CONFIGS: A config that
   converges fast at 1000 iters may not be the same one that's best
   at 4000 iters. Heavier models in particular are penalized at
   short budgets. After HPO finishes, you can use --topk N to
   re-train the top N configs at full budget and pick the best of
   those — that's the safe pattern.
"""
import argparse
import json
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ── Optional dependency: Optuna ─────────────────────────────────
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    optuna = None
    OPTUNA_AVAILABLE = False


# ── Random-trial shim (used when Optuna isn't installed) ────────

class _RandomTrial:
    """Minimal Optuna-trial-API shim for random search.

    Implements suggest_int, suggest_float, suggest_categorical with
    the same signatures as Optuna's Trial. Stores chosen values in
    `params` for logging.
    """

    def __init__(self, rng, number=0):
        self._rng = rng
        self.params = {}
        self.number = number

    def suggest_int(self, name, low, high, step=1, log=False):
        if log:
            import math
            lo = math.log(low); hi = math.log(high)
            v = int(round(math.exp(self._rng.uniform(lo, hi))))
            v = max(low, min(high, v))
        else:
            choices = list(range(low, high + 1, step))
            v = int(self._rng.choice(choices))
        self.params[name] = v
        return v

    def suggest_float(self, name, low, high, step=None, log=False):
        if log:
            import math
            v = math.exp(self._rng.uniform(math.log(low), math.log(high)))
        elif step is not None:
            n = int(round((high - low) / step)) + 1
            v = float(low + step * self._rng.integers(0, n))
        else:
            v = float(self._rng.uniform(low, high))
        self.params[name] = v
        return v

    def suggest_categorical(self, name, choices):
        v = choices[int(self._rng.integers(0, len(choices)))]
        self.params[name] = v
        return v


# ── HPO space loader ────────────────────────────────────────────

def load_space(algo: str):
    """Load hpo_spaces/<algo>.py and return (BASE_CONFIG, suggest_fn, DESCRIPTION)."""
    import importlib.util
    spec_path = ROOT / "hpo_spaces" / f"{algo}.py"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"No HPO space spec for '{algo}'. Create "
            f"hpo_spaces/{algo}.py with BASE_CONFIG and suggest(trial)."
        )
    spec = importlib.util.spec_from_file_location(f"hpo_spaces.{algo}",
                                                    str(spec_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "BASE_CONFIG") or not hasattr(mod, "suggest"):
        raise AttributeError(
            f"hpo_spaces/{algo}.py must define BASE_CONFIG dict and "
            f"suggest(trial) function."
        )
    desc = getattr(mod, "DESCRIPTION", "")
    return mod.BASE_CONFIG, mod.suggest, desc


# ── Trial runner ────────────────────────────────────────────────

def run_trial(
    *, trial_number, params, base_config, algo,
    stack_pairs, results_dir, figures_dir, group_id, config_name,
    reduce_factor, metric, aggregate, verbose,
):
    """
    Run one HPO trial across the chosen stacks.

    Returns (objective_value, per_stack_metrics_dict).

    `params` is the dict returned by space.suggest(trial). It's merged
    on top of BASE_CONFIG.
    """
    from runner.core import run_one

    # Build the full config for this trial
    cfg = dict(base_config)
    cfg.update(params)
    cfg.pop("algo", None)
    cfg.pop("paper_frame", None)  # framework-level
    cfg.pop("name", None)
    cfg.pop("description", None)

    # Apply reduce_factor to schedule
    if "warmup_iters" in cfg:
        cfg["warmup_iters"] = max(
            int(round(cfg["warmup_iters"] * reduce_factor)), 10,
        )
    if "n2v_iters" in cfg:
        cfg["n2v_iters"] = max(
            int(round(cfg["n2v_iters"] * reduce_factor)), 50,
        )
    if "fm2s_iters" in cfg:
        cfg["fm2s_iters"] = max(
            int(round(cfg["fm2s_iters"] * reduce_factor)), 50,
        )
    if "vit_iters" in cfg:
        cfg["vit_iters"] = max(
            int(round(cfg["vit_iters"] * reduce_factor)), 50,
        )

    per_stack = {}
    for stack_name, noisy_path, clean_path in stack_pairs:
        # Each trial-stack run gets its own config_name so the CSV row
        # is distinguishable.
        trial_cfg_name = f"{config_name}__trial{trial_number:04d}"
        try:
            summary = run_one(
                algo=algo, config=dict(cfg),
                noisy_path=noisy_path, clean_path=clean_path,
                results_dir=results_dir, figures_dir=figures_dir,
                group_id=group_id, config_name=trial_cfg_name,
                paper_frame=base_config.get("paper_frame", 750),
                save_checkpoint=False, save_figures=False,
                verbose=verbose,
            )
        except Exception as e:
            traceback.print_exc()
            return None, {sn: None for sn, *_ in stack_pairs}

        if summary["status"] != "success":
            per_stack[stack_name] = None
            continue
        m = summary.get("metrics") or {}
        v = m.get(metric)
        if v is None:
            per_stack[stack_name] = None
            continue
        try:
            v = float(v)
            if v != v:  # NaN
                per_stack[stack_name] = None
                continue
        except (TypeError, ValueError):
            per_stack[stack_name] = None
            continue
        per_stack[stack_name] = v

    # Aggregate across stacks
    vals = [v for v in per_stack.values() if v is not None]
    if not vals:
        return None, per_stack
    if aggregate == "mean":
        objective = sum(vals) / len(vals)
    elif aggregate == "min":
        objective = min(vals)
    elif aggregate == "max":
        objective = max(vals)
    else:
        raise ValueError(f"Unknown aggregate: {aggregate}")
    return objective, per_stack


# ── Trial logging ───────────────────────────────────────────────

def init_trials_csv(path: Path, metric: str):
    if path.exists():
        return  # append-only
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(
            "trial,started_at,duration_sec,objective,status,"
            f"{metric}_per_stack,params\n"
        )


def append_trial_row(
    path: Path, *, trial_number, started_at, duration_sec,
    objective, status, per_stack_metrics, params,
):
    import csv
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            trial_number, started_at, f"{duration_sec:.1f}",
            "" if objective is None else f"{objective:.6f}",
            status,
            json.dumps({k: (None if v is None else round(v, 4))
                       for k, v in per_stack_metrics.items()}),
            json.dumps(params, default=str),
        ])


# ── Best-config writer ──────────────────────────────────────────

def write_best_config(out_path: Path, algo: str, base_config: dict,
                       best_params: dict, best_obj: float, metric: str,
                       reduce_factor: float):
    """Emit a configs/<algo>_hpo_best.py-style file with the merged config.

    The schedule is restored to the FULL budget (un-reduced) so this
    file is ready to use as a production config.
    """
    cfg = dict(base_config)
    cfg.update(best_params)
    # Restore production iter counts (we trained at reduced budget).
    # The values in best_params are the SUGGESTED counts (pre-reduce),
    # so we just keep them as-is.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f'"""\n')
        f.write(f"Best config found by HPO for {algo}.\n\n")
        f.write(f"Best {metric}: {best_obj:.4f}\n")
        f.write(f"HPO was run at reduced budget "
                f"(--reduce {reduce_factor:g}); the iter counts\n")
        f.write(f"in this config are the FULL budget — use as a "
                f"production config.\n")
        f.write(f'"""\n\n')
        f.write("CONFIG = {\n")
        for k, v in cfg.items():
            if isinstance(v, str):
                f.write(f"    {k!r:20s}: {v!r},\n")
            else:
                f.write(f"    {k!r:20s}: {v!r},\n")
        f.write("}\n")


# ── Main ────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Hyperparameter optimization for one algorithm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--algo", required=True,
                    help="Algorithm name (must have a spec in hpo_spaces/).")
    p.add_argument("--noisy-dir", required=True,
                    help="Directory with noisy .tif stacks.")
    p.add_argument("--clean-dir", required=True,
                    help="Directory with clean GT .tif stacks.")
    p.add_argument("--stacks", nargs="+", default=None,
                    help="Stack names to use (e.g. F0 F1). Default: all "
                         "stacks found in --noisy-dir.")
    p.add_argument("--trials", type=int, default=20,
                    help="Number of HPO trials. Default: 20.")
    p.add_argument("--metric", default="stSNR",
                    help="Metric to optimize. Default: stSNR.")
    p.add_argument("--aggregate", choices=("mean", "min", "max"),
                    default="mean",
                    help="How to aggregate across stacks when more than "
                         "one is given. Default: mean.")
    p.add_argument("--reduce", type=float, default=0.25,
                    help="Reduce iters by this factor for each trial. "
                         "Default: 0.25 (4x faster than production). "
                         "Set to 1.0 to disable reduction.")
    p.add_argument("--method", choices=("optuna", "random"), default="auto",
                    help="Search method. Default: optuna if installed, "
                         "else random. Force with optuna|random.")
    p.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the search.")
    p.add_argument("--timeout", type=float, default=None,
                    help="Wall-clock limit in seconds. HPO stops "
                         "cleanly when reached (Optuna only).")
    p.add_argument("--results-dir", default=str(ROOT / "hpo_results"))
    p.add_argument("--figures-dir", default=str(ROOT / "paper_figures"))
    p.add_argument("--resume", default=None,
                    help="Path to an existing Optuna study .db to resume.")
    p.add_argument("--quiet", action="store_true",
                    help="Silence per-trial run_one logs.")
    args = p.parse_args()

    # Resolve search method
    if args.method == "auto":
        method = "optuna" if OPTUNA_AVAILABLE else "random"
    else:
        method = args.method
    if method == "optuna" and not OPTUNA_AVAILABLE:
        sys.exit("Optuna not installed. Run `pip install optuna` or use "
                 "--method random.")

    # Load space + algo registry check
    from runner import io as io_
    base_config, suggest_fn, desc = load_space(args.algo)

    # Discover stacks
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
        sys.exit("HPO requires clean GT for every chosen stack "
                 "(it's the only way to compute the metric).")

    # Build group id & output dirs
    if args.resume:
        # Resuming an existing study — recover its group_id from the db path
        group_id = Path(args.resume).parent.name
        results_dir = Path(args.resume).parent.parent
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        group_id = f"hpo_{args.algo}_{ts}"
        results_dir = Path(args.results_dir)

    group_dir = results_dir / group_id
    group_dir.mkdir(parents=True, exist_ok=True)
    trials_csv = group_dir / "trials.csv"
    best_cfg_path = group_dir / "best_config.py"
    init_trials_csv(trials_csv, args.metric)

    # Header
    print("=" * 70)
    print(f"HPO — {args.algo}")
    print(f"  description: {desc}")
    print(f"  search:      {method}")
    print(f"  trials:      {args.trials}")
    print(f"  metric:      {args.metric} (maximize)")
    print(f"  stacks:      {[n for n, *_ in pairs]}")
    print(f"  aggregate:   {args.aggregate}")
    print(f"  reduce:      {args.reduce} (iters × {args.reduce} per trial)")
    print(f"  group:       {group_id}")
    print(f"  results dir: {results_dir}")
    if args.timeout:
        print(f"  timeout:     {args.timeout:.0f}s")
    print("=" * 70)

    # ── Trial dispatcher (shared between optuna and random) ────────

    best_so_far = {"obj": float("-inf"), "params": None, "trial": -1}

    def _run_one_trial(trial, trial_number):
        nonlocal best_so_far
        t0 = time.time()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            params = suggest_fn(trial)
        except Exception as e:
            print(f"   suggest() failed: {e}")
            append_trial_row(
                trials_csv, trial_number=trial_number,
                started_at=started_at, duration_sec=time.time() - t0,
                objective=None, status="suggest_error",
                per_stack_metrics={}, params={},
            )
            return None

        print(f"\nTrial {trial_number}: params = {params}")
        obj, per_stack = run_trial(
            trial_number=trial_number, params=params,
            base_config=base_config, algo=args.algo,
            stack_pairs=pairs, results_dir=results_dir,
            figures_dir=Path(args.figures_dir),
            group_id=group_id,
            config_name=f"{args.algo}_hpo",
            reduce_factor=args.reduce, metric=args.metric,
            aggregate=args.aggregate, verbose=not args.quiet,
        )
        duration = time.time() - t0
        status = "success" if obj is not None else "no_metric"
        append_trial_row(
            trials_csv, trial_number=trial_number,
            started_at=started_at, duration_sec=duration,
            objective=obj, status=status,
            per_stack_metrics=per_stack, params=params,
        )
        if obj is not None and obj > best_so_far["obj"]:
            best_so_far.update(obj=obj, params=params, trial=trial_number)
            # Write best_config.py incrementally
            write_best_config(
                best_cfg_path, args.algo, base_config, params,
                obj, args.metric, args.reduce,
            )
            print(f"   NEW BEST: {args.metric}={obj:.4f}  "
                  f"(trial {trial_number})")
        print(f"   per-stack: {per_stack}")
        print(f"   trial took {duration:.0f}s, "
              f"best so far = {best_so_far['obj']:.4f} "
              f"(trial {best_so_far['trial']})")
        return obj

    # ── Run Optuna or random ───────────────────────────────────────

    overall_t0 = time.time()
    if method == "optuna":
        db_path = group_dir / "study.db"
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            study_name=group_id,
            storage=f"sqlite:///{db_path}",
            sampler=sampler,
            direction="maximize",
            load_if_exists=True,
        )

        def _objective(trial):
            obj = _run_one_trial(trial, trial.number)
            if obj is None:
                # Optuna requires a number; use a very-negative sentinel
                raise optuna.TrialPruned()
            return obj

        try:
            study.optimize(
                _objective, n_trials=args.trials, timeout=args.timeout,
            )
        except KeyboardInterrupt:
            print("\nInterrupted by user — partial results saved.")
    else:
        import numpy as np
        rng = np.random.default_rng(args.seed)
        try:
            for i in range(args.trials):
                if args.timeout and (time.time() - overall_t0) > args.timeout:
                    print(f"Wall-clock timeout {args.timeout:.0f}s reached.")
                    break
                trial = _RandomTrial(rng, number=i)
                _run_one_trial(trial, i)
        except KeyboardInterrupt:
            print("\nInterrupted by user — partial results saved.")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"HPO complete: {args.trials} trial(s) requested.")
    print(f"Best {args.metric}: {best_so_far['obj']:.4f} "
          f"(trial {best_so_far['trial']})")
    print(f"Best params:")
    for k, v in (best_so_far["params"] or {}).items():
        print(f"  {k} = {v}")
    print(f"\nResults:    {trials_csv}")
    print(f"Best config (FULL budget): {best_cfg_path}")
    print(f"\nTo train at full budget with the best config:")
    print(f"  cp {best_cfg_path} configs/")
    print(f"  python scripts/run_one.py \\")
    print(f"    --config configs/{best_cfg_path.name} \\")
    print(f"    --noisy <path> --clean <path>")
    print("=" * 70)


if __name__ == "__main__":
    main()
