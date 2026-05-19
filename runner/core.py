"""
runner.core — the top-level `run_one(...)` function.

Given an algo name, a config dict, a noisy stack and an optional clean
stack, it:
    1. Generates a unique run_id.
    2. Starts a GPU sampler thread.
    3. Times stages (training, inference, eval) separately.
    4. Calls the algo's train_self_supervised + denoise_stack.
    5. Saves the denoised TIFF, the checkpoint, and a paper figure.
    6. Computes metrics if a clean stack is given.
    7. Writes one row each to runs.csv / config.csv / timing.csv /
       metrics.csv / stacks.csv (and continuous samples to gpu_log.csv).
    8. Returns a summary dict so the caller can print or chain.

Failures are caught and logged with status="error"; a stack trace is
saved to <results_dir>/errors/<run_id>.log.
"""

import os
import sys
import time
import uuid
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np

import algos
from . import io as io_, csv_db, runtime_log, eval_runner, plots


def _make_run_id(algo: str, stack_name: str) -> str:
    """Stable but unique id: <algo>__<stack>__<utc-yyyymmdd-hhmmss>__<sha8>."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    sha = uuid.uuid4().hex[:8]
    return f"{algo}__{stack_name}__{ts}__{sha}"


def run_one(
    algo: str,
    config: Dict[str, Any],
    noisy_path,
    clean_path: Optional[Path] = None,
    *,
    results_dir,
    figures_dir,
    paper_frame: int = 750,
    save_checkpoint: bool = True,
    save_figures: bool = True,
    gpu_sample_interval: float = 2.0,
    verbose: bool = True,
):
    """
    Run a single (algo, noisy_path) job. Returns a summary dict.

    Side effects:
        - One row added to runs.csv
        - Rows added to config.csv, timing.csv, metrics.csv (if clean),
          stacks.csv, gpu_log.csv
        - Denoised TIFF saved under outputs/<run_id>/
        - Checkpoint saved under checkpoints/<run_id>/ (if save_checkpoint)
        - Paper figure under paper_figures/<run_id>/ (if save_figures)
    """
    import torch

    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    noisy_path = Path(noisy_path)
    stack_name = noisy_path.stem
    run_id = _make_run_id(algo, stack_name)

    if verbose:
        print(f"\n{'='*70}\n RUN  algo={algo}  stack={stack_name}\n"
              f"      run_id={run_id}\n{'='*70}")

    # ── Static env info written eagerly so we can debug if something
    # blows up below. ──────────────────────────────────────────
    env = runtime_log.env_info()

    # GPU sampler in the background
    sampler = runtime_log.GPUSampler(
        results_dir=results_dir, run_id=run_id,
        interval_sec=gpu_sample_interval,
    )
    sampler.start()

    timer = runtime_log.StageTimer()
    status = "success"
    error_msg = ""
    summary: Dict[str, Any] = {"run_id": run_id, "algo": algo,
                                "stack_name": stack_name}

    denoised = None
    metrics: Dict[str, float] = {}

    try:
        # ── Load noisy ────────────────────────────────────────
        with timer.stage("load_noisy"):
            noisy = io_.load_stack(noisy_path)
        info_n = io_.stack_info(stack_name, noisy, noisy_path)
        info_n["role"] = "noisy"
        info_n["run_id"] = run_id
        csv_db.write_stack_info(results_dir, info_n)
        if verbose:
            print(f"   Noisy: shape={noisy.shape} dtype={noisy.dtype} "
                  f"range=[{noisy.min()}, {noisy.max()}]")

        # ── Load clean if provided ────────────────────────────
        clean = None
        if clean_path is not None and Path(clean_path).exists():
            with timer.stage("load_clean"):
                clean = io_.load_stack(clean_path)
            info_c = io_.stack_info(f"{stack_name}_clean", clean, clean_path)
            info_c["role"] = "clean"
            info_c["run_id"] = run_id
            csv_db.write_stack_info(results_dir, info_c)
            if verbose:
                print(f"   Clean: shape={clean.shape} dtype={clean.dtype} "
                      f"range=[{clean.min()}, {clean.max()}]")

        # ── Resolve algo module ───────────────────────────────
        mod = algos.get_algo(algo)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ── Training ──────────────────────────────────────────
        if verbose:
            print(f"   Training on {device}…")
        with timer.stage("train"):
            model, returned_cfg = mod.train_self_supervised(
                stack=noisy, device=device, config=dict(config),
                verbose=verbose,
            )

        # ── Inference ─────────────────────────────────────────
        if verbose:
            print(f"   Inference…")
        with timer.stage("inference"):
            denoised = mod.denoise_stack(
                model=model, stack=noisy.astype(np.float32),
                config=returned_cfg, device=device, verbose=verbose,
            )

        # Convert to a sensible dtype for output (match input)
        if np.issubdtype(noisy.dtype, np.integer):
            info_ = np.iinfo(noisy.dtype)
            denoised_save = np.clip(denoised, info_.min, info_.max)
            denoised_save = np.round(denoised_save).astype(noisy.dtype)
        else:
            denoised_save = denoised.astype(np.float32)

        # ── Save outputs ──────────────────────────────────────
        out_dir = io_.run_output_dir(results_dir, run_id)
        out_path = out_dir / f"{stack_name}.tif"
        with timer.stage("save_output"):
            io_.save_stack(denoised_save, out_path)
        summary["output_path"] = str(out_path)

        if save_checkpoint:
            ckpt_dir = io_.run_checkpoint_dir(results_dir, run_id)
            ckpt_path = ckpt_dir / f"{stack_name}.pth"
            try:
                mod.save_checkpoint(model, returned_cfg, str(ckpt_path))
                summary["checkpoint_path"] = str(ckpt_path)
            except Exception as e:
                if verbose:
                    print(f"   (checkpoint save failed: {e})")

        # ── Evaluate if clean given ───────────────────────────
        if clean is not None:
            if verbose:
                print(f"   Evaluating…")
            with timer.stage("evaluate"):
                metrics = eval_runner.evaluate_pair(denoised, clean)
            csv_db.write_metrics(results_dir, run_id, metrics)
            if verbose:
                key_metrics = ("stSNR", "stPSNR", "stSI_PSNR",
                               "sSNR", "tSNR")
                print("   Metrics:")
                for k in key_metrics:
                    if k in metrics:
                        print(f"     {k:12s} = {metrics[k]:.4f}")

        # ── Paper figure ──────────────────────────────────────
        if save_figures:
            fig_dir = io_.run_figure_dir(figures_dir, run_id)
            fig_path = fig_dir / f"{stack_name}_frame{paper_frame:04d}.png"
            metric_str = ""
            if metrics:
                metric_str = (
                    f"stSNR={metrics.get('stSNR', float('nan')):.2f}  "
                    f"stPSNR={metrics.get('stPSNR', float('nan')):.2f}  "
                    f"stSI_PSNR={metrics.get('stSI_PSNR', float('nan')):.2f}"
                )
            try:
                with timer.stage("paper_figure"):
                    plots.comparison_grid(
                        noisy_stack=noisy,
                        clean_stack=clean,
                        denoised_stack=denoised,
                        frame=paper_frame,
                        save_path=fig_path,
                        title=f"{algo}  /  {stack_name}",
                        metric_str=metric_str,
                    )
                summary["figure_path"] = str(fig_path)
            except Exception as e:
                if verbose:
                    print(f"   (paper figure failed: {e})")

    except Exception as e:
        status = "error"
        error_msg = f"{type(e).__name__}: {e}"
        err_dir = results_dir / "errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        with open(err_dir / f"{run_id}.log", "w") as f:
            f.write(traceback.format_exc())
        if verbose:
            print(f"   ERROR: {error_msg}")
            traceback.print_exc()
    finally:
        sampler.stop()

    # ── Per-stage timings ─────────────────────────────────────
    csv_db.write_timing(results_dir, run_id, timer.timings)

    # ── GPU/CPU summary ───────────────────────────────────────
    gpu_summary = runtime_log.summarize_gpu_log(results_dir, run_id)

    # ── Config (flattened) ────────────────────────────────────
    cfg_for_log = {k: v for k, v in config.items()
                    if not k.startswith("_")}
    cfg_for_log["__resolved_device"] = str(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    csv_db.write_config(results_dir, run_id, cfg_for_log)

    # ── Master row in runs.csv ────────────────────────────────
    run_row = {
        "run_id":     run_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "algo":       algo,
        "stack_name": stack_name,
        "status":     status,
        "error":      error_msg,
        "noisy_path": str(noisy_path),
        "clean_path": str(clean_path) if clean_path else "",
        "output_path": summary.get("output_path", ""),
        "checkpoint_path": summary.get("checkpoint_path", ""),
        "figure_path": summary.get("figure_path", ""),
        # primary metrics in the master row for convenience
        "stSNR":      metrics.get("stSNR", ""),
        "stPSNR":     metrics.get("stPSNR", ""),
        "stSI_PSNR":  metrics.get("stSI_PSNR", ""),
        "sSNR":       metrics.get("sSNR", ""),
        "tSNR":       metrics.get("tSNR", ""),
        # stage timings inline
        "train_sec":      timer.timings.get("train", ""),
        "inference_sec":  timer.timings.get("inference", ""),
        "evaluate_sec":   timer.timings.get("evaluate", ""),
        "total_sec":      sum(timer.timings.values()),
        # gpu summary
        **gpu_summary,
        # static env
        "host":           env.get("host", ""),
        "python":         env.get("python", ""),
        "torch":          env.get("torch", ""),
        "gpu_name":       env.get("gpu_name", ""),
        "gpu_total_mib":  env.get("gpu_total_mem_mib", ""),
    }
    csv_db.write_run(results_dir, run_row)

    summary.update(run_row)
    summary["metrics"] = metrics
    summary["timings"] = timer.timings
    summary["status"] = status
    return summary
