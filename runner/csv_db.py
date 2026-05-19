"""
CSV "database" for benchmark results.

We use a set of CSV files instead of SQLite for simplicity (no extra deps,
no schema migrations, trivially viewable in Excel / pandas). Each table is
a separate CSV. All tables share `run_id` as the primary key so they can
be joined.

Tables:
    runs.csv         — one row per (algo, stack) run. Master table.
    metrics.csv      — one row per (run_id, metric_name).
    config.csv       — one row per (run_id, config_key). Flattened config.
    timing.csv       — one row per (run_id, stage). Per-stage durations.
    gpu_log.csv      — periodic GPU samples taken during the run.
    algos.csv        — one row per algorithm (the registry).
    stacks.csv       — one row per input stack (shape, dtype, range, etc.)

All writes are APPEND-ONLY: if a run is re-executed, a new run_id is
created. The benchmark runner uses runs.csv to detect already-completed
(algo, stack) pairs and skip them.
"""

import csv
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


# Thread lock so background GPU logger + main thread don't interleave writes
_locks: Dict[str, threading.Lock] = {}


def _get_lock(path: Path) -> threading.Lock:
    key = str(path)
    if key not in _locks:
        _locks[key] = threading.Lock()
    return _locks[key]


def append_row(csv_path: Path, row: Dict[str, Any]):
    """Append a single row to `csv_path`. Creates the file with a header
    on first call.

    Safe under threading via per-file locks.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _get_lock(csv_path)
    with lock:
        is_new = not csv_path.exists()
        # When the header was already written, prefer that header. We
        # rebuild the row dict in that column order, filling unknown keys
        # as empty strings — keeps the schema stable across calls.
        if not is_new:
            with open(csv_path, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
            if header is None:
                is_new = True
        if is_new:
            header = list(row.keys())
        # Align row to header
        out = {k: row.get(k, "") for k in header}
        # If the row has new keys not in header, add them (rewrite header
        # only on first write). For simplicity we drop any extra keys —
        # the schema is fixed at first write per table. To extend a
        # schema, delete the CSV and rerun.
        extra = [k for k in row.keys() if k not in header]
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if is_new:
                writer.writeheader()
            writer.writerow(out)
        return extra  # caller can choose to warn on dropped keys


def append_rows(csv_path: Path, rows: Iterable[Dict[str, Any]]):
    for r in rows:
        append_row(csv_path, r)


# ── High-level helpers for each table ──────────────────────────

def write_run(results_dir: Path, run: Dict[str, Any]):
    """One row in runs.csv. Required keys: run_id, algo, stack, status."""
    append_row(Path(results_dir) / "runs.csv", run)


def write_metrics(results_dir: Path, run_id: str, metrics: Dict[str, float]):
    """One row per metric for this run."""
    rows = []
    for name, value in metrics.items():
        rows.append({
            "run_id": run_id,
            "metric": name,
            "value": value,
        })
    append_rows(Path(results_dir) / "metrics.csv", rows)


def write_config(results_dir: Path, run_id: str, config: Dict[str, Any]):
    """Flatten a config dict to one row per key."""
    rows = []
    for k, v in config.items():
        # Skip non-scalar entries that aren't useful in a flat CSV
        if isinstance(v, (dict, list, tuple)):
            v = str(v)
        rows.append({
            "run_id": run_id,
            "key": k,
            "value": v,
        })
    append_rows(Path(results_dir) / "config.csv", rows)


def write_timing(results_dir: Path, run_id: str,
                  timings: Dict[str, float]):
    """One row per timing stage."""
    rows = []
    for stage, seconds in timings.items():
        rows.append({
            "run_id": run_id,
            "stage": stage,
            "seconds": seconds,
        })
    append_rows(Path(results_dir) / "timing.csv", rows)


def write_stack_info(results_dir: Path, info: Dict[str, Any]):
    """One row per input stack (name, shape, dtype, range, etc)."""
    append_row(Path(results_dir) / "stacks.csv", info)


def write_algo_registry(results_dir: Path, rows: List[Dict[str, Any]]):
    """One row per algorithm. Called once at benchmark start."""
    path = Path(results_dir) / "algos.csv"
    # Overwrite if it exists — algos registry is canonical, not append.
    if path.exists():
        path.unlink()
    append_rows(path, rows)


# ── Read helpers ───────────────────────────────────────────────

def read_runs(results_dir: Path) -> List[Dict[str, Any]]:
    path = Path(results_dir) / "runs.csv"
    if not path.exists():
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def completed_pairs(results_dir: Path):
    """Return set of (algo, stack_name) pairs that already succeeded."""
    done = set()
    for row in read_runs(results_dir):
        if row.get("status", "").lower() == "success":
            done.add((row["algo"], row["stack_name"]))
    return done
