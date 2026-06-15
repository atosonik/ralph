"""Noise-floor calibration aggregator for the downstream-eval harness (B1).

The Cross-Scale Downstream Pareto kernel (`aggregate_pareto`) uses a
per-task noise floor `eta_task` as the minimum delta that counts as a
significant win or loss in a cell. eta_task answers: "what's the
typical spread in this task's accuracy across statistically equivalent
baseline runs?" — i.e., the magnitude of noise we should attribute to
randomness rather than to skill differences.

This module ships:

  * `aggregate_noise_floors(reports, *, margin_multiplier=2.0,
    recipe_sha="")` — pure aggregation. Takes N `DownstreamReport`
    objects from N statistically equivalent baseline runs and returns
    a `NoiseFloorTable` mapping task name → eta_task.

    eta_task is conservative across scales: for a given task, it is
    `margin_multiplier * max(stddev(accs at scale)) for scale in
    scales`. The "max across scales" choice means the per-task floor
    is at least as wide as the noisiest scale's spread — a delta
    smaller than the worst-scale noise is rejected for that task on
    every scale, not just the noisy one.

  * `compute_per_cell_stddev(reports)` — diagnostic helper that
    returns the per-`(task:scale)` stddev BEFORE the per-task
    aggregation. Useful for the calibration report ("how does noise
    vary across scales for task X?"); the Pareto kernel itself uses
    only the per-task aggregated value.

  * `_sample_stddev(xs)` — Bessel-corrected (ddof=1) sample stddev
    with two surgical short-circuits:
      - n < 2 → 0.0 (no within-sample variance to estimate)
      - all xs equal → 0.0 exactly (avoids the `sum/N != X` fp
        rounding flake that bit
        `test_multiseed_three_runs_produce_byte_identical_results`
        before; matches `validator/multiseed.py::_mean_stderr`'s
        same short-circuit).

  * `write_noise_floor_table_json(table, path)` — atomic JSON output
    via `.tmp` + rename, with parent-dir mkdir.

  * `read_noise_floor_table_json(path)` — inverse, with explicit
    schema validation on required keys.

What this module does NOT ship (separate H100 PR):

  * The driver that actually trains N baseline checkpoints, runs
    them through `run_eval_in_subprocess`, and feeds the resulting
    reports into `aggregate_noise_floors`. That driver is gated on
    (a) `load_task_examples` being implemented (B1-D1 follow-up)
    and (b) a real H100 instance for the baseline-training pass
    (~$15 wall-clock per the master plan).

  * Cross-scale eta variants (per-cell etas, per-rung-pair etas).
    The single per-task eta is the v0.10 convention; multi-scale
    variants are a B3 / aggregate-v2 concern.

Closes B1-D8 at the aggregation surface: the CODE path is wired;
the operational H100 run produces the actual `noise_floors_v1.json`
file in a follow-up.

Reference scope: docs/build_scope/02_scope_B1.md "calibration.py".
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .types import (
    HARNESS_VERSION,
    DownstreamReport,
    NoiseFloorTable,
)

# ----------------------------------------------------------------------------
# Sample stddev
# ----------------------------------------------------------------------------


def _sample_stddev(xs: list[float]) -> float:
    """Bessel-corrected (ddof=1) sample standard deviation.

    Edge cases:
      * n == 0 or n == 1 → 0.0 (no within-sample variance to estimate)
      * all values equal → 0.0 (exact short-circuit; avoids the
        `sum/N != X` floating-point flake)
      * otherwise → sqrt(sum((x - mean)² for x in xs) / (n - 1))
    """
    n = len(xs)
    if n < 2:
        return 0.0
    first = xs[0]
    if all(x == first for x in xs):
        return 0.0
    mean = sum(xs) / n
    var = sum((x - mean) * (x - mean) for x in xs) / (n - 1)
    return math.sqrt(var)


# ----------------------------------------------------------------------------
# Per-cell stddev (diagnostic)
# ----------------------------------------------------------------------------


def compute_per_cell_stddev(
    reports: list[DownstreamReport],
) -> dict[str, float]:
    """Per-`(task:scale)` sample stddev across the N reports.

    Returns dict `cell_key -> stddev`. Cells that don't appear in EVERY
    report contribute the stddev of however many they appear in (the
    aggregator's caller is responsible for ensuring report shapes line
    up; this helper is tolerant).

    Diagnostic only — the Pareto kernel reads per-TASK floors, not
    per-cell. The per-cell breakdown is for the human-readable
    calibration report ("does the task get noisier at smaller scale?").
    """
    if not reports:
        return {}
    per_cell: dict[str, list[float]] = {}
    for report in reports:
        for cell_key, cell in report.cells.items():
            per_cell.setdefault(cell_key, []).append(cell.accuracy)
    return {k: _sample_stddev(v) for k, v in per_cell.items()}


# ----------------------------------------------------------------------------
# Aggregator
# ----------------------------------------------------------------------------


def aggregate_noise_floors(
    reports: list[DownstreamReport],
    *,
    margin_multiplier: float = 2.0,
    recipe_sha: str = "",
) -> NoiseFloorTable:
    """Aggregate N baseline DownstreamReports into a NoiseFloorTable.

    Algorithm:
      1. Group accuracies by cell key (`task:scale`).
      2. For each cell key, compute the across-baseline stddev.
      3. For each TASK (cell_key.split(":", 1)[0]), take the MAX
         stddev across all of its scales.
      4. eta_task = margin_multiplier * max_stddev.

    The "max across scales" choice ensures the per-task floor is
    conservative — at least as wide as the noisiest scale's measured
    noise. A delta smaller than that floor is rejected on every scale
    for that task, not just the noisy one.

    Args:
      reports: at least one DownstreamReport. Empty → ValueError.
      margin_multiplier: scalar applied to the stddev. Default 2.0
        matches `scripts/noise_floor.py`'s "decisively beats the king"
        margin. Must be >= 0; negative → ValueError.
      recipe_sha: stamped into the returned NoiseFloorTable. The
        validator emits this on chain so a future auditor can replay
        the calibration run. Empty string is allowed (B1-D8 follow-up
        will populate it when the H100 driver lands).

    Returns:
      `NoiseFloorTable` with floors keyed by task name, plus
      `harness_version`, `recipe_sha`, `n_baselines` stamped.
    """
    if not reports:
        raise ValueError(
            "aggregate_noise_floors requires at least one report; got 0"
        )
    if margin_multiplier < 0:
        raise ValueError(
            f"margin_multiplier must be >= 0; got {margin_multiplier}"
        )

    per_cell_stddev = compute_per_cell_stddev(reports)

    per_task_max_stddev: dict[str, float] = {}
    for cell_key, stddev in per_cell_stddev.items():
        task = cell_key.split(":", 1)[0]
        if task not in per_task_max_stddev or stddev > per_task_max_stddev[task]:
            per_task_max_stddev[task] = stddev

    floors = {
        task: margin_multiplier * stddev
        for task, stddev in per_task_max_stddev.items()
    }

    return NoiseFloorTable(
        floors=floors,
        harness_version=HARNESS_VERSION,
        recipe_sha=recipe_sha,
        n_baselines=len(reports),
    )


# ----------------------------------------------------------------------------
# JSON I/O
# ----------------------------------------------------------------------------


_META_MARKER = "ralph-noise-floor-table"


def write_noise_floor_table_json(
    table: NoiseFloorTable,
    path: Path,
) -> None:
    """Write a NoiseFloorTable to disk as JSON.

    Atomic-write via `<path>.tmp` + rename so a crashed write never
    leaves a half-written file the validator consumes. Output is
    human-readable (indent=2, sort_keys=True) — the file is small
    (~one entry per task, ~26 tasks total) and the audit trail
    benefits from a clean diff.

    Schema:
      {
        "_meta": "ralph-noise-floor-table",
        "harness_version": "...",
        "recipe_sha": "...",
        "n_baselines": <int>,
        "floors": {"<task>": <eta>, ...}
      }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": _META_MARKER,
        "harness_version": table.harness_version,
        "recipe_sha": table.recipe_sha,
        "n_baselines": table.n_baselines,
        "floors": dict(table.floors),
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    tmp.replace(path)


def read_noise_floor_table_json(path: Path) -> NoiseFloorTable:
    """Inverse of `write_noise_floor_table_json`.

    Validates the `_meta` marker so a different JSON file accidentally
    handed to this reader fails clean instead of producing a
    NoiseFloorTable with arbitrary numbers. Missing optional fields
    fall back to defaults; missing required keys raise ValueError.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"noise-floor JSON at {path} is not valid JSON: {e}"
        ) from e

    meta = payload.get("_meta")
    if meta != _META_MARKER:
        raise ValueError(
            f"unexpected _meta marker {meta!r} in {path}; "
            f"expected {_META_MARKER!r}"
        )

    if "floors" not in payload:
        raise ValueError(
            f"noise-floor JSON at {path} missing required 'floors' key"
        )

    floors_raw = payload["floors"]
    if not isinstance(floors_raw, dict):
        raise ValueError(
            f"'floors' must be a dict; got {type(floors_raw).__name__}"
        )
    floors = {str(k): float(v) for k, v in floors_raw.items()}

    return NoiseFloorTable(
        floors=floors,
        harness_version=str(payload.get("harness_version", HARNESS_VERSION)),
        recipe_sha=str(payload.get("recipe_sha", "")),
        n_baselines=int(payload.get("n_baselines", 0)),
    )
