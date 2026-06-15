"""B5 H100 calibration orchestrator — produces `noise_floors_v1.json`.

Runs N=10 baseline submissions through the full v0.11-lite ladder and
aggregates the per-task across-baseline variance into a per-task
`NoiseFloorTable` that the Pareto kernel consumes at king-selection
time (per-task `eta_task = margin_multiplier * max(stddev across
scales)`).

USAGE:
    python scripts/b5_h100_calibration.py \\
        --ralph-root /path/to/ralph_root \\
        --bundle-dir eval/private/downstream_pool/bundle_v1 \\
        --bundle-sha-sha256 <sha> \\
        --baseline-checkpoint /path/to/baseline.pt \\
        --output eval/private/calibration/noise_floors_v1.json \\
        [--n-baselines 10] \\
        [--seeds 0,1,...,9] \\
        [--budget-gpu-hr-cap 50] \\
        [--task arc_easy --task piqa ...] \\
        [--per-task-cap 0]

OPERATIONAL ENVELOPE:
  * N=10 baselines × 3 rungs (S1+S2+S3) × ~5 H100-min per S3 rung ≈
    ~$92 at $2/H100-hr spot pricing (Shadeform).
  * Per-rung subprocess timeout default 600s; total wall-clock <3 days
    spot-tolerant.
  * Failure of one baseline does NOT abort the run; aggregation falls
    back to the surviving baselines (n_baselines stamped on the
    NoiseFloorTable reflects the surviving count).

The CPU-testable path of this script (`run_calibration`) accepts a
custom `command_prefix` so the synthetic subprocess entry can stand in
for the real CLI in unit tests. The CLI entry point (`main`) wires
real arguments through.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401, E402
from eval.downstream.calibration import (  # noqa: E402
    aggregate_noise_floors,
    write_noise_floor_table_json,
)
from eval.downstream.runner import RALPH_VOCAB_SIZE  # noqa: E402
from eval.downstream.runner_subprocess import EvalSubprocessError  # noqa: E402
from validator.ladder import (  # noqa: E402
    EVAL_MODE_V011,
    LadderEvalConfig,
    LadderRungSpec,
    Submission,
    run_ladder_eval,
)

DEFAULT_N_BASELINES = 10


@dataclass
class CalibrationSummary:
    """Returned by `run_calibration`. Carries the table + per-baseline
    survival stats for the CLI to log to the operator."""

    n_baselines_requested: int
    n_baselines_succeeded: int
    failures: list[dict]
    output_path: Path


def _make_baseline_submission(seed: int) -> Submission:
    """Construct a synthetic baseline `Submission` for calibration.

    Baselines are NOT real miner submissions — they're operator-driven
    runs of the canonical recipe with varying seeds. We tag them with a
    well-known `miner_hotkey` so the chain event log shows them as
    baselines, not real submissions.
    """
    return Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,  # baselines are genesis
        branch_id="main",
        bundle_hash=f"calibration_baseline_seed_{seed}",
        miner_hotkey="5F_calibration_baseline",
        vocab_size=RALPH_VOCAB_SIZE,
    )


def run_calibration(
    *,
    ralph_root: Path,
    baseline_checkpoint: Path,
    bundle_dir: Path,
    bundle_sha256: str,
    tasks: tuple[str, ...],
    output_path: Path,
    n_baselines: int = DEFAULT_N_BASELINES,
    seeds: Optional[Sequence[int]] = None,
    rungs: Optional[Sequence[LadderRungSpec]] = None,
    margin_multiplier: float = 2.0,
    timeout_s_per_rung: float = 600.0,
    command_prefix: Optional[Sequence[str]] = None,
    per_task_cap: int = 0,
) -> CalibrationSummary:
    """Drive N baseline runs and write `noise_floors_v1.json`.

    Args:
      ralph_root: passed to `run_ladder_eval` as `--ralph-root`.
      baseline_checkpoint: the canonical baseline checkpoint to evaluate.
      bundle_dir: DCLM bundle root (per-task JSONLs + private_hard/ subdir).
      bundle_sha256: pinned bundle SHA.
      tasks: tuple of task names to include in calibration.
      output_path: where to write the NoiseFloorTable JSON.
      n_baselines: number of baseline runs to attempt (default 10).
      seeds: explicit seeds (must have length n_baselines if provided).
        Default is `range(n_baselines)`.
      rungs: explicit rung list (default S1+S2+S3 standard).
      margin_multiplier: per-task floor multiplier (default 2.0).
      timeout_s_per_rung: subprocess timeout per rung.
      command_prefix: for tests — synthetic CLI command.
      per_task_cap: `n_examples_per_task` per rung (0 = all).

    Returns:
      `CalibrationSummary` with survival stats and the output path.
    """
    if seeds is None:
        seeds = tuple(range(n_baselines))
    if len(seeds) != n_baselines:
        raise ValueError(
            f"n_baselines={n_baselines} but {len(seeds)} seeds given"
        )

    if rungs is None:
        rungs_for_config = (
            LadderRungSpec(scale_label="S1", dim=256, n_layers=4,
                           n_examples_per_task=per_task_cap),
            LadderRungSpec(scale_label="S2", dim=512, n_layers=12,
                           n_examples_per_task=per_task_cap),
            LadderRungSpec(scale_label="S3", dim=768, n_layers=12,
                           n_examples_per_task=per_task_cap),
        )
    else:
        rungs_for_config = tuple(rungs)

    reports = []
    failures: list[dict] = []

    for seed in seeds:
        config = LadderEvalConfig(
            rungs=rungs_for_config,
            tasks=tasks,
            bundle_dir=bundle_dir,
            bundle_sha256=bundle_sha256,
            seed=seed,
        )
        try:
            result = run_ladder_eval(
                _make_baseline_submission(seed),
                config,
                checkpoint_path=baseline_checkpoint,
                ralph_root=ralph_root,
                mode=EVAL_MODE_V011,
                command_prefix=command_prefix,
                timeout_s_per_rung=timeout_s_per_rung,
            )
            reports.append(result.combined_report)
        except EvalSubprocessError as e:
            failures.append({
                "seed": seed,
                "reason": e.reason,
                "exit_code": e.exit_code,
                "stderr_tail": e.stderr_tail[:512],
            })
        except Exception as e:
            failures.append({
                "seed": seed,
                "reason": "exception",
                "exception_type": type(e).__name__,
                "message": str(e)[:512],
            })

    if not reports:
        raise RuntimeError(
            "calibration failed: 0 of "
            f"{n_baselines} baselines succeeded; "
            f"failures: {failures}"
        )

    table = aggregate_noise_floors(
        reports,
        margin_multiplier=margin_multiplier,
        recipe_sha=str(bundle_sha256),
    )
    write_noise_floor_table_json(table, output_path)

    return CalibrationSummary(
        n_baselines_requested=n_baselines,
        n_baselines_succeeded=len(reports),
        failures=failures,
        output_path=output_path,
    )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts.b5_h100_calibration")
    p.add_argument("--ralph-root", required=True, type=Path)
    p.add_argument("--baseline-checkpoint", required=True, type=Path)
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--bundle-sha-sha256", required=True, dest="bundle_sha256")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--task", required=True, action="append", dest="tasks",
                   help="repeat for each task (e.g. --task arc_easy --task piqa)")
    p.add_argument("--n-baselines", type=int, default=DEFAULT_N_BASELINES)
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds (overrides n_baselines if given)")
    p.add_argument("--margin-multiplier", type=float, default=2.0)
    p.add_argument("--timeout-s-per-rung", type=float, default=600.0)
    p.add_argument("--per-task-cap", type=int, default=0,
                   help="n_examples_per_task per rung; 0 = use all")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    seeds = None
    if args.seeds is not None:
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        if not seeds:
            print("ERROR: --seeds must list at least one seed", file=sys.stderr)
            return 2

    started = time.time()
    try:
        summary = run_calibration(
            ralph_root=args.ralph_root,
            baseline_checkpoint=args.baseline_checkpoint,
            bundle_dir=args.bundle_dir,
            bundle_sha256=args.bundle_sha256,
            tasks=tuple(args.tasks),
            output_path=args.output,
            n_baselines=args.n_baselines,
            seeds=seeds,
            margin_multiplier=args.margin_multiplier,
            timeout_s_per_rung=args.timeout_s_per_rung,
            per_task_cap=args.per_task_cap,
        )
    except Exception as e:
        print(f"B5 calibration FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    elapsed = time.time() - started

    report = {
        "n_baselines_requested": summary.n_baselines_requested,
        "n_baselines_succeeded": summary.n_baselines_succeeded,
        "n_failures": len(summary.failures),
        "wall_clock_s": elapsed,
        "output_path": str(summary.output_path),
        "failures": summary.failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if summary.n_baselines_succeeded == summary.n_baselines_requested else 0
    # Return 0 even on partial survival — the operator inspects the JSON
    # report. Hard failure (zero survivors) raises and returns 1 above.


if __name__ == "__main__":
    sys.exit(main())
