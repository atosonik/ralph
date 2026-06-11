"""Karpa downstream-eval harness (B1).

RESTRICTED: this package is part of the validator-controlled eval surface.
Miner-submitted patches must not modify any file under eval/downstream/ — the
restricted-file scanner (proof/runner.py::scan_diff_for_restricted) enforces
this at op1.

What B1 ships (this commit):
  * types.py: the DownstreamReport / NoiseFloorTable / ParetoVerdict
    dataclasses + the cell-key conventions + HARNESS_VERSION pin
  * aggregate.py: the Cross-Scale Downstream Pareto kernel — the core
    algorithm of the new king-selection gate

What B1 will ship (separate commits within the phase):
  * scorer.py: score_mc / score_schema / score_lm kernels
  * core22.py: the 22-task DCLM CORE eval bundle adapter
  * private_hard.py: the 4-task private hardness subset adapter
  * grader.py: one-shot offline grader → eval/private/hardness/index.parquet
  * calibration.py: N=10 baseline runs → noise_floors_v1.json
  * runner.py: subprocess-isolated entrypoint

Reference scope: docs/build_scope/02_scope_B1.md.
"""
from __future__ import annotations

from .aggregate import aggregate_pareto
from .types import (
    BPB_SUFFIX,
    HARNESS_VERSION,
    POOL_CORE22,
    POOL_PRIVATE_HARD,
    POOL_S2_VAL_BPB,
    CellResult,
    DownstreamReport,
    NoiseFloorTable,
    ParetoOutcome,
    ParetoVerdict,
    TaskSpec,
)

__all__ = [
    "BPB_SUFFIX",
    "CellResult",
    "DownstreamReport",
    "HARNESS_VERSION",
    "NoiseFloorTable",
    "POOL_CORE22",
    "POOL_PRIVATE_HARD",
    "POOL_S2_VAL_BPB",
    "ParetoOutcome",
    "ParetoVerdict",
    "TaskSpec",
    "aggregate_pareto",
]
