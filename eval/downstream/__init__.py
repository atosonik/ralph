"""Karpa downstream-eval harness (B1).

RESTRICTED: this package is part of the validator-controlled eval surface.
Miner-submitted patches must not modify any file under eval/downstream/ — the
restricted-file scanner (proof/runner.py::scan_diff_for_restricted) enforces
this at op1.

What B1 has shipped so far:
  * types.py: the DownstreamReport / NoiseFloorTable / ParetoVerdict
    dataclasses + the cell-key conventions + HARNESS_VERSION pin
  * aggregate.py: the Cross-Scale Downstream Pareto kernel — the core
    algorithm of the new king-selection gate
  * scorer.py: score_mc / score_schema / score_lm kernels (pure functions
    over pre-tokenized examples; model-agnostic via a forward_logits
    callable)
  * core22.py: the 22-task registry + per-task evaluators that wire scorer
    kernels to CellResult outputs. Bundle URL pinned (B1-D2); bundle
    download itself stubbed until the SHA-pin commit.

What B1 will ship (separate commits within the phase):
  * private_hard.py: the 4-task private hardness subset adapter
  * grader.py: one-shot offline grader → eval/private/hardness/index.parquet
  * calibration.py: N=10 baseline runs → noise_floors_v1.json
  * runner.py: subprocess-isolated entrypoint

Reference scope: docs/build_scope/02_scope_B1.md.
"""
from __future__ import annotations

from .aggregate import aggregate_pareto
from .core22 import (
    DCLM_CORE_22_TASKS,
    DCLM_EVAL_BUNDLE_SHA256,
    DCLM_EVAL_BUNDLE_URL,
    TASK_SPECS,
    evaluate_lm_task_lambada,
    evaluate_mc_task,
    evaluate_schema_task,
    to_cell_result,
)
from .scorer import (
    LMExample,
    MCExample,
    SchemaExample,
    score_lm,
    score_mc,
    score_schema,
)
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
    "DCLM_CORE_22_TASKS",
    "DCLM_EVAL_BUNDLE_SHA256",
    "DCLM_EVAL_BUNDLE_URL",
    "DownstreamReport",
    "HARNESS_VERSION",
    "LMExample",
    "MCExample",
    "NoiseFloorTable",
    "POOL_CORE22",
    "POOL_PRIVATE_HARD",
    "POOL_S2_VAL_BPB",
    "ParetoOutcome",
    "ParetoVerdict",
    "SchemaExample",
    "TASK_SPECS",
    "TaskSpec",
    "aggregate_pareto",
    "evaluate_lm_task_lambada",
    "evaluate_mc_task",
    "evaluate_schema_task",
    "score_lm",
    "score_mc",
    "score_schema",
    "to_cell_result",
]
