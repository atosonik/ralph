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
    callable) + score_mc_logprobs / score_schema_logprobs for grader
  * core22.py: the 22-task registry + per-task evaluators that wire scorer
    kernels to CellResult outputs. Bundle URL pinned (B1-D2); bundle
    download itself stubbed until the SHA-pin commit.
  * private_hard.py: the 4-task hardness subset registry + HardnessIndex
    contract + bottom-quintile filter
  * grader.py: gold_margin_bits computation + per-task graders +
    HardnessIndex assembly + JSONL round-trip

What B1 will ship (separate commits within the phase):
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
    LMRawRow,
    MCRawRow,
    SchemaRawRow,
    evaluate_lm_task_lambada,
    evaluate_mc_task,
    evaluate_schema_task,
    make_lm_example,
    make_mc_example,
    make_schema_example,
    to_cell_result,
)
from .grader import (
    assemble_hardness_index,
    compute_bottom_quintile,
    gold_margin_bits,
    grade_mc_task,
    grade_schema_task,
    read_hardness_index_jsonl,
    write_hardness_index_jsonl,
)
from .private_hard import (
    HF_DATASET_IDS,
    PRIVATE_HARD_TASK_SPECS,
    PRIVATE_HARD_TASKS,
    HardnessIndex,
    HardnessIndexRow,
    evaluate_private_hard_task,
    select_hardness_subset,
    to_private_hard_cell_result,
)
from .scorer import (
    LMExample,
    MCExample,
    SchemaExample,
    score_lm,
    score_mc,
    score_mc_logprobs,
    score_schema,
    score_schema_logprobs,
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
    "HF_DATASET_IDS",
    "HardnessIndex",
    "HardnessIndexRow",
    "LMExample",
    "LMRawRow",
    "MCExample",
    "MCRawRow",
    "NoiseFloorTable",
    "POOL_CORE22",
    "POOL_PRIVATE_HARD",
    "POOL_S2_VAL_BPB",
    "PRIVATE_HARD_TASKS",
    "PRIVATE_HARD_TASK_SPECS",
    "ParetoOutcome",
    "ParetoVerdict",
    "SchemaExample",
    "SchemaRawRow",
    "TASK_SPECS",
    "TaskSpec",
    "aggregate_pareto",
    "assemble_hardness_index",
    "compute_bottom_quintile",
    "evaluate_lm_task_lambada",
    "evaluate_mc_task",
    "evaluate_private_hard_task",
    "evaluate_schema_task",
    "gold_margin_bits",
    "grade_mc_task",
    "grade_schema_task",
    "make_lm_example",
    "make_mc_example",
    "make_schema_example",
    "read_hardness_index_jsonl",
    "score_lm",
    "score_mc",
    "score_mc_logprobs",
    "score_schema",
    "score_schema_logprobs",
    "select_hardness_subset",
    "to_cell_result",
    "to_private_hard_cell_result",
    "write_hardness_index_jsonl",
]
