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
  * runner.py (kernel): EvalConfig + vocab/determinism guards +
    in-process run_downstream_eval driver (B1-D6, B1-D7 closed)
  * runner_subprocess.py: caller-side subprocess wrapper +
    EvalSubprocessError + JSON IPC contract
  * runner_cli.py: production CLI entrypoint — argparse +
    `torch.load(weights_only=True)` checkpoint load + KarpaBase
    model construction + tiktoken GPT-2 BPE tokenizer wiring +
    structural-patch CLI args (B1-D5, B1-D13 closed)
  * calibration.py: aggregate_noise_floors + per-cell stddev
    diagnostic + noise_floors_v1.json round-trip (B1-D8 closed
    at the aggregation surface; the H100 training-driver follow-up
    will produce the actual file)

B1 module sweep complete — types, aggregate, scorer, core22,
private_hard, grader, runner (kernel + subprocess + CLI),
calibration.

Reference scope: docs/build_scope/02_scope_B1.md.
"""
from __future__ import annotations

from .aggregate import aggregate_pareto
from .calibration import (
    aggregate_noise_floors,
    compute_per_cell_stddev,
    read_noise_floor_table_json,
    write_noise_floor_table_json,
)
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
from .runner import (
    KARPA_VOCAB_SIZE,
    EvalConfig,
    check_vocab_compatibility,
    run_downstream_eval,
    set_eval_determinism,
)
from .runner_subprocess import (
    DEFAULT_COMMAND_PREFIX,
    STDERR_TAIL_LIMIT,
    EvalSubprocessError,
    deserialize_report,
    run_eval_in_subprocess,
    serialize_report,
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
    "DEFAULT_COMMAND_PREFIX",
    "DownstreamReport",
    "EvalConfig",
    "EvalSubprocessError",
    "HARNESS_VERSION",
    "HF_DATASET_IDS",
    "HardnessIndex",
    "HardnessIndexRow",
    "KARPA_VOCAB_SIZE",
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
    "STDERR_TAIL_LIMIT",
    "SchemaExample",
    "SchemaRawRow",
    "TASK_SPECS",
    "TaskSpec",
    "aggregate_noise_floors",
    "aggregate_pareto",
    "assemble_hardness_index",
    "check_vocab_compatibility",
    "compute_bottom_quintile",
    "compute_per_cell_stddev",
    "deserialize_report",
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
    "read_noise_floor_table_json",
    "run_downstream_eval",
    "run_eval_in_subprocess",
    "score_lm",
    "score_mc",
    "score_mc_logprobs",
    "score_schema",
    "score_schema_logprobs",
    "select_hardness_subset",
    "serialize_report",
    "set_eval_determinism",
    "to_cell_result",
    "to_private_hard_cell_result",
    "write_hardness_index_jsonl",
    "write_noise_floor_table_json",
]
