"""In-process downstream-eval driver (B1).

Composes core22.py + private_hard.py evaluators into one call that
returns a `DownstreamReport`. The validator's subprocess wrapper (next
PR) shells out to a CLI that imports this kernel + a forward function
loaded from a miner checkpoint.

What this module ships:

  * `EvalConfig` — JSON-serializable run parameters (tasks,
    n_examples_per_task, seed, scale_label, length_normalize_mc).
    Frozen dataclass with `to_dict` / `from_dict` so the subprocess
    wrapper can ship configs over IPC without ad-hoc serialization
    glue.
  * `RALPH_VOCAB_SIZE = 50257` — the GPT-2 BPE vocab lock. Submissions
    whose checkpoint config reports a different vocab are rejected at
    runner time. Closes **B1-D6**.
  * `check_vocab_compatibility(actual_vocab_size)` — raises ValueError
    with a clear message when actual != 50257.
  * `set_eval_determinism(seed)` — pins
    `torch.use_deterministic_algorithms(True)` +
    `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var + `torch.manual_seed` (+
    CUDA seed if available). Closes **B1-D7**.
  * `run_downstream_eval(forward_logits, *, config, task_loaders,
    tokenize, bundle_sha256, vocab_size, hardness_index, wall_clock_s)`
    — the pure in-process driver. Dispatches core22 tasks through
    `evaluate_mc_task` / `evaluate_schema_task` /
    `evaluate_lm_task_lambada` by `TaskSpec.mode`; dispatches
    private_hard tasks through `evaluate_private_hard_task` with the
    provided `HardnessIndex`. Returns a `DownstreamReport`.

What this module does NOT ship (next PR):

  * Subprocess wrapper (subprocess.run + JSON IPC + stderr handling).
  * Checkpoint loading (torch.load with weights_only=True). Closes
    **B1-D5** (the "weights_only blocks pickle RCE but not forward()
    code execution" caveat is documented as part of that wrapper).
  * The argparse CLI entrypoint.
  * Structural-patch handling (`--patch` / `--ralph-root` args).
    Closes **B1-D13**.

Cell-key construction: `{task_name}:{scale_label}` for accuracy cells.
The `:bpb` suffix (lower-is-better direction-flip) is reserved for the
report-assembly layer where bytes-per-token is known; the in-process
kernel emits the raw NLL in `CellResult.accuracy` for LM tasks and the
caller does the bpb conversion if needed.

Wall-clock: defaults to 0.0. The in-process kernel is deterministic
data-flow; the subprocess wrapper measures real elapsed seconds and
injects them via the `wall_clock_s` argument.

Reference scope: docs/build_scope/02_scope_B1.md "runner.py".
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .core22 import (
    TASK_SPECS,
    evaluate_lm_task_lambada,
    evaluate_mc_task,
    evaluate_schema_task,
    make_lm_example,
    make_mc_example,
    make_schema_example,
    to_cell_result,
)
from .private_hard import (
    PRIVATE_HARD_TASK_SPECS,
    HardnessIndex,
    evaluate_private_hard_task,
    to_private_hard_cell_result,
)
from .types import (
    HARNESS_VERSION,
    CellResult,
    DownstreamReport,
)

# GPT-2 BPE vocab size. The runner accepts only models whose `vocab_size`
# matches; mismatched-vocab submissions fail clean here instead of
# silently producing nonsensical scores. Closes B1-D6.
RALPH_VOCAB_SIZE = 50257


# ----------------------------------------------------------------------------
# EvalConfig
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalConfig:
    """Inputs to `run_downstream_eval`, JSON-serializable.

    Frozen so the config can be hashed into cache keys / chain logs
    without defensive copying.

    Fields:
      * `tasks` — tuple of task names. Every task must be in `TASK_SPECS`
        (core22) OR `PRIVATE_HARD_TASK_SPECS` (the hardness subset).
        Empty tuple rejected.
      * `n_examples_per_task` — truncate task loaders to the first N
        rows. 0 = use all. Negative rejected.
      * `seed` — passed to `torch.manual_seed` via `set_eval_determinism`
        and stamped into every `CellResult.seed` so multi-seed scaffolding
        downstream can re-derive provenance.
      * `scale_label` — cell-key suffix. Validator passes "S1"/"S2"/"S3"
        for the ladder rungs. Empty string rejected.
      * `length_normalize_mc` — pass-through to `evaluate_mc_task`. Default
        True matches DCLM / lm-eval-harness convention for MMLU/ARC.
    """

    tasks: tuple[str, ...]
    n_examples_per_task: int = 0
    seed: int = 0
    scale_label: str = "S3"
    length_normalize_mc: bool = True

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("EvalConfig.tasks must be non-empty")
        if self.n_examples_per_task < 0:
            raise ValueError(
                f"EvalConfig.n_examples_per_task must be >= 0; "
                f"got {self.n_examples_per_task}"
            )
        if not self.scale_label:
            raise ValueError("EvalConfig.scale_label must be non-empty")

    def to_dict(self) -> dict:
        """Serialize to a plain dict (JSON-safe). Round-trips via from_dict."""
        return {
            "tasks": list(self.tasks),
            "n_examples_per_task": self.n_examples_per_task,
            "seed": self.seed,
            "scale_label": self.scale_label,
            "length_normalize_mc": self.length_normalize_mc,
        }

    @classmethod
    def from_dict(cls, d: dict) -> EvalConfig:
        """Inverse of `to_dict`. Tolerant of missing optional fields.

        Converts the tasks list back to a tuple for frozen-dataclass
        hashability. Missing `tasks` raises KeyError so the caller sees
        the bug immediately instead of an empty-tasks ValueError.
        """
        return cls(
            tasks=tuple(d["tasks"]),
            n_examples_per_task=d.get("n_examples_per_task", 0),
            seed=d.get("seed", 0),
            scale_label=d.get("scale_label", "S3"),
            length_normalize_mc=d.get("length_normalize_mc", True),
        )


# ----------------------------------------------------------------------------
# Vocab + determinism guards
# ----------------------------------------------------------------------------


def check_vocab_compatibility(actual_vocab_size: int) -> None:
    """Reject submissions whose tokenizer vocab != 50257 (GPT-2 BPE).

    Why this is necessary (not just nice-to-have): tokens for the same
    input string differ across vocabs, so the per-token log-probability
    space is incomparable across models. Two models that report different
    `vocab_size` cannot be compared on the SAME private hardness subset
    without an entirely different scoring protocol. Rather than silently
    produce non-comparable numbers, we fail clean at runner time.

    Closes B1-D6.
    """
    if actual_vocab_size != RALPH_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size mismatch: forward function reports "
            f"{actual_vocab_size}, runner requires {RALPH_VOCAB_SIZE} "
            "(GPT-2 BPE). Submissions with divergent vocab must extend "
            "or pad to vocab=50257 or be evaluated under a different "
            "harness."
        )


def set_eval_determinism(seed: int) -> None:
    """Configure torch for byte-identical results across runs.

    Sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` env var (required for CUDA
    deterministic matmul kernels) BEFORE flipping the deterministic
    algorithm flag, then seeds CPU + (if present) CUDA RNGs.

    Idempotent: safe to call multiple times in the same process.
    Side-effecting: the deterministic-algorithms flag is process-wide;
    the subprocess wrapper (next PR) keeps each run in a fresh process so
    this state doesn't leak. In-process tests rely on the same property.

    Closes B1-D7.
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# In-process driver
# ----------------------------------------------------------------------------

# A task loader is a no-arg callable that returns the raw rows for one
# task. For core22 tasks the elements are MCRawRow / SchemaRawRow /
# LMRawRow; for private_hard tasks the elements are (item_id, raw_row)
# tuples. The loader's identity (HF, local cache, synthetic, etc.) is
# the wrapper's concern, not the kernel's.
TaskLoader = Callable[[], list]


def run_downstream_eval(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    *,
    config: EvalConfig,
    task_loaders: dict[str, TaskLoader],
    tokenize: Callable[[str], list[int]],
    bundle_sha256: str,
    vocab_size: int,
    hardness_index: HardnessIndex | None = None,
    wall_clock_s: float = 0.0,
) -> DownstreamReport:
    """Run the configured tasks and assemble a `DownstreamReport`.

    Dispatching:
      * task in TASK_SPECS → dispatch by spec.mode:
          mode == "mc"     → evaluate_mc_task
          mode == "schema" → evaluate_schema_task
          mode == "lm"     → evaluate_lm_task_lambada
      * task in PRIVATE_HARD_TASK_SPECS → evaluate_private_hard_task
        with `hardness_index` (None → ValueError).
      * task in neither → ValueError.

    Truncation: `config.n_examples_per_task > 0` slices the loaded raw
    rows BEFORE tokenization. For private_hard tasks it slices BEFORE
    the hardness filter — the runner exposes a row budget, not a hardness
    budget. In production the validator passes 0 (use all).

    Cell keys: `{task_name}:{config.scale_label}` for every cell. The
    `:bpb` suffix is constructed by the report-assembly layer when LM
    accuracy gets converted to bits-per-byte.

    Determinism: `set_eval_determinism(config.seed)` is called BEFORE
    the first task runs, so any RNG inside the forward function or
    tokenize callable gets a pinned seed.
    """
    check_vocab_compatibility(vocab_size)
    set_eval_determinism(config.seed)

    cells: dict[str, CellResult] = {}
    total_examples = 0

    for task_name in config.tasks:
        in_core22 = task_name in TASK_SPECS
        in_private_hard = task_name in PRIVATE_HARD_TASK_SPECS
        if not in_core22 and not in_private_hard:
            raise ValueError(
                f"unknown task {task_name!r}: not in TASK_SPECS "
                "(core22) or PRIVATE_HARD_TASK_SPECS"
            )
        if task_name not in task_loaders:
            raise ValueError(
                f"no loader registered for task {task_name!r}; pass "
                "it in task_loaders"
            )

        raw = task_loaders[task_name]()
        if config.n_examples_per_task > 0:
            raw = raw[: config.n_examples_per_task]

        if in_private_hard:
            if hardness_index is None:
                raise ValueError(
                    f"private-hard task {task_name!r} requires a "
                    "hardness_index (got None). The validator constructs "
                    "the index once per calibration cycle and reuses it "
                    "across submissions."
                )
            acc, n = evaluate_private_hard_task(
                forward_logits,
                raw,
                hardness_index,
                task_name,
                tokenize,
                length_normalize=config.length_normalize_mc,
            )
            cell = to_private_hard_cell_result(
                task_name,
                acc,
                n,
                scale=config.scale_label,
                seed=config.seed,
            )
        else:
            spec = TASK_SPECS[task_name]
            if spec.mode == "mc":
                examples = [make_mc_example(r, tokenize) for r in raw]
                acc, n = evaluate_mc_task(
                    forward_logits,
                    examples,
                    length_normalize=config.length_normalize_mc,
                )
            elif spec.mode == "schema":
                examples = [make_schema_example(r, tokenize) for r in raw]
                acc, n = evaluate_schema_task(forward_logits, examples)
            elif spec.mode == "lm":
                examples = [make_lm_example(r, tokenize) for r in raw]
                acc, n = evaluate_lm_task_lambada(forward_logits, examples)
            else:
                raise ValueError(
                    f"task {task_name!r} has unsupported mode "
                    f"{spec.mode!r}; expected mc/schema/lm"
                )
            cell = to_cell_result(
                task_name,
                acc,
                n,
                scale=config.scale_label,
                seed=config.seed,
            )

        cells[f"{task_name}:{config.scale_label}"] = cell
        total_examples += n

    return DownstreamReport(
        harness_version=HARNESS_VERSION,
        bundle_sha256=bundle_sha256,
        seed=config.seed,
        total_examples=total_examples,
        wall_clock_s=wall_clock_s,
        cells=cells,
    )
