"""Private hardness-graded subset (B1).

The 4-task private subset that scores submissions at the S₃ rung alongside
the public CORE-22 (see core22.py). The subset's purpose is twofold:

  1. Hold a slice of the eval surface PRIVATE so a miner cannot trivially
     overfit to the public bundle's contents.
  2. Focus on the HIGH-DIFFICULTY tail of each task — the items where
     models near the 124M-param coordination point separate the most.

The 4 tasks are pinned at:

  * ARC-Challenge bottom-quintile — `allenai/ai2_arc` config `ARC-Challenge`,
    retain the 20% of val items with the smallest gold_margin (per a
    grader.py reference-model pre-pass).
  * winogrande — `allenai/winogrande` config `winogrande_xl`. Schema mode.
  * tinyARC — `tinyBenchmarks/tinyARC` (or constructed from `ai2_arc` if
    the tinyBenchmarks distribution does not ship it; verified at B1 start).
  * tinyMMLU — `tinyBenchmarks/tinyMMLU`, all 14 task subsets, scored via
    the IRT++ projection to full-MMLU rank.

Provenance of this 4-task set: `docs/license/hardness_subset_decision.md`
(2026-06-10 decision; closed B1-D1, pre-swap of OpenBookQA + SciQ which
were CC-BY-NC and therefore incompatible with the protocol's commercial
emission use). Per B1-D11, this subset is bundled into Ralph's eval
surface but is NOT yet part of the attested container measurement — the
container bump is a mainnet-activation deliverable, not B1's.

What this commit ships:

  * PRIVATE_HARD_TASKS — the 4-task name tuple.
  * PRIVATE_HARD_TASK_SPECS — mode/baseline/pool dict matching core22.py
    conventions.
  * HF_DATASET_IDS — the HuggingFace dataset identifiers each task pulls
    from.
  * HardnessIndexRow / HardnessIndex — the contract grader.py emits and
    private_hard.py consumes for the bottom-quintile selection.
  * select_hardness_subset(rows, index, task_name) — deterministic filter
    that keeps only the rows whose item_id appears in the index.
  * evaluate_private_hard_task — per-task driver wrapping evaluate_mc_task
    / evaluate_schema_task with the hardness filter applied first.
  * load_task_examples — stub that raises NotImplementedError pointing at
    the protocol the first downloader commit must follow.

What this commit does NOT ship:

  * HuggingFace dataset download + caching — gated on the runner's HF
    auth posture and the per-task split/configuration parsing.
  * Hardness-graded item construction (grader.py) — separate B1 module.
  * tinyMMLU IRT++ projection — defers to the tinyBenchmarks package,
    pulled in by the runner.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .core22 import (
    LMRawRow,
    MCRawRow,
    SchemaRawRow,
    evaluate_mc_task,
    evaluate_schema_task,
    make_mc_example,
    make_schema_example,
)
from .types import (
    POOL_PRIVATE_HARD,
    CellResult,
    TaskSpec,
)

# ----------------------------------------------------------------------------
# Task registry (per docs/license/hardness_subset_decision.md)
# ----------------------------------------------------------------------------

PRIVATE_HARD_TASKS: tuple[str, ...] = (
    "arc_challenge_hard",
    "winogrande_hard",
    "tiny_arc",
    "tiny_mmlu",
)
"""The 4 tasks in the private hardness subset, in canonical Ralph order.

Suffix conventions: `_hard` for bottom-quintile-by-margin filters of
public datasets; `tiny_*` for the tinyBenchmarks-curated IRT++ subsets
(which are themselves already hardness-graded by their construction).
"""

assert len(PRIVATE_HARD_TASKS) == 4, "private hardness subset must be exactly 4 tasks"


PRIVATE_HARD_TASK_SPECS: dict[str, TaskSpec] = {
    "arc_challenge_hard": TaskSpec("arc_challenge_hard", "mc",     0.25, POOL_PRIVATE_HARD),
    "winogrande_hard":    TaskSpec("winogrande_hard",    "schema", 0.50, POOL_PRIVATE_HARD),
    "tiny_arc":           TaskSpec("tiny_arc",           "mc",     0.25, POOL_PRIVATE_HARD),
    "tiny_mmlu":          TaskSpec("tiny_mmlu",          "mc",     0.25, POOL_PRIVATE_HARD),
}

assert set(PRIVATE_HARD_TASK_SPECS.keys()) == set(PRIVATE_HARD_TASKS)


HF_DATASET_IDS: dict[str, tuple[str, str | None]] = {
    "arc_challenge_hard": ("allenai/ai2_arc",            "ARC-Challenge"),
    "winogrande_hard":    ("allenai/winogrande",         "winogrande_xl"),
    "tiny_arc":           ("tinyBenchmarks/tinyAI2_arc", None),
    "tiny_mmlu":          ("tinyBenchmarks/tinyMMLU",    None),
}
"""HuggingFace dataset identifiers per task: `(repo_id, config)`.

`config` is `None` for datasets whose default config we accept (the tiny*
sets ship a single config each). The runner.py author confirms the exact
identifiers at B1 start (HF repos rotate occasionally; pin by SHA if a
stable version is required) and adds a `test_hf_dataset_ids_resolve`
test against a frozen HF Hub snapshot.
"""

assert set(HF_DATASET_IDS.keys()) == set(PRIVATE_HARD_TASKS), \
    "every private-hard task must have an HF dataset identifier"


# ----------------------------------------------------------------------------
# HardnessIndex — the contract grader.py emits, this module consumes
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class HardnessIndexRow:
    """One row in the hardness index `parquet` grader.py produces.

    `gold_margin_bits` is `log_p(gold) - max_{d != gold} log_p(d)` under a
    fixed reference model (a 50M-param Ralph baseline run once at
    calibration time). Smaller margin → harder item; the bottom 20% by
    margin are selected as the hardness subset.

    For tinyBenchmarks tasks the parquet still ships row-per-item but the
    margin column is informational only — the tinyBenchmarks set is
    already hardness-curated externally via IRT++.
    """

    dataset: str        # one of PRIVATE_HARD_TASKS
    item_id: str        # the upstream HF dataset's identifier for the row
    gold_margin_bits: float


@dataclass(frozen=True)
class HardnessIndex:
    """The full index — emitted once by grader.py, consumed by this module.

    Stored on disk as `eval/private/hardness/index.parquet` (gitignored).
    The `version` field bumps every time grader.py regenerates against a
    new reference checkpoint or with a new bottom-quintile rule; the
    runner records this version into the DownstreamReport so a future
    auditor can replay the selection deterministically.
    """

    version: str
    rows: list[HardnessIndexRow] = field(default_factory=list)

    def for_task(self, task_name: str) -> set[str]:
        """Return the set of item_ids that belong to the bottom-quintile
        filter for the given task. Empty set if the task is unknown or
        absent from the index."""
        return {r.item_id for r in self.rows if r.dataset == task_name}


# ----------------------------------------------------------------------------
# Hardness selection
# ----------------------------------------------------------------------------


def select_hardness_subset(
    rows: list[tuple[str, MCRawRow | SchemaRawRow | LMRawRow]],
    index: HardnessIndex,
    task_name: str,
) -> list[MCRawRow | SchemaRawRow | LMRawRow]:
    """Keep only the rows whose item_id is in the index for `task_name`.

    `rows` is a list of `(item_id, raw_row)` tuples. The loader supplies the
    item_id from the upstream HF dataset (e.g., ARC's `id` column); the
    grader_index marks which item_ids belong to the bottom-quintile.

    Returns the raw_row objects in the same order they appeared in `rows`
    (stable, deterministic). Empty input → empty output (no error).
    """
    accept = index.for_task(task_name)
    if not accept:
        return []
    return [raw for item_id, raw in rows if item_id in accept]


# ----------------------------------------------------------------------------
# Per-task evaluator
# ----------------------------------------------------------------------------

Tokenize = Callable[[str], list[int]]


def evaluate_private_hard_task(
    forward_logits: Callable,
    raw_rows: list[tuple[str, MCRawRow | SchemaRawRow | LMRawRow]],
    index: HardnessIndex,
    task_name: str,
    tokenize: Tokenize,
    *,
    length_normalize: bool = True,
) -> tuple[float, int]:
    """Apply the hardness filter, tokenize, dispatch to the right scorer.

    Returns (accuracy, n_examples_after_filter). If the index is empty for
    this task or the filter leaves zero rows, returns (0.0, 0) — the
    aggregator treats an empty cell as a no-data sentinel rather than a
    legitimate zero accuracy.
    """
    if task_name not in PRIVATE_HARD_TASK_SPECS:
        raise ValueError(f"unknown private-hard task {task_name!r}")

    spec = PRIVATE_HARD_TASK_SPECS[task_name]
    filtered = select_hardness_subset(raw_rows, index, task_name)
    if not filtered:
        return 0.0, 0

    if spec.mode == "mc":
        examples = [make_mc_example(r, tokenize) for r in filtered if isinstance(r, MCRawRow)]
        return evaluate_mc_task(forward_logits, examples, length_normalize=length_normalize)
    elif spec.mode == "schema":
        examples = [make_schema_example(r, tokenize) for r in filtered if isinstance(r, SchemaRawRow)]
        return evaluate_schema_task(forward_logits, examples)
    else:
        # LM mode is reserved — no current private-hard task uses it. If a
        # future task lands as LM mode, plug make_lm_example +
        # evaluate_lm_task_lambada here.
        raise NotImplementedError(
            f"LM mode not yet supported in private_hard.py; task={task_name}"
        )


def to_private_hard_cell_result(
    task_name: str,
    accuracy: float,
    n_examples: int,
    *,
    scale: str = "S3",  # noqa: ARG001 — accepted for symmetry with to_cell_result
    seed: int = 0,
) -> CellResult:
    """Wrap a private-hard task measurement as a CellResult.

    Validates against PRIVATE_HARD_TASK_SPECS specifically (not the
    union of core22 + private_hard) so a task name typo doesn't silently
    misroute through the wrong pool. The Pareto kernel reads `cell.task`
    to look up the noise floor; the pool selection is implicit in the
    cell-key suffix the runner constructs (`{task_name}:S3` for
    accuracy cells; `:bpb` reserved for val_bpb carriers).
    """
    if task_name not in PRIVATE_HARD_TASK_SPECS:
        raise ValueError(
            f"unknown private-hard task {task_name!r}; "
            f"expected one of {sorted(PRIVATE_HARD_TASKS)}"
        )
    return CellResult(
        task=task_name,
        accuracy=accuracy,
        accuracy_stderr=0.0,
        n_examples=n_examples,
        seed=seed,
    )


# ----------------------------------------------------------------------------
# Bundle / dataset loader — stub
# ----------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file → list of dicts. Skips blank lines, raises with
    a line number on invalid JSON.

    Independent of `core22._read_jsonl` (same shape, kept private to each
    module to avoid cross-module coupling on a 10-line helper)."""
    rows: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"invalid JSON at {path}:{line_no}: {e}"
                ) from e
    return rows


def _parse_id(row: dict, line_no: int) -> str:
    """Extract and validate the `id` field — required for every
    private-hard row so HardnessIndex lookup can route correctly."""
    if "id" not in row:
        raise ValueError(
            f"row {line_no}: missing required 'id' field (private-hard "
            "tasks require an item id for HardnessIndex lookup)"
        )
    item_id = row["id"]
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(
            f"row {line_no}: 'id' must be a non-empty string; got "
            f"{item_id!r}"
        )
    return item_id


def _parse_mc_with_id(row: dict, line_no: int) -> tuple[str, MCRawRow]:
    """Parse a private-hard MC row with id.

    Schema: {"id": str, "query": str, "choices": [str, ...], "gold": int}
    Same as core22 MC + the id is mandatory and preserved.
    """
    item_id = _parse_id(row, line_no)
    try:
        query = str(row["query"])
        choices = list(row["choices"])
        gold = int(row["gold"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"row {line_no}: expected private-hard MC schema "
            f"{{id, query, choices, gold}}; got error: {e}"
        ) from e
    if not all(isinstance(c, str) for c in choices):
        raise ValueError(
            f"row {line_no}: 'choices' must be a list of strings"
        )
    if not 0 <= gold < len(choices):
        raise ValueError(
            f"row {line_no}: gold={gold} out of range for "
            f"{len(choices)} choices"
        )
    return item_id, MCRawRow(query=query, choices=choices, gold=gold)


def _parse_schema_with_id(
    row: dict, line_no: int,
) -> tuple[str, SchemaRawRow]:
    """Parse a private-hard schema row with id.

    Schema: {"id": str, "contexts": [str, ...], "continuations": [str, ...],
             "gold": int}
    """
    item_id = _parse_id(row, line_no)
    try:
        contexts = list(row["contexts"])
        continuations = list(row["continuations"])
        gold = int(row["gold"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"row {line_no}: expected private-hard schema "
            f"{{id, contexts, continuations, gold}}; got error: {e}"
        ) from e
    if len(contexts) != len(continuations):
        raise ValueError(
            f"row {line_no}: contexts ({len(contexts)}) and "
            f"continuations ({len(continuations)}) length mismatch"
        )
    if not all(isinstance(c, str) for c in contexts):
        raise ValueError(
            f"row {line_no}: 'contexts' must be a list of strings"
        )
    if not all(isinstance(c, str) for c in continuations):
        raise ValueError(
            f"row {line_no}: 'continuations' must be a list of strings"
        )
    if not 0 <= gold < len(contexts):
        raise ValueError(
            f"row {line_no}: gold={gold} out of range for "
            f"{len(contexts)} variants"
        )
    return item_id, SchemaRawRow(
        contexts=contexts, continuations=continuations, gold=gold,
    )


def load_task_examples(
    cache_dir: Path | str,
    task_name: str,
) -> list[tuple[str, MCRawRow | SchemaRawRow]]:
    """Load `(item_id, raw_row)` pairs for a private-hard task.

    Reads `{cache_dir}/{task_name}.jsonl` and parses each line per the
    task's mode (`PRIVATE_HARD_TASK_SPECS[task_name].mode`). The item
    id is preserved in the returned tuples so
    `select_hardness_subset` can filter by the HardnessIndex.

    JSONL schema (canonical Ralph form):
      * mc:     {"id": str, "query": str, "choices": [str, ...], "gold": int}
      * schema: {"id": str, "contexts": [str, ...],
                 "continuations": [str, ...], "gold": int}

    The `cache_dir` should be a local directory populated by the HF
    download step (a separate operational concern — typically
    `eval/private/downstream_pool/private_hard/`). If the HF datasets'
    raw schemas differ (e.g. ai2_arc uses `question`/`choices.text` keys),
    the downloader script re-keys them into the canonical form during
    the cache-prep step.

    Args:
      cache_dir: directory containing per-task `<task>.jsonl` files.
      task_name: one of `PRIVATE_HARD_TASKS`.

    Raises:
      ValueError if `task_name` is unknown.
      FileNotFoundError if `{cache_dir}/{task_name}.jsonl` doesn't exist.
      ValueError if any row fails per-mode parsing, with a message
        naming the line number + missing/invalid field.
      NotImplementedError if a task is registered with `mode == "lm"`
        — none of today's private-hard tasks use LM mode; if a future
        one does, plug `_parse_lm_with_id` here.
    """
    if task_name not in PRIVATE_HARD_TASK_SPECS:
        raise ValueError(
            f"unknown private-hard task {task_name!r}; expected one of "
            f"{sorted(PRIVATE_HARD_TASKS)}"
        )
    cache_dir = Path(cache_dir)
    path = cache_dir / f"{task_name}.jsonl"
    if not path.exists():
        upstream = HF_DATASET_IDS.get(task_name)
        raise FileNotFoundError(
            f"private-hard task file not found: {path}. Operator step: "
            f"download {upstream!r} from HuggingFace Hub, re-key into the "
            "canonical {id,...} schema, and cache the JSONL at this path."
        )

    spec = PRIVATE_HARD_TASK_SPECS[task_name]
    raw_rows = _read_jsonl(path)
    parsed: list[tuple[str, MCRawRow | SchemaRawRow]] = []
    for i, row in enumerate(raw_rows, start=1):
        if spec.mode == "mc":
            parsed.append(_parse_mc_with_id(row, i))
        elif spec.mode == "schema":
            parsed.append(_parse_schema_with_id(row, i))
        else:
            raise NotImplementedError(
                f"private-hard task {task_name!r} has mode "
                f"{spec.mode!r}; only mc/schema are wired today"
            )
    return parsed
