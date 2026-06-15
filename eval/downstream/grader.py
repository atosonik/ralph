"""One-shot offline grader for the private hardness subset (B1).

Computes per-item `gold_margin_bits` under a fixed reference model — typically
a 50M-param Ralph baseline trained once at calibration time. The bottom 20%
of items by margin become the hardness subset that `private_hard.py`
consumes via `select_hardness_subset` at scoring time.

This module ships the GRADING + INDEX-ASSEMBLY logic:

  * `gold_margin_bits(choice_logprobs, gold)` — the core scalar.
    Margin = log_p(gold) − max_{d != gold} log_p(d), converted from nats
    to bits (divide by log 2). Tasks with a single choice trivially
    return +∞ (infinitely confident); empty input returns 0.0.
  * `grade_mc_task` / `grade_schema_task` — per-task drivers that take
    pre-keyed `(item_id, raw_row)` inputs, tokenize, score, and emit
    `HardnessIndexRow` per item.
  * `compute_bottom_quintile(rows, quintile=0.20)` — deterministic sort
    + slice. Stable across re-runs because Python's sort is stable.
  * `assemble_hardness_index(per_task, version, *, quintile=0.20)` —
    combine across tasks, apply the bottom-quintile rule per task,
    return a `HardnessIndex` ready for `write_hardness_index_jsonl`.
  * `write_hardness_index_jsonl` / `read_hardness_index_jsonl` — JSONL
    round-trip. Chosen over parquet for B1 because (a) zero new
    dependencies (b) human-readable for audit (c) row counts are tiny
    (~hundreds per task × 4 tasks ≈ few thousand rows total).
    A parquet upgrade path is recorded in `WRITER_FORMAT` for a future
    runner.py that needs columnar reads.

This module does NOT ship the reference-model checkpoint or the HF dataset
download — those are runner.py / calibration.py concerns. Tests drive the
grader with synthetic logits, which is enough to pin the math, the
sort-and-slice, the per-task assembly, and the JSONL round-trip.

Reference scope: docs/build_scope/02_scope_B1.md "grader.py".
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path

from .core22 import (
    MCRawRow,
    SchemaRawRow,
    make_mc_example,
    make_schema_example,
)
from .private_hard import (
    HardnessIndex,
    HardnessIndexRow,
)
from .scorer import (
    score_mc_logprobs,
    score_schema_logprobs,
)

# JSONL on disk today; switch to parquet via a one-line dispatcher when
# runner.py needs columnar reads at scale. Bumping this string is the
# signal that the on-disk format has changed.
WRITER_FORMAT = "jsonl-v1"


# ----------------------------------------------------------------------------
# Core scalar
# ----------------------------------------------------------------------------


def gold_margin_bits(choice_logprobs: list[float], gold: int) -> float:
    """Return `log_p(gold) - max_{d != gold} log_p(d)`, in bits.

    Input log-probabilities are in NATS (the scorer's natural-log
    log_softmax output). Output is in bits via division by `log(2)`.

    Edge cases (each tested in test_downstream_grader.py):
      * Empty input → 0.0 (the empty argmax convention).
      * Single choice → +inf (no distractors → infinitely confident).
      * `gold` out of range → ValueError (caller bug).
    """
    if not choice_logprobs:
        return 0.0
    if gold < 0 or gold >= len(choice_logprobs):
        raise ValueError(
            f"gold={gold} out of range for {len(choice_logprobs)} choices"
        )
    if len(choice_logprobs) == 1:
        return float("inf")
    gold_lp = choice_logprobs[gold]
    other_lps = [lp for i, lp in enumerate(choice_logprobs) if i != gold]
    return (gold_lp - max(other_lps)) / math.log(2)


# ----------------------------------------------------------------------------
# Per-task graders
# ----------------------------------------------------------------------------

Tokenize = Callable[[str], list[int]]


def grade_mc_task(
    forward_logits: Callable,
    items: list[tuple[str, MCRawRow]],
    tokenize: Tokenize,
    task_name: str,
    *,
    length_normalize: bool = True,
) -> list[HardnessIndexRow]:
    """Grade every (item_id, MCRawRow) under the reference model.

    Returns one `HardnessIndexRow` per item with `gold_margin_bits`
    computed as above. Order preserved from input.
    """
    if not items:
        return []
    examples = [make_mc_example(row, tokenize) for _, row in items]
    per_choice = score_mc_logprobs(
        forward_logits, examples, length_normalize=length_normalize,
    )
    out: list[HardnessIndexRow] = []
    for (item_id, row), choice_logprobs in zip(items, per_choice):
        margin = gold_margin_bits(choice_logprobs, row.gold)
        out.append(HardnessIndexRow(
            dataset=task_name, item_id=item_id, gold_margin_bits=margin,
        ))
    return out


def grade_schema_task(
    forward_logits: Callable,
    items: list[tuple[str, SchemaRawRow]],
    tokenize: Tokenize,
    task_name: str,
) -> list[HardnessIndexRow]:
    """Same as grade_mc_task but for schema rows (winogrande_hard etc.)."""
    if not items:
        return []
    examples = [make_schema_example(row, tokenize) for _, row in items]
    per_variant = score_schema_logprobs(forward_logits, examples)
    out: list[HardnessIndexRow] = []
    for (item_id, row), variant_logprobs in zip(items, per_variant):
        margin = gold_margin_bits(variant_logprobs, row.gold)
        out.append(HardnessIndexRow(
            dataset=task_name, item_id=item_id, gold_margin_bits=margin,
        ))
    return out


# ----------------------------------------------------------------------------
# Bottom-quintile selection
# ----------------------------------------------------------------------------


def compute_bottom_quintile(
    rows: list[HardnessIndexRow],
    quintile_fraction: float = 0.20,
) -> list[HardnessIndexRow]:
    """Return the `quintile_fraction` of rows with the smallest
    gold_margin_bits values.

    Smaller margin = harder. Sort is stable (Python's Timsort), so two
    runs with identical inputs produce identical outputs.

    For N input rows, exactly `round(N * quintile_fraction)` are
    returned — clipped to `[0, N]`. Empty input → empty output. Quintile
    fraction outside `[0, 1]` raises ValueError.

    The rows are returned in ascending-margin order, NOT in the input
    order. This makes the hardness subset itself a useful diagnostic:
    the first row is the hardest item, the last row is the
    boundary-of-quintile item.
    """
    if not 0.0 <= quintile_fraction <= 1.0:
        raise ValueError(
            f"quintile_fraction must be in [0, 1]; got {quintile_fraction}"
        )
    if not rows:
        return []
    n = len(rows)
    take = round(n * quintile_fraction)
    take = max(0, min(take, n))
    return sorted(rows, key=lambda r: r.gold_margin_bits)[:take]


# ----------------------------------------------------------------------------
# Index assembly
# ----------------------------------------------------------------------------


def assemble_hardness_index(
    per_task_rows: dict[str, list[HardnessIndexRow]],
    version: str,
    *,
    quintile_fraction: float = 0.20,
) -> HardnessIndex:
    """Apply the bottom-quintile filter per task, then merge into a single
    `HardnessIndex`.

    `per_task_rows` is a dict `task_name → all rows graded for that task`.
    The output index contains only the bottom-quintile rows from each
    task. Tasks with empty input contribute zero rows.

    Determinism: the merged row order is `(task_name in dict order,
    margin ascending within task)`. The dict iteration order in Python
    3.7+ is insertion order, so callers control the merged order by the
    order they assemble the input dict.
    """
    if not version:
        raise ValueError(
            "version must be non-empty; grader.py is the only caller, "
            "supply a grader-vX.Y.Z-<commit-sha> identifier so future "
            "auditors can replay the index"
        )
    merged: list[HardnessIndexRow] = []
    for task_name, rows in per_task_rows.items():
        bottom = compute_bottom_quintile(rows, quintile_fraction)
        # Re-stamp the dataset field if a caller fed in rows from a
        # different task. The grader emits one task per call; mismatches
        # here are caller bugs but we silently coerce because the result
        # is unambiguous given the dict key.
        for r in bottom:
            if r.dataset != task_name:
                r = HardnessIndexRow(
                    dataset=task_name,
                    item_id=r.item_id,
                    gold_margin_bits=r.gold_margin_bits,
                )
            merged.append(r)
    return HardnessIndex(version=version, rows=merged)


# ----------------------------------------------------------------------------
# JSONL I/O
# ----------------------------------------------------------------------------


def write_hardness_index_jsonl(index: HardnessIndex, path: Path) -> None:
    """Write a `HardnessIndex` to disk as JSONL.

    Line 1 is a header `{"_meta": ..., "version": ..., "format": ...}`.
    Lines 2..N are one `HardnessIndexRow` per line, encoded as JSON.

    Atomic-write via `path.with_suffix(".tmp")` + rename so a crashed
    write never leaves a half-written file the runner consumes.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tmp.open("w") as f:
        header = {
            "_meta": "ralph-hardness-index",
            "version": index.version,
            "format": WRITER_FORMAT,
            "n_rows": len(index.rows),
        }
        f.write(json.dumps(header, sort_keys=True) + "\n")
        for row in index.rows:
            f.write(json.dumps({
                "dataset": row.dataset,
                "item_id": row.item_id,
                "gold_margin_bits": row.gold_margin_bits,
            }, sort_keys=True) + "\n")
    tmp.replace(path)


def read_hardness_index_jsonl(path: Path) -> HardnessIndex:
    """Inverse of `write_hardness_index_jsonl`.

    Raises ValueError if the header is missing / malformed, or if the
    format string doesn't match WRITER_FORMAT. The format-check exists
    so a forward-compat parquet writer doesn't get silently consumed by
    an old JSONL reader.
    """
    path = Path(path)
    text = path.read_text()
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"empty hardness-index file at {path}")
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as e:
        raise ValueError(f"header line is not valid JSON: {e}") from e
    if header.get("_meta") != "ralph-hardness-index":
        raise ValueError(
            f"unexpected _meta marker {header.get('_meta')!r}; "
            "is this a ralph hardness-index file?"
        )
    if header.get("format") != WRITER_FORMAT:
        raise ValueError(
            f"format mismatch: file is {header.get('format')!r}, "
            f"this reader expects {WRITER_FORMAT!r}"
        )
    version = header.get("version", "")
    rows: list[HardnessIndexRow] = []
    for ln in lines[1:]:
        d = json.loads(ln)
        rows.append(HardnessIndexRow(
            dataset=d["dataset"],
            item_id=d["item_id"],
            gold_margin_bits=float(d["gold_margin_bits"]),
        ))
    return HardnessIndex(version=version, rows=rows)
