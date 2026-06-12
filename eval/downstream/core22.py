"""DCLM CORE-22 task registry + per-task evaluator (B1).

Adapts DCLM's CORE-22 eval bundle to the Karpa downstream-eval harness.
The 22 tasks are pinned from DCLM's `low_variance_datasets` aggregation
(see eval/downstream/DEFERRED.md B1-D3 for provenance). Each task is
catalogued by:

  * `name` — canonical DCLM task identifier
  * `mode` — "mc" / "schema" / "lm", routing to score_mc / score_schema /
    score_lm in scorer.py
  * `random_baseline` — DCLM's published random-guessing-or-majority
    baseline as a fraction in [0, 1]

The bundle URL + SHA pin are constants here so the runner can verify the
on-disk mirror against the expected upstream version. `load_task_examples`
parses per-task JSONLs from a local bundle copy and dispatches to the
per-mode parser (mc / schema / lm) to produce typed raw rows. The
actual bundle FETCH (download + SHA verification + local mirror) is a
separate operational step — `load_task_examples` takes the bundle
directory as an input and reads from there.

Reference scope: docs/build_scope/02_scope_B1.md "core22.py".
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .scorer import (
    LMExample,
    MCExample,
    SchemaExample,
    score_lm,
    score_mc,
    score_schema,
)
from .types import (
    POOL_CORE22,
    CellResult,
    TaskSpec,
)

# ----------------------------------------------------------------------------
# Bundle constants (per B1-D2)
# ----------------------------------------------------------------------------

DCLM_EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
"""DCLM CORE-22 eval bundle source URL.

Verified 2026-06-10 via nanochat/scripts/base_eval.py constant
EVAL_BUNDLE_URL. The bundle is hosted in Karpathy's personal S3, which
means it CAN rotate without notice — the SHA pin below is the guard.
"""

DCLM_EVAL_BUNDLE_SHA256: str | None = (
    "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
)
"""SHA256 of the canonical bundle. Pinned 2026-06-12 against the
`download_dclm_bundle.py` manifest (`extracted_member_count = 86`).

If a future pull mismatches, treat that as an upstream rotation: do
NOT silently bump this constant. Re-derive the bundle's provenance
(diff member list against the prior 86 entries) and bump only with
a paired commit that lists what changed. Per B1-D2, this constant
is FROZEN once pinned; downstream callers (`download_dclm_bundle`,
`core22` integrity check) rely on a single-value match.
"""


# ----------------------------------------------------------------------------
# Task registry (per B1-D3)
# ----------------------------------------------------------------------------

DCLM_CORE_22_TASKS: tuple[str, ...] = (
    "hellaswag_zeroshot",
    "jeopardy",
    "bigbench_qa_wikidata",
    "arc_easy",
    "arc_challenge",
    "copa",
    "commonsense_qa",
    "piqa",
    "openbook_qa",
    "lambada_openai",
    "hellaswag",
    "winograd",
    "winogrande",
    "bigbench_dyck_languages",
    "agi_eval_lsat_ar",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
)
"""The verbatim 22-task list from DCLM's `low_variance_datasets`
aggregation (additional_aggregation.json). Order preserved from the
source so cross-references against DCLM tooling stay byte-equal."""

assert len(DCLM_CORE_22_TASKS) == 22, "CORE-22 must be exactly 22 tasks"


# Per-task metadata sourced verbatim from DCLM's eval/eval_meta_data.csv on
# 2026-06-10. Random baselines are decimal fractions in [0, 1]; DCLM
# reports percents which are converted here. The scoring mode determines
# which scorer.py entrypoint the per-task evaluator dispatches to.
TASK_SPECS: dict[str, TaskSpec] = {
    "hellaswag_zeroshot":               TaskSpec("hellaswag_zeroshot",               "mc",     0.25, POOL_CORE22),
    "jeopardy":                         TaskSpec("jeopardy",                         "lm",     0.0,  POOL_CORE22),
    "bigbench_qa_wikidata":             TaskSpec("bigbench_qa_wikidata",             "lm",     0.0,  POOL_CORE22),
    "arc_easy":                         TaskSpec("arc_easy",                         "mc",     0.25, POOL_CORE22),
    "arc_challenge":                    TaskSpec("arc_challenge",                    "mc",     0.25, POOL_CORE22),
    "copa":                             TaskSpec("copa",                             "mc",     0.50, POOL_CORE22),
    "commonsense_qa":                   TaskSpec("commonsense_qa",                   "mc",     0.403, POOL_CORE22),
    "piqa":                             TaskSpec("piqa",                             "mc",     0.50, POOL_CORE22),
    "openbook_qa":                      TaskSpec("openbook_qa",                      "mc",     0.25, POOL_CORE22),
    "lambada_openai":                   TaskSpec("lambada_openai",                   "lm",     0.0,  POOL_CORE22),
    "hellaswag":                        TaskSpec("hellaswag",                        "mc",     0.25, POOL_CORE22),
    "winograd":                         TaskSpec("winograd",                         "schema", 0.50, POOL_CORE22),
    "winogrande":                       TaskSpec("winogrande",                       "schema", 0.50, POOL_CORE22),
    "bigbench_dyck_languages":          TaskSpec("bigbench_dyck_languages",          "lm",     0.0,  POOL_CORE22),
    "agi_eval_lsat_ar":                 TaskSpec("agi_eval_lsat_ar",                 "mc",     0.25, POOL_CORE22),
    "bigbench_cs_algorithms":           TaskSpec("bigbench_cs_algorithms",           "lm",     0.0,  POOL_CORE22),
    "bigbench_operators":               TaskSpec("bigbench_operators",               "lm",     0.0,  POOL_CORE22),
    "bigbench_repeat_copy_logic":       TaskSpec("bigbench_repeat_copy_logic",       "lm",     0.0,  POOL_CORE22),
    "squad":                            TaskSpec("squad",                            "lm",     0.0,  POOL_CORE22),
    "coqa":                             TaskSpec("coqa",                             "lm",     0.0,  POOL_CORE22),
    # boolq's 0.62 is DCLM's majority-class baseline, not the 0.50 you'd
    # expect from a binary task — boolq has skewed True/False prevalence.
    "boolq":                            TaskSpec("boolq",                            "mc",     0.62, POOL_CORE22),
    "bigbench_language_identification": TaskSpec("bigbench_language_identification", "mc",     0.25, POOL_CORE22),
}

assert set(TASK_SPECS.keys()) == set(DCLM_CORE_22_TASKS), \
    "TASK_SPECS keys must exactly match DCLM_CORE_22_TASKS"


# ----------------------------------------------------------------------------
# Raw-row → tokenized-example converters
# ----------------------------------------------------------------------------

# A tokenize callable: text → list of token ids. Caller supplies the actual
# tokenizer (tiktoken GPT-2 BPE per DCLM convention); we don't import it
# here so tests can mock it trivially.
Tokenize = Callable[[str], list[int]]


@dataclass(frozen=True)
class MCRawRow:
    """A multiple-choice raw row from the DCLM bundle's jsonl."""

    query: str
    choices: list[str]
    gold: int


@dataclass(frozen=True)
class SchemaRawRow:
    """A schema raw row — each variant has its own context/continuation."""

    contexts: list[str]
    continuations: list[str]
    gold: int


@dataclass(frozen=True)
class LMRawRow:
    """A language-modeling raw row.

    `accept_set` is the set of valid completion strings — LAMBADA-style
    tasks have exactly one accept; some other LM tasks have multiple
    (e.g., a question with several phrasings of the same answer all
    counted as correct). Empty set → never-correct (test signal only).
    """

    context: str
    target: str
    accept_set: tuple[str, ...] = ()


def make_mc_example(row: MCRawRow, tokenize: Tokenize) -> MCExample:
    return MCExample(
        context_ids=tokenize(row.query),
        choice_ids=[tokenize(c) for c in row.choices],
        gold=row.gold,
    )


def make_schema_example(row: SchemaRawRow, tokenize: Tokenize) -> SchemaExample:
    return SchemaExample(
        context_ids=[tokenize(c) for c in row.contexts],
        continuation_ids=[tokenize(c) for c in row.continuations],
        gold=row.gold,
    )


def make_lm_example(row: LMRawRow, tokenize: Tokenize) -> LMExample:
    return LMExample(
        context_ids=tokenize(row.context),
        target_ids=tokenize(row.target),
    )


# ----------------------------------------------------------------------------
# Per-task evaluators
# ----------------------------------------------------------------------------


def evaluate_mc_task(
    forward_logits: Callable,
    examples: list[MCExample],
    *,
    length_normalize: bool = True,
) -> tuple[float, int]:
    """Run score_mc on `examples` and return (accuracy, n_examples).

    Accuracy = fraction of examples where the predicted choice index
    equals the gold index. Returns (0.0, 0) on an empty list (no
    division-by-zero; the caller's aggregator handles the empty cell).
    """
    if not examples:
        return 0.0, 0
    preds = score_mc(forward_logits, examples, length_normalize=length_normalize)
    correct = sum(1 for p, ex in zip(preds, examples) if p == ex.gold)
    return correct / len(examples), len(examples)


def evaluate_schema_task(
    forward_logits: Callable,
    examples: list[SchemaExample],
) -> tuple[float, int]:
    if not examples:
        return 0.0, 0
    preds = score_schema(forward_logits, examples)
    correct = sum(1 for p, ex in zip(preds, examples) if p == ex.gold)
    return correct / len(examples), len(examples)


def evaluate_lm_task_lambada(
    forward_logits: Callable,
    examples: list[LMExample],
) -> tuple[float, int]:
    """Score an LM task using LAMBADA-style "is the gold target the most
    likely continuation" accuracy.

    For each example the per-token NLL of the target is the only signal.
    For LAMBADA-style scoring we don't have distractors here; this is the
    simplest LM-accuracy proxy and the same one DCLM uses for
    `lambada_openai`. Tasks that need different accuracy logic (e.g.
    SQuAD F1, CoQA bytes-per-byte) will get their own evaluators wired
    by the runner.

    Returns (mean target NLL in nats, n_examples). The Pareto kernel
    treats LM cells as `:bpb`-suffixed (lower is better); the bpb
    conversion happens at the report-assembly layer where bytes-per-token
    is known.
    """
    if not examples:
        return 0.0, 0
    nlls = score_lm(forward_logits, examples)
    return sum(nlls) / len(nlls), len(nlls)


# ----------------------------------------------------------------------------
# CellResult builder
# ----------------------------------------------------------------------------


def to_cell_result(
    task_name: str,
    accuracy: float,
    n_examples: int,
    *,
    scale: str = "S3",
    seed: int = 0,
) -> CellResult:
    """Wrap a (accuracy, n) measurement as a CellResult keyed for the
    aggregate kernel.

    The Pareto kernel keys cells by `{task_name}:{scale}` for downstream
    accuracy cells, or `{task_name}:bpb` for val_bpb cells (lower-is-better
    direction-flip via BPB_SUFFIX). This helper produces the accuracy form;
    LM tasks that the runner converts to BPB pass the bpb value as
    `accuracy` and the cell key gets the `:bpb` suffix at a higher layer.
    """
    if task_name not in TASK_SPECS:
        raise ValueError(
            f"unknown task {task_name!r}; not in DCLM_CORE_22_TASKS"
        )
    return CellResult(
        task=task_name,
        accuracy=accuracy,
        accuracy_stderr=0.0,  # B1: deterministic eval, single-seed
        n_examples=n_examples,
        seed=seed,
    )


# ----------------------------------------------------------------------------
# Bundle loader — stub until B1-D2 SHA pin lands
# ----------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Iterate a JSONL file into a list of dicts. Skips blank lines.

    Raises ValueError naming the offending line on the first invalid
    JSON line, so a downloader / mirror that produced a corrupt file
    fails clean rather than silently dropping examples.
    """
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


def _parse_mc_row(row: dict, line_no: int) -> MCRawRow:
    """Parse one canonical MC JSONL row.

    Schema (canonical Karpa form):
      {"id": str, "query": str, "choices": [str, ...], "gold": int}

    `id` is consumed by callers that need item-level keying (e.g.
    private-hard's HardnessIndex lookup). `_parse_mc_row` ignores it —
    the loader returns plain MCRawRow / SchemaRawRow / LMRawRow without
    the id; private-hard's `load_task_examples` (separate follow-up)
    is the variant that preserves `(item_id, row)` pairs.
    """
    try:
        query = str(row["query"])
        choices = list(row["choices"])
        gold = int(row["gold"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"row {line_no}: expected canonical MC schema "
            f"{{query, choices, gold}}; got error: {e}"
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
    return MCRawRow(query=query, choices=choices, gold=gold)


def _parse_schema_row(row: dict, line_no: int) -> SchemaRawRow:
    """Parse one canonical schema JSONL row.

    Schema: {"id": str, "contexts": [str, ...], "continuations": [str, ...],
             "gold": int}

    Each variant has its own (context, continuation) pair. `contexts` and
    `continuations` must have the same length (one entry per variant).
    """
    try:
        contexts = list(row["contexts"])
        continuations = list(row["continuations"])
        gold = int(row["gold"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(
            f"row {line_no}: expected canonical schema "
            f"{{contexts, continuations, gold}}; got error: {e}"
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
    return SchemaRawRow(
        contexts=contexts, continuations=continuations, gold=gold,
    )


def _parse_lm_row(row: dict, line_no: int) -> LMRawRow:
    """Parse one canonical LM JSONL row.

    Schema: {"id": str, "context": str, "target": str,
             "accept_set": [str, ...] | optional}

    `accept_set` is the set of accepted completions for LM tasks that
    allow multiple gold answers (e.g., a question whose answer can be
    phrased several ways). Optional — defaults to () for LAMBADA-style
    tasks with a single target.
    """
    try:
        context = str(row["context"])
        target = str(row["target"])
    except (KeyError, TypeError) as e:
        raise ValueError(
            f"row {line_no}: expected canonical LM schema "
            f"{{context, target}}; got error: {e}"
        ) from e
    accept_raw = row.get("accept_set", [])
    if not isinstance(accept_raw, list):
        raise ValueError(
            f"row {line_no}: 'accept_set' must be a list if provided"
        )
    if not all(isinstance(a, str) for a in accept_raw):
        raise ValueError(
            f"row {line_no}: 'accept_set' must be a list of strings"
        )
    return LMRawRow(
        context=context, target=target, accept_set=tuple(accept_raw),
    )


def load_task_examples(
    bundle_dir: Path | str,
    task_name: str,
) -> list[MCRawRow | SchemaRawRow | LMRawRow]:
    """Load raw rows for a CORE-22 task from a local DCLM bundle copy.

    Reads `{bundle_dir}/{task_name}.jsonl` and parses each line per
    the task's mode (per `TASK_SPECS[task_name].mode`):
      * mode == "mc"     → MCRawRow
      * mode == "schema" → SchemaRawRow
      * mode == "lm"     → LMRawRow

    The JSONL schema is the canonical Karpa form documented in
    `_parse_mc_row` / `_parse_schema_row` / `_parse_lm_row`. If the
    DCLM bundle's per-task JSONLs use different keys, the
    downloader / mirror script (separate follow-up) is responsible for
    re-keying into this canonical schema during the bundle-prep step.

    Args:
      bundle_dir: directory containing per-task `<task>.jsonl` files.
      task_name: one of `DCLM_CORE_22_TASKS`.

    Raises:
      ValueError if `task_name` is unknown.
      FileNotFoundError if `{bundle_dir}/{task_name}.jsonl` doesn't exist.
      ValueError if any row in the JSONL fails per-mode parsing, with a
        message naming the line number and the missing/invalid field.
    """
    if task_name not in TASK_SPECS:
        raise ValueError(
            f"unknown task {task_name!r}; not in DCLM_CORE_22_TASKS"
        )
    bundle_dir = Path(bundle_dir)
    path = bundle_dir / f"{task_name}.jsonl"
    if not path.exists():
        # DCLM bundle layout nests tasks in category subdirs
        # (eval_bundle/eval_data/{category}/{task}.jsonl). Walk the tree
        # so callers can pass either the flat-mirror dir or the raw
        # upstream bundle root and have lookups still resolve.
        candidates = sorted(bundle_dir.rglob(f"{task_name}.jsonl"))
        if len(candidates) == 1:
            path = candidates[0]
        elif len(candidates) > 1:
            raise ValueError(
                f"ambiguous task file location for {task_name!r}: "
                f"{[str(c) for c in candidates]}. Flatten the bundle "
                "or pass a more specific bundle_dir."
            )
        else:
            raise FileNotFoundError(
                f"CORE-22 task file {task_name}.jsonl not found under "
                f"{bundle_dir} (searched flat + recursive). The bundle "
                f"download (URL {DCLM_EVAL_BUNDLE_URL}) must place "
                f"per-task JSONLs under the supplied bundle_dir."
            )

    spec = TASK_SPECS[task_name]
    raw_rows = _read_jsonl(path)
    parsed: list[MCRawRow | SchemaRawRow | LMRawRow] = []
    for i, row in enumerate(raw_rows, start=1):
        if spec.mode == "mc":
            parsed.append(_parse_mc_row(row, i))
        elif spec.mode == "schema":
            parsed.append(_parse_schema_row(row, i))
        elif spec.mode == "lm":
            parsed.append(_parse_lm_row(row, i))
        else:
            raise ValueError(
                f"task {task_name!r} has unsupported mode "
                f"{spec.mode!r}; expected mc/schema/lm"
            )
    return parsed
