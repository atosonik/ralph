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
on-disk mirror against the expected upstream version. The actual bundle
fetch + jsonl parsing is deferred to a follow-up commit once the bundle
is downloaded and its SHA is pinned (B1-D2 protocol). For now the
loader stub raises a clear NotImplementedError that the runner.py
author will resolve.

Reference scope: docs/build_scope/02_scope_B1.md "core22.py".
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

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

DCLM_EVAL_BUNDLE_SHA256: str | None = None
"""SHA256 of the canonical bundle. Pinned at first download.

Today (B1 foundation) this is None — the bundle has not been
downloaded yet. The first B1 code commit that actually fetches the
bundle MUST update this constant with the verbatim sha256 and a
matching `test_dclm_bundle_sha_pinned` test. Until then, callers
should skip integrity verification with a clear log message.
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


def load_task_examples(bundle_dir, task_name: str):
    """Load raw rows for a task from the DCLM bundle.

    NOT IMPLEMENTED in this commit. The bundle layout follows DCLM's
    `eval_bundle/<task_name>.jsonl` convention with per-task schemas
    that the runner.py author will parse.

    Until the bundle SHA is pinned (B1-D2), this raises a clear error
    pointing at the protocol the first downloader commit must follow.
    Callers that test against fixture data can use
    `make_mc_example` / `make_schema_example` / `make_lm_example`
    directly without going through this loader.
    """
    raise NotImplementedError(
        f"load_task_examples({task_name!r}) is not implemented in B1 "
        "foundation. The first commit that downloads the bundle from "
        f"{DCLM_EVAL_BUNDLE_URL} must (1) compute its SHA256, (2) update "
        "DCLM_EVAL_BUNDLE_SHA256 in this module, (3) implement this "
        "loader to parse <bundle_dir>/<task_name>.jsonl, and (4) add "
        "tests/test_downstream_core22_bundle.py with one task driven "
        "end-to-end against the real bundle. See eval/downstream/DEFERRED.md "
        "items B1-D2 / B1-D13 / B1-D8 for the protocol."
    )
