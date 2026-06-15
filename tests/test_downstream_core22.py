"""Tests for the CORE-22 task registry + per-task evaluators (B1).

Pins the 22-task list verbatim against DCLM's `low_variance_datasets`,
verifies the per-task TaskSpec metadata (mode + random_baseline) matches
DCLM's eval_meta_data.csv, exercises the per-task evaluators with
synthetic logits, and confirms the bundle-loader stub raises a clear
error pointing at the B1-D2 protocol.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream import (
    POOL_CORE22,
    CellResult,
    LMExample,
    MCExample,
    SchemaExample,
)
from eval.downstream.core22 import (
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
    load_task_examples,
    make_lm_example,
    make_mc_example,
    make_schema_example,
    to_cell_result,
)

# ----------------------------------------------------------------------------
# Pinned constants
# ----------------------------------------------------------------------------


def test_dclm_core_22_count():
    """Exactly 22 tasks. Pinned by DCLM's low_variance_datasets list +
    Karpathy's nanochat #420 confirmation. Bumping this number requires a
    matching change in DEFERRED.md B1-D3."""
    assert len(DCLM_CORE_22_TASKS) == 22


def test_dclm_core_22_verbatim_order():
    """Order preserved from DCLM source. If DCLM rotates the order, a
    future PR must update both this list and the TASK_SPECS dict so
    cross-references stay byte-equal."""
    expected = (
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
    assert DCLM_CORE_22_TASKS == expected


def test_bigbench_language_identification_included():
    """B1-D3 explicitly settled the 22-vs-23 question. The
    bigbench_language_identification task is IN the CORE-22 list."""
    assert "bigbench_language_identification" in DCLM_CORE_22_TASKS


def test_eval_bundle_url_pinned():
    """The bundle source URL is pinned per B1-D2 from nanochat's
    EVAL_BUNDLE_URL constant. Karpathy's personal S3 hosting means the
    bundle CAN rotate without notice; the SHA pin (below) is the guard."""
    assert DCLM_EVAL_BUNDLE_URL == (
        "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"
    )


def test_eval_bundle_sha_pinned():
    """SHA pin is FROZEN to the 2026-06-12 download manifest. A mismatch
    means upstream rotated the bundle; per B1-D2 that requires a paired
    commit (re-derive provenance, list diffs) — NOT a silent bump."""
    assert DCLM_EVAL_BUNDLE_SHA256 == (
        "90a7c19e28ee7a52b4f6e1f87658deb9fde7f63deba2379045bdb1fe9ea5d200"
    )


# ----------------------------------------------------------------------------
# TASK_SPECS — mode + pool + random_baseline are all correct
# ----------------------------------------------------------------------------


def test_task_specs_keys_match_task_list():
    """Every task in DCLM_CORE_22_TASKS has a TaskSpec; no extras."""
    assert set(TASK_SPECS.keys()) == set(DCLM_CORE_22_TASKS)


def test_task_specs_all_in_core22_pool():
    """Every CORE-22 TaskSpec belongs to the CORE-22 pool."""
    for spec in TASK_SPECS.values():
        assert spec.pool == POOL_CORE22


def test_task_specs_valid_modes():
    """Every TaskSpec mode is one of mc / schema / lm (TaskSpec's own
    __post_init__ would have caught anything else; the test pins the
    invariant explicitly)."""
    for spec in TASK_SPECS.values():
        assert spec.mode in ("mc", "schema", "lm")


def test_task_specs_known_mode_assignments():
    """Spot-check the mode assignments against DCLM's eval_meta_data.csv:
    multiple_choice → "mc", schema → "schema", language_modeling → "lm"."""
    assert TASK_SPECS["winograd"].mode == "schema"
    assert TASK_SPECS["winogrande"].mode == "schema"
    assert TASK_SPECS["arc_easy"].mode == "mc"
    assert TASK_SPECS["arc_challenge"].mode == "mc"
    assert TASK_SPECS["lambada_openai"].mode == "lm"
    assert TASK_SPECS["jeopardy"].mode == "lm"
    assert TASK_SPECS["squad"].mode == "lm"
    assert TASK_SPECS["coqa"].mode == "lm"
    assert TASK_SPECS["copa"].mode == "mc"  # DCLM labels copa as MC
    assert TASK_SPECS["boolq"].mode == "mc"
    assert TASK_SPECS["bigbench_language_identification"].mode == "mc"


def test_task_specs_random_baselines_are_fractions():
    """Random baselines are stored as fractions in [0, 1]. DCLM's CSV
    reports them as percents; the registry MUST convert."""
    for spec in TASK_SPECS.values():
        assert 0.0 <= spec.random_baseline <= 1.0, (
            f"{spec.name} baseline {spec.random_baseline} out of [0,1]"
        )


def test_task_specs_known_baselines():
    """Spot-check baselines against DCLM's eval_meta_data.csv."""
    assert TASK_SPECS["hellaswag_zeroshot"].random_baseline == 0.25
    assert TASK_SPECS["arc_easy"].random_baseline == 0.25
    assert TASK_SPECS["piqa"].random_baseline == 0.50
    assert TASK_SPECS["winogrande"].random_baseline == 0.50
    assert TASK_SPECS["boolq"].random_baseline == 0.62  # majority-class baseline
    assert TASK_SPECS["lambada_openai"].random_baseline == 0.0
    assert TASK_SPECS["commonsense_qa"].random_baseline == pytest.approx(0.403)


# ----------------------------------------------------------------------------
# Raw-row → Example conversion
# ----------------------------------------------------------------------------


def _char_tokenize(text: str) -> list[int]:
    """Trivial deterministic tokenizer for tests: each character → its ord."""
    return [ord(c) for c in text]


def test_make_mc_example_round_trip():
    row = MCRawRow(query="Q?", choices=["a", "bb", "ccc"], gold=1)
    ex = make_mc_example(row, _char_tokenize)
    assert ex.context_ids == [ord("Q"), ord("?")]
    assert ex.choice_ids == [[ord("a")], [ord("b"), ord("b")], [ord("c"), ord("c"), ord("c")]]
    assert ex.gold == 1


def test_make_schema_example_round_trip():
    row = SchemaRawRow(
        contexts=["one", "two"],
        continuations=["x", "yy"],
        gold=0,
    )
    ex = make_schema_example(row, _char_tokenize)
    assert ex.context_ids == [[ord("o"), ord("n"), ord("e")], [ord("t"), ord("w"), ord("o")]]
    assert ex.continuation_ids == [[ord("x")], [ord("y"), ord("y")]]
    assert ex.gold == 0


def test_make_lm_example_round_trip():
    row = LMRawRow(context="abc", target="d")
    ex = make_lm_example(row, _char_tokenize)
    assert ex.context_ids == [ord("a"), ord("b"), ord("c")]
    assert ex.target_ids == [ord("d")]


# ----------------------------------------------------------------------------
# evaluate_mc_task / evaluate_schema_task / evaluate_lm_task_lambada
# ----------------------------------------------------------------------------


VOCAB = 16


def _uniform_forward(input_ids: torch.Tensor) -> torch.Tensor:
    """Returns uniform logits of shape (1, T, V) for any input."""
    return torch.zeros((1, input_ids.size(1), VOCAB))


def _favor_token_forward(token: int, position: int):
    """Returns a forward fn that boosts `token` at `position` in the
    log-softmax — used to deterministically pick a specific choice."""

    def fwd(input_ids: torch.Tensor) -> torch.Tensor:
        T = input_ids.size(1)
        logits = torch.zeros((1, T, VOCAB))
        if 0 <= position < T:
            logits[0, position, token] = 10.0
        return logits

    return fwd


def test_evaluate_mc_task_empty_returns_zero():
    """Empty example list → (0.0, 0). No division-by-zero."""
    acc, n = evaluate_mc_task(_uniform_forward, [])
    assert acc == 0.0
    assert n == 0


def test_evaluate_mc_task_uniform_logits_picks_choice_zero():
    """All choices tied → argmax tie-break picks index 0 → only the
    examples with gold==0 are 'correct'."""
    examples = [
        MCExample(context_ids=[1], choice_ids=[[2], [3]], gold=0),
        MCExample(context_ids=[4], choice_ids=[[5], [6]], gold=1),
        MCExample(context_ids=[7], choice_ids=[[8], [9]], gold=0),
    ]
    acc, n = evaluate_mc_task(_uniform_forward, examples)
    # All examples score uniform → choice 0 wins → 2 of 3 correct
    assert acc == pytest.approx(2 / 3)
    assert n == 3


def test_evaluate_mc_task_perfect_accuracy():
    """When the forward function favors EXACTLY the gold-choice input
    (not just any choice), accuracy is 1.0."""
    # 2 examples each with 2 choices. Gold continuation tokens are 5 and 11.
    examples = [
        MCExample(context_ids=[1], choice_ids=[[2], [5]], gold=1),
        MCExample(context_ids=[1], choice_ids=[[11], [2]], gold=0),
    ]
    # GOLD inputs are [1, 5] (ex0 gold) and [1, 11] (ex1 gold);
    # WRONG inputs are both [1, 2]. The forward fn only boosts the gold
    # inputs so the wrong choices score at the uniform-logits baseline.
    def gold_only_forward(input_ids: torch.Tensor) -> torch.Tensor:
        ids = tuple(input_ids.flatten().tolist())
        T = input_ids.size(1)
        logits = torch.zeros((1, T, VOCAB))
        if ids == (1, 5):
            logits[0, 0, 5] = 10.0
        elif ids == (1, 11):
            logits[0, 0, 11] = 10.0
        return logits

    acc, n = evaluate_mc_task(gold_only_forward, examples)
    assert acc == 1.0
    assert n == 2


def test_evaluate_schema_task_empty():
    acc, n = evaluate_schema_task(_uniform_forward, [])
    assert acc == 0.0
    assert n == 0


def test_evaluate_schema_task_picks_higher_logprob_variant():
    """Schema variants have different (context, continuation) pairs.
    The forward function favors variant 1's continuation."""
    ex = SchemaExample(
        context_ids=[[1, 2], [3, 4]],
        continuation_ids=[[5], [6]],
        gold=1,
    )

    def smart_forward(input_ids: torch.Tensor) -> torch.Tensor:
        ids = tuple(input_ids.flatten().tolist())
        T = input_ids.size(1)
        logits = torch.zeros((1, T, VOCAB))
        # Boost the variant-1 path: input [3,4,6] gets a sharper logit at pos 1
        if ids == (3, 4, 6):
            logits[0, 1, 6] = 8.0
        return logits

    acc, n = evaluate_schema_task(smart_forward, [ex])
    assert acc == 1.0
    assert n == 1


def test_evaluate_lm_task_lambada_returns_mean_nll():
    """For 2 examples with deterministic uniform logits, mean NLL equals
    per-token-NLL × mean-target-length."""
    examples = [
        LMExample(context_ids=[1], target_ids=[2]),       # 1 target token
        LMExample(context_ids=[1, 2], target_ids=[3, 4]), # 2 target tokens
    ]
    mean_nll, n = evaluate_lm_task_lambada(_uniform_forward, examples)
    # Uniform → per-token NLL = log(VOCAB)
    # Example 0 NLL = 1 * log(VOCAB); Example 1 NLL = 2 * log(VOCAB)
    # Mean = 1.5 * log(VOCAB)
    assert mean_nll == pytest.approx(1.5 * math.log(VOCAB), rel=1e-5)
    assert n == 2


# ----------------------------------------------------------------------------
# to_cell_result wrapper
# ----------------------------------------------------------------------------


def test_to_cell_result_default_scale_and_seed():
    cr = to_cell_result("arc_easy", 0.42, 100)
    assert isinstance(cr, CellResult)
    assert cr.task == "arc_easy"
    assert cr.accuracy == 0.42
    assert cr.n_examples == 100
    assert cr.accuracy_stderr == 0.0
    assert cr.seed == 0


def test_to_cell_result_custom_scale_seed():
    cr = to_cell_result("mmlu" if False else "lambada_openai", 0.31, 50, scale="S2", seed=7)
    assert cr.seed == 7


def test_to_cell_result_rejects_unknown_task():
    with pytest.raises(ValueError, match=r"unknown task"):
        to_cell_result("not_a_real_task", 0.5, 10)


# ----------------------------------------------------------------------------
# load_task_examples — canonical JSONL parser
# ----------------------------------------------------------------------------


def _write_jsonl(path, rows):
    """Helper: write a list of dicts as a JSONL file."""
    import json as _json
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")


def test_load_task_examples_unknown_task_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"unknown task"):
        load_task_examples(tmp_path, "not_a_real_task")


def test_load_task_examples_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"CORE-22 task file"):
        load_task_examples(tmp_path, "arc_easy")


def test_load_task_examples_mc_happy_path(tmp_path):
    """An MC task (arc_easy) parses to MCRawRow list."""
    _write_jsonl(tmp_path / "arc_easy.jsonl", [
        {"id": "a1", "query": "Q1?", "choices": ["A", "B", "C", "D"], "gold": 0},
        {"id": "a2", "query": "Q2?", "choices": ["W", "X"], "gold": 1},
    ])
    rows = load_task_examples(tmp_path, "arc_easy")
    assert len(rows) == 2
    assert all(isinstance(r, MCRawRow) for r in rows)
    assert rows[0].query == "Q1?"
    assert rows[0].choices == ["A", "B", "C", "D"]
    assert rows[0].gold == 0
    assert rows[1].gold == 1


def test_load_task_examples_schema_happy_path(tmp_path):
    """A schema task (winogrande) parses to SchemaRawRow list."""
    _write_jsonl(tmp_path / "winogrande.jsonl", [
        {"id": "w1",
         "contexts": ["The trophy didn't fit in the brown suitcase because it",
                      "The trophy didn't fit in the brown suitcase because it"],
         "continuations": ["was too big", "was too small"],
         "gold": 0},
    ])
    rows = load_task_examples(tmp_path, "winogrande")
    assert len(rows) == 1
    assert isinstance(rows[0], SchemaRawRow)
    assert len(rows[0].contexts) == 2
    assert rows[0].gold == 0


def test_load_task_examples_lm_happy_path(tmp_path):
    """An LM task (lambada_openai) parses to LMRawRow list."""
    _write_jsonl(tmp_path / "lambada_openai.jsonl", [
        {"id": "l1", "context": "Once upon a", "target": " time"},
        {"id": "l2", "context": "The quick brown", "target": " fox",
         "accept_set": [" fox", " Fox"]},
    ])
    rows = load_task_examples(tmp_path, "lambada_openai")
    assert len(rows) == 2
    assert all(isinstance(r, LMRawRow) for r in rows)
    assert rows[0].context == "Once upon a"
    assert rows[0].target == " time"
    assert rows[0].accept_set == ()
    assert rows[1].accept_set == (" fox", " Fox")


def test_load_task_examples_skips_blank_lines(tmp_path):
    """Blank lines in the JSONL are skipped, not parsed as bad JSON."""
    path = tmp_path / "arc_easy.jsonl"
    path.write_text(
        '{"id": "a", "query": "q", "choices": ["x", "y"], "gold": 0}\n'
        '\n'
        '   \n'
        '{"id": "b", "query": "q", "choices": ["x", "y"], "gold": 1}\n'
    )
    rows = load_task_examples(tmp_path, "arc_easy")
    assert len(rows) == 2


def test_load_task_examples_invalid_json_raises(tmp_path):
    """A bad line raises with the line number."""
    path = tmp_path / "arc_easy.jsonl"
    path.write_text(
        '{"id": "a", "query": "q", "choices": ["x", "y"], "gold": 0}\n'
        '{not valid json\n'
    )
    with pytest.raises(ValueError, match=r":2"):
        load_task_examples(tmp_path, "arc_easy")


def test_load_task_examples_mc_missing_field_raises(tmp_path):
    _write_jsonl(tmp_path / "arc_easy.jsonl", [
        {"id": "a", "query": "q", "choices": ["x", "y"]},  # no 'gold'
    ])
    with pytest.raises(ValueError, match=r"canonical MC schema"):
        load_task_examples(tmp_path, "arc_easy")


def test_load_task_examples_mc_out_of_range_gold_raises(tmp_path):
    _write_jsonl(tmp_path / "arc_easy.jsonl", [
        {"id": "a", "query": "q", "choices": ["x", "y"], "gold": 5},
    ])
    with pytest.raises(ValueError, match=r"gold=5 out of range"):
        load_task_examples(tmp_path, "arc_easy")


def test_load_task_examples_mc_non_string_choice_raises(tmp_path):
    _write_jsonl(tmp_path / "arc_easy.jsonl", [
        {"id": "a", "query": "q", "choices": ["x", 42], "gold": 0},
    ])
    with pytest.raises(ValueError, match=r"list of strings"):
        load_task_examples(tmp_path, "arc_easy")


def test_load_task_examples_schema_length_mismatch_raises(tmp_path):
    _write_jsonl(tmp_path / "winogrande.jsonl", [
        {"id": "w", "contexts": ["a", "b"], "continuations": ["x"], "gold": 0},
    ])
    with pytest.raises(ValueError, match=r"length mismatch"):
        load_task_examples(tmp_path, "winogrande")


def test_load_task_examples_lm_missing_context_raises(tmp_path):
    _write_jsonl(tmp_path / "lambada_openai.jsonl", [
        {"id": "l", "target": " time"},  # missing 'context'
    ])
    with pytest.raises(ValueError, match=r"canonical LM schema"):
        load_task_examples(tmp_path, "lambada_openai")


def test_load_task_examples_lm_bad_accept_set_type_raises(tmp_path):
    _write_jsonl(tmp_path / "lambada_openai.jsonl", [
        {"id": "l", "context": "x", "target": "y", "accept_set": "not a list"},
    ])
    with pytest.raises(ValueError, match=r"accept_set.*list"):
        load_task_examples(tmp_path, "lambada_openai")
