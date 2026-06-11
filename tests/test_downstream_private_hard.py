"""Tests for the private hardness subset (B1).

Pins the 4-task registry, the HF dataset identifiers, the HardnessIndex
contract, the bottom-quintile selection logic, and the per-task evaluator
dispatch. Covers the loader stub's NotImplementedError + the
to_private_hard_cell_result wrapper.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from eval.downstream import (
    POOL_PRIVATE_HARD,
    CellResult,
)
from eval.downstream.core22 import (
    MCRawRow,
    SchemaRawRow,
)
from eval.downstream.private_hard import (
    HF_DATASET_IDS,
    PRIVATE_HARD_TASK_SPECS,
    PRIVATE_HARD_TASKS,
    HardnessIndex,
    HardnessIndexRow,
    evaluate_private_hard_task,
    load_task_examples,
    select_hardness_subset,
    to_private_hard_cell_result,
)

# ----------------------------------------------------------------------------
# Pinned registry
# ----------------------------------------------------------------------------


def test_private_hard_task_count():
    """Exactly 4 tasks, per docs/license/hardness_subset_decision.md."""
    assert len(PRIVATE_HARD_TASKS) == 4


def test_private_hard_task_names_pinned():
    """Names locked to the license decision's pool."""
    assert PRIVATE_HARD_TASKS == (
        "arc_challenge_hard",
        "winogrande_hard",
        "tiny_arc",
        "tiny_mmlu",
    )


def test_private_hard_no_openbookqa():
    """OpenBookQA pre-swapped out (CC-BY-NC-SA-4.0 incompatible with
    Karpa's commercial emission use)."""
    assert "openbook_qa" not in PRIVATE_HARD_TASKS
    assert "openbookqa" not in PRIVATE_HARD_TASKS


def test_private_hard_no_sciq():
    """SciQ pre-swapped out (CC-BY-NC-3.0)."""
    assert "sciq" not in PRIVATE_HARD_TASKS


# ----------------------------------------------------------------------------
# TASK_SPECS
# ----------------------------------------------------------------------------


def test_task_specs_keys_match_task_list():
    assert set(PRIVATE_HARD_TASK_SPECS.keys()) == set(PRIVATE_HARD_TASKS)


def test_task_specs_all_in_private_hard_pool():
    for spec in PRIVATE_HARD_TASK_SPECS.values():
        assert spec.pool == POOL_PRIVATE_HARD


def test_task_specs_modes():
    """ARC variants + tinyMMLU = mc; winogrande_hard = schema (matches
    DCLM's winogrande mode in core22)."""
    assert PRIVATE_HARD_TASK_SPECS["arc_challenge_hard"].mode == "mc"
    assert PRIVATE_HARD_TASK_SPECS["winogrande_hard"].mode == "schema"
    assert PRIVATE_HARD_TASK_SPECS["tiny_arc"].mode == "mc"
    assert PRIVATE_HARD_TASK_SPECS["tiny_mmlu"].mode == "mc"


def test_task_specs_random_baselines():
    """ARC + tinyMMLU baselines = 0.25 (4-choice); winogrande = 0.50
    (2-choice schema)."""
    assert PRIVATE_HARD_TASK_SPECS["arc_challenge_hard"].random_baseline == 0.25
    assert PRIVATE_HARD_TASK_SPECS["winogrande_hard"].random_baseline == 0.50
    assert PRIVATE_HARD_TASK_SPECS["tiny_arc"].random_baseline == 0.25
    assert PRIVATE_HARD_TASK_SPECS["tiny_mmlu"].random_baseline == 0.25


# ----------------------------------------------------------------------------
# HF_DATASET_IDS
# ----------------------------------------------------------------------------


def test_hf_dataset_ids_cover_every_task():
    assert set(HF_DATASET_IDS.keys()) == set(PRIVATE_HARD_TASKS)


def test_hf_dataset_ids_pin_canonical_paths():
    """Pin the exact HF repo IDs the loader will pull from. If HF rotates
    a repo, this test must be updated in lock-step with the loader."""
    assert HF_DATASET_IDS["arc_challenge_hard"] == ("allenai/ai2_arc", "ARC-Challenge")
    assert HF_DATASET_IDS["winogrande_hard"] == ("allenai/winogrande", "winogrande_xl")
    assert HF_DATASET_IDS["tiny_arc"] == ("tinyBenchmarks/tinyArc", None)
    assert HF_DATASET_IDS["tiny_mmlu"] == ("tinyBenchmarks/tinyMMLU", None)


# ----------------------------------------------------------------------------
# HardnessIndex + HardnessIndexRow
# ----------------------------------------------------------------------------


def test_hardness_index_default_empty():
    idx = HardnessIndex(version="v0")
    assert idx.rows == []
    assert idx.for_task("arc_challenge_hard") == set()


def test_hardness_index_for_task_filters_by_dataset():
    rows = [
        HardnessIndexRow(dataset="arc_challenge_hard", item_id="a1", gold_margin_bits=0.5),
        HardnessIndexRow(dataset="arc_challenge_hard", item_id="a2", gold_margin_bits=0.3),
        HardnessIndexRow(dataset="winogrande_hard",    item_id="w1", gold_margin_bits=0.2),
    ]
    idx = HardnessIndex(version="v1", rows=rows)
    assert idx.for_task("arc_challenge_hard") == {"a1", "a2"}
    assert idx.for_task("winogrande_hard") == {"w1"}
    assert idx.for_task("tiny_arc") == set()


def test_hardness_index_version_required():
    """No default version — every emit MUST carry a version string from
    grader.py so a future auditor can re-derive the selection."""
    idx = HardnessIndex(version="grader-v0.0.1-abc123")
    assert idx.version == "grader-v0.0.1-abc123"


# ----------------------------------------------------------------------------
# select_hardness_subset
# ----------------------------------------------------------------------------


def _mc_rows(items: list[tuple[str, str]]) -> list:
    """Build (item_id, MCRawRow) pairs from (id, query) tuples."""
    return [
        (item_id, MCRawRow(query=q, choices=["a", "b", "c", "d"], gold=0))
        for item_id, q in items
    ]


def test_select_filters_to_indexed_item_ids():
    rows = _mc_rows([("a1", "q1"), ("a2", "q2"), ("a3", "q3")])
    idx = HardnessIndex(
        version="v1",
        rows=[
            HardnessIndexRow(dataset="arc_challenge_hard", item_id="a1", gold_margin_bits=0.1),
            HardnessIndexRow(dataset="arc_challenge_hard", item_id="a3", gold_margin_bits=0.2),
        ],
    )
    selected = select_hardness_subset(rows, idx, "arc_challenge_hard")
    assert len(selected) == 2
    queries = [r.query for r in selected]
    assert queries == ["q1", "q3"]  # original order preserved


def test_select_empty_index_returns_empty():
    rows = _mc_rows([("a1", "q1")])
    idx = HardnessIndex(version="v1")
    assert select_hardness_subset(rows, idx, "arc_challenge_hard") == []


def test_select_empty_rows_returns_empty():
    idx = HardnessIndex(
        version="v1",
        rows=[HardnessIndexRow(dataset="arc_challenge_hard", item_id="a1", gold_margin_bits=0.1)],
    )
    assert select_hardness_subset([], idx, "arc_challenge_hard") == []


def test_select_unknown_task_returns_empty():
    rows = _mc_rows([("a1", "q1")])
    idx = HardnessIndex(version="v1", rows=[])
    assert select_hardness_subset(rows, idx, "not_a_real_task") == []


# ----------------------------------------------------------------------------
# evaluate_private_hard_task
# ----------------------------------------------------------------------------


VOCAB = 256  # Covers ord(c) for all ASCII chars, used by _char_tokenize.


def _char_tokenize(text: str) -> list[int]:
    return [ord(c) for c in text]


def _uniform_forward(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, input_ids.size(1), VOCAB))


def test_evaluate_private_hard_task_mc_path():
    """MC task → routes through evaluate_mc_task. With uniform logits +
    gold=0 across all examples, accuracy = 1.0 (argmax tie picks index 0)."""
    rows = _mc_rows([("a1", "q1"), ("a2", "q2"), ("a3", "q3")])
    idx = HardnessIndex(
        version="v1",
        rows=[
            HardnessIndexRow(dataset="arc_challenge_hard", item_id="a1", gold_margin_bits=0.0),
            HardnessIndexRow(dataset="arc_challenge_hard", item_id="a2", gold_margin_bits=0.1),
        ],
    )
    acc, n = evaluate_private_hard_task(
        _uniform_forward, rows, idx, "arc_challenge_hard", _char_tokenize,
    )
    assert n == 2
    # All gold=0, uniform → ties → tie-break picks 0 → all correct
    assert acc == 1.0


def test_evaluate_private_hard_task_schema_path():
    """Schema task → routes through evaluate_schema_task."""
    rows = [
        ("w1", SchemaRawRow(contexts=["a"], continuations=["b"], gold=0)),
        ("w2", SchemaRawRow(contexts=["c"], continuations=["d"], gold=0)),
    ]
    idx = HardnessIndex(
        version="v1",
        rows=[
            HardnessIndexRow(dataset="winogrande_hard", item_id="w1", gold_margin_bits=0.05),
        ],
    )
    acc, n = evaluate_private_hard_task(
        _uniform_forward, rows, idx, "winogrande_hard", _char_tokenize,
    )
    # Only w1 selected. 1 variant → trivially scoring picks 0 → matches gold
    assert n == 1
    assert acc == 1.0


def test_evaluate_private_hard_task_empty_index_returns_zero():
    """No index entries for task → filter yields zero rows → (0.0, 0)."""
    rows = _mc_rows([("a1", "q1")])
    idx = HardnessIndex(version="v1")
    acc, n = evaluate_private_hard_task(
        _uniform_forward, rows, idx, "arc_challenge_hard", _char_tokenize,
    )
    assert acc == 0.0
    assert n == 0


def test_evaluate_private_hard_task_rejects_unknown_task():
    rows = _mc_rows([("a1", "q1")])
    idx = HardnessIndex(version="v1")
    with pytest.raises(ValueError, match=r"unknown private-hard task"):
        evaluate_private_hard_task(
            _uniform_forward, rows, idx, "not_a_real_task", _char_tokenize,
        )


# ----------------------------------------------------------------------------
# to_private_hard_cell_result
# ----------------------------------------------------------------------------


def test_to_private_hard_cell_result_happy():
    cr = to_private_hard_cell_result("tiny_arc", 0.78, 100)
    assert isinstance(cr, CellResult)
    assert cr.task == "tiny_arc"
    assert cr.accuracy == 0.78
    assert cr.n_examples == 100
    assert cr.seed == 0


def test_to_private_hard_cell_result_rejects_unknown_task():
    with pytest.raises(ValueError, match=r"unknown private-hard task"):
        to_private_hard_cell_result("mmlu", 0.5, 10)


def test_to_private_hard_cell_result_rejects_core22_task_name():
    """Even a valid CORE-22 task name gets rejected — pool routing is by
    name, not by mode. The runner constructs cell keys based on which
    helper it calls; cross-pool name leakage would silently misroute."""
    with pytest.raises(ValueError, match=r"unknown private-hard task"):
        to_private_hard_cell_result("hellaswag", 0.5, 10)


# ----------------------------------------------------------------------------
# load_task_examples — canonical JSONL parser
# ----------------------------------------------------------------------------


def _write_jsonl(path, rows):
    """Helper: write a list of dicts as a JSONL file."""
    import json as _json
    path.write_text("\n".join(_json.dumps(r) for r in rows) + "\n")


def test_load_task_examples_unknown_task_rejected(tmp_path):
    with pytest.raises(ValueError, match=r"unknown private-hard task"):
        load_task_examples(tmp_path, "not_a_real_task")


def test_load_task_examples_missing_file_raises(tmp_path):
    """Missing file raises with the upstream HF dataset id in the
    message so the operator knows what to download."""
    with pytest.raises(FileNotFoundError, match=r"ai2_arc"):
        load_task_examples(tmp_path, "arc_challenge_hard")


def test_load_task_examples_mc_happy_path(tmp_path):
    """ARC-Challenge-hard parses to (id, MCRawRow) pairs."""
    _write_jsonl(tmp_path / "arc_challenge_hard.jsonl", [
        {"id": "ARC-456", "query": "Q1?", "choices": ["A", "B", "C", "D"], "gold": 0},
        {"id": "ARC-789", "query": "Q2?", "choices": ["W", "X"], "gold": 1},
    ])
    rows = load_task_examples(tmp_path, "arc_challenge_hard")
    assert len(rows) == 2
    item_id_0, row_0 = rows[0]
    assert item_id_0 == "ARC-456"
    assert isinstance(row_0, MCRawRow)
    assert row_0.query == "Q1?"
    assert row_0.gold == 0
    item_id_1, row_1 = rows[1]
    assert item_id_1 == "ARC-789"
    assert row_1.gold == 1


def test_load_task_examples_schema_happy_path(tmp_path):
    """winogrande-hard parses to (id, SchemaRawRow) pairs."""
    _write_jsonl(tmp_path / "winogrande_hard.jsonl", [
        {"id": "wg-1",
         "contexts": ["The X did not Y because it", "The X did not Y because it"],
         "continuations": ["was Z", "was W"],
         "gold": 0},
    ])
    rows = load_task_examples(tmp_path, "winogrande_hard")
    assert len(rows) == 1
    item_id, row = rows[0]
    assert item_id == "wg-1"
    assert isinstance(row, SchemaRawRow)
    assert row.gold == 0


def test_load_task_examples_tiny_arc(tmp_path):
    """tiny_arc routes through MC dispatch (mode='mc')."""
    _write_jsonl(tmp_path / "tiny_arc.jsonl", [
        {"id": "t1", "query": "Q?", "choices": ["a", "b", "c", "d"], "gold": 2},
    ])
    rows = load_task_examples(tmp_path, "tiny_arc")
    assert len(rows) == 1
    item_id, row = rows[0]
    assert item_id == "t1"
    assert row.gold == 2


def test_load_task_examples_tiny_mmlu(tmp_path):
    """tiny_mmlu routes through MC dispatch."""
    _write_jsonl(tmp_path / "tiny_mmlu.jsonl", [
        {"id": "m1", "query": "Q?", "choices": ["a", "b", "c", "d"], "gold": 1},
    ])
    rows = load_task_examples(tmp_path, "tiny_mmlu")
    assert len(rows) == 1
    item_id, row = rows[0]
    assert item_id == "m1"
    assert row.gold == 1


def test_load_task_examples_skips_blank_lines(tmp_path):
    path = tmp_path / "arc_challenge_hard.jsonl"
    path.write_text(
        '{"id": "a", "query": "q", "choices": ["x", "y"], "gold": 0}\n'
        '\n'
        '   \n'
        '{"id": "b", "query": "q", "choices": ["x", "y"], "gold": 1}\n'
    )
    rows = load_task_examples(tmp_path, "arc_challenge_hard")
    assert len(rows) == 2


def test_load_task_examples_invalid_json_raises(tmp_path):
    path = tmp_path / "arc_challenge_hard.jsonl"
    path.write_text(
        '{"id": "a", "query": "q", "choices": ["x", "y"], "gold": 0}\n'
        '{not valid json\n'
    )
    with pytest.raises(ValueError, match=r":2"):
        load_task_examples(tmp_path, "arc_challenge_hard")


def test_load_task_examples_missing_id_raises(tmp_path):
    """Private-hard requires `id` on every row (HardnessIndex routing)."""
    _write_jsonl(tmp_path / "arc_challenge_hard.jsonl", [
        {"query": "q", "choices": ["x", "y"], "gold": 0},  # no id
    ])
    with pytest.raises(ValueError, match=r"missing required 'id'"):
        load_task_examples(tmp_path, "arc_challenge_hard")


def test_load_task_examples_empty_id_raises(tmp_path):
    _write_jsonl(tmp_path / "arc_challenge_hard.jsonl", [
        {"id": "", "query": "q", "choices": ["x", "y"], "gold": 0},
    ])
    with pytest.raises(ValueError, match=r"non-empty string"):
        load_task_examples(tmp_path, "arc_challenge_hard")


def test_load_task_examples_non_string_id_raises(tmp_path):
    _write_jsonl(tmp_path / "arc_challenge_hard.jsonl", [
        {"id": 42, "query": "q", "choices": ["x", "y"], "gold": 0},
    ])
    with pytest.raises(ValueError, match=r"non-empty string"):
        load_task_examples(tmp_path, "arc_challenge_hard")


def test_load_task_examples_mc_out_of_range_gold_raises(tmp_path):
    _write_jsonl(tmp_path / "arc_challenge_hard.jsonl", [
        {"id": "a", "query": "q", "choices": ["x", "y"], "gold": 5},
    ])
    with pytest.raises(ValueError, match=r"gold=5 out of range"):
        load_task_examples(tmp_path, "arc_challenge_hard")


def test_load_task_examples_schema_length_mismatch_raises(tmp_path):
    _write_jsonl(tmp_path / "winogrande_hard.jsonl", [
        {"id": "w", "contexts": ["a", "b"], "continuations": ["x"], "gold": 0},
    ])
    with pytest.raises(ValueError, match=r"length mismatch"):
        load_task_examples(tmp_path, "winogrande_hard")


def test_load_task_examples_output_routes_through_evaluate(tmp_path):
    """End-to-end: the loaded rows are the right shape for
    evaluate_private_hard_task + a HardnessIndex."""
    _write_jsonl(tmp_path / "arc_challenge_hard.jsonl", [
        {"id": "a", "query": "q", "choices": ["x", "y"], "gold": 0},
        {"id": "b", "query": "q", "choices": ["x", "y"], "gold": 0},
    ])
    rows = load_task_examples(tmp_path, "arc_challenge_hard")
    idx = HardnessIndex(
        version="v1",
        rows=[HardnessIndexRow(
            dataset="arc_challenge_hard",
            item_id="a",
            gold_margin_bits=0.0,
        )],
    )
    selected = select_hardness_subset(rows, idx, "arc_challenge_hard")
    # Only item_id "a" is in the index → 1 row.
    assert len(selected) == 1
    assert selected[0].query == "q"
