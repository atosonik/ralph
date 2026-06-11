"""Tests for grader.py — gold-margin computation, bottom-quintile selection,
index assembly, and JSONL round-trip.

20-ish cases cover the math (margin sign, unit conversion, edge cases),
the deterministic sort+slice, the index merge, and the JSONL header
+ format-mismatch rejection.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from eval.downstream.core22 import (
    MCRawRow,
    SchemaRawRow,
)
from eval.downstream.grader import (
    WRITER_FORMAT,
    assemble_hardness_index,
    compute_bottom_quintile,
    gold_margin_bits,
    grade_mc_task,
    grade_schema_task,
    read_hardness_index_jsonl,
    write_hardness_index_jsonl,
)
from eval.downstream.private_hard import (
    HardnessIndex,
    HardnessIndexRow,
)

# ----------------------------------------------------------------------------
# gold_margin_bits
# ----------------------------------------------------------------------------


def test_gold_margin_empty_returns_zero():
    assert gold_margin_bits([], 0) == 0.0


def test_gold_margin_single_choice_returns_inf():
    """Only one choice → no distractors → infinitely confident."""
    assert gold_margin_bits([-2.0], 0) == float("inf")


def test_gold_margin_simple_math():
    """log_p values [-1.0, -3.0] in nats, gold=0:
       margin_nats = -1.0 - (-3.0) = 2.0
       margin_bits = 2.0 / log(2) ≈ 2.885"""
    m = gold_margin_bits([-1.0, -3.0], gold=0)
    assert m == pytest.approx(2.0 / math.log(2), rel=1e-6)


def test_gold_margin_negative_when_distractor_beats_gold():
    """log_p values [-3.0, -1.0], gold=0 → -2.0 nats → ≈-2.885 bits."""
    m = gold_margin_bits([-3.0, -1.0], gold=0)
    assert m < 0
    assert m == pytest.approx(-2.0 / math.log(2), rel=1e-6)


def test_gold_margin_max_over_distractors():
    """With 3 distractors at [-1, -2, -3], the best (largest) is -1.
       margin = log_p(gold) - (-1) = log_p(gold) + 1."""
    m = gold_margin_bits([-1.0, -2.0, -3.0, 0.0], gold=3)
    # gold log_p = 0.0; max distractor = -1.0
    # margin_nats = 0.0 - (-1.0) = 1.0; bits = 1/log(2)
    assert m == pytest.approx(1.0 / math.log(2), rel=1e-6)


def test_gold_margin_rejects_gold_out_of_range():
    with pytest.raises(ValueError, match=r"gold=5 out of range"):
        gold_margin_bits([-1.0, -2.0], gold=5)
    with pytest.raises(ValueError, match=r"gold=-1 out of range"):
        gold_margin_bits([-1.0, -2.0], gold=-1)


# ----------------------------------------------------------------------------
# grade_mc_task / grade_schema_task with synthetic logits
# ----------------------------------------------------------------------------


VOCAB = 256


def _char_tokenize(text: str) -> list[int]:
    return [ord(c) for c in text]


def _uniform_forward(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, input_ids.size(1), VOCAB))


def test_grade_mc_task_empty_returns_empty():
    assert grade_mc_task(_uniform_forward, [], _char_tokenize, "arc_challenge_hard") == []


def test_grade_mc_task_uniform_logits_zero_margin():
    """Uniform logits → all choices have identical log-probs → margin = 0
    bits regardless of gold index."""
    items = [
        ("i1", MCRawRow(query="q", choices=["a", "b"], gold=0)),
        ("i2", MCRawRow(query="q", choices=["a", "b"], gold=1)),
    ]
    rows = grade_mc_task(_uniform_forward, items, _char_tokenize, "arc_challenge_hard")
    assert len(rows) == 2
    for r in rows:
        assert r.dataset == "arc_challenge_hard"
        assert r.gold_margin_bits == pytest.approx(0.0, abs=1e-9)
    assert [r.item_id for r in rows] == ["i1", "i2"]


def test_grade_mc_task_preserves_input_order():
    """grade_mc_task emits one row per input item in INPUT order — not
    sorted. Sorting happens in compute_bottom_quintile."""
    items = [
        ("z_last", MCRawRow(query="q", choices=["a", "b"], gold=0)),
        ("a_first", MCRawRow(query="q", choices=["a", "b"], gold=0)),
    ]
    rows = grade_mc_task(_uniform_forward, items, _char_tokenize, "arc_challenge_hard")
    assert [r.item_id for r in rows] == ["z_last", "a_first"]


def test_grade_schema_task_uniform_logits_zero_margin():
    items = [
        ("w1", SchemaRawRow(contexts=["a", "b"], continuations=["x", "y"], gold=0)),
    ]
    rows = grade_schema_task(_uniform_forward, items, _char_tokenize, "winogrande_hard")
    assert len(rows) == 1
    assert rows[0].dataset == "winogrande_hard"
    assert rows[0].gold_margin_bits == pytest.approx(0.0, abs=1e-9)


def test_grade_schema_task_empty_returns_empty():
    assert grade_schema_task(_uniform_forward, [], _char_tokenize, "winogrande_hard") == []


# ----------------------------------------------------------------------------
# compute_bottom_quintile
# ----------------------------------------------------------------------------


def _row(item_id: str, margin: float, dataset: str = "arc_challenge_hard") -> HardnessIndexRow:
    return HardnessIndexRow(dataset=dataset, item_id=item_id, gold_margin_bits=margin)


def test_bottom_quintile_takes_lowest_20_percent():
    """10 rows × 20% = 2 rows. The 2 returned have the smallest margins."""
    rows = [_row(f"i{i}", float(i)) for i in range(10)]  # margins 0..9
    bottom = compute_bottom_quintile(rows, 0.20)
    assert len(bottom) == 2
    assert [r.item_id for r in bottom] == ["i0", "i1"]


def test_bottom_quintile_returns_ascending_margin_order():
    """Output is sorted ascending by margin — first row is hardest item."""
    rows = [_row("a", 5.0), _row("b", 1.0), _row("c", 3.0), _row("d", 0.5), _row("e", 4.0)]
    bottom = compute_bottom_quintile(rows, 0.40)  # 5 × 0.40 = 2
    # Margins sorted: [0.5 (d), 1.0 (b), 3.0 (c), 4.0 (e), 5.0 (a)] → take 2
    assert [r.item_id for r in bottom] == ["d", "b"]


def test_bottom_quintile_empty_input_returns_empty():
    assert compute_bottom_quintile([], 0.20) == []


def test_bottom_quintile_zero_fraction_returns_empty():
    rows = [_row("a", 1.0), _row("b", 2.0)]
    assert compute_bottom_quintile(rows, 0.0) == []


def test_bottom_quintile_one_fraction_returns_all_sorted():
    rows = [_row("a", 3.0), _row("b", 1.0), _row("c", 2.0)]
    bottom = compute_bottom_quintile(rows, 1.0)
    assert len(bottom) == 3
    assert [r.item_id for r in bottom] == ["b", "c", "a"]


def test_bottom_quintile_rejects_invalid_fraction():
    rows = [_row("a", 1.0)]
    with pytest.raises(ValueError, match=r"quintile_fraction must be"):
        compute_bottom_quintile(rows, 1.5)
    with pytest.raises(ValueError, match=r"quintile_fraction must be"):
        compute_bottom_quintile(rows, -0.1)


def test_bottom_quintile_deterministic_across_runs():
    """Stable sort guarantees byte-equal outputs on identical inputs."""
    rows = [_row("a", 1.0), _row("b", 2.0), _row("c", 1.0)]  # a, c tie
    r1 = compute_bottom_quintile(rows, 0.67)  # take 2
    r2 = compute_bottom_quintile(rows, 0.67)
    assert r1 == r2
    # Among the tied pair (a, c), the stable sort preserves input order
    assert [r.item_id for r in r1] == ["a", "c"]


def test_bottom_quintile_rounds_to_nearest_integer():
    """N=10, fraction=0.25 → 2.5 → round to 2. N=10, fraction=0.27 → 2.7 → round to 3."""
    rows = [_row(f"i{i}", float(i)) for i in range(10)]
    assert len(compute_bottom_quintile(rows, 0.25)) == 2
    assert len(compute_bottom_quintile(rows, 0.27)) == 3


# ----------------------------------------------------------------------------
# assemble_hardness_index
# ----------------------------------------------------------------------------


def test_assemble_index_applies_quintile_per_task():
    """Each task contributes its OWN bottom-quintile slice; the index
    merge does NOT compute a global quintile across all tasks."""
    arc_rows = [_row(f"a{i}", float(i), dataset="arc_challenge_hard") for i in range(10)]
    win_rows = [_row(f"w{i}", float(i), dataset="winogrande_hard") for i in range(10)]
    idx = assemble_hardness_index(
        {"arc_challenge_hard": arc_rows, "winogrande_hard": win_rows},
        version="grader-v0.1",
        quintile_fraction=0.20,
    )
    # 2 rows per task × 2 tasks = 4 total
    assert len(idx.rows) == 4
    assert idx.for_task("arc_challenge_hard") == {"a0", "a1"}
    assert idx.for_task("winogrande_hard") == {"w0", "w1"}


def test_assemble_index_carries_version():
    idx = assemble_hardness_index({}, version="grader-v0.0.1-abc123")
    assert idx.version == "grader-v0.0.1-abc123"


def test_assemble_index_rejects_empty_version():
    with pytest.raises(ValueError, match=r"version must be non-empty"):
        assemble_hardness_index({}, version="")


def test_assemble_index_skips_empty_task():
    """A task with zero rows contributes zero to the merged index."""
    idx = assemble_hardness_index(
        {"arc_challenge_hard": [], "tiny_arc": [_row("t1", 0.5, dataset="tiny_arc")]},
        version="v",
        quintile_fraction=1.0,
    )
    assert idx.for_task("arc_challenge_hard") == set()
    assert idx.for_task("tiny_arc") == {"t1"}


def test_assemble_index_coerces_dataset_field_from_key():
    """If a caller passes rows whose `dataset` field doesn't match the dict
    key, the assembler re-stamps the field to match the key. This is the
    correct invariant because the dict key is the source of truth."""
    bogus_rows = [_row("x1", 0.5, dataset="wrong_dataset")]
    idx = assemble_hardness_index(
        {"arc_challenge_hard": bogus_rows},
        version="v",
        quintile_fraction=1.0,
    )
    assert idx.for_task("arc_challenge_hard") == {"x1"}
    assert idx.for_task("wrong_dataset") == set()


# ----------------------------------------------------------------------------
# JSONL round-trip
# ----------------------------------------------------------------------------


def test_write_then_read_round_trip(tmp_path):
    rows = [
        HardnessIndexRow(dataset="arc_challenge_hard", item_id="a1", gold_margin_bits=0.1),
        HardnessIndexRow(dataset="winogrande_hard",    item_id="w1", gold_margin_bits=0.2),
    ]
    idx_in = HardnessIndex(version="grader-v0.0.1", rows=rows)
    path = tmp_path / "out.jsonl"
    write_hardness_index_jsonl(idx_in, path)
    idx_out = read_hardness_index_jsonl(path)
    assert idx_out.version == idx_in.version
    assert len(idx_out.rows) == 2
    assert idx_out.rows[0] == rows[0]
    assert idx_out.rows[1] == rows[1]


def test_write_creates_parent_dirs(tmp_path):
    """Caller may pass a path whose parent doesn't exist yet — write
    creates it. (Mirrors the master plan's atomic-write expectation.)"""
    idx = HardnessIndex(version="v", rows=[])
    path = tmp_path / "nested" / "dir" / "out.jsonl"
    write_hardness_index_jsonl(idx, path)
    assert path.exists()


def test_write_is_atomic_no_tmp_leftover(tmp_path):
    """After a successful write, the .tmp file is gone (rename replaced
    the destination)."""
    idx = HardnessIndex(version="v", rows=[])
    path = tmp_path / "out.jsonl"
    write_hardness_index_jsonl(idx, path)
    assert path.exists()
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert not tmp.exists()


def test_read_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ValueError, match=r"empty hardness-index"):
        read_hardness_index_jsonl(path)


def test_read_rejects_missing_meta_marker(tmp_path):
    path = tmp_path / "bogus.jsonl"
    path.write_text('{"format": "jsonl-v1", "version": "v"}\n')
    with pytest.raises(ValueError, match=r"_meta marker"):
        read_hardness_index_jsonl(path)


def test_read_rejects_format_mismatch(tmp_path):
    """Forward-compat: a parquet-v1 file (or any other format string)
    should not be silently consumed by the JSONL reader."""
    path = tmp_path / "wrong.jsonl"
    path.write_text(
        '{"_meta": "karpa-hardness-index", "format": "parquet-v1", "version": "v"}\n'
    )
    with pytest.raises(ValueError, match=r"format mismatch"):
        read_hardness_index_jsonl(path)


def test_read_rejects_invalid_json_header(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json at all\n")
    with pytest.raises(ValueError, match=r"header line is not valid JSON"):
        read_hardness_index_jsonl(path)


def test_writer_format_pinned():
    """If you change WRITER_FORMAT you MUST also update read_hardness_index_jsonl
    AND every test that asserts the format string."""
    assert WRITER_FORMAT == "jsonl-v1"


def test_round_trip_preserves_float_precision(tmp_path):
    """JSON encoding of a float64 should round-trip via json.dumps/loads
    to within machine epsilon."""
    rows = [
        HardnessIndexRow(
            dataset="arc_challenge_hard",
            item_id="i1",
            gold_margin_bits=-0.123456789012345,
        ),
    ]
    idx_in = HardnessIndex(version="v", rows=rows)
    path = tmp_path / "out.jsonl"
    write_hardness_index_jsonl(idx_in, path)
    idx_out = read_hardness_index_jsonl(path)
    assert idx_out.rows[0].gold_margin_bits == rows[0].gold_margin_bits
