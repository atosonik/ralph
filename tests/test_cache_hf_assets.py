"""Tests for scripts/cache_hf_assets.py.

The `datasets` HF library is heavyweight + network-required. All HF
calls are mocked. We verify:
  * Per-task row converters produce canonical-schema dicts
  * cache_one_task writes JSONL atomically + correct manifest entry
  * Unknown task name rejected
  * CLI arg parsing including --revision=task=<sha> form
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from scripts.cache_hf_assets import (
    TASK_CONVERTERS,
    _build_parser,
    cache_all,
    cache_one_task,
    convert_arc_row,
    convert_tinybench_mc_row,
    convert_winogrande_row,
)

# ============================================================================
# convert_arc_row
# ============================================================================


class TestConvertArcRow:
    def test_canonical_arc_dict_choices(self):
        row = {
            "id": "ARC-1",
            "question": "Why does X happen?",
            "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]},
            "answerKey": "C",
        }
        out = convert_arc_row(row)
        assert out == {"id": "ARC-1", "query": "Why does X happen?",
                       "choices": ["a", "b", "c", "d"], "gold": 2}

    def test_numeric_label(self):
        row = {
            "id": "ARC-2",
            "question": "Q",
            "choices": {"text": ["x", "y"], "label": ["1", "2"]},
            "answerKey": "2",
        }
        out = convert_arc_row(row)
        assert out["gold"] == 1

    def test_unknown_answer_key_returns_none(self):
        row = {
            "id": "x",
            "question": "Q",
            "choices": {"text": ["a", "b"], "label": ["A", "B"]},
            "answerKey": "Z",
        }
        assert convert_arc_row(row) is None

    def test_choices_as_list_of_dicts(self):
        """Some HF revisions ship choices as a list of {text, label} dicts."""
        row = {
            "id": "x",
            "question": "Q",
            "choices": [
                {"text": "alpha", "label": "A"},
                {"text": "beta", "label": "B"},
            ],
            "answerKey": "B",
        }
        out = convert_arc_row(row)
        assert out["choices"] == ["alpha", "beta"]
        assert out["gold"] == 1

    def test_missing_field_returns_none(self):
        assert convert_arc_row({"id": "x"}) is None


# ============================================================================
# convert_winogrande_row
# ============================================================================


class TestConvertWinogrande:
    def test_basic_two_variants(self):
        row = {
            "qID": "wino-1",
            "sentence": "The trophy didn't fit in the suitcase because _ was too small.",
            "option1": "the trophy",
            "option2": "the suitcase",
            "answer": "2",
        }
        out = convert_winogrande_row(row)
        assert out is not None
        assert out["id"] == "wino-1"
        assert out["gold"] == 1
        # Both contexts identical (the prefix)
        assert out["contexts"][0] == out["contexts"][1]
        assert out["continuations"][0].startswith("the trophy")
        assert out["continuations"][1].startswith("the suitcase")

    def test_missing_underscore_returns_none(self):
        row = {
            "qID": "wino-bad",
            "sentence": "no placeholder here",
            "option1": "a", "option2": "b", "answer": "1",
        }
        assert convert_winogrande_row(row) is None

    def test_bad_answer_returns_none(self):
        row = {
            "qID": "wino-bad-ans",
            "sentence": "X _ Y", "option1": "a", "option2": "b", "answer": "3",
        }
        assert convert_winogrande_row(row) is None


# ============================================================================
# convert_tinybench_mc_row
# ============================================================================


class TestConvertTinybenchMc:
    def test_canonical_mc(self):
        row = {
            "id": "tiny-1",
            "question": "Q?",
            "choices": ["a", "b", "c", "d"],
            "answer": 2,
        }
        out = convert_tinybench_mc_row(row)
        assert out == {"id": "tiny-1", "query": "Q?",
                       "choices": ["a", "b", "c", "d"], "gold": 2}

    def test_dict_choices(self):
        row = {
            "id": "tiny-1",
            "question": "Q?",
            "choices": {"text": ["a", "b"]},
            "answer": 0,
        }
        out = convert_tinybench_mc_row(row)
        assert out["choices"] == ["a", "b"]
        assert out["gold"] == 0

    def test_out_of_range_gold_returns_none(self):
        row = {"id": "x", "question": "Q?", "choices": ["a", "b"], "answer": 5}
        assert convert_tinybench_mc_row(row) is None

    def test_input_id_fallback(self):
        row = {"input_id": "ii-1", "question": "Q", "choices": ["a", "b"], "answer": 0}
        out = convert_tinybench_mc_row(row)
        assert out["id"] == "ii-1"


# ============================================================================
# cache_one_task — mocked HF
# ============================================================================


class _FakeDataset:
    """Minimal HF dataset stand-in. Iterates a list of dicts; has an
    `info` attribute with a `version` field."""

    def __init__(self, rows: list[dict], version: str = "v_fake"):
        self._rows = rows
        self.info = type("I", (), {"version": version})()

    def __iter__(self):
        return iter(self._rows)


def _mock_load_hf(monkeypatch, rows: list[dict], version: str = "v_fake"):
    fake = _FakeDataset(rows, version=version)
    monkeypatch.setattr(
        "scripts.cache_hf_assets._load_hf",
        lambda hf_id, hf_config, *, hf_token, revision, split: fake,
    )


def test_cache_one_task_writes_jsonl(tmp_path, monkeypatch):
    rows = [
        {"id": "1", "question": "Q1", "choices": {"text": ["a", "b"],
         "label": ["A", "B"]}, "answerKey": "A"},
        {"id": "2", "question": "Q2", "choices": {"text": ["x", "y"],
         "label": ["A", "B"]}, "answerKey": "B"},
    ]
    _mock_load_hf(monkeypatch, rows)
    entry = cache_one_task(
        "arc_challenge_hard",
        output_dir=tmp_path / "out",
    )
    assert entry["n_rows_in"] == 2
    assert entry["n_rows_written"] == 2
    assert entry["n_rows_skipped"] == 0
    out_path = tmp_path / "out" / "arc_challenge_hard.jsonl"
    assert out_path.exists()
    lines = [json.loads(ln) for ln in out_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 2
    assert lines[0]["gold"] == 0
    assert lines[1]["gold"] == 1


def test_cache_one_task_atomic_no_tmp_leftover(tmp_path, monkeypatch):
    _mock_load_hf(monkeypatch, [])
    cache_one_task("arc_challenge_hard", output_dir=tmp_path / "out")
    assert not (tmp_path / "out" / "arc_challenge_hard.jsonl.tmp").exists()


def test_cache_one_task_skips_malformed_rows(tmp_path, monkeypatch):
    """Rows that fail conversion are counted in n_rows_skipped."""
    rows = [
        {"id": "ok", "question": "Q", "choices": {"text": ["a", "b"],
         "label": ["A", "B"]}, "answerKey": "A"},
        {"id": "bad", "question": "Q"},  # missing choices/answerKey
    ]
    _mock_load_hf(monkeypatch, rows)
    entry = cache_one_task("arc_challenge_hard", output_dir=tmp_path / "out")
    assert entry["n_rows_in"] == 2
    assert entry["n_rows_written"] == 1
    assert entry["n_rows_skipped"] == 1


def test_cache_one_task_unknown_task_raises(tmp_path):
    with pytest.raises(ValueError, match=r"unknown private-hard task"):
        cache_one_task("not_a_task", output_dir=tmp_path / "out")


def test_cache_all_writes_manifest(tmp_path, monkeypatch):
    _mock_load_hf(monkeypatch, [])
    manifest = cache_all(output_dir=tmp_path / "out", tasks=("arc_challenge_hard",))
    assert manifest["_meta"] == "karpa-private-hard-cache-manifest"
    assert manifest["version"] == "v1"
    assert len(manifest["tasks"]) == 1
    on_disk = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert on_disk == manifest


def test_task_converters_cover_all_4_private_hard_tasks():
    from eval.downstream.private_hard import PRIVATE_HARD_TASKS
    assert set(TASK_CONVERTERS.keys()) == set(PRIVATE_HARD_TASKS)


# ============================================================================
# CLI
# ============================================================================


def test_cli_default_runs_all_tasks(tmp_path):
    parser = _build_parser()
    args = parser.parse_args([])
    assert args.tasks is None  # cache_all picks default = all 4


def test_cli_subset_via_repeat(tmp_path):
    parser = _build_parser()
    args = parser.parse_args([
        "--task", "arc_challenge_hard",
        "--task", "winogrande_hard",
    ])
    assert args.tasks == ["arc_challenge_hard", "winogrande_hard"]


def test_cli_revision_parsing(tmp_path, monkeypatch):
    """The --revision task=<sha> form parses into a per-task dict."""
    from scripts.cache_hf_assets import main
    _mock_load_hf(monkeypatch, [])
    rc = main([
        "--output-dir", str(tmp_path / "out"),
        "--task", "arc_challenge_hard",
        "--revision", "arc_challenge_hard=abc123",
    ])
    assert rc == 0


def test_cli_bad_revision_format_rejected(tmp_path):
    """Missing `=` in --revision arg returns exit 2."""
    from scripts.cache_hf_assets import main
    rc = main([
        "--output-dir", str(tmp_path / "out"),
        "--revision", "no_equals_sign",
    ])
    assert rc == 2
