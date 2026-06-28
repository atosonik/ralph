"""Measurement-pinning (B): the secret held-out eval under eval/private/ must
NOT contribute to the container_measurement.

The validator has eval/private/active_*.json; miners never do. If it's hashed
into the measurement, no honest miner can ever reproduce the validator's value
and every real_tdx_nvcc submission rejects at op2 with "container measurement
mismatch". This pins that down.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from proof.sources import compute_container_measurement, list_proof_sources


def _make_ralph(root: Path) -> None:
    (root / "eval" / "downstream").mkdir(parents=True)
    (root / "eval" / "hidden_eval.py").write_text("x = 1\n")
    (root / "eval" / "downstream" / "private_hard.py").write_text("y = 2\n")  # tracked → kept
    (root / "calibration").mkdir()
    (root / "calibration" / "c.py").write_text("c = 1\n")
    (root / "proof").mkdir()
    (root / "proof" / "p.py").write_text("p = 1\n")
    (root / "restricted_files.yaml").write_text("a: b\n")
    (root / "README.md").write_text("# r\n")


def _make_recipe(root: Path) -> None:
    (root / "configs").mkdir(parents=True)
    (root / "configs" / "cfg.json").write_text('{"k": 1}')
    (root / "recipe").mkdir()
    (root / "recipe" / "train.py").write_text("t = 1\n")


def test_secret_eval_private_does_not_affect_measurement(tmp_path):
    ralph, recipe = tmp_path / "ralph", tmp_path / "recipe"
    _make_ralph(ralph)
    _make_recipe(recipe)

    before = compute_container_measurement(ralph, recipe)
    rels_before = [r.as_posix() for _, r in list_proof_sources(ralph, recipe)]

    # deploy a SECRET held-out eval into eval/private/ (exactly what the validator does)
    (ralph / "eval" / "private").mkdir()
    (ralph / "eval" / "private" / "active_benchmark.json").write_text('[{"secret": true}]')

    after = compute_container_measurement(ralph, recipe)
    rels_after = [r.as_posix() for _, r in list_proof_sources(ralph, recipe)]

    assert before == after, "secret eval/private/ leaked into the container_measurement"
    assert rels_before == rels_after
    assert "eval/private/active_benchmark.json" not in rels_after
    # the tracked downstream code is still measured (only the eval/private/ dir is dropped)
    assert "eval/downstream/private_hard.py" in rels_after


# ---- op2 canonical source-version check (clear drift error) ----

def test_canonical_version_check_off_by_default(tmp_path, monkeypatch):
    from validator.validator import _check_canonical_source_version
    monkeypatch.delenv("RALPH_CANONICAL_SOURCE_COMMITS", raising=False)
    (tmp_path / "bundle_manifest.json").write_text('{"ralph_source_commit": "deadbeef"}')
    assert _check_canonical_source_version(tmp_path) == (True, "")


def test_canonical_version_match_tolerates_short_sha(tmp_path, monkeypatch):
    from validator.validator import _check_canonical_source_version
    (tmp_path / "bundle_manifest.json").write_text(
        '{"ralph_source_commit": "abc1234def", "recipe_source_commit": "999888"}')
    monkeypatch.setenv("RALPH_CANONICAL_SOURCE_COMMITS", "ralph=abc1234,recipe=999888")
    assert _check_canonical_source_version(tmp_path) == (True, "")


def test_canonical_version_mismatch_rejects_with_actionable_detail(tmp_path, monkeypatch):
    from validator.validator import _check_canonical_source_version
    (tmp_path / "bundle_manifest.json").write_text(
        '{"ralph_source_commit": "abc1234def", "recipe_source_commit": "999888"}')
    monkeypatch.setenv("RALPH_CANONICAL_SOURCE_COMMITS", "ralph=ffffffff,recipe=999888")
    ok, detail = _check_canonical_source_version(tmp_path)
    assert not ok
    assert "non-canonical" in detail and "ralph=abc1234def" in detail and "rebuild" in detail
