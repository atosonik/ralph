"""Tests for validator.audit_scheduler.

Covers the audit dispatcher's enqueue logic + decision determinism. Does NOT
exercise run_pending_audits (that re-trains the model, too slow for unit
tests; covered by the integration suite when wired up).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from validator.audit_scheduler import (
    _deterministic_audit_decision,
    archive_audit_job,
    list_pending_audits,
    maybe_enqueue_audit,
)


def test_decision_deterministic():
    """Same bundle_hash → same audit decision."""
    bh = "a" * 64
    d1 = _deterministic_audit_decision(bh, 0.1)
    d2 = _deterministic_audit_decision(bh, 0.1)
    assert d1 == d2


def test_decision_rate_zero():
    assert _deterministic_audit_decision("ff" * 32, 0.0) is False


def test_decision_rate_one():
    assert _deterministic_audit_decision("00" * 32, 1.0) is True


def test_decision_rate_approximates_target():
    """200 distinct bundles at 10% sample → ~20 audited (95% binomial CI)."""
    import hashlib
    n_audited = sum(
        _deterministic_audit_decision(hashlib.sha256(f"b{i}".encode()).hexdigest(), 0.10)
        for i in range(200)
    )
    # Wide range to avoid flaky test under random-seed shifts. Tight
    # determinism is checked by test_decision_deterministic.
    assert 8 <= n_audited <= 35


def test_plain_failure_never_enqueued(tmp_path):
    job = maybe_enqueue_audit(
        chain_dir=tmp_path,
        bundle_id="b1",
        miner_hotkey="hk",
        miner_github="",
        bundle_hash="x" * 64,
        val_bpb=2.5,
        king_val_bpb=1.5,
        quality_gain=-1.0,
        classification="plain_failure",
        proof_dir=tmp_path / "x",
        noise_floor_margin=0.013,
        random_audit_rate=1.0,  # would otherwise audit everything
    )
    assert job is None


def test_close_margin_king_change_always_audited(tmp_path):
    """quality_gain < 2× noise_floor → always enqueue (king_margin trigger)."""
    job = maybe_enqueue_audit(
        chain_dir=tmp_path,
        bundle_id="b_close",
        miner_hotkey="hk_close",
        miner_github="",
        bundle_hash="00" * 32,  # won't trigger at random_rate=0
        val_bpb=1.49,
        king_val_bpb=1.5,
        quality_gain=0.01,  # well under 2 * 0.013 = 0.026
        classification="king_change",
        proof_dir=tmp_path / "b_close",
        noise_floor_margin=0.013,
        random_audit_rate=0.0,
    )
    assert job is not None
    assert "king_margin" in job.reason


def test_audit_job_persisted(tmp_path):
    job = maybe_enqueue_audit(
        chain_dir=tmp_path,
        bundle_id="b_persist",
        miner_hotkey="hk",
        miner_github="gh_user",
        bundle_hash="aa" * 32,
        val_bpb=1.45,
        king_val_bpb=1.5,
        quality_gain=0.05,
        classification="king_change",
        proof_dir=tmp_path / "b_persist",
        noise_floor_margin=0.013,
        random_audit_rate=1.0,  # force enqueue
    )
    assert job is not None
    pending = list_pending_audits(tmp_path)
    assert len(pending) == 1
    assert pending[0].bundle_id == "b_persist"
    assert pending[0].miner_hotkey == "hk"


def test_audit_archive_passed(tmp_path):
    maybe_enqueue_audit(
        chain_dir=tmp_path,
        bundle_id="b_arch",
        miner_hotkey="hk",
        miner_github="",
        bundle_hash="bb" * 32,
        val_bpb=1.45,
        king_val_bpb=1.5,
        quality_gain=0.05,
        classification="king_change",
        proof_dir=tmp_path / "b_arch",
        noise_floor_margin=0.013,
        random_audit_rate=1.0,
    )
    assert len(list_pending_audits(tmp_path)) == 1
    archive_audit_job(tmp_path, "b_arch", "passed")
    assert len(list_pending_audits(tmp_path)) == 0
    assert (tmp_path / "audit_queue" / "passed" / "b_arch.json").exists()
