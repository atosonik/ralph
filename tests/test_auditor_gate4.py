"""Gate-4 (GPU re-eval) decision-spine tests — pure CPU, no torch/GPU.

Covers tier selection, sampling fractions, the val_bpb match verdict (pass /
diverge / escalation band / tier-none deferral), exit codes, and the
re-eval sampling math (with compute_val_bpb mocked so no model/GPU is needed).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from auditor.gate4 import (
    DEFAULT_GATE4_TOLERANCE,
    EXIT_EVAL_DIVERGE,
    ESCALATION_BAND,
    TIER_CHEAP,
    TIER_H100_FULL,
    TIER_H200_KINGCHANGE,
    TIER_NONE,
    Gate4Verdict,
    evaluate_val_bpb_match,
    reeval_val_bpb_on_sample,
    sample_fraction,
    select_tier,
)


# --- select_tier ---------------------------------------------------------
@pytest.mark.parametrize("eo", [
    {"decisive_vs_king": True},
    {"is_first": True},
    {"gate": "king_change"},
])
def test_select_tier_king_change_candidates_get_h200(eo):
    assert select_tier(eo) == TIER_H200_KINGCHANGE


def test_select_tier_meaningful_failure_is_cheap():
    assert select_tier({"gate": "meaningful_failure"}) == TIER_CHEAP


def test_select_tier_plain_failure_is_none():
    assert select_tier({"gate": "plain_failure"}) == TIER_NONE


def test_select_tier_king_change_beats_gate_label():
    # decisive flag wins even if the gate string says otherwise
    assert select_tier({"gate": "meaningful_failure", "decisive_vs_king": True}) \
        == TIER_H200_KINGCHANGE


# --- sample_fraction -----------------------------------------------------
def test_sample_fraction_per_tier():
    assert sample_fraction(TIER_NONE) == 0.0
    assert sample_fraction(TIER_CHEAP) == 0.25
    assert sample_fraction(TIER_H100_FULL) == 1.0
    assert sample_fraction(TIER_H200_KINGCHANGE) == 1.0


def test_sample_fraction_unknown_tier_raises():
    with pytest.raises(ValueError, match="unknown Gate-4 tier"):
        sample_fraction("quantum_gpu")


# --- evaluate_val_bpb_match ----------------------------------------------
def test_verdict_pass_within_tolerance():
    v = evaluate_val_bpb_match(1.5000, 1.5010, tier=TIER_H200_KINGCHANGE,
                               tolerance=0.0032)
    assert v.passed and not v.escalate
    assert v.delta == pytest.approx(0.0010)
    assert v.exit_code == 0


def test_verdict_diverged_beyond_band():
    v = evaluate_val_bpb_match(1.5000, 1.5200, tier=TIER_H200_KINGCHANGE,
                               tolerance=0.0032)
    assert not v.passed and not v.escalate
    assert v.exit_code == EXIT_EVAL_DIVERGE
    assert "DIVERGED" in v.reason


def test_verdict_escalation_band_for_sampled_tier():
    # |Δ|=0.005 is in (0.0032, 0.0064] → escalate (cheap sampled pass is noisier)
    v = evaluate_val_bpb_match(1.5000, 1.5050, tier=TIER_CHEAP, tolerance=0.0032)
    assert not v.passed and v.escalate
    assert "escalation band" in v.reason


def test_verdict_no_escalation_for_full_king_change_tier():
    # H200 is already a full pass — no escalation rung above it
    v = evaluate_val_bpb_match(1.5000, 1.5050, tier=TIER_H200_KINGCHANGE,
                               tolerance=0.0032)
    assert not v.passed and not v.escalate


def test_verdict_tier_none_passes_by_deferral():
    v = evaluate_val_bpb_match(1.5, None, tier=TIER_NONE)
    assert v.passed and v.recomputed_val_bpb is None
    assert "no re-eval" in v.reason


def test_verdict_missing_val_bpb_fails():
    v = evaluate_val_bpb_match(None, 1.5, tier=TIER_CHEAP)
    assert not v.passed
    assert v.exit_code == EXIT_EVAL_DIVERGE


def test_verdict_nonpositive_tolerance_raises():
    with pytest.raises(ValueError, match="tolerance must be > 0"):
        evaluate_val_bpb_match(1.5, 1.5, tier=TIER_CHEAP, tolerance=0.0)


def test_escalation_band_boundary_is_inclusive():
    # exactly tolerance × ESCALATION_BAND is still escalate, not hard-diverge
    tol = DEFAULT_GATE4_TOLERANCE
    edge = tol * ESCALATION_BAND
    v = evaluate_val_bpb_match(1.5, 1.5 + edge, tier=TIER_CHEAP, tolerance=tol)
    assert v.escalate and not v.passed


# --- reeval_val_bpb_on_sample (sampling math; compute_val_bpb mocked) -----
class _FakeSpec:
    bytes_per_token = 4.0
    id = "stream_a"


class _FakeBatch:
    spec = _FakeSpec()

    def __init__(self, n_tokens):
        self.tokens = np.zeros(n_tokens, dtype=np.int64)


def test_reeval_samples_ceil_fraction_of_windows(monkeypatch):
    seen = {}

    def fake_compute(model, tokens, seq_len, batch_size, device, *, bytes_per_token):
        seen["n_tokens"] = len(tokens)
        seen["bytes_per_token"] = bytes_per_token
        return {"val_bpb": 1.234}

    import eval.val_bpb as vb
    monkeypatch.setattr(vb, "compute_val_bpb", fake_compute)

    # 10 windows of seq_len 16; fraction 0.25 → ceil(2.5)=3 windows kept
    batch = _FakeBatch(n_tokens=160)
    out = reeval_val_bpb_on_sample(model=object(), batch=batch, seq_len=16,
                                   fraction=0.25)
    assert out == pytest.approx(1.234)
    assert seen["n_tokens"] == 3 * 16
    assert seen["bytes_per_token"] == 4.0


def test_reeval_full_fraction_keeps_all_windows(monkeypatch):
    seen = {}

    def fake_compute(model, tokens, seq_len, batch_size, device, *, bytes_per_token):
        seen["n_tokens"] = len(tokens)
        return {"val_bpb": 1.0}

    import eval.val_bpb as vb
    monkeypatch.setattr(vb, "compute_val_bpb", fake_compute)

    batch = _FakeBatch(n_tokens=160)
    reeval_val_bpb_on_sample(model=object(), batch=batch, seq_len=16, fraction=1.0)
    assert seen["n_tokens"] == 160


def test_reeval_bad_fraction_raises():
    batch = _FakeBatch(n_tokens=160)
    with pytest.raises(ValueError, match="fraction must be in"):
        reeval_val_bpb_on_sample(model=object(), batch=batch, seq_len=16, fraction=0.0)


def test_gate4verdict_is_frozen():
    v = Gate4Verdict(tier=TIER_NONE, claimed_val_bpb=1.0, recomputed_val_bpb=None,
                     delta=None, tolerance=0.003, passed=True, escalate=False,
                     reason="x")
    with pytest.raises(Exception):
        v.passed = False  # type: ignore[misc]
