"""Compute-aware crown gate (calibrated net-score). A challenger must beat the
king on quality/benchmark AND be net-positive once its compute is charged against
the *calibrated* H100 reference. Rewards efficiency, rejects runaway compute, with
no hard hour cap. Fully tunable (RALPH_COMPUTE_COST_WEIGHT / _H100_MATMUL_MS_REF /
_CROWN_GATE)."""
from validator.scoring import (
    DEFAULT_H100_MATMUL_MS_REF,
    _h100_matmul_ms_ref,
    score_bundle,
)


def _score(wall_h, gain=0.05, matmul_ms=0.51, **kw):
    # king at val_bpb 1.50 / benchmark 0.95; challenger improves val_bpb by `gain`
    base = dict(
        val_bpb=1.50 - gain, benchmark_accuracy=0.95,
        king_val_bpb=1.50, king_benchmark=0.95,
        noise_floor_margin=0.014, matmul_ms=matmul_ms, wall_clock_s=wall_h * 3600,
    )
    base.update(kw)
    return score_bundle(**base)


def test_calibrated_reference_not_placeholder():
    assert _h100_matmul_ms_ref() == DEFAULT_H100_MATMUL_MS_REF == 0.51


def test_efficient_incremental_win_crowns():
    # 0.05 bpb gain at ~10 H100h -> net-positive -> crowns
    r = _score(wall_h=10, gain=0.05)
    assert r.decisively_beats_king and r.score > 0


def test_runaway_compute_rejected_despite_quality_win():
    # same 0.05 gain at 100 H100h -> net-negative -> the gate blocks the crown
    r = _score(wall_h=100, gain=0.05)
    assert r.score < 0 and not r.decisively_beats_king


def test_gate_disabled_reverts_to_quality_only(monkeypatch):
    monkeypatch.setenv("RALPH_COMPUTE_CROWN_GATE", "0")
    r = _score(wall_h=100, gain=0.05)
    assert r.decisively_beats_king  # quality win crowns regardless of compute


def test_cost_weight_tunable_up(monkeypatch):
    monkeypatch.setenv("RALPH_COMPUTE_COST_WEIGHT", "0.05")  # aggressive pressure
    r = _score(wall_h=10, gain=0.05)  # 0.05 - 0.05*10 = -0.45 -> rejected
    assert not r.decisively_beats_king


def test_reference_tunable(monkeypatch):
    monkeypatch.setenv("RALPH_H100_MATMUL_MS_REF", "1.02")  # 2x -> doubles cost
    assert _h100_matmul_ms_ref() == 1.02
