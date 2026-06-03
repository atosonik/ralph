"""Tests for validator.scoring.score_bundle.

Covers:
  - NaN/Inf safety (deep_review_2026-05-31 high #3)
  - decisively_beats_king OR-of-AND condition
  - v1.2 single-tier semantics (no α discount)
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from validator.scoring import ALPHA_VERIFIED, score_bundle

_BASE = dict(
    benchmark_accuracy=0.5,
    king_val_bpb=1.5,
    king_benchmark=0.45,
    noise_floor_margin=0.013,
    matmul_ms=5.0,
    wall_clock_s=60.0,
)


def test_decisive_on_bpb_gain():
    """val_bpb 0.05 better than king (>> noise floor) → decisive."""
    sr = score_bundle(val_bpb=1.45, **_BASE)
    assert sr.decisively_beats_king is True
    assert sr.quality_gain > _BASE["noise_floor_margin"]


def test_not_decisive_inside_noise_band():
    """val_bpb 0.005 better AND bench 0.005 better — both inside noise band
    → not decisive. The OR-of-AND condition requires at least one axis to
    exceed the noise floor."""
    sr = score_bundle(
        val_bpb=1.495,  # 0.005 better — inside noise band
        benchmark_accuracy=0.455,  # 0.005 better — inside noise band
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_not_decisive_when_bench_regresses_too_much():
    """bpb wins but bench regresses past noise floor → not decisive
    (the OR-of-AND condition rules both axes together)."""
    sr = score_bundle(
        val_bpb=1.45,
        benchmark_accuracy=0.40,  # 0.05 worse — outside -noise_floor
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_decisive_via_benchmark_axis():
    """Benchmark wins, val_bpb roughly tied — decisive via the benchmark
    arm of the OR-of-AND condition."""
    sr = score_bundle(
        val_bpb=1.50,  # tied
        benchmark_accuracy=0.50,  # 0.05 better — outside noise floor
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is True


def test_nan_val_bpb_not_decisive():
    """NaN val_bpb must NOT crown a king — non-finite metrics rejected."""
    sr = score_bundle(val_bpb=float("nan"), **_BASE)
    assert sr.decisively_beats_king is False
    assert sr.quality_gain == 0.0


def test_inf_val_bpb_not_decisive():
    sr = score_bundle(val_bpb=float("inf"), **_BASE)
    assert sr.decisively_beats_king is False
    assert sr.quality_gain == 0.0


def test_nan_benchmark_not_decisive():
    sr = score_bundle(
        val_bpb=1.45,
        benchmark_accuracy=float("nan"),
        king_val_bpb=1.5,
        king_benchmark=0.45,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False


def test_first_submission_no_king():
    """king_val_bpb=None → no decisive comparison possible; caller treats
    is_first separately."""
    sr = score_bundle(
        val_bpb=1.45,
        benchmark_accuracy=0.5,
        king_val_bpb=None,
        king_benchmark=None,
        noise_floor_margin=0.013,
        matmul_ms=5.0, wall_clock_s=60.0,
    )
    assert sr.decisively_beats_king is False
    assert sr.quality_gain == 0.0


def test_v12_single_tier_no_alpha_discount():
    """ScoreReport.tier always returns 'verified' (v1.2) regardless of
    requested tier; cost_effective == cost (no α factor)."""
    sr = score_bundle(val_bpb=1.45, tier="verified", **_BASE)
    assert sr.tier == "verified"
    # cost_effective must equal cost (no α division)
    assert math.isclose(sr.compute_cost_effective, sr.compute_cost)


def test_alpha_verified_kept_for_back_compat():
    """ALPHA_VERIFIED remains importable for any external caller but
    represents the no-discount default."""
    assert ALPHA_VERIFIED == 1.0
