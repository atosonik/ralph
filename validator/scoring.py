"""
Scoring — the `score = quality + benchmark + stability - cost - complexity -
regression` formula from whitepaper §5.5.

v1.2 §5.4: single attested-execution tier. The two-tier credibility factor
(α = 1.0 verified / α = 0.5 unverified) is RETIRED. Every miner runs the
official Karpa container inside CC; anything without a valid attestation
chain is rejected at op2. No discount factor on the cost denominator.

For Phase 0 we keep it minimal and tunable: only quality and cost terms are
implemented; stability, complexity, regression are zero placeholders with
hooks so we can wire them in once we have multi-seed runs.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

# Kept for back-compat with any external caller that still imports it; v1.2
# code never multiplies by it (single attested-execution tier).
ALPHA_VERIFIED = 1.0

# A quality_gain >= DOMINANT_QUALITY_MULTIPLIER * noise_floor_margin lets a
# challenger crown via the dominant-quality branch (Branch A). The multiplier
# is intentionally generous so this branch fires only on unambiguous wins.
#
# v0.10 CHANGE: Branch A now ALSO requires benchmark_gain >= -noise_floor_margin.
# The unguarded prior form was a Goodhart vector — a miner could seed-search 200
# random init seeds, pick the best val_bpb (expected ≈ √(2 ln N) · σ ≈ 0.042 bpb,
# i.e. exactly at the 3x noise-floor threshold) and crown regardless of what
# benchmark did. The matching condition with the other two clauses removes that
# backdoor. See docs/king_criterion_review/00_RECOMMENDATION.md §6 (v0.10) and
# the published verdict at docs/direction_reframe/00_VERDICT.md §6.
DOMINANT_QUALITY_MULTIPLIER = 3.0


def get_king_rule() -> str:
    """Return the currently-active king-selection rule.

    Reads the KARPA_KING_RULE environment variable; defaults to "legacy" which
    is the v0.10-guarded version of the original score_bundle gate. The
    forthcoming "cross_scale_v1" rule (B3 of the Cross-Scale Downstream Pareto
    build) will be selected via KARPA_KING_RULE=cross_scale_v1 once it ships;
    until then this function only returns "legacy" and a request for any other
    value falls back to "legacy" with a one-line warning to stderr.
    """
    val = os.environ.get("KARPA_KING_RULE", "legacy").strip()
    if val in ("legacy", "cross_scale_v1"):
        return val
    import sys as _sys
    print(
        f"[scoring] WARNING: KARPA_KING_RULE={val!r} not recognised; "
        "falling back to 'legacy'. Valid values: legacy, cross_scale_v1.",
        file=_sys.stderr,
    )
    return "legacy"


@dataclass
class ScoreReport:
    val_bpb: float
    benchmark_accuracy: float
    quality_gain: float
    benchmark_gain: float
    compute_cost: float
    compute_cost_effective: float
    tier: str
    score: float
    king_val_bpb: float | None
    king_benchmark: float | None
    decisively_beats_king: bool


def _hours_to_normalized_h100(
    matmul_ms: float,
    wall_clock_s: float,
    h100_matmul_ms_ref: float = 5.0,
) -> float:
    """Translate wall-clock into normalized H100-hours via the calibration
    benchmark's matmul timing. The miner's machine took (matmul_ms / h100_ref)
    times as long as an H100 would have, so each wall-clock hour counts as
    (h100_ref / matmul_ms) normalized H100-hours.

    Reference: h100_matmul_ms_ref is the matmul_ms on an H100 for the
    calibration workload, to be measured in Phase 0.5 and pinned in this file.
    The 5.0 default is a placeholder.
    """
    if matmul_ms <= 0:
        return wall_clock_s / 3600.0
    speed_factor = h100_matmul_ms_ref / matmul_ms
    return (wall_clock_s / 3600.0) * speed_factor


def score_bundle(
    val_bpb: float,
    benchmark_accuracy: float,
    king_val_bpb: float | None,
    king_benchmark: float | None,
    noise_floor_margin: float,
    matmul_ms: float,
    wall_clock_s: float,
    tier: str = "verified",
    bpb_weight: float = 1.0,
    benchmark_weight: float = 1.0,
    cost_weight: float = 0.1,
) -> ScoreReport:
    """
    Quality gain on val_bpb is computed as (king - challenger) since lower
    val_bpb is better. Benchmark gain is (challenger - king) since higher
    accuracy is better.

    v1.2 §5.4: single attested-execution tier. No α discount on the cost
    denominator — verification is binary, enforced at op2.

    NaN/Inf-safe (deep_review_2026-05-31 high #3): if val_bpb or
    benchmark_accuracy is non-finite the ScoreReport is returned with
    decisively_beats_king=False and quality_gain/benchmark_gain zero so the
    caller can reject without crowning a null king.
    """
    finite_metrics = (
        isinstance(val_bpb, (int, float)) and math.isfinite(val_bpb)
        and isinstance(benchmark_accuracy, (int, float)) and math.isfinite(benchmark_accuracy)
    )

    if not finite_metrics or king_val_bpb is None:
        quality_gain = 0.0
        benchmark_gain = 0.0
        decisively = False
    else:
        quality_gain = (king_val_bpb - val_bpb)
        benchmark_gain = (benchmark_accuracy - (king_benchmark or 0.0))
        # Three branches, all v0.10-guarded so that no clause crowns a
        # challenger that regressed past the noise floor on the OTHER axis:
        #   Branch A: dominant quality (≥3× noise) AND benchmark-no-regress
        #   Branch B: quality > noise              AND benchmark-no-regress
        #   Branch C: benchmark > noise            AND quality-no-regress
        # The Branch A guard (the AND ... clause) is the v0.10 Goodhart fix.
        # See the constant docstring above for the seed-search vector it closes.
        decisively = (
            (quality_gain >= DOMINANT_QUALITY_MULTIPLIER * noise_floor_margin
             and benchmark_gain >= -noise_floor_margin)
            or
            (quality_gain > noise_floor_margin and benchmark_gain >= -noise_floor_margin)
            or
            (benchmark_gain > noise_floor_margin and quality_gain >= -noise_floor_margin)
        )

    cost_h100h = _hours_to_normalized_h100(matmul_ms, wall_clock_s)
    # v1.2: no α factor.
    cost_effective = cost_h100h

    score = (
        bpb_weight * quality_gain
        + benchmark_weight * benchmark_gain
        - cost_weight * cost_effective
    )

    return ScoreReport(
        val_bpb=val_bpb,
        benchmark_accuracy=benchmark_accuracy,
        quality_gain=quality_gain,
        benchmark_gain=benchmark_gain,
        compute_cost=cost_h100h,
        compute_cost_effective=cost_effective,
        tier="verified",  # v1.2 single tier
        score=score,
        king_val_bpb=king_val_bpb,
        king_benchmark=king_benchmark,
        decisively_beats_king=decisively,
    )
