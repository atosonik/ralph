"""Tests for the Cross-Scale Downstream Pareto kernel — the core algorithm
of the new king-selection gate.

15 cases cover the decision-rule surface end-to-end: no-regression gating,
≥1 sig-win requirement, per-cell threshold = max(2·pooled_stderr, eta_task),
NaN handling, missing-cell forward-compat, and the BPB_SUFFIX direction flip.

Reference: docs/build_scope/02_scope_B1.md "test_downstream_aggregate" + the
predicate spec at docs/king_criterion_review/00_RECOMMENDATION.md §4.6.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream import (
    BPB_SUFFIX,
    HARNESS_VERSION,
    CellResult,
    DownstreamReport,
    NoiseFloorTable,
    ParetoOutcome,
    aggregate_pareto,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _report(cells: dict[str, CellResult], seed: int = 0) -> DownstreamReport:
    """Build a minimal DownstreamReport for tests. Bundle SHA / wall-clock
    are not consulted by aggregate_pareto."""
    return DownstreamReport(
        harness_version=HARNESS_VERSION,
        bundle_sha256="0" * 64,
        seed=seed,
        total_examples=sum(c.n_examples for c in cells.values()),
        wall_clock_s=1.0,
        cells=dict(cells),
    )


# Standard noise-floor table used across tests. ARC-Challenge has the largest
# floor (0.02) so it's the easiest task to land inside the noise band.
_FLOORS = NoiseFloorTable(
    floors={
        "mmlu": 0.012,
        "hellaswag": 0.010,
        "arc_challenge": 0.020,
        "arc_easy": 0.010,
        "winogrande": 0.012,
    },
    harness_version=HARNESS_VERSION,
    recipe_sha="abc123",
    n_baselines=10,
)


# ----------------------------------------------------------------------------
# Case 1: No regressions, multiple sig wins → KING_CHANGE
# ----------------------------------------------------------------------------


def test_clean_win_no_regressions():
    """3 sig wins across CORE-22, no cell regresses → KING_CHANGE."""
    challenger = _report({
        "mmlu:S3":         CellResult(task="mmlu",         accuracy=0.50),
        "hellaswag:S3":    CellResult(task="hellaswag",    accuracy=0.65),
        "arc_easy:S3":     CellResult(task="arc_easy",     accuracy=0.62),
    })
    king = _report({
        "mmlu:S3":         CellResult(task="mmlu",         accuracy=0.45),
        "hellaswag:S3":    CellResult(task="hellaswag",    accuracy=0.60),
        "arc_easy:S3":     CellResult(task="arc_easy",     accuracy=0.55),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.KING_CHANGE
    assert sorted(v.sig_wins) == ["arc_easy:S3", "hellaswag:S3", "mmlu:S3"]
    assert v.sig_losses == []
    assert v.cell_deltas["mmlu:S3"] == pytest.approx(0.05)
    assert "3 sig win" in v.reason


# ----------------------------------------------------------------------------
# Case 2: One cell sig-regresses past threshold → PLAIN_FAILURE
# ----------------------------------------------------------------------------


def test_one_regression_kills_otherwise_clean_wins():
    """3 sig wins on other cells don't rescue: any cell regressing past
    threshold = PLAIN_FAILURE. This is the v0.10 anti-Goodhart rule."""
    challenger = _report({
        "mmlu:S3":         CellResult(task="mmlu",         accuracy=0.50),
        "hellaswag:S3":    CellResult(task="hellaswag",    accuracy=0.65),
        "arc_easy:S3":     CellResult(task="arc_easy",     accuracy=0.40),  # tanked
    })
    king = _report({
        "mmlu:S3":         CellResult(task="mmlu",         accuracy=0.45),
        "hellaswag:S3":    CellResult(task="hellaswag",    accuracy=0.60),
        "arc_easy:S3":     CellResult(task="arc_easy",     accuracy=0.55),  # was 0.55
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.PLAIN_FAILURE
    assert "arc_easy:S3" in v.sig_losses
    assert sorted(v.sig_wins) == ["hellaswag:S3", "mmlu:S3"]
    assert v.cell_deltas["arc_easy:S3"] == pytest.approx(-0.15)
    assert "regression at" in v.reason


# ----------------------------------------------------------------------------
# Case 3: Regression INSIDE the noise band → not a sig loss
# ----------------------------------------------------------------------------


def test_regression_inside_noise_band_does_not_count():
    """ARC-Challenge eta is 0.020; a 0.005 regression is inside the floor —
    counted as neither win nor loss. Pair with a real win elsewhere →
    KING_CHANGE."""
    challenger = _report({
        "arc_challenge:S3": CellResult(task="arc_challenge", accuracy=0.395),  # -0.005
        "mmlu:S3":          CellResult(task="mmlu",          accuracy=0.500),  # +0.05
    })
    king = _report({
        "arc_challenge:S3": CellResult(task="arc_challenge", accuracy=0.400),
        "mmlu:S3":          CellResult(task="mmlu",          accuracy=0.450),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.KING_CHANGE
    assert v.sig_wins == ["mmlu:S3"]
    assert v.sig_losses == []
    assert v.cell_deltas["arc_challenge:S3"] == pytest.approx(-0.005)  # delta recorded
    assert "arc_challenge:S3" not in v.sig_wins  # not a win either


# ----------------------------------------------------------------------------
# Case 4: No regressions but no sig wins either → MEANINGFUL_FAILURE
# ----------------------------------------------------------------------------


def test_inside_noise_band_on_all_cells_is_meaningful_failure():
    """All cells within ±eta_task: no losses, no wins. Verdict is
    MEANINGFUL_FAILURE; the validator layer pairs this with rationale +
    diff coherence checks before crediting the 10% pool."""
    challenger = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.455),  # +0.005, inside 0.012
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.605),  # +0.005, inside 0.010
    })
    king = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.450),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.600),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.MEANINGFUL_FAILURE
    assert v.sig_wins == []
    assert v.sig_losses == []
    assert v.reason == "no regression, no significant win"


# ----------------------------------------------------------------------------
# Case 5: NaN cell → PLAIN_FAILURE regardless of other cells
# ----------------------------------------------------------------------------


def test_nan_cell_short_circuits_to_plain_failure():
    """Any NaN/Inf measurement in challenger or king triggers PLAIN_FAILURE
    with the offending cell named. Defense against bundles whose eval ran
    out of memory or hit a numerical issue mid-eval."""
    challenger = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=float("nan")),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.65),
    })
    king = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.45),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.60),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.PLAIN_FAILURE
    assert "nan_cells" in v.reason
    assert "mmlu:S3" in v.reason


def test_inf_cell_also_triggers_plain_failure():
    challenger = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=float("inf")),
    })
    king = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.45),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.PLAIN_FAILURE
    assert "nan_cells" in v.reason


# ----------------------------------------------------------------------------
# Case 6: pooled_stderr math against a hand-computed example
# ----------------------------------------------------------------------------


def test_pooled_stderr_used_when_larger_than_eta():
    """When a cell has real seed-to-seed variance (B2+ era), the pooled
    stderr term takes over from the eta_task floor.
    Pooled = sqrt(0.05² + 0.05²) ≈ 0.0707; threshold = 2 × 0.0707 ≈ 0.1414.
    A 0.10 delta is INSIDE the threshold → neither win nor loss."""
    challenger = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.55, accuracy_stderr=0.05),
    })
    king = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.45, accuracy_stderr=0.05),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    # 0.10 delta inside 0.1414 threshold → MEANINGFUL_FAILURE (no win, no loss)
    assert v.outcome == ParetoOutcome.MEANINGFUL_FAILURE
    assert v.sig_wins == []


def test_eta_task_floor_overrides_zero_pooled_stderr():
    """B1 era: stderr is 0 for both arms (single seed deterministic eval).
    Per-cell threshold collapses to eta_task. A 0.005 delta with eta=0.012
    is INSIDE the threshold → not a sig win."""
    challenger = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.455),  # +0.005
    })
    king = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.450),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.MEANINGFUL_FAILURE


# ----------------------------------------------------------------------------
# Case 7: BPB_SUFFIX cells (lower is better) — direction sign flip
# ----------------------------------------------------------------------------


def test_bpb_cell_lower_is_better():
    """val_bpb cells use the :bpb suffix; aggregator inverts delta sign.
    challenger=1.40 vs king=1.50 → challenger is BETTER on this axis."""
    # Use s2_val_bpb pool with a wide floor so the test focuses on direction
    # math, not on threshold tuning.
    floors = NoiseFloorTable(floors={"s2_val_bpb": 0.005})
    challenger = _report({
        f"s2_val_bpb{BPB_SUFFIX}": CellResult(task="s2_val_bpb", accuracy=1.40),
    })
    king = _report({
        f"s2_val_bpb{BPB_SUFFIX}": CellResult(task="s2_val_bpb", accuracy=1.50),
    })
    v = aggregate_pareto(challenger, king, floors)
    # Delta is (king - challenger) = 0.10 → POSITIVE → sig win
    assert v.outcome == ParetoOutcome.KING_CHANGE
    assert v.sig_wins == [f"s2_val_bpb{BPB_SUFFIX}"]
    assert v.cell_deltas[f"s2_val_bpb{BPB_SUFFIX}"] == pytest.approx(0.10)


def test_bpb_cell_higher_is_a_loss():
    """challenger has WORSE val_bpb (higher value) → counted as a loss."""
    floors = NoiseFloorTable(floors={"s2_val_bpb": 0.005})
    challenger = _report({
        f"s2_val_bpb{BPB_SUFFIX}": CellResult(task="s2_val_bpb", accuracy=1.60),  # WORSE
    })
    king = _report({
        f"s2_val_bpb{BPB_SUFFIX}": CellResult(task="s2_val_bpb", accuracy=1.50),
    })
    v = aggregate_pareto(challenger, king, floors)
    assert v.outcome == ParetoOutcome.PLAIN_FAILURE
    assert v.sig_losses == [f"s2_val_bpb{BPB_SUFFIX}"]
    # cell_deltas is "challenger improvement" — should be negative (worse)
    assert v.cell_deltas[f"s2_val_bpb{BPB_SUFFIX}"] == pytest.approx(-0.10)


# ----------------------------------------------------------------------------
# Case 8: missing king cell → forward-compat skip
# ----------------------------------------------------------------------------


def test_challenger_cell_missing_from_king_is_skipped():
    """B2 might ship more cells than B1's king. The kernel takes the
    intersection — does NOT crown on cells the king never had."""
    challenger = _report({
        "mmlu:S3":          CellResult(task="mmlu",          accuracy=0.50),
        "new_task_S3":      CellResult(task="new_task",      accuracy=0.99),  # missing
    })
    king = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.45),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.KING_CHANGE
    assert v.sig_wins == ["mmlu:S3"]
    assert "new_task_S3" not in v.cell_deltas


# ----------------------------------------------------------------------------
# Case 9: All cells exactly tied → MEANINGFUL_FAILURE
# ----------------------------------------------------------------------------


def test_exact_tie_on_all_cells():
    """Exactly tied delta = 0; under noise band → no win, no loss →
    MEANINGFUL_FAILURE."""
    challenger = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.45),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.60),
    })
    king = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.45),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.60),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.MEANINGFUL_FAILURE
    assert v.cell_deltas == {"mmlu:S3": 0.0, "hellaswag:S3": 0.0}


# ----------------------------------------------------------------------------
# Case 10: sig_multiplier override
# ----------------------------------------------------------------------------


def test_sig_multiplier_override_loosens_threshold():
    """Lowering the multiplier from 2.0 to 0.5 lets a smaller win count.
    A +0.013 delta with eta=0.012 is at the boundary; at multiplier=2 it's
    inside (no win), at multiplier=0.5 it's still inside because the eta
    floor dominates. Verify the eta floor still binds."""
    challenger = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.463),  # +0.013
    })
    king = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.450),
    })
    # Under default mult=2, threshold = max(2*1e-12, 0.012) = 0.012; 0.013 > 0.012 → win
    v_default = aggregate_pareto(challenger, king, _FLOORS)
    assert v_default.outcome == ParetoOutcome.KING_CHANGE

    # With a tiny multiplier the eta floor still binds → same outcome
    v_loose = aggregate_pareto(challenger, king, _FLOORS, sig_multiplier=0.5)
    assert v_loose.outcome == ParetoOutcome.KING_CHANGE


def test_sig_multiplier_with_real_stderr_changes_threshold():
    """Real stderr case: 0.06 challenger stderr + 0.06 king stderr.
    pooled = sqrt(2 * 0.06²) ≈ 0.0849.
    multiplier=2 → threshold = 0.1697; a +0.10 delta is INSIDE → no win.
    multiplier=1 → threshold = 0.0849; a +0.10 delta is OUTSIDE → win.
    """
    floors = NoiseFloorTable(floors={"mmlu": 0.001})  # tiny eta so stderr binds
    challenger = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.55, accuracy_stderr=0.06),
    })
    king = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.45, accuracy_stderr=0.06),
    })
    v_strict = aggregate_pareto(challenger, king, floors, sig_multiplier=2.0)
    assert v_strict.outcome == ParetoOutcome.MEANINGFUL_FAILURE

    v_loose = aggregate_pareto(challenger, king, floors, sig_multiplier=1.0)
    assert v_loose.outcome == ParetoOutcome.KING_CHANGE


# ----------------------------------------------------------------------------
# Case 11: cell_deltas dict populated for downstream serialization
# ----------------------------------------------------------------------------


def test_cell_deltas_populated_for_every_intersected_cell():
    """Every cell that exists in BOTH reports gets a delta entry, even ties
    and inside-noise-band cases. This is what the LadderScore chain event
    serializes."""
    challenger = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.50),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.60),  # tied
        "arc_easy:S3":  CellResult(task="arc_easy",  accuracy=0.40),  # regressed
    })
    king = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.45),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.60),
        "arc_easy:S3":  CellResult(task="arc_easy",  accuracy=0.55),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.PLAIN_FAILURE
    assert v.cell_deltas["mmlu:S3"] == pytest.approx(0.05)
    assert v.cell_deltas["hellaswag:S3"] == 0.0
    assert math.isclose(v.cell_deltas["arc_easy:S3"], -0.15)


# ----------------------------------------------------------------------------
# Case 12: sig_wins / sig_losses are disjoint
# ----------------------------------------------------------------------------


def test_sig_wins_and_sig_losses_disjoint():
    """A cell cannot be both a sig win and a sig loss simultaneously."""
    challenger = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.55),  # win
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.45),  # loss
    })
    king = _report({
        "mmlu:S3":      CellResult(task="mmlu",      accuracy=0.45),
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.60),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert set(v.sig_wins).isdisjoint(set(v.sig_losses))


# ----------------------------------------------------------------------------
# Case 13: Empty challenger cells → MEANINGFUL_FAILURE (defensible default)
# ----------------------------------------------------------------------------


def test_empty_intersection_is_meaningful_failure():
    """Edge case: challenger and king have non-overlapping cell sets. The
    intersection is empty so there's no decision to make. We return
    MEANINGFUL_FAILURE — caller layer should detect empty intersection
    earlier as a config error."""
    challenger = _report({
        "mmlu:S3": CellResult(task="mmlu", accuracy=0.50),
    })
    king = _report({
        "hellaswag:S3": CellResult(task="hellaswag", accuracy=0.60),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.MEANINGFUL_FAILURE
    assert v.cell_deltas == {}


# ----------------------------------------------------------------------------
# Case 14: NaN floor for unknown task → effectively pooled_stderr threshold
# ----------------------------------------------------------------------------


def test_unknown_task_falls_back_to_pooled_stderr_threshold():
    """Task not in the noise-floor table: eta=0, threshold collapses to
    2·pooled_stderr (or the pooled_stderr floor, whichever is larger)."""
    floors = NoiseFloorTable(floors={})  # empty
    # Both stderrs are 0 (B1 deterministic case) → pooled = 0 → threshold
    # = max(2 * 1e-12, 0) = 2e-12. Any nonzero delta is a sig win/loss.
    challenger = _report({
        "unknown_task:S3": CellResult(task="unknown_task", accuracy=0.501),
    })
    king = _report({
        "unknown_task:S3": CellResult(task="unknown_task", accuracy=0.500),
    })
    v = aggregate_pareto(challenger, king, floors)
    # 0.001 delta > 2e-12 → KING_CHANGE on the unknown task
    assert v.outcome == ParetoOutcome.KING_CHANGE
    assert v.sig_wins == ["unknown_task:S3"]


# ----------------------------------------------------------------------------
# Case 15: reason string is well-formed and useful
# ----------------------------------------------------------------------------


def test_reason_string_clean_win():
    challenger = _report({"mmlu:S3": CellResult(task="mmlu", accuracy=0.50)})
    king = _report({"mmlu:S3": CellResult(task="mmlu", accuracy=0.45)})
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert "1 sig win" in v.reason


def test_reason_string_regression():
    challenger = _report({
        "mmlu:S3":     CellResult(task="mmlu",     accuracy=0.30),
        "arc_easy:S3": CellResult(task="arc_easy", accuracy=0.30),
    })
    king = _report({
        "mmlu:S3":     CellResult(task="mmlu",     accuracy=0.45),
        "arc_easy:S3": CellResult(task="arc_easy", accuracy=0.55),
    })
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.outcome == ParetoOutcome.PLAIN_FAILURE
    assert "regression at" in v.reason
    # Reason names at least one offending cell
    assert "mmlu:S3" in v.reason or "arc_easy:S3" in v.reason


def test_reason_string_meaningful_failure():
    challenger = _report({"mmlu:S3": CellResult(task="mmlu", accuracy=0.45)})
    king = _report({"mmlu:S3": CellResult(task="mmlu", accuracy=0.45)})
    v = aggregate_pareto(challenger, king, _FLOORS)
    assert v.reason == "no regression, no significant win"
