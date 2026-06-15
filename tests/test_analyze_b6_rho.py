"""C3-PREREG analyze_b6_rho.py FROZEN analysis tests.

Verifies the pinned constants + the math + the gate logic. Pinned
constants are SHA-anchored in the script; any test failure means the
analysis script was modified post-freeze.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from scripts.analyze_b6_rho import (
    BOOTSTRAP_CI,
    BOOTSTRAP_N_RESAMPLES,
    BOOTSTRAP_SEED,
    PASS_LOWER_CI_THRESHOLD,
    PASS_OLMO_POINT_THRESHOLD,
    PASS_OLMO_REFERENCE_NAME,
    PINNED_REFERENCES,
    RALPH_AXIS_FOR_RHO,
    _average_rank,
    _pearson,
    _ralph_s3_overall_from_report,
    analyze_b6,
    bootstrap_spearman_ci,
    spearman_rho,
)

# ============================================================================
# FROZEN CONSTANT PINS — failing means post-freeze edit
# ============================================================================


def test_pinned_references_locked():
    assert PINNED_REFERENCES == (
        "olmo_2_1b_step_30b",
        "pythia_1_4b",
        "tinyllama_1_1b_3t",
    )


def test_pass_thresholds_locked():
    assert PASS_LOWER_CI_THRESHOLD == 0.5
    assert PASS_OLMO_POINT_THRESHOLD == 0.6
    assert PASS_OLMO_REFERENCE_NAME == "olmo_2_1b_step_30b"


def test_bootstrap_constants_locked():
    assert BOOTSTRAP_N_RESAMPLES == 10_000
    assert BOOTSTRAP_CI == 0.95
    assert BOOTSTRAP_SEED == 0


def test_ralph_axis_locked():
    assert RALPH_AXIS_FOR_RHO == "s3_overall"


# ============================================================================
# _average_rank
# ============================================================================


class TestAverageRank:
    def test_strictly_increasing(self):
        assert _average_rank([1.0, 2.0, 3.0]) == [1.0, 2.0, 3.0]

    def test_strictly_decreasing(self):
        assert _average_rank([3.0, 2.0, 1.0]) == [3.0, 2.0, 1.0]

    def test_all_ties(self):
        # All four tied at rank avg(1,2,3,4) = 2.5
        assert _average_rank([5.0, 5.0, 5.0, 5.0]) == [2.5, 2.5, 2.5, 2.5]

    def test_partial_ties(self):
        # Values: [1, 2, 2, 3] → ranks [1, 2.5, 2.5, 4]
        assert _average_rank([1.0, 2.0, 2.0, 3.0]) == [1.0, 2.5, 2.5, 4.0]


# ============================================================================
# spearman_rho + _pearson
# ============================================================================


class TestSpearman:
    def test_perfectly_correlated(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [10.0, 20.0, 30.0, 40.0, 50.0]
        assert spearman_rho(x, y) == pytest.approx(1.0)

    def test_perfectly_anti_correlated(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [50.0, 40.0, 30.0, 20.0, 10.0]
        assert spearman_rho(x, y) == pytest.approx(-1.0)

    def test_rank_invariant_to_monotone_transform(self):
        """Spearman is rank-based — log(y) gives same ρ as y for positive y."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.5, 2.7, 4.1, 5.8, 9.2]
        y_log = [math.log(v) for v in y]
        assert spearman_rho(x, y) == pytest.approx(spearman_rho(x, y_log))

    def test_length_mismatch_rejected(self):
        with pytest.raises(ValueError, match=r"length mismatch"):
            spearman_rho([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_constant_vector_rejected(self):
        with pytest.raises(ZeroDivisionError):
            _pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])


# ============================================================================
# bootstrap_spearman_ci
# ============================================================================


class TestBootstrapCi:
    def test_perfect_correlation_tight_ci(self):
        """Perfect rank correlation → CI tightly around 1.0."""
        x = list(range(10))
        y = [v * 2.0 for v in x]
        lo, hi, n_used = bootstrap_spearman_ci(x, y, n_resamples=2000)
        assert n_used > 0
        assert lo >= 0.5
        assert hi == pytest.approx(1.0, abs=0.05)

    def test_short_input_rejected(self):
        with pytest.raises(ValueError, match=r">= 3 pairs"):
            bootstrap_spearman_ci([1.0, 2.0], [3.0, 4.0])

    def test_deterministic_with_pinned_seed(self):
        """Same seed → same CI bounds across runs."""
        x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        y = [2.0, 3.0, 1.0, 6.0, 4.0, 5.0]
        a = bootstrap_spearman_ci(x, y, n_resamples=1000, seed=42)
        b = bootstrap_spearman_ci(x, y, n_resamples=1000, seed=42)
        assert a == b


# ============================================================================
# Ralph S3 extraction
# ============================================================================


class TestRalphS3Extract:
    def test_picks_only_s3_cells(self):
        report = {
            "cells": {
                "arc_easy:S1": {"accuracy": 0.3},
                "arc_easy:S2": {"accuracy": 0.5},
                "arc_easy:S3": {"accuracy": 0.7},
                "piqa:S3": {"accuracy": 0.6},
                "boolq:S3": {"accuracy": 0.8},
            }
        }
        # Mean of three S3 cells: (0.7 + 0.6 + 0.8) / 3 = 0.7
        assert _ralph_s3_overall_from_report(report) == pytest.approx(0.7)

    def test_no_s3_cells_returns_none(self):
        report = {"cells": {"arc_easy:S1": {"accuracy": 0.3}}}
        assert _ralph_s3_overall_from_report(report) is None

    def test_empty_cells_returns_none(self):
        assert _ralph_s3_overall_from_report({"cells": {}}) is None


# ============================================================================
# analyze_b6 end-to-end
# ============================================================================


def _write_recipe_report(path: Path, *, s3_value: float):
    """Write a per-recipe DownstreamReport JSON with one S3 cell."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "harness_version": "1.0.0-b1",
        "bundle_sha256": "x",
        "seed": 0,
        "total_examples": 10,
        "wall_clock_s": 1.0,
        "cells": {
            "arc_easy:S3": {
                "task": "arc_easy",
                "accuracy": s3_value,
                "accuracy_stderr": 0.0,
                "n_examples": 10,
                "seed": 0,
            }
        },
    }))


def _make_b6_inputs(tmp_path: Path, *, ralph_scores: list[float],
                    ref_scores: list[dict] | None = None,
                    statuses: list[str] | None = None):
    """Build run_result, refs, and per-recipe reports inside tmp_path."""
    if statuses is None:
        statuses = ["success"] * len(ralph_scores)
    if ref_scores is None:
        ref_scores = [
            {
                "olmo_2_1b_step_30b": 0.5 + i * 0.05,
                "pythia_1_4b": 0.5 + i * 0.04,
                "tinyllama_1_1b_3t": 0.5 + i * 0.03,
            }
            for i in range(len(ralph_scores))
        ]
    per_recipe_dir = tmp_path / "per_recipe"
    per_recipe_dir.mkdir(parents=True, exist_ok=True)
    recipes = []
    refs = {}
    for i, (score, status) in enumerate(zip(ralph_scores, statuses)):
        recipe_id = f"r{i}"
        report_path = per_recipe_dir / f"{recipe_id}.json"
        if status == "success":
            _write_recipe_report(report_path, s3_value=score)
        recipes.append({
            "id": recipe_id,
            "status": status,
            "combined_report_path": str(report_path) if status == "success" else None,
            "seed_used": i,
            "wall_clock_s": 1.0,
            "h100_hr": 0.1,
            "cost_usd": 0.2,
            "abort_reasons": [],
        })
        refs[recipe_id] = ref_scores[i]
    run_result = {
        "recipes": recipes,
        "bundle_sha256": "x",
    }
    return run_result, refs, per_recipe_dir


class TestAnalyzeB6:
    def test_perfect_correlation_passes(self, tmp_path):
        # 6 recipes with perfectly correlated Ralph S3 and reference scores.
        ralph = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        run_result, refs, prd = _make_b6_inputs(tmp_path, ralph_scores=ralph)
        result = analyze_b6(run_result, refs, prd)
        assert result.decision == "PASS"
        olmo = next(r for r in result.per_reference if r.reference_name == "olmo_2_1b_step_30b")
        assert olmo.rho_point_estimate == pytest.approx(1.0)
        assert olmo.ci_lower > 0.5

    def test_zero_correlation_fails(self, tmp_path):
        # Ralph S3 perfectly correlated with itself; reference set
        # uncorrelated with Ralph (random-ish).
        ralph = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
        ref_scores = []
        bad = [0.7, 0.3, 0.6, 0.4, 0.5, 0.2]
        for i in range(6):
            ref_scores.append({
                "olmo_2_1b_step_30b": bad[i],
                "pythia_1_4b": bad[(i + 2) % 6],
                "tinyllama_1_1b_3t": bad[(i + 4) % 6],
            })
        run_result, refs, prd = _make_b6_inputs(
            tmp_path, ralph_scores=ralph, ref_scores=ref_scores,
        )
        result = analyze_b6(run_result, refs, prd)
        assert result.decision == "FAIL"

    def test_aborted_recipes_dropped(self, tmp_path):
        ralph = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        statuses = ["success", "success", "aborted", "success", "success", "success"]
        run_result, refs, prd = _make_b6_inputs(
            tmp_path, ralph_scores=ralph, statuses=statuses,
        )
        result = analyze_b6(run_result, refs, prd)
        olmo = next(r for r in result.per_reference if r.reference_name == "olmo_2_1b_step_30b")
        assert olmo.n_pairs_used == 5  # the aborted one dropped

    def test_nan_reference_scores_dropped(self, tmp_path):
        ralph = [0.3, 0.4, 0.5, 0.6, 0.7]
        # Recipe 2 has NaN OLMo score → dropped from OLMo pair set.
        refs_in = [
            {"olmo_2_1b_step_30b": 0.5, "pythia_1_4b": 0.5, "tinyllama_1_1b_3t": 0.5},
            {"olmo_2_1b_step_30b": 0.6, "pythia_1_4b": 0.55, "tinyllama_1_1b_3t": 0.45},
            {"olmo_2_1b_step_30b": float("nan"), "pythia_1_4b": 0.6, "tinyllama_1_1b_3t": 0.4},
            {"olmo_2_1b_step_30b": 0.7, "pythia_1_4b": 0.65, "tinyllama_1_1b_3t": 0.35},
            {"olmo_2_1b_step_30b": 0.8, "pythia_1_4b": 0.7, "tinyllama_1_1b_3t": 0.3},
        ]
        run_result, refs, prd = _make_b6_inputs(
            tmp_path, ralph_scores=ralph, ref_scores=refs_in,
        )
        result = analyze_b6(run_result, refs, prd)
        olmo = next(r for r in result.per_reference if r.reference_name == "olmo_2_1b_step_30b")
        assert olmo.n_pairs_used == 4

    def test_decision_includes_all_pinned_references(self, tmp_path):
        ralph = [0.3, 0.4, 0.5, 0.6, 0.7]
        run_result, refs, prd = _make_b6_inputs(tmp_path, ralph_scores=ralph)
        result = analyze_b6(run_result, refs, prd)
        names = [r.reference_name for r in result.per_reference]
        assert names == list(PINNED_REFERENCES)

    def test_empty_recipes_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"recipes is empty"):
            analyze_b6({"recipes": []}, {}, tmp_path)

    def test_no_ralph_scores_rejected(self, tmp_path):
        # All recipes aborted → no Ralph scores → can't compute rho.
        ralph = [0.3, 0.4, 0.5]
        statuses = ["aborted", "aborted", "aborted"]
        run_result, refs, prd = _make_b6_inputs(
            tmp_path, ralph_scores=ralph, statuses=statuses,
        )
        with pytest.raises(ValueError, match=r"no Ralph S3 scores"):
            analyze_b6(run_result, refs, prd)


# ============================================================================
# CLI
# ============================================================================


def test_cli_main_pass_returns_zero(tmp_path):
    from scripts.analyze_b6_rho import main
    ralph = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    run_result, refs, _ = _make_b6_inputs(tmp_path, ralph_scores=ralph)
    rr_path = tmp_path / "result.json"
    rr_path.write_text(json.dumps(run_result))
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps(refs))
    out_path = tmp_path / "analysis.json"
    rc = main([
        "--run-result", str(rr_path),
        "--refs", str(refs_path),
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    analysis = json.loads(out_path.read_text())
    assert analysis["decision"] == "PASS"


def test_cli_main_fail_returns_one(tmp_path):
    from scripts.analyze_b6_rho import main
    ralph = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    bad = [0.7, 0.3, 0.6, 0.4, 0.5, 0.2]
    ref_scores = [
        {"olmo_2_1b_step_30b": bad[i], "pythia_1_4b": bad[(i + 2) % 6],
         "tinyllama_1_1b_3t": bad[(i + 4) % 6]}
        for i in range(6)
    ]
    run_result, refs, _ = _make_b6_inputs(
        tmp_path, ralph_scores=ralph, ref_scores=ref_scores,
    )
    rr_path = tmp_path / "result.json"
    rr_path.write_text(json.dumps(run_result))
    refs_path = tmp_path / "refs.json"
    refs_path.write_text(json.dumps(refs))
    rc = main([
        "--run-result", str(rr_path),
        "--refs", str(refs_path),
        "--output", str(tmp_path / "analysis.json"),
    ])
    assert rc == 1
