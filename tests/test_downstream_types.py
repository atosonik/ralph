"""Schema-stability tests for the B1 downstream-eval harness data contract.

These pin the dataclass shapes + conventions so B2's validator/scoring.py
rewrite and B3's ladder orchestration can be authored against a stable contract
even before the scorer / runner code lands.
"""
from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream import (
    BPB_SUFFIX,
    HARNESS_VERSION,
    POOL_CORE22,
    POOL_PRIVATE_HARD,
    POOL_S2_VAL_BPB,
    CellResult,
    DownstreamReport,
    NoiseFloorTable,
    ParetoOutcome,
    ParetoVerdict,
    TaskSpec,
)

# ----------------------------------------------------------------------------
# Pinned constants — bump in lock-step with intentional schema changes
# ----------------------------------------------------------------------------


def test_harness_version_is_pinned():
    """If you bump HARNESS_VERSION you must also bump this test value.
    Catches accidental version drift across the harness modules."""
    assert HARNESS_VERSION == "1.0.0-b1"


def test_bpb_suffix_is_pinned():
    """The reserved direction-flip suffix. Aggregator inverts delta sign
    for cells whose key ends in this suffix."""
    assert BPB_SUFFIX == ":bpb"


def test_pool_identifiers_are_pinned():
    """Pools are referenced by string across the codebase + chain payloads;
    drift would silently disagree on which cells belong to which pool."""
    assert POOL_CORE22 == "core22"
    assert POOL_PRIVATE_HARD == "private_hard"
    assert POOL_S2_VAL_BPB == "s2_val_bpb"


# ----------------------------------------------------------------------------
# TaskSpec — frozen, validated at construction
# ----------------------------------------------------------------------------


def test_taskspec_accepts_valid_mc():
    t = TaskSpec(name="mmlu", mode="mc", random_baseline=0.25, pool=POOL_CORE22)
    assert t.name == "mmlu"
    assert t.mode == "mc"
    assert t.random_baseline == 0.25
    assert t.pool == "core22"


def test_taskspec_rejects_bad_mode():
    with pytest.raises(ValueError, match=r"mode must be one of"):
        TaskSpec(name="x", mode="bogus", random_baseline=0.5, pool=POOL_CORE22)


def test_taskspec_rejects_bad_pool():
    with pytest.raises(ValueError, match=r"pool must be one of"):
        TaskSpec(name="x", mode="mc", random_baseline=0.5, pool="garbage")


def test_taskspec_rejects_out_of_range_baseline():
    with pytest.raises(ValueError, match=r"random_baseline must be in"):
        TaskSpec(name="x", mode="mc", random_baseline=1.5, pool=POOL_CORE22)
    with pytest.raises(ValueError, match=r"random_baseline must be in"):
        TaskSpec(name="x", mode="mc", random_baseline=-0.1, pool=POOL_CORE22)


def test_taskspec_frozen():
    """Frozen so two references to the same task are identity-comparable."""
    t = TaskSpec(name="mmlu", mode="mc", random_baseline=0.25, pool=POOL_CORE22)
    with pytest.raises((AttributeError, Exception)):
        t.name = "changed"  # type: ignore[misc]


# ----------------------------------------------------------------------------
# CellResult — defaults reasonable for B1 single-seed era
# ----------------------------------------------------------------------------


def test_cellresult_minimal_init():
    """Single-seed B1: stderr defaults to 0, n_examples to 0."""
    c = CellResult(task="mmlu", accuracy=0.42)
    assert c.task == "mmlu"
    assert c.accuracy == 0.42
    assert c.accuracy_stderr == 0.0
    assert c.n_examples == 0
    assert c.seed == 0


def test_cellresult_full_init():
    c = CellResult(
        task="hellaswag", accuracy=0.555, accuracy_stderr=0.012,
        n_examples=10042, seed=42,
    )
    assert c.accuracy_stderr == 0.012
    assert c.n_examples == 10042
    assert c.seed == 42


# ----------------------------------------------------------------------------
# DownstreamReport — the cross-phase contract
# ----------------------------------------------------------------------------


def test_downstreamreport_fields_are_pinned():
    """If you add / remove / rename fields you break B2 + B3 + chain payload.
    This test fails to remind the author to coordinate the schema change."""
    expected = {
        "harness_version",
        "bundle_sha256",
        "seed",
        "total_examples",
        "wall_clock_s",
        "cells",
    }
    actual = {f.name for f in fields(DownstreamReport)}
    assert actual == expected, (
        f"DownstreamReport fields drifted. Added: {actual - expected}, "
        f"Removed: {expected - actual}. Coordinate with B2/B3 + chain payload "
        "schema before bumping."
    )


def test_downstreamreport_empty_cells_default():
    r = DownstreamReport(
        harness_version=HARNESS_VERSION,
        bundle_sha256="0" * 64,
        seed=0,
        total_examples=0,
        wall_clock_s=0.0,
    )
    assert r.cells == {}


def test_downstreamreport_independent_cells_per_instance():
    """Default factory mutability check: appending to one instance's cells
    must not affect another's."""
    a = DownstreamReport(harness_version=HARNESS_VERSION, bundle_sha256="a",
                         seed=0, total_examples=0, wall_clock_s=0.0)
    b = DownstreamReport(harness_version=HARNESS_VERSION, bundle_sha256="b",
                         seed=0, total_examples=0, wall_clock_s=0.0)
    a.cells["mmlu:S3"] = CellResult(task="mmlu", accuracy=0.42)
    assert "mmlu:S3" not in b.cells


# ----------------------------------------------------------------------------
# NoiseFloorTable — the calibration-output contract
# ----------------------------------------------------------------------------


def test_noisefloortable_eta_for_returns_zero_default():
    """Unknown task → 0.0 (no floor). Conservative: kernel falls back to
    pooled_stderr × multiplier alone."""
    t = NoiseFloorTable(floors={"mmlu": 0.012})
    assert t.eta_for("mmlu") == 0.012
    assert t.eta_for("not_in_table") == 0.0


def test_noisefloortable_eta_for_returns_float():
    """Float coercion catches accidental int storage from a JSON load."""
    t = NoiseFloorTable(floors={"mmlu": 1})  # int on purpose
    assert isinstance(t.eta_for("mmlu"), float)
    assert t.eta_for("mmlu") == 1.0


def test_noisefloortable_carries_harness_version_default():
    t = NoiseFloorTable()
    assert t.harness_version == HARNESS_VERSION


# ----------------------------------------------------------------------------
# ParetoOutcome enum — the three-class verdict
# ----------------------------------------------------------------------------


def test_pareto_outcome_three_classes():
    """The verdict space is exactly three values — KING_CHANGE,
    MEANINGFUL_FAILURE, PLAIN_FAILURE — mapped from the existing service.py
    classification dispatch."""
    assert ParetoOutcome.KING_CHANGE.value == "king_change"
    assert ParetoOutcome.MEANINGFUL_FAILURE.value == "meaningful_failure"
    assert ParetoOutcome.PLAIN_FAILURE.value == "plain_failure"
    assert len(ParetoOutcome) == 3


def test_pareto_verdict_default_fields():
    v = ParetoVerdict(outcome=ParetoOutcome.KING_CHANGE)
    assert v.outcome == ParetoOutcome.KING_CHANGE
    assert v.sig_wins == []
    assert v.sig_losses == []
    assert v.cell_deltas == {}
    assert v.reason == ""
