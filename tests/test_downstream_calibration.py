"""Tests for eval/downstream/calibration.py — noise-floor aggregator.

Covers:
  * _sample_stddev edge cases (empty / single / identical /
    floating-point precision)
  * compute_per_cell_stddev diagnostic helper
  * aggregate_noise_floors: happy path, multi-scale max-across, single
    report / identical reports → 0 floors, margin_multiplier scaling,
    metadata propagation, empty-input + negative-multiplier rejection
  * write/read JSON round-trip + atomicity + schema validation
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream.calibration import (
    _META_MARKER,
    _sample_stddev,
    aggregate_noise_floors,
    compute_per_cell_stddev,
    read_noise_floor_table_json,
    write_noise_floor_table_json,
)
from eval.downstream.types import (
    HARNESS_VERSION,
    CellResult,
    DownstreamReport,
    NoiseFloorTable,
)

# ----------------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------------


def _report(cells: dict[str, float], *, seed: int = 0) -> DownstreamReport:
    """Build a DownstreamReport from a {cell_key: accuracy} dict."""
    return DownstreamReport(
        harness_version=HARNESS_VERSION,
        bundle_sha256="test-sha",
        seed=seed,
        total_examples=sum(1 for _ in cells),
        wall_clock_s=0.0,
        cells={
            key: CellResult(
                task=key.split(":", 1)[0],
                accuracy=acc,
                accuracy_stderr=0.0,
                n_examples=1,
                seed=seed,
            )
            for key, acc in cells.items()
        },
    )


# ============================================================================
# _sample_stddev
# ============================================================================


class TestSampleStddev:
    def test_empty(self):
        assert _sample_stddev([]) == 0.0

    def test_single(self):
        assert _sample_stddev([0.5]) == 0.0

    def test_all_identical(self):
        """Short-circuit avoids the fp-rounding flake."""
        x = 3.9048486872424535
        assert _sample_stddev([x, x, x]) == 0.0

    def test_two_distinct(self):
        # ddof=1: var = ((0-0.5)² + (1-0.5)²) / 1 = 0.5
        assert _sample_stddev([0.0, 1.0]) == pytest.approx(math.sqrt(0.5))

    def test_three_distinct(self):
        # xs = [1, 2, 3]; mean = 2; var = ((1-2)² + (2-2)² + (3-2)²) / 2 = 1.0
        assert _sample_stddev([1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_ddof_one_not_population(self):
        """Population stddev of [0, 1] is 0.5; sample (ddof=1) is sqrt(0.5)."""
        result = _sample_stddev([0.0, 1.0])
        assert result == pytest.approx(math.sqrt(0.5))
        assert result != pytest.approx(0.5)

    def test_negative_values(self):
        # [-1, 1]; mean=0; var = (1+1)/1 = 2; stddev = sqrt(2)
        assert _sample_stddev([-1.0, 1.0]) == pytest.approx(math.sqrt(2.0))

    def test_floating_point_precision(self):
        """Three near-identical values shouldn't blow up."""
        x = 0.5
        # Values that differ by a few ULPs.
        result = _sample_stddev([x, x + 1e-15, x - 1e-15])
        assert result < 1e-14


# ============================================================================
# compute_per_cell_stddev
# ============================================================================


class TestComputePerCellStddev:
    def test_empty_reports(self):
        assert compute_per_cell_stddev([]) == {}

    def test_single_report(self):
        rep = _report({"arc_easy:S3": 0.5})
        result = compute_per_cell_stddev([rep])
        # n=1 → stddev=0
        assert result == {"arc_easy:S3": 0.0}

    def test_multi_report_per_cell(self):
        reps = [
            _report({"arc_easy:S3": 0.5}),
            _report({"arc_easy:S3": 0.7}),
            _report({"arc_easy:S3": 0.9}),
        ]
        result = compute_per_cell_stddev(reps)
        # [0.5, 0.7, 0.9] → mean=0.7, var = (0.04+0+0.04)/2 = 0.04
        assert result["arc_easy:S3"] == pytest.approx(0.2)

    def test_separates_cell_keys(self):
        reps = [
            _report({"arc_easy:S3": 0.5, "piqa:S3": 0.6}),
            _report({"arc_easy:S3": 0.7, "piqa:S3": 0.8}),
        ]
        result = compute_per_cell_stddev(reps)
        assert "arc_easy:S3" in result
        assert "piqa:S3" in result
        assert len(result) == 2

    def test_tolerates_missing_cells(self):
        """One report has fewer cells than others — that cell's stddev
        is over the present subset."""
        reps = [
            _report({"arc_easy:S3": 0.5, "piqa:S3": 0.6}),
            _report({"arc_easy:S3": 0.7}),  # missing piqa
        ]
        result = compute_per_cell_stddev(reps)
        assert "arc_easy:S3" in result
        # piqa appears only once → stddev=0 (single sample)
        assert result["piqa:S3"] == 0.0


# ============================================================================
# aggregate_noise_floors — happy path + scaling
# ============================================================================


class TestAggregateHappyPath:
    def test_returns_noise_floor_table(self):
        reps = [_report({"arc_easy:S3": 0.5 + i * 0.1}) for i in range(3)]
        table = aggregate_noise_floors(reps)
        assert isinstance(table, NoiseFloorTable)

    def test_single_task_single_scale(self):
        reps = [_report({"arc_easy:S3": v}) for v in [0.5, 0.7, 0.9]]
        # stddev([0.5, 0.7, 0.9]) = 0.2; eta = 2 * 0.2 = 0.4
        table = aggregate_noise_floors(reps, margin_multiplier=2.0)
        assert table.floors["arc_easy"] == pytest.approx(0.4)

    def test_margin_multiplier_scales(self):
        reps = [_report({"arc_easy:S3": v}) for v in [0.5, 0.7, 0.9]]
        # stddev = 0.2; with 3.0 multiplier → 0.6
        table = aggregate_noise_floors(reps, margin_multiplier=3.0)
        assert table.floors["arc_easy"] == pytest.approx(0.6)

    def test_zero_multiplier_yields_zero(self):
        reps = [_report({"arc_easy:S3": v}) for v in [0.5, 0.7, 0.9]]
        table = aggregate_noise_floors(reps, margin_multiplier=0.0)
        assert table.floors["arc_easy"] == 0.0

    def test_single_report_yields_zero_floors(self):
        """n=1 → no within-sample variance → all floors = 0."""
        rep = _report({"arc_easy:S3": 0.5, "piqa:S3": 0.7})
        table = aggregate_noise_floors([rep])
        assert table.floors["arc_easy"] == 0.0
        assert table.floors["piqa"] == 0.0

    def test_identical_reports_yield_zero_floors(self):
        """N copies of the same report → all floors = 0."""
        rep = _report({"arc_easy:S3": 0.5})
        table = aggregate_noise_floors([rep, rep, rep, rep])
        assert table.floors["arc_easy"] == 0.0


# ============================================================================
# aggregate_noise_floors — multi-scale per-task
# ============================================================================


class TestAggregateMultiScale:
    def test_takes_max_across_scales(self):
        """Per-task eta = max stddev across all scales for that task."""
        reps = [
            _report({"arc_easy:S1": 0.5, "arc_easy:S3": 0.5}),
            _report({"arc_easy:S1": 0.7, "arc_easy:S3": 0.55}),
            _report({"arc_easy:S1": 0.9, "arc_easy:S3": 0.6}),
        ]
        # S1 stddev = 0.2; S3 stddev ≈ 0.05; max = 0.2
        # eta = 2 * 0.2 = 0.4
        table = aggregate_noise_floors(reps, margin_multiplier=2.0)
        assert table.floors["arc_easy"] == pytest.approx(0.4)

    def test_max_across_scales_picks_noisier(self):
        """Reversal: S3 noisier than S1 → eta picks S3's stddev."""
        reps = [
            _report({"arc_easy:S1": 0.5, "arc_easy:S3": 0.2}),
            _report({"arc_easy:S1": 0.51, "arc_easy:S3": 0.5}),
            _report({"arc_easy:S1": 0.52, "arc_easy:S3": 0.8}),
        ]
        # S1 stddev ≈ 0.01; S3 stddev = 0.3; max = 0.3
        # eta = 2 * 0.3 = 0.6
        table = aggregate_noise_floors(reps, margin_multiplier=2.0)
        assert table.floors["arc_easy"] == pytest.approx(0.6)

    def test_separate_tasks_independent(self):
        reps = [
            _report({"arc_easy:S3": v, "piqa:S3": w})
            for v, w in [(0.5, 0.8), (0.7, 0.8), (0.9, 0.8)]
        ]
        # arc_easy stddev = 0.2; piqa stddev = 0 (all 0.8)
        table = aggregate_noise_floors(reps, margin_multiplier=2.0)
        assert table.floors["arc_easy"] == pytest.approx(0.4)
        assert table.floors["piqa"] == 0.0


# ============================================================================
# aggregate_noise_floors — metadata
# ============================================================================


class TestAggregateMetadata:
    def test_harness_version_stamped(self):
        rep = _report({"arc_easy:S3": 0.5})
        table = aggregate_noise_floors([rep])
        assert table.harness_version == HARNESS_VERSION

    def test_recipe_sha_threaded_through(self):
        rep = _report({"arc_easy:S3": 0.5})
        table = aggregate_noise_floors([rep], recipe_sha="abc123")
        assert table.recipe_sha == "abc123"

    def test_recipe_sha_default_empty(self):
        rep = _report({"arc_easy:S3": 0.5})
        table = aggregate_noise_floors([rep])
        assert table.recipe_sha == ""

    def test_n_baselines_count(self):
        reps = [_report({"arc_easy:S3": v}) for v in [0.5, 0.7, 0.9]]
        table = aggregate_noise_floors(reps)
        assert table.n_baselines == 3

    def test_eta_for_returns_zero_for_unknown_task(self):
        rep = _report({"arc_easy:S3": 0.5})
        table = aggregate_noise_floors([rep])
        assert table.eta_for("not_a_real_task") == 0.0


# ============================================================================
# aggregate_noise_floors — error paths
# ============================================================================


class TestAggregateErrors:
    def test_empty_reports_rejected(self):
        with pytest.raises(ValueError, match=r"at least one report"):
            aggregate_noise_floors([])

    def test_negative_multiplier_rejected(self):
        rep = _report({"arc_easy:S3": 0.5})
        with pytest.raises(ValueError, match=r"margin_multiplier"):
            aggregate_noise_floors([rep], margin_multiplier=-0.5)

    def test_zero_multiplier_accepted(self):
        rep = _report({"arc_easy:S3": 0.5})
        aggregate_noise_floors([rep], margin_multiplier=0.0)  # no raise


# ============================================================================
# JSON I/O — write
# ============================================================================


class TestWriteJson:
    def test_round_trip(self, tmp_path):
        table = NoiseFloorTable(
            floors={"arc_easy": 0.02, "piqa": 0.015},
            harness_version=HARNESS_VERSION,
            recipe_sha="sha-abc",
            n_baselines=10,
        )
        path = tmp_path / "nf.json"
        write_noise_floor_table_json(table, path)
        restored = read_noise_floor_table_json(path)
        assert restored.floors == table.floors
        assert restored.harness_version == table.harness_version
        assert restored.recipe_sha == table.recipe_sha
        assert restored.n_baselines == table.n_baselines

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "nested" / "deep" / "nf.json"
        table = NoiseFloorTable(floors={"x": 0.1}, n_baselines=1)
        write_noise_floor_table_json(table, path)
        assert path.exists()

    def test_atomic_via_tmp(self, tmp_path):
        """Write succeeds → no leftover .tmp file."""
        path = tmp_path / "nf.json"
        table = NoiseFloorTable(floors={"x": 0.1}, n_baselines=1)
        write_noise_floor_table_json(table, path)
        assert not (tmp_path / "nf.json.tmp").exists()

    def test_meta_marker_present(self, tmp_path):
        path = tmp_path / "nf.json"
        table = NoiseFloorTable(floors={"x": 0.1}, n_baselines=1)
        write_noise_floor_table_json(table, path)
        loaded = json.loads(path.read_text())
        assert loaded["_meta"] == _META_MARKER

    def test_human_readable_sorted(self, tmp_path):
        """sort_keys=True + indent=2 produces an audit-friendly diff."""
        path = tmp_path / "nf.json"
        table = NoiseFloorTable(
            floors={"zebra": 0.1, "alpha": 0.2},
            n_baselines=1,
        )
        write_noise_floor_table_json(table, path)
        text = path.read_text()
        assert text.index("alpha") < text.index("zebra")
        assert "\n  " in text  # indent=2


# ============================================================================
# JSON I/O — read
# ============================================================================


class TestReadJson:
    def test_rejects_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not valid json")
        with pytest.raises(ValueError, match=r"not valid JSON"):
            read_noise_floor_table_json(path)

    def test_rejects_missing_meta(self, tmp_path):
        path = tmp_path / "no-meta.json"
        path.write_text(json.dumps({"floors": {}, "n_baselines": 0}))
        with pytest.raises(ValueError, match=r"_meta marker"):
            read_noise_floor_table_json(path)

    def test_rejects_wrong_meta(self, tmp_path):
        path = tmp_path / "wrong-meta.json"
        path.write_text(json.dumps({
            "_meta": "some-other-format",
            "floors": {},
        }))
        with pytest.raises(ValueError, match=r"_meta marker"):
            read_noise_floor_table_json(path)

    def test_rejects_missing_floors_key(self, tmp_path):
        path = tmp_path / "no-floors.json"
        path.write_text(json.dumps({"_meta": _META_MARKER}))
        with pytest.raises(ValueError, match=r"'floors'"):
            read_noise_floor_table_json(path)

    def test_rejects_non_dict_floors(self, tmp_path):
        path = tmp_path / "bad-floors.json"
        path.write_text(json.dumps({
            "_meta": _META_MARKER,
            "floors": [0.1, 0.2],  # list, not dict
        }))
        with pytest.raises(ValueError, match=r"'floors' must be a dict"):
            read_noise_floor_table_json(path)

    def test_tolerates_missing_optional_fields(self, tmp_path):
        """harness_version / recipe_sha / n_baselines all have defaults."""
        path = tmp_path / "minimal.json"
        path.write_text(json.dumps({
            "_meta": _META_MARKER,
            "floors": {"arc_easy": 0.05},
        }))
        table = read_noise_floor_table_json(path)
        assert table.floors == {"arc_easy": 0.05}
        assert table.harness_version == HARNESS_VERSION
        assert table.recipe_sha == ""
        assert table.n_baselines == 0

    def test_coerces_floor_values_to_float(self, tmp_path):
        """Even ints in the JSON come back as floats."""
        path = tmp_path / "ints.json"
        path.write_text(json.dumps({
            "_meta": _META_MARKER,
            "floors": {"arc_easy": 0, "piqa": 1},  # ints
        }))
        table = read_noise_floor_table_json(path)
        assert table.floors["arc_easy"] == 0.0
        assert isinstance(table.floors["arc_easy"], float)
