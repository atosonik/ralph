"""C2-LITE validator/ladder.py eval-driver tests.

Covers:
  * LadderRungSpec validation
  * LadderEvalConfig validation + standard_s1_s2_s3 factory
  * merge_rung_reports happy path + collision + mismatch
  * run_ladder_eval v0.11 mode (with synthetic CLI)
  * run_ladder_eval legacy mode (byte-equivalent HiddenEvalResult)
  * Per-rung subprocess failure propagation
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream.runner import RALPH_VOCAB_SIZE
from eval.downstream.runner_subprocess import EvalSubprocessError
from eval.downstream.types import (
    HARNESS_VERSION,
    CellResult,
    DownstreamReport,
)
from validator.ladder import (
    EVAL_MODE_LEGACY,
    EVAL_MODE_V011,
    LadderEvalConfig,
    LadderEvalResult,
    LadderRungSpec,
    Submission,
    merge_rung_reports,
    run_ladder_eval,
)

# Synthetic CLI entry that ships with the test suite.
_TEST_ENTRY = Path(__file__).resolve().parent / "_runner_subprocess_test_entry.py"
_TEST_COMMAND_PREFIX = (sys.executable, str(_TEST_ENTRY))


@pytest.fixture(autouse=True)
def _set_test_runner_mode(monkeypatch):
    """Default the synthetic entry to success mode for the eval-driver tests."""
    monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "success")
    yield


def _genesis_submission() -> Submission:
    return Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="bh_test",
        miner_hotkey="5F_test",
        vocab_size=RALPH_VOCAB_SIZE,
    )


def _single_rung_config(tmp_path: Path, scale_label: str = "S3") -> LadderEvalConfig:
    """Build a single-rung config to use with the synthetic CLI (which
    hard-codes its cell key, so multi-rung tests must construct fake
    reports directly)."""
    return LadderEvalConfig(
        rungs=(LadderRungSpec(scale_label=scale_label, dim=768, n_layers=12),),
        tasks=("arc_easy",),
        bundle_dir=tmp_path / "bundle",
        bundle_sha256="test-sha",
        seed=0,
    )


# ============================================================================
# LadderRungSpec
# ============================================================================


class TestLadderRungSpec:
    def test_minimum_valid(self):
        s = LadderRungSpec(scale_label="S3", dim=768, n_layers=12)
        assert s.scale_label == "S3"
        assert s.n_examples_per_task == 0

    def test_empty_label_rejected(self):
        with pytest.raises(ValueError, match=r"scale_label"):
            LadderRungSpec(scale_label="", dim=768, n_layers=12)

    def test_zero_dim_rejected(self):
        with pytest.raises(ValueError, match=r"dim/n_layers"):
            LadderRungSpec(scale_label="X", dim=0, n_layers=12)

    def test_negative_n_examples_rejected(self):
        with pytest.raises(ValueError, match=r"n_examples_per_task"):
            LadderRungSpec(scale_label="X", dim=64, n_layers=4, n_examples_per_task=-1)


# ============================================================================
# LadderEvalConfig
# ============================================================================


class TestLadderEvalConfig:
    def test_empty_rungs_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"rungs"):
            LadderEvalConfig(
                rungs=(),
                tasks=("arc_easy",),
                bundle_dir=tmp_path,
                bundle_sha256="x",
            )

    def test_empty_tasks_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"tasks"):
            LadderEvalConfig(
                rungs=(LadderRungSpec("S3", 768, 12),),
                tasks=(),
                bundle_dir=tmp_path,
                bundle_sha256="x",
            )

    def test_duplicate_scale_labels_rejected(self, tmp_path):
        with pytest.raises(ValueError, match=r"duplicate"):
            LadderEvalConfig(
                rungs=(
                    LadderRungSpec("S3", 768, 12),
                    LadderRungSpec("S3", 256, 4),
                ),
                tasks=("arc_easy",),
                bundle_dir=tmp_path,
                bundle_sha256="x",
            )

    def test_standard_s1_s2_s3_factory(self, tmp_path):
        cfg = LadderEvalConfig.standard_s1_s2_s3(
            tasks=("arc_easy", "piqa"),
            bundle_dir=tmp_path,
            bundle_sha256="abc",
            seed=42,
            n_examples_per_task=10,
        )
        assert [r.scale_label for r in cfg.rungs] == ["S1", "S2", "S3"]
        assert cfg.rungs[0].dim == 256
        assert cfg.rungs[2].dim == 768
        assert all(r.n_examples_per_task == 10 for r in cfg.rungs)
        assert cfg.seed == 42


# ============================================================================
# merge_rung_reports
# ============================================================================


def _make_report(*, scale: str, task: str = "arc_easy", n: int = 5,
                 wall_clock: float = 1.0, bundle_sha: str = "x",
                 seed: int = 0) -> DownstreamReport:
    return DownstreamReport(
        harness_version=HARNESS_VERSION,
        bundle_sha256=bundle_sha,
        seed=seed,
        total_examples=n,
        wall_clock_s=wall_clock,
        cells={
            f"{task}:{scale}": CellResult(
                task=task, accuracy=0.5, accuracy_stderr=0.0,
                n_examples=n, seed=seed,
            ),
        },
    )


class TestMergeRungReports:
    def test_empty_input_rejected(self):
        with pytest.raises(ValueError, match=r"at least one"):
            merge_rung_reports({})

    def test_merge_three_rungs(self):
        reports = {
            "S1": _make_report(scale="S1", n=5, wall_clock=2.0),
            "S2": _make_report(scale="S2", n=10, wall_clock=5.0),
            "S3": _make_report(scale="S3", n=15, wall_clock=20.0),
        }
        merged = merge_rung_reports(reports)
        assert set(merged.cells.keys()) == {
            "arc_easy:S1", "arc_easy:S2", "arc_easy:S3",
        }
        assert merged.total_examples == 30
        # wall_clock is MAX, not sum
        assert merged.wall_clock_s == 20.0

    def test_duplicate_cell_key_rejected(self):
        # Same scale label produces same cell key — must error.
        a = _make_report(scale="S3")
        b = _make_report(scale="S3", n=99)
        with pytest.raises(ValueError, match=r"duplicate cell key"):
            merge_rung_reports({"a": a, "b": b})

    def test_bundle_sha_mismatch_rejected(self):
        a = _make_report(scale="S1", bundle_sha="sha_a")
        b = _make_report(scale="S2", bundle_sha="sha_b")
        with pytest.raises(ValueError, match=r"bundle_sha256 mismatch"):
            merge_rung_reports({"a": a, "b": b})

    def test_seed_mismatch_rejected(self):
        a = _make_report(scale="S1", seed=0)
        b = _make_report(scale="S2", seed=1)
        with pytest.raises(ValueError, match=r"seed mismatch"):
            merge_rung_reports({"a": a, "b": b})

    def test_first_report_provides_bundle_sha_and_seed(self):
        a = _make_report(scale="S3", bundle_sha="my_sha", seed=7)
        merged = merge_rung_reports({"a": a})
        assert merged.bundle_sha256 == "my_sha"
        assert merged.seed == 7
        assert merged.harness_version == HARNESS_VERSION


# ============================================================================
# run_ladder_eval — v0.11 mode
# ============================================================================


class TestRunLadderEvalV011:
    def test_single_rung_happy_path(self, tmp_path):
        cfg = _single_rung_config(tmp_path)
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"")
        result = run_ladder_eval(
            _genesis_submission(),
            cfg,
            checkpoint_path=ckpt,
            ralph_root=tmp_path,
            command_prefix=_TEST_COMMAND_PREFIX,
            timeout_s_per_rung=15.0,
        )
        assert isinstance(result, LadderEvalResult)
        assert result.mode == EVAL_MODE_V011
        assert set(result.per_rung_reports.keys()) == {"S3"}
        # Combined report = the single rung's report
        assert result.combined_report.cells == result.per_rung_reports["S3"].cells
        assert result.hidden_eval.downstream is not None
        assert result.hidden_eval.downstream.cells == result.combined_report.cells

    def test_subprocess_failure_propagates(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "nonzero")
        monkeypatch.setenv("RALPH_TEST_RUNNER_EXIT_CODE", "9")
        cfg = _single_rung_config(tmp_path)
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"")
        with pytest.raises(EvalSubprocessError) as exc_info:
            run_ladder_eval(
                _genesis_submission(),
                cfg,
                checkpoint_path=ckpt,
                ralph_root=tmp_path,
                command_prefix=_TEST_COMMAND_PREFIX,
                timeout_s_per_rung=15.0,
            )
        assert exc_info.value.exit_code == 9

    def test_unknown_mode_rejected(self, tmp_path):
        cfg = _single_rung_config(tmp_path)
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"")
        with pytest.raises(ValueError, match=r"unknown mode"):
            run_ladder_eval(
                _genesis_submission(),
                cfg,
                checkpoint_path=ckpt,
                ralph_root=tmp_path,
                mode="not_a_mode",
            )


# ============================================================================
# run_ladder_eval — legacy mode (byte-equivalent HiddenEvalResult)
# ============================================================================


class TestRunLadderEvalLegacy:
    def test_legacy_mode_does_not_invoke_subprocess(self, tmp_path):
        """A legacy-mode call with a command_prefix pointing at a FAILING
        synthetic entry must NOT actually call it — the eval is skipped."""
        cfg = _single_rung_config(tmp_path)
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"")
        result = run_ladder_eval(
            _genesis_submission(),
            cfg,
            checkpoint_path=ckpt,
            ralph_root=tmp_path,
            mode=EVAL_MODE_LEGACY,
            command_prefix=[sys.executable, "-c", "import sys; sys.exit(1)"],
            legacy_val_bpb=1.234,
            legacy_benchmark_accuracy=0.5,
            legacy_tokens_evaluated=1000,
            legacy_benchmark_examples=50,
            legacy_eval_set_hash="legacy_hash",
        )
        assert result.mode == EVAL_MODE_LEGACY
        assert result.per_rung_reports == {}
        assert result.combined_report.cells == {}
        assert result.hidden_eval.downstream is None
        assert result.hidden_eval.val_bpb == 1.234

    def test_legacy_to_legacy_dict_byte_equivalent(self, tmp_path):
        """to_legacy_dict() output omits the `downstream` key when None,
        matching the v0.10 chain shape exactly."""
        cfg = _single_rung_config(tmp_path)
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"")
        result = run_ladder_eval(
            _genesis_submission(),
            cfg,
            checkpoint_path=ckpt,
            ralph_root=tmp_path,
            mode=EVAL_MODE_LEGACY,
            legacy_val_bpb=2.0,
            legacy_benchmark_accuracy=0.3,
            legacy_tokens_evaluated=500,
            legacy_benchmark_examples=25,
            legacy_eval_set_hash="x",
        )
        d = result.hidden_eval.to_legacy_dict()
        assert d == {
            "val_bpb": 2.0,
            "benchmark_accuracy": 0.3,
            "tokens_evaluated": 500,
            "benchmark_examples": 25,
            "eval_set_hash": "x",
        }
        assert "downstream" not in d

    def test_v011_mode_populates_downstream(self, tmp_path):
        """In v0.11 mode, to_legacy_dict still INCLUDES downstream when set."""
        cfg = _single_rung_config(tmp_path)
        ckpt = tmp_path / "fake.ckpt"
        ckpt.write_bytes(b"")
        result = run_ladder_eval(
            _genesis_submission(),
            cfg,
            checkpoint_path=ckpt,
            ralph_root=tmp_path,
            mode=EVAL_MODE_V011,
            command_prefix=_TEST_COMMAND_PREFIX,
            timeout_s_per_rung=15.0,
        )
        d = result.hidden_eval.to_legacy_dict()
        assert "downstream" in d
        assert d["downstream"]["cells"]
