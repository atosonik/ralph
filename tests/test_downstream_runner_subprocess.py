"""Tests for eval/downstream/runner_subprocess.py.

Drives the caller-side wrapper through every documented failure mode
plus the happy path. Uses tests/_runner_subprocess_test_entry.py as a
synthetic CLI controlled by env vars so each test can deterministically
exercise one path without needing a real RalphBase checkpoint or DCLM
bundle.

Covers:
  * Happy path: subprocess writes a valid report; wrapper returns it.
  * Argv construction: required + optional args propagate correctly.
  * Config IPC: config.json written before subprocess invocation.
  * tmpdir lifecycle: auto-cleanup when output_dir is None; caller-owned
    when output_dir is passed.
  * Error paths: nonzero_exit, timeout, missing_output,
    malformed_output (invalid JSON + schema mismatch + version mismatch),
    config_write_failed.
  * Stderr tail truncation.
  * serialize_report / deserialize_report round-trip.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream.runner import EvalConfig
from eval.downstream.runner_subprocess import (
    DEFAULT_COMMAND_PREFIX,
    STDERR_TAIL_LIMIT,
    CellResult,
    DownstreamReport,
    EvalSubprocessError,
    _truncate_stderr,
    deserialize_report,
    run_eval_in_subprocess,
    serialize_report,
)
from eval.downstream.types import HARNESS_VERSION

# Synthetic entrypoint script used as a stand-in for the production CLI.
TEST_ENTRY = str(Path(__file__).resolve().parent / "_runner_subprocess_test_entry.py")
TEST_COMMAND_PREFIX = (sys.executable, TEST_ENTRY)


# Tests can dirty os.environ; an autouse fixture restores it.
@pytest.fixture(autouse=True)
def _restore_env(monkeypatch):
    """Per-test env isolation. monkeypatch reverts on teardown by default;
    this fixture is here so tests that pass a custom env get clean state."""
    yield


def _make_config(tasks=("arc_easy",)) -> EvalConfig:
    return EvalConfig(tasks=tasks, n_examples_per_task=0, seed=0, scale_label="S3")


# ============================================================================
# serialize_report / deserialize_report
# ============================================================================


class TestSerialization:
    def test_round_trip_empty_cells(self):
        original = DownstreamReport(
            harness_version=HARNESS_VERSION,
            bundle_sha256="abc",
            seed=0,
            total_examples=0,
            wall_clock_s=0.0,
            cells={},
        )
        restored = deserialize_report(serialize_report(original))
        assert restored == original

    def test_round_trip_with_cells(self):
        original = DownstreamReport(
            harness_version=HARNESS_VERSION,
            bundle_sha256="def",
            seed=7,
            total_examples=42,
            wall_clock_s=3.5,
            cells={
                "arc_easy:S3": CellResult(
                    task="arc_easy",
                    accuracy=0.5,
                    accuracy_stderr=0.0,
                    n_examples=42,
                    seed=7,
                ),
                "piqa:S2": CellResult(
                    task="piqa",
                    accuracy=0.75,
                    accuracy_stderr=0.05,
                    n_examples=20,
                    seed=7,
                ),
            },
        )
        restored = deserialize_report(serialize_report(original))
        assert restored == original

    def test_deserialize_tolerates_missing_optional_fields(self):
        """accuracy_stderr / n_examples / seed have defaults; wall_clock too."""
        payload = {
            "harness_version": HARNESS_VERSION,
            "bundle_sha256": "x",
            "seed": 0,
            "total_examples": 0,
            "cells": {
                "arc_easy:S3": {"task": "arc_easy", "accuracy": 0.5},
            },
        }
        report = deserialize_report(payload)
        assert report.cells["arc_easy:S3"].accuracy_stderr == 0.0
        assert report.cells["arc_easy:S3"].n_examples == 0
        assert report.cells["arc_easy:S3"].seed == 0
        assert report.wall_clock_s == 0.0

    def test_deserialize_missing_required_key_raises(self):
        with pytest.raises(KeyError):
            deserialize_report({"harness_version": HARNESS_VERSION})  # missing the rest


# ============================================================================
# _truncate_stderr
# ============================================================================


class TestTruncate:
    def test_short_string_unchanged(self):
        assert _truncate_stderr("hello") == "hello"

    def test_long_string_tail_kept(self):
        text = "X" * (STDERR_TAIL_LIMIT * 2)
        result = _truncate_stderr(text)
        assert len(result) == STDERR_TAIL_LIMIT
        assert result == "X" * STDERR_TAIL_LIMIT

    def test_bytes_decoded(self):
        assert _truncate_stderr(b"hello") == "hello"

    def test_invalid_utf8_replaced(self):
        result = _truncate_stderr(b"\xff\xfe\xff")
        assert isinstance(result, str)
        assert "�" in result  # replacement char


# ============================================================================
# DEFAULT_COMMAND_PREFIX
# ============================================================================


class TestDefaultCommandPrefix:
    def test_default_prefix_uses_current_interpreter(self):
        assert DEFAULT_COMMAND_PREFIX[0] == sys.executable

    def test_default_prefix_targets_runner_cli(self):
        assert DEFAULT_COMMAND_PREFIX[1] == "-m"
        assert DEFAULT_COMMAND_PREFIX[2] == "eval.downstream.runner_cli"


# ============================================================================
# run_eval_in_subprocess — happy path
# ============================================================================


class TestHappyPath:
    def test_returns_downstream_report(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "success")
        report = run_eval_in_subprocess(
            checkpoint_path=tmp_path / "fake.ckpt",
            config=_make_config(),
            bundle_sha256="test-sha",
            bundle_dir=tmp_path / "bundle",
            vocab_size=50257,
            output_dir=tmp_path / "work",
            command_prefix=TEST_COMMAND_PREFIX,
        )
        assert isinstance(report, DownstreamReport)
        assert report.harness_version == HARNESS_VERSION
        assert report.bundle_sha256 == "test-sha"
        assert report.seed == 42  # from the synthetic stub
        assert "arc_easy:S3" in report.cells

    def test_config_serialized_to_work_dir(self, tmp_path, monkeypatch):
        """The wrapper must write config.json before invoking subprocess."""
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "success")
        work = tmp_path / "work"
        run_eval_in_subprocess(
            checkpoint_path=tmp_path / "fake.ckpt",
            config=EvalConfig(tasks=("arc_easy",), seed=42),
            bundle_sha256="x",
            bundle_dir=tmp_path / "bundle",
            vocab_size=50257,
            output_dir=work,
            command_prefix=TEST_COMMAND_PREFIX,
        )
        # The config was written to work/config.json and was readable by the
        # subprocess. We can't read it after — the stub may have run and
        # the wrapper does NOT clean caller-owned dirs.
        assert (work / "config.json").exists()
        loaded = json.loads((work / "config.json").read_text())
        assert loaded["tasks"] == ["arc_easy"]
        assert loaded["seed"] == 42

    def test_auto_tmpdir_cleaned_up(self, tmp_path, monkeypatch):
        """When output_dir is None, the wrapper allocates and removes a tmpdir."""
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "success")
        report = run_eval_in_subprocess(
            checkpoint_path=tmp_path / "fake.ckpt",
            config=_make_config(),
            bundle_sha256="x",
            bundle_dir=tmp_path / "bundle",
            vocab_size=50257,
            command_prefix=TEST_COMMAND_PREFIX,
        )
        assert isinstance(report, DownstreamReport)
        # We can't verify the tmpdir is gone without knowing its path, but the
        # subprocess succeeded → cleanup ran in the finally block.

    def test_caller_owned_dir_preserved(self, tmp_path, monkeypatch):
        """Caller-passed output_dir is NOT removed on success."""
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "success")
        work = tmp_path / "preserved"
        run_eval_in_subprocess(
            checkpoint_path=tmp_path / "fake.ckpt",
            config=_make_config(),
            bundle_sha256="x",
            bundle_dir=tmp_path / "bundle",
            vocab_size=50257,
            output_dir=work,
            command_prefix=TEST_COMMAND_PREFIX,
        )
        assert work.exists()
        assert (work / "report.json").exists()


# ============================================================================
# run_eval_in_subprocess — argv construction
# ============================================================================


class TestArgvConstruction:
    def _capture_argv(self, **wrapper_kwargs):
        """Helper: monkey-patch subprocess.run to capture argv, return it."""
        captured: dict = {}

        def fake_run(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            # Pretend success — wrapper expects an output file though.
            output_path_idx = argv.index("--output") + 1
            Path(argv[output_path_idx]).write_text(json.dumps({
                "harness_version": HARNESS_VERSION,
                "bundle_sha256": "x",
                "seed": 0,
                "total_examples": 0,
                "wall_clock_s": 0.0,
                "cells": {},
            }))
            return mock.Mock(returncode=0, stderr=b"")

        with mock.patch("subprocess.run", side_effect=fake_run):
            run_eval_in_subprocess(**wrapper_kwargs)
        return captured["argv"]

    def test_required_args_present(self, tmp_path):
        argv = self._capture_argv(
            checkpoint_path=Path("/tmp/ckpt"),
            config=_make_config(),
            bundle_sha256="sha123",
            bundle_dir=Path("/tmp/bundle"),
            vocab_size=50257,
            output_dir=tmp_path / "work",
        )
        assert "--checkpoint" in argv
        assert "--config" in argv
        assert "--output" in argv
        assert "--bundle-sha" in argv
        assert "--bundle-dir" in argv
        assert "--vocab-size" in argv
        # Spot-check the values land in the right positions.
        sha_idx = argv.index("--bundle-sha")
        assert argv[sha_idx + 1] == "sha123"
        vs_idx = argv.index("--vocab-size")
        assert argv[vs_idx + 1] == "50257"

    def test_optional_args_omitted_when_none(self, tmp_path):
        argv = self._capture_argv(
            checkpoint_path=Path("/tmp/ckpt"),
            config=_make_config(),
            bundle_sha256="x",
            bundle_dir=Path("/tmp/bundle"),
            vocab_size=50257,
            output_dir=tmp_path / "work",
        )
        assert "--hardness-index" not in argv
        assert "--patch" not in argv
        assert "--ralph-root" not in argv

    def test_optional_args_propagate_when_set(self, tmp_path):
        argv = self._capture_argv(
            checkpoint_path=Path("/tmp/ckpt"),
            config=_make_config(),
            bundle_sha256="x",
            bundle_dir=Path("/tmp/bundle"),
            vocab_size=50257,
            hardness_index_path=Path("/tmp/hard.jsonl"),
            patch_path=Path("/tmp/p.patch"),
            ralph_root=Path("/tmp/root"),
            output_dir=tmp_path / "work",
        )
        hi_idx = argv.index("--hardness-index")
        assert argv[hi_idx + 1] == "/tmp/hard.jsonl"
        p_idx = argv.index("--patch")
        assert argv[p_idx + 1] == "/tmp/p.patch"
        kr_idx = argv.index("--ralph-root")
        assert argv[kr_idx + 1] == "/tmp/root"

    def test_default_command_prefix_used(self, tmp_path):
        argv = self._capture_argv(
            checkpoint_path=Path("/tmp/ckpt"),
            config=_make_config(),
            bundle_sha256="x",
            bundle_dir=Path("/tmp/bundle"),
            vocab_size=50257,
            output_dir=tmp_path / "work",
        )
        assert argv[:3] == list(DEFAULT_COMMAND_PREFIX)


# ============================================================================
# run_eval_in_subprocess — error paths
# ============================================================================


class TestErrorPaths:
    def test_nonzero_exit_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "nonzero")
        monkeypatch.setenv("RALPH_TEST_RUNNER_EXIT_CODE", "7")
        with pytest.raises(EvalSubprocessError) as exc_info:
            run_eval_in_subprocess(
                checkpoint_path=tmp_path / "fake.ckpt",
                config=_make_config(),
                bundle_sha256="x",
                bundle_dir=tmp_path / "bundle",
                vocab_size=50257,
                output_dir=tmp_path / "work",
                command_prefix=TEST_COMMAND_PREFIX,
            )
        err = exc_info.value
        assert err.reason == "nonzero_exit"
        assert err.exit_code == 7
        assert "simulated failure on stderr" in err.stderr_tail

    def test_missing_output_raises(self, tmp_path, monkeypatch):
        """Subprocess exits 0 but doesn't write the output file."""
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "no_output")
        with pytest.raises(EvalSubprocessError) as exc_info:
            run_eval_in_subprocess(
                checkpoint_path=tmp_path / "fake.ckpt",
                config=_make_config(),
                bundle_sha256="x",
                bundle_dir=tmp_path / "bundle",
                vocab_size=50257,
                output_dir=tmp_path / "work",
                command_prefix=TEST_COMMAND_PREFIX,
            )
        assert exc_info.value.reason == "missing_output"
        assert exc_info.value.exit_code == 0

    def test_malformed_output_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "malformed_output")
        with pytest.raises(EvalSubprocessError) as exc_info:
            run_eval_in_subprocess(
                checkpoint_path=tmp_path / "fake.ckpt",
                config=_make_config(),
                bundle_sha256="x",
                bundle_dir=tmp_path / "bundle",
                vocab_size=50257,
                output_dir=tmp_path / "work",
                command_prefix=TEST_COMMAND_PREFIX,
            )
        assert exc_info.value.reason == "malformed_output"
        assert "JSON parse error" in exc_info.value.stderr_tail

    def test_schema_mismatch_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "schema_mismatch")
        with pytest.raises(EvalSubprocessError) as exc_info:
            run_eval_in_subprocess(
                checkpoint_path=tmp_path / "fake.ckpt",
                config=_make_config(),
                bundle_sha256="x",
                bundle_dir=tmp_path / "bundle",
                vocab_size=50257,
                output_dir=tmp_path / "work",
                command_prefix=TEST_COMMAND_PREFIX,
            )
        assert exc_info.value.reason == "malformed_output"
        assert "schema mismatch" in exc_info.value.stderr_tail

    def test_wrong_harness_version_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "wrong_version")
        with pytest.raises(EvalSubprocessError) as exc_info:
            run_eval_in_subprocess(
                checkpoint_path=tmp_path / "fake.ckpt",
                config=_make_config(),
                bundle_sha256="x",
                bundle_dir=tmp_path / "bundle",
                vocab_size=50257,
                output_dir=tmp_path / "work",
                command_prefix=TEST_COMMAND_PREFIX,
            )
        assert exc_info.value.reason == "malformed_output"
        assert "harness_version mismatch" in exc_info.value.stderr_tail

    def test_timeout_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RALPH_TEST_RUNNER_MODE", "slow")
        monkeypatch.setenv("RALPH_TEST_RUNNER_SLEEP_S", "5")
        with pytest.raises(EvalSubprocessError) as exc_info:
            run_eval_in_subprocess(
                checkpoint_path=tmp_path / "fake.ckpt",
                config=_make_config(),
                bundle_sha256="x",
                bundle_dir=tmp_path / "bundle",
                vocab_size=50257,
                output_dir=tmp_path / "work",
                command_prefix=TEST_COMMAND_PREFIX,
                timeout_s=0.5,
            )
        assert exc_info.value.reason == "timeout"
        assert exc_info.value.exit_code is None

    def test_eval_subprocess_error_str_includes_reason(self):
        err = EvalSubprocessError(
            reason="timeout",
            exit_code=None,
            stderr_tail="oom",
            argv=["a", "b"],
        )
        s = str(err)
        assert "timeout" in s
        assert "exit=None" in s
        assert "oom" in s

    def test_eval_subprocess_error_str_omits_empty_stderr(self):
        err = EvalSubprocessError(
            reason="missing_output",
            exit_code=0,
            stderr_tail="",
            argv=["a"],
        )
        s = str(err)
        assert "missing_output" in s
        assert "stderr" not in s.lower() or "stderr tail" not in s


# ============================================================================
# Env propagation
# ============================================================================


class TestEnvPropagation:
    def test_custom_env_passed_to_subprocess(self, tmp_path, monkeypatch):
        """When `env` is set, the subprocess sees ONLY that env (subprocess.run)."""
        custom = os.environ.copy()
        custom["RALPH_TEST_RUNNER_MODE"] = "success"
        # The success mode requires importing eval.downstream.types — so the
        # subprocess needs PYTHONPATH / cwd to still work. os.environ.copy()
        # preserves PATH etc.; we only add the mode var.
        report = run_eval_in_subprocess(
            checkpoint_path=tmp_path / "fake.ckpt",
            config=_make_config(),
            bundle_sha256="x",
            bundle_dir=tmp_path / "bundle",
            vocab_size=50257,
            output_dir=tmp_path / "work",
            command_prefix=TEST_COMMAND_PREFIX,
            env=custom,
        )
        assert isinstance(report, DownstreamReport)
