"""Caller-side subprocess wrapper for the downstream eval CLI (B1).

This module is the validator-side bridge between in-process orchestration
(scheduler / ladder / king-rule code) and the isolated subprocess that
actually loads a miner-submitted checkpoint and runs the eval. The
isolation matters because miner-submitted code can execute arbitrary
operations inside the model's `forward()`, even with
`torch.load(weights_only=True)` blocking pickle-deserialization RCE.
Subprocess-level isolation (separate PID, separate Python interpreter)
is the only containment B1 provides; seccomp / landlock are deferred to
a named follow-up phase before mainnet activation (per B1-D5 in
DEFERRED.md).

What this module ships:

  * `EvalSubprocessError` — exception class for any failure during a
    subprocess eval (non-zero exit, timeout, missing / malformed
    output). Carries `stderr_tail` and `exit_code` so callers can
    distinguish "miner crashed our subprocess" from "we didn't
    write the args right".
  * `run_eval_in_subprocess(checkpoint_path, config, *, bundle_sha256,
    bundle_dir, vocab_size, hardness_index_path=None, patch_path=None,
    ralph_root=None, output_dir=None, timeout_s=600.0,
    command_prefix=None) -> DownstreamReport` — the wrapper.

The wrapper:
  1. Allocates a tmpdir under `output_dir` (caller-owned) or via
     `tempfile.mkdtemp()` (auto-cleaned).
  2. Serializes `config` to a JSON file in the tmpdir.
  3. Builds an argv: `command_prefix + [--checkpoint, --config,
     --output, --bundle-sha, --bundle-dir, --vocab-size, ...]`. The
     `command_prefix` defaults to `[sys.executable, "-m",
     "eval.downstream.runner_cli"]`. Tests pass a synthetic command
     prefix that points at a test entrypoint script.
  4. Invokes `subprocess.run(argv, capture_output=True, timeout=…)`.
  5. On non-zero exit / timeout / missing output / malformed JSON:
     raises `EvalSubprocessError` with the most recent stderr tail
     (last `STDERR_TAIL_LIMIT` chars) so the caller logs an actionable
     error.
  6. On success: parses the output JSON into a `DownstreamReport` and
     returns it.

The IPC contract (config in, report out) is intentionally identical to
what the in-process `run_downstream_eval` accepts and emits, so a future
refactor can swap subprocess invocation for in-process invocation by
just bypassing this wrapper. Today's CLI (`runner_cli.py`, separate PR)
deserializes the config the same way `EvalConfig.from_dict` does, calls
`run_downstream_eval`, and serializes the report via the helper here.

What this module does NOT ship:

  * The CLI entrypoint itself (separate PR). The wrapper's tests use a
    synthetic in-tree entrypoint script that produces a known
    `DownstreamReport` JSON without loading a real checkpoint.
  * Subprocess sandboxing beyond what the OS provides (seccomp /
    landlock — deferred).
  * Per-task resource limits (memory caps, GPU partitioning) — those
    are a B5 deployment concern, not a B1 protocol concern.

Reference scope: docs/build_scope/02_scope_B1.md "runner.py subprocess
wrapper".
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .runner import EvalConfig
from .types import (
    HARNESS_VERSION,
    CellResult,
    DownstreamReport,
)

# Truncate stderr to this many chars in EvalSubprocessError. Full stderr
# is still in the subprocess's own logs; this is a budget for the
# in-process exception message that the validator's log line will carry.
STDERR_TAIL_LIMIT = 4096

# Default argv prefix for the production CLI. Tests pass a custom one.
DEFAULT_COMMAND_PREFIX: tuple[str, ...] = (
    sys.executable,
    "-m",
    "eval.downstream.runner_cli",
)

# Ralph repo root, resolved at import time. We prepend this to the
# subprocess's PYTHONPATH so `eval.downstream.runner_cli` (the production
# CLI) and `eval.downstream.types` (its imports) resolve regardless of
# the caller's cwd or how the package was installed.
RALPH_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------------------------------------------------------
# Exception
# ----------------------------------------------------------------------------


@dataclass
class EvalSubprocessError(Exception):
    """A subprocess eval call failed before producing a valid report.

    Attributes:
      reason — one-line failure category ("nonzero_exit", "timeout",
        "missing_output", "malformed_output", "config_write_failed").
      exit_code — subprocess exit code, or None if the subprocess never
        completed (e.g., timeout).
      stderr_tail — last STDERR_TAIL_LIMIT chars of subprocess stderr.
        Empty string if no stderr was captured.
      argv — the argv passed to subprocess.run, for reproducibility in
        post-mortem logs.
    """

    reason: str
    exit_code: int | None
    stderr_tail: str
    argv: list[str]

    def __str__(self) -> str:
        head = f"{self.reason} (exit={self.exit_code})"
        if self.stderr_tail:
            return f"{head}\nstderr tail:\n{self.stderr_tail}"
        return head


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _truncate_stderr(stderr: str | bytes) -> str:
    """Decode + tail-truncate to STDERR_TAIL_LIMIT chars."""
    if isinstance(stderr, bytes):
        try:
            stderr = stderr.decode("utf-8", errors="replace")
        except Exception:
            stderr = repr(stderr)
    if len(stderr) <= STDERR_TAIL_LIMIT:
        return stderr
    return stderr[-STDERR_TAIL_LIMIT:]


def _parse_report(payload: dict) -> DownstreamReport:
    """Inverse of `_serialize_report`. Raises ValueError on missing keys.

    Used by the wrapper to decode the subprocess's output JSON. Also
    used by tests that want to assert on report contents without
    running the real subprocess.
    """
    cells_in = payload.get("cells", {})
    cells: dict[str, CellResult] = {}
    for key, cell_dict in cells_in.items():
        cells[key] = CellResult(
            task=cell_dict["task"],
            accuracy=float(cell_dict["accuracy"]),
            accuracy_stderr=float(cell_dict.get("accuracy_stderr", 0.0)),
            n_examples=int(cell_dict.get("n_examples", 0)),
            seed=int(cell_dict.get("seed", 0)),
        )
    return DownstreamReport(
        harness_version=str(payload["harness_version"]),
        bundle_sha256=str(payload["bundle_sha256"]),
        seed=int(payload["seed"]),
        total_examples=int(payload["total_examples"]),
        wall_clock_s=float(payload.get("wall_clock_s", 0.0)),
        cells=cells,
    )


def serialize_report(report: DownstreamReport) -> dict:
    """Serialize a `DownstreamReport` to a JSON-safe dict.

    Public because the CLI uses this to write its output too — keeping
    the writer and reader on the same helper avoids drift in the IPC
    contract.
    """
    return {
        "harness_version": report.harness_version,
        "bundle_sha256": report.bundle_sha256,
        "seed": report.seed,
        "total_examples": report.total_examples,
        "wall_clock_s": report.wall_clock_s,
        "cells": {
            key: {
                "task": cell.task,
                "accuracy": cell.accuracy,
                "accuracy_stderr": cell.accuracy_stderr,
                "n_examples": cell.n_examples,
                "seed": cell.seed,
            }
            for key, cell in report.cells.items()
        },
    }


def deserialize_report(payload: dict) -> DownstreamReport:
    """Public alias for `_parse_report` — same role for the IPC contract."""
    return _parse_report(payload)


# ----------------------------------------------------------------------------
# The wrapper
# ----------------------------------------------------------------------------


def run_eval_in_subprocess(
    checkpoint_path: Path,
    config: EvalConfig,
    *,
    bundle_sha256: str,
    bundle_dir: Path,
    vocab_size: int,
    hardness_index_path: Path | None = None,
    patch_path: Path | None = None,
    ralph_root: Path | None = None,
    output_dir: Path | None = None,
    timeout_s: float = 600.0,
    command_prefix: Sequence[str] | None = None,
    env: dict[str, str] | None = None,
) -> DownstreamReport:
    """Run a downstream eval in an isolated subprocess and return the report.

    Args:
      checkpoint_path: path to the miner-submitted checkpoint. The
        subprocess loads it with `weights_only=True`. The wrapper
        does NOT validate the file exists — the subprocess raises
        cleanly if it doesn't, and the wrapper converts that into an
        `EvalSubprocessError`.
      config: `EvalConfig`. Serialized to a JSON file in the run's
        working dir and passed via `--config <path>`.
      bundle_sha256: pinned SHA of the DCLM eval bundle (per B1-D2).
        Passed through to the subprocess via `--bundle-sha`.
      bundle_dir: path to the local cached DCLM bundle (the
        unzipped `eval_bundle/`). Passed via `--bundle-dir`.
      vocab_size: tokenizer vocab the CLI must validate against
        (must be 50257 today; per B1-D6).
      hardness_index_path: optional JSONL hardness-index path.
        Required when `config.tasks` includes a private_hard task.
      patch_path: optional structural-patch file path (per B1-D13).
      ralph_root: optional ralph repo root for structural-patch
        application (per B1-D13).
      output_dir: caller-controlled directory for the run's working
        files (config, output, hardness-index copy). If None, a
        tempfile.mkdtemp() is used and removed on return.
      timeout_s: wall-clock budget for the subprocess. Default 600s
        (10 min) — enough for one S₃ ladder rung on H100; the
        validator should tune this per scale.
      command_prefix: argv prefix for the subprocess. Default
        `(sys.executable, "-m", "eval.downstream.runner_cli")`. Tests
        pass a custom prefix pointing at a synthetic entrypoint.
      env: optional environment dict for the subprocess. Default is
        the wrapper's current env (subprocess.run inherits it).

    Raises:
      EvalSubprocessError on any failure path (non-zero exit, timeout,
      missing output, malformed JSON, output schema mismatch).

    Returns:
      `DownstreamReport` parsed from the subprocess's output JSON.
    """
    if command_prefix is None:
        command_prefix = DEFAULT_COMMAND_PREFIX

    owned_tmpdir = output_dir is None
    work_dir = Path(tempfile.mkdtemp()) if owned_tmpdir else Path(output_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    config_path = work_dir / "config.json"
    output_path = work_dir / "report.json"

    try:
        try:
            config_path.write_text(json.dumps(config.to_dict()))
        except OSError as e:
            raise EvalSubprocessError(
                reason="config_write_failed",
                exit_code=None,
                stderr_tail=str(e),
                argv=list(command_prefix),
            ) from e

        argv: list[str] = list(command_prefix) + [
            "--checkpoint", str(checkpoint_path),
            "--config", str(config_path),
            "--output", str(output_path),
            "--bundle-sha", bundle_sha256,
            "--bundle-dir", str(bundle_dir),
            "--vocab-size", str(vocab_size),
        ]
        if hardness_index_path is not None:
            argv += ["--hardness-index", str(hardness_index_path)]
        if patch_path is not None:
            argv += ["--patch", str(patch_path)]
        if ralph_root is not None:
            argv += ["--ralph-root", str(ralph_root)]

        sub_env = (env if env is not None else os.environ).copy()
        existing_pp = sub_env.get("PYTHONPATH", "")
        ralph_root_str = str(RALPH_ROOT)
        sub_env["PYTHONPATH"] = (
            f"{ralph_root_str}{os.pathsep}{existing_pp}"
            if existing_pp
            else ralph_root_str
        )

        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                timeout=timeout_s,
                check=False,
                env=sub_env,
            )
        except subprocess.TimeoutExpired as e:
            stderr_tail = _truncate_stderr(e.stderr or b"")
            raise EvalSubprocessError(
                reason="timeout",
                exit_code=None,
                stderr_tail=stderr_tail,
                argv=argv,
            ) from e

        if completed.returncode != 0:
            raise EvalSubprocessError(
                reason="nonzero_exit",
                exit_code=completed.returncode,
                stderr_tail=_truncate_stderr(completed.stderr),
                argv=argv,
            )

        if not output_path.exists():
            raise EvalSubprocessError(
                reason="missing_output",
                exit_code=completed.returncode,
                stderr_tail=_truncate_stderr(completed.stderr),
                argv=argv,
            )

        try:
            payload = json.loads(output_path.read_text())
        except json.JSONDecodeError as e:
            raise EvalSubprocessError(
                reason="malformed_output",
                exit_code=completed.returncode,
                stderr_tail=f"output JSON parse error: {e}\n"
                            + _truncate_stderr(completed.stderr),
                argv=argv,
            ) from e

        try:
            report = _parse_report(payload)
        except (KeyError, ValueError, TypeError) as e:
            raise EvalSubprocessError(
                reason="malformed_output",
                exit_code=completed.returncode,
                stderr_tail=f"report schema mismatch: {e}\n"
                            + _truncate_stderr(completed.stderr),
                argv=argv,
            ) from e

        if report.harness_version != HARNESS_VERSION:
            raise EvalSubprocessError(
                reason="malformed_output",
                exit_code=completed.returncode,
                stderr_tail=(
                    f"harness_version mismatch: subprocess reported "
                    f"{report.harness_version!r}, wrapper expects "
                    f"{HARNESS_VERSION!r}"
                ),
                argv=argv,
            )

        return report

    finally:
        if owned_tmpdir:
            shutil.rmtree(work_dir, ignore_errors=True)
