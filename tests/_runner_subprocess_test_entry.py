"""Synthetic subprocess entrypoint for test_downstream_runner_subprocess.py.

This is NOT the production CLI. It's a minimal stub that the subprocess
wrapper tests invoke as a subprocess instead of the real
`eval.downstream.runner_cli`. Its behaviour is controlled by a few
env vars so each test can drive the success / failure paths
deterministically without needing a real RalphBase checkpoint.

Env vars:
  RALPH_TEST_RUNNER_MODE — one of:
    "success"           — write a valid DownstreamReport JSON and exit 0
    "nonzero"           — write nothing, exit with code from RALPH_TEST_RUNNER_EXIT_CODE
    "no_output"         — exit 0 without writing the output file
    "malformed_output"  — write invalid JSON, exit 0
    "schema_mismatch"   — write a JSON without required keys, exit 0
    "wrong_version"     — write a JSON with a bad harness_version, exit 0
    "slow"              — sleep RALPH_TEST_RUNNER_SLEEP_S seconds, then succeed
  RALPH_TEST_RUNNER_EXIT_CODE — exit code for "nonzero" mode (default 1).
  RALPH_TEST_RUNNER_SLEEP_S — sleep duration for "slow" mode.

The args parsed match the real CLI's surface so the wrapper can pass
its argv unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--bundle-sha", required=True)
    p.add_argument("--bundle-dir", required=True)
    p.add_argument("--vocab-size", required=True, type=int)
    p.add_argument("--hardness-index", default=None)
    p.add_argument("--patch", default=None)
    p.add_argument("--ralph-root", default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    mode = os.environ.get("RALPH_TEST_RUNNER_MODE", "success")

    if mode == "slow":
        sleep_s = float(os.environ.get("RALPH_TEST_RUNNER_SLEEP_S", "0"))
        time.sleep(sleep_s)
        mode = "success"  # fall through

    if mode == "nonzero":
        exit_code = int(os.environ.get("RALPH_TEST_RUNNER_EXIT_CODE", "1"))
        print("simulated failure on stderr", file=sys.stderr)
        return exit_code

    if mode == "no_output":
        # Successfully exit but produce no output file.
        return 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "malformed_output":
        output_path.write_text("{this is not valid json")
        return 0

    if mode == "schema_mismatch":
        # Missing required keys (e.g. harness_version, seed).
        output_path.write_text(json.dumps({"cells": {}}))
        return 0

    if mode == "wrong_version":
        # Valid schema but wrong harness_version.
        output_path.write_text(json.dumps({
            "harness_version": "999.999.999-test",
            "bundle_sha256": args.bundle_sha,
            "seed": 0,
            "total_examples": 0,
            "wall_clock_s": 0.0,
            "cells": {},
        }))
        return 0

    # mode == "success" — emit a deterministic report so tests can assert
    # on its contents. We import the live HARNESS_VERSION so the test
    # stub never lies about the schema version it claims to produce.
    from eval.downstream.types import HARNESS_VERSION
    report_dict = {
        "harness_version": HARNESS_VERSION,
        "bundle_sha256": args.bundle_sha,
        "seed": 42,
        "total_examples": 7,
        "wall_clock_s": 0.5,
        "cells": {
            "arc_easy:S3": {
                "task": "arc_easy",
                "accuracy": 0.875,
                "accuracy_stderr": 0.0,
                "n_examples": 7,
                "seed": 42,
            },
        },
    }
    output_path.write_text(json.dumps(report_dict))
    return 0


if __name__ == "__main__":
    sys.exit(main())
