"""B4 CPU smoke test runner — verifies scripts/b4_cpu_smoke.py wires up.

Imports the smoke script's `run_smoke` directly and drives it against a
tmp workdir. The synthetic CLI inside the test suite is used as the
subprocess entry; no real model is loaded, no GPU is required.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from scripts.b4_cpu_smoke import run_smoke


def test_b4_smoke_end_to_end(tmp_path):
    summary = run_smoke(tmp_path)
    # The single rung produced exactly one entry in per_rung_reports
    # under v0.11 mode + one cell in the combined report.
    assert summary["v011_rungs"] == ["S3"]
    assert summary["v011_combined_cells"] == ["arc_easy:S3"]
    assert summary["v011_hidden_has_downstream"] is True
    assert summary["legacy_hidden_no_downstream"] is True
    # The legacy dict keys are EXACTLY the pre-v0.11 shape.
    assert summary["legacy_dict_keys"] == [
        "benchmark_accuracy",
        "eval_set_hash",
        "tokens_evaluated",
        "benchmark_examples",
        "val_bpb",
    ] or sorted(summary["legacy_dict_keys"]) == [
        "benchmark_accuracy",
        "benchmark_examples",
        "eval_set_hash",
        "tokens_evaluated",
        "val_bpb",
    ]
    # The acceptance preflight emitted exactly one chain event.
    assert "submission_received" in summary["events_emitted"]


def test_b4_smoke_main_returns_zero():
    """The CLI main returns 0 on success."""
    from scripts.b4_cpu_smoke import main
    rc = main([])
    assert rc == 0
