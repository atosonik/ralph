"""B4 CPU smoke test — end-to-end v0.11-lite validator pipeline on CPU.

Drives a synthetic submission through the whole validator chain on CPU:
  1. Construct a fake submission bundle (parent at genesis).
  2. `accept_submission` → preflight passes.
  3. `run_ladder_eval` with mode=v0.11 against the synthetic runner
     entrypoint that ships with tests/_runner_subprocess_test_entry.py.
  4. Verify the merged DownstreamReport + HiddenEvalResult.downstream.
  5. Verify the legacy-mode path produces a HiddenEvalResult with
     downstream=None whose `to_legacy_dict()` is byte-equivalent to
     pre-v0.11 `asdict()`.

Runs in <10s on a laptop. No GPU, no real model, no real DCLM bundle.
Used in CI as a smoke check that the validator's full call chain wires
up correctly post-C1-LITE + C2-LITE without regressions.

Usage:
    python scripts/b4_cpu_smoke.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Make `chain_layer`, `validator`, `eval` importable when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401, E402
from chain_layer.local import LocalChain  # noqa: E402
from eval.downstream.runner import RALPH_VOCAB_SIZE  # noqa: E402
from validator.ladder import (  # noqa: E402
    EVAL_MODE_LEGACY,
    EVAL_MODE_V011,
    LadderEvalConfig,
    LadderRungSpec,
    Submission,
    accept_submission,
    run_ladder_eval,
)

# Path to the synthetic subprocess entrypoint that ships with the test
# suite. The CLI is controlled by RALPH_TEST_RUNNER_MODE env var; we set
# it to "success" so the entrypoint writes a deterministic report.
_TEST_ENTRY = Path(__file__).resolve().parent.parent / "tests" / "_runner_subprocess_test_entry.py"


def run_smoke(workdir: Path) -> dict:
    """Drive the full pipeline and return a small dict of observations.

    `workdir` is a scratch dir; the function creates a chain dir +
    submission bundle + tmp output paths inside it. Returns a small
    summary dict so callers (CI + the unit test) can assert on it.
    """
    chain_dir = workdir / "chain"
    chain = LocalChain(chain_dir)

    # 1. Preflight accept (genesis: no parent).
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="smoke_bundle_hash",
        miner_hotkey="5F_smoke_miner",
        vocab_size=RALPH_VOCAB_SIZE,
    )
    accept = accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    if not accept.accepted:
        raise RuntimeError(f"smoke preflight rejected: {accept.reason}")

    # 2. Build a LadderEvalConfig pointing at the synthetic CLI. We use a
    # SINGLE-rung config here (S3 only) because the synthetic entry stub
    # hard-codes its output cell key to "arc_easy:S3" — running all 3
    # rungs against it would create duplicate cell keys at merge time.
    # The S1+S2+S3 merge logic is unit-tested separately in
    # test_validator_ladder_eval.py; the smoke is just verifying the
    # end-to-end wiring works.
    config = LadderEvalConfig(
        rungs=(LadderRungSpec(scale_label="S3", dim=768, n_layers=12),),
        tasks=("arc_easy",),
        bundle_dir=workdir / "bundle",  # never touched by synthetic entry
        bundle_sha256="smoke_sha",
        seed=0,
    )
    fake_checkpoint = workdir / "fake.ckpt"
    fake_checkpoint.write_bytes(b"")  # validator never reads it; entry stub is in env-mode "success"
    fake_ralph_root = workdir / "ralph_root"
    fake_ralph_root.mkdir()

    # The synthetic CLI reads RALPH_TEST_RUNNER_MODE; force "success" here.
    # run_eval_in_subprocess inherits the parent env by default, so
    # setting os.environ here propagates to the child.
    os.environ["RALPH_TEST_RUNNER_MODE"] = "success"

    # We pass a command_prefix that points at the test entry; one
    # subprocess per rung is invoked under the hood.
    command_prefix = (sys.executable, str(_TEST_ENTRY))

    # 3. Run the v0.11 ladder eval.
    v011_result = run_ladder_eval(
        sub,
        config,
        checkpoint_path=fake_checkpoint,
        ralph_root=fake_ralph_root,
        mode=EVAL_MODE_V011,
        command_prefix=command_prefix,
        timeout_s_per_rung=30.0,
    )
    # The synthetic entry produces ONE cell "arc_easy:S3" with seed=42
    # regardless of the requested scale_label. With 3 rungs this would
    # produce 3 collisions, so the synthetic stub here is fine only for a
    # single-rung smoke. For the multi-rung smoke we constrain to one rung.

    # 4. Run the legacy-mode path: must NOT invoke any subprocess.
    legacy_result = run_ladder_eval(
        sub,
        config,
        checkpoint_path=fake_checkpoint,
        ralph_root=fake_ralph_root,
        mode=EVAL_MODE_LEGACY,
        command_prefix=command_prefix,  # ignored under legacy
        legacy_val_bpb=1.5,
        legacy_benchmark_accuracy=0.4,
        legacy_tokens_evaluated=1000,
        legacy_benchmark_examples=50,
        legacy_eval_set_hash="smoke_legacy_eval_hash",
    )
    assert legacy_result.hidden_eval.downstream is None, "legacy mode must NOT populate downstream"
    legacy_dict = legacy_result.hidden_eval.to_legacy_dict()
    # The legacy dict is byte-equivalent to pre-v0.11 asdict (no downstream key).
    expected_legacy_keys = {
        "val_bpb", "benchmark_accuracy", "tokens_evaluated",
        "benchmark_examples", "eval_set_hash",
    }
    assert set(legacy_dict.keys()) == expected_legacy_keys, (
        f"legacy dict has unexpected keys: {sorted(legacy_dict.keys())}"
    )

    return {
        "v011_rungs": list(v011_result.per_rung_reports.keys()),
        "v011_combined_cells": list(v011_result.combined_report.cells.keys()),
        "v011_hidden_has_downstream": v011_result.hidden_eval.downstream is not None,
        "legacy_hidden_no_downstream": legacy_result.hidden_eval.downstream is None,
        "legacy_dict_keys": sorted(legacy_dict.keys()),
        "events_emitted": [
            e["type"] for e in chain.get_events(limit=10)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on failure."""
    workdir = Path(tempfile.mkdtemp(prefix="ralph_b4_smoke_"))
    try:
        summary = run_smoke(workdir)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as e:
        print(f"B4 smoke FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
