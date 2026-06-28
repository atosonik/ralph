"""op4 failure must reject cleanly, not crash scoring.

When op4_hidden_eval can't evaluate a checkpoint (e.g. it won't load into the
validator's RalphBase and the patched-workdir re-eval also fails) it returns
(False, detail, None). Previously judge_submission stored hidden_eval=None and
returned a "passing" result, so score_and_decide crashed on None.val_bpb and
took down the whole epoch loop (a DoS for any unloadable checkpoint). This pins
the clean rejection.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
import validator.validator as V


def test_op4_failure_rejects_cleanly(tmp_path, monkeypatch):
    (tmp_path / "submission.json").write_text(json.dumps(
        {"miner_hotkey": "5Test", "bundle_hash": "bh", "handshake_nonce": "0x00"}))

    # op1-op3 pass; op4 fails like an unloadable checkpoint (returns None eval)
    monkeypatch.setattr(V, "op1_diff_and_integrity", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(V, "op2_attestation_verify", lambda *a, **k: (True, "ok", "verified"))
    monkeypatch.setattr(V, "op3_log_plausibility", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(
        V, "op4_hidden_eval",
        lambda *a, **k: (False, "state_dict shape mismatch + subprocess exit=1", None),
    )

    # ralph_root = the protocol repo root — derive it so the test is portable
    # across nodes (it's /workspace/ralph on the validator, never a hardcoded
    # checkout path). The op gates above are mocked, so its only role here is
    # the call signature.
    ralph_root = Path(__file__).resolve().parent.parent
    res = V.judge_submission(ralph_root, tmp_path)

    assert res.rejected is not None
    assert res.rejected.reason == "op4_hidden_eval"
    assert "shape mismatch" in res.rejected.detail
    assert res.hidden_eval is None  # not a "passing" result
