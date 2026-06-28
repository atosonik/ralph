"""Validator-pinned hidden-eval window (EVAL_SEQ_LEN).

The eval seq_len must be the VALIDATOR's, not the miner's. Previously every path
derived it from the miner's checkpoint (cfg.max_seq_len // 2), letting a miner
enlarge max_seq_len to be scored on an easier, longer-context eval than the king
faced — and making different models non-comparable. `pinned_eval_seq_len` is the
single source of truth; the sandbox host independently re-derives it and REJECTS
a container that echoes anything else (verified, not trusted).
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.val_bpb import EVAL_SEQ_LEN, pinned_eval_seq_len

RECIPE_DIR = str(ralph_bootstrap.RECIPE_DIR)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)
try:
    from model import RalphBase, RalphConfig  # noqa: E402
    _HAVE_MODEL = True
except Exception:  # noqa: BLE001
    _HAVE_MODEL = False


def test_pinned_seq_len_caps_floors_and_never_miner_widened():
    assert pinned_eval_seq_len(2048) == EVAL_SEQ_LEN     # large context capped DOWN to the pin
    assert pinned_eval_seq_len(EVAL_SEQ_LEN) == EVAL_SEQ_LEN
    assert pinned_eval_seq_len(256) == 256               # small-context model: its own max
    assert pinned_eval_seq_len(1) == 2                   # floored at 2 (needs seq_len+1 tokens)
    # The old miner-controlled value for a long-context model is NOT what we pin.
    assert pinned_eval_seq_len(4096) != 4096 // 2


@pytest.mark.skipif(not _HAVE_MODEL, reason="canonical model package not importable")
def test_sandbox_host_rejects_tampered_seq_len(tmp_path, monkeypatch):
    """A container that echoes seq_len != the host-pinned value is REJECTED — the
    host re-derives the window itself and never trusts the manifest."""
    import validator.sandbox as sbx
    import validator.validator as vv
    from validator.sandbox import SandboxResult

    cfg = RalphConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_mult=2.0, max_seq_len=16,
    )
    model = RalphBase(cfg)
    proof = tmp_path / "proof"
    (proof / "training").mkdir(parents=True)
    torch.save(
        {"model": model.state_dict(), "config": dataclasses.asdict(cfg)},
        proof / "training" / "checkpoint.pt",
    )
    (proof / "patch.diff").write_text("")
    ralph_root = tmp_path / "root"
    evdir = ralph_root / "eval" / "private"
    evdir.mkdir(parents=True)
    np.random.default_rng(1).integers(0, cfg.vocab_size, size=300, dtype=np.uint16).tofile(
        evdir / "active_tokens.bin"
    )

    monkeypatch.setenv("RALPH_SANDBOX", "1")
    monkeypatch.setenv("RALPH_SANDBOX_IMAGE", "ralph-eval-sandbox@sha256:" + "a" * 64)

    bad = cfg.max_seq_len // 2  # 8, the OLD miner-controlled value != pinned 16

    def tampering_sandbox(cfg_, *, container_argv, mounts, out_dir, timeout_s, **kw):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        n_windows = (300 - 1) // bad
        np.save(out / "nlls.npy", np.ones(n_windows * bad, dtype=np.float32))
        (out / "manifest.json").write_text(json.dumps({
            "status": "ok", "seq_len": bad, "tokens_emitted": n_windows * bad,
            "benchmark_accuracy": 0.0, "benchmark_examples": 0, "model_config": {},
        }))
        return SandboxResult(returncode=0, stdout="ok", stderr="", timed_out=False)

    monkeypatch.setattr(sbx, "run_in_sandbox", tampering_sandbox)

    assert pinned_eval_seq_len(cfg.max_seq_len) == 16 and bad == 8  # the gap under test
    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof)
    assert not ok and result is None
    assert "seq_len mismatch" in detail
