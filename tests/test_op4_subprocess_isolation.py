"""op4 eval runs in a subprocess (crash isolation) AND the child never inherits
the validator's secrets (the regression that sank PR#61's first cut)."""
from __future__ import annotations

import subprocess as _sp

import pytest

from validator.validator import _run_eval_subprocess

_OK_LINE = (
    "RALPH_EVAL_RESULT val_bpb=1.234 benchmark_acc=0.6 tokens_evaluated=50 "
    "benchmark_examples=5 eval_set_hash=" + "b" * 64 + " tail_val_bpb=1.30\n"
)


class _Proc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_subprocess_gets_sanitized_env(monkeypatch, tmp_path):
    # miner model code runs in this child — secrets must NOT reach it.
    monkeypatch.setenv("RALPH_VALIDATOR_PRIVKEY", "seal_should_not_leak_123")
    monkeypatch.setenv("BT_WALLET_PASSWORD", "wallet_pw_should_not_leak")
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak_abcdefgh")
    captured = {}

    def fake(argv, **kw):
        captured["env"] = kw.get("env")
        return _Proc(out=_OK_LINE)

    monkeypatch.setattr(_sp, "run", fake)
    ok, detail, _ = _run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert ok, detail
    env = captured["env"]
    assert env is not None, "subprocess spawned with no explicit env (would inherit secrets)"
    for k in ("RALPH_VALIDATOR_PRIVKEY", "BT_WALLET_PASSWORD", "HF_TOKEN",
              "RALPH_SKIP_HANDSHAKE", "RALPH_ALLOW_MOCK_ATTESTATION", "RALPH_TEST_MODE"):
        assert k not in env, f"{k} leaked to the miner-code subprocess"
    leaked = {"seal_should_not_leak_123", "wallet_pw_should_not_leak", "hf_should_not_leak_abcdefgh"}
    assert not (leaked & set(env.values())), "a secret value leaked into the subprocess env"


def test_synthetic_eval_toggle_is_forwarded_but_not_set_on_mainnet(monkeypatch, tmp_path):
    # The canonical eval harness in the child needs RALPH_ALLOW_SYNTHETIC_EVAL;
    # it's the validator's own toggle, so it IS forwarded when set...
    captured = {}
    monkeypatch.setattr(_sp, "run", lambda argv, **kw: (captured.update(env=kw.get("env")), _Proc(out=_OK_LINE))[1])
    monkeypatch.setenv("RALPH_ALLOW_SYNTHETIC_EVAL", "1")
    _run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert captured["env"].get("RALPH_ALLOW_SYNTHETIC_EVAL") == "1"
    # ...and absent (fail-closed) when the validator hasn't set it (mainnet).
    monkeypatch.delenv("RALPH_ALLOW_SYNTHETIC_EVAL", raising=False)
    _run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert "RALPH_ALLOW_SYNTHETIC_EVAL" not in captured["env"]


def test_nonzero_exit_is_rejected_not_crash(monkeypatch, tmp_path):
    # a CUDA abort in the child surfaces as a rejection, not a validator crash.
    monkeypatch.setattr(_sp, "run", lambda argv, **kw: _Proc(rc=1, err="CUDA error: device-side assert"))
    ok, detail, result = _run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert not ok and result is None and "exit=1" in detail


def test_timeout_is_rejected(monkeypatch, tmp_path):
    def boom(argv, **kw):
        raise _sp.TimeoutExpired(argv, kw.get("timeout", 1))
    monkeypatch.setattr(_sp, "run", boom)
    ok, detail, result = _run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "patched-eval")
    assert not ok and result is None and "timed out" in detail


def test_parses_result_line(monkeypatch, tmp_path):
    monkeypatch.setattr(_sp, "run", lambda argv, **kw: _Proc(out="noise\n" + _OK_LINE))
    ok, detail, result = _run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert ok and result.val_bpb == pytest.approx(1.234) and result.tail_val_bpb == pytest.approx(1.30)


def test_no_result_line_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(_sp, "run", lambda argv, **kw: _Proc(out="nothing useful"))
    ok, detail, result = _run_eval_subprocess(tmp_path, tmp_path / "ckpt.pt", tmp_path, "canonical-eval")
    assert not ok and result is None and "RALPH_EVAL_RESULT" in detail
