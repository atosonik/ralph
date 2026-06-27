"""Stopgap: the op4 patched-eval subprocess must NOT inherit validator secrets.

`validator/eval_in_workdir.py` imports and executes the miner's patched
`model.py`. Before this fix the subprocess was spawned with no `env=` kwarg, so
the miner's code inherited the validator's full `os.environ` — including the
libsodium seal privkey (`RALPH_VALIDATOR_PRIVKEY`, decrypts every bundle), the
Bittensor wallet, and cloud tokens. These tests pin the allowlist-only env and
the extended blocklist.
"""
from __future__ import annotations

import subprocess as _subprocess
from pathlib import Path

import pytest

from proof.runner import _TRAINING_ENV_BLOCKLIST, _sanitized_env

# Secrets / escape hatches the eval subprocess must never see.
_MUST_BLOCK = (
    "RALPH_VALIDATOR_PRIVKEY",
    "RALPH_VALIDATOR_PRIVKEY_FILE",
    "RALPH_TEST_MODE",
    "RALPH_ALLOW_SYNTHETIC_EVAL",
    "RALPH_ALLOW_MOCK_ATTESTATION",
    "RALPH_SKIP_HANDSHAKE",
)


def test_blocklist_covers_validator_secrets():
    for name in _MUST_BLOCK:
        assert name in _TRAINING_ENV_BLOCKLIST, f"{name} missing from blocklist"


def test_sanitized_env_scrubs_secrets(monkeypatch):
    secret_val = "supersecret_privkey_value_0123456789"
    monkeypatch.setenv("RALPH_VALIDATOR_PRIVKEY", secret_val)
    monkeypatch.setenv("RALPH_VALIDATOR_PRIVKEY_FILE", "/root/.ralph_validator_enc_key.json")
    monkeypatch.setenv("BT_WALLET_PASSWORD", "wallet_password_value")
    monkeypatch.setenv("HF_TOKEN", "hf_tokenvalue_abcdefgh")
    # RALPH_ALLOW_SYNTHETIC_EVAL is set by the autouse conftest fixture.

    env = _sanitized_env(extra={"PYTHONPATH": "/tmp/workdir"})

    for name in (*_MUST_BLOCK, "BT_WALLET_PASSWORD", "HF_TOKEN"):
        assert name not in env, f"{name} leaked into sanitized env"
    assert secret_val not in env.values()
    # The allowlist + extra still pass through.
    assert env.get("PYTHONPATH") == "/tmp/workdir"


def test_sanitized_env_rejects_blocklisted_extra():
    with pytest.raises(ValueError):
        _sanitized_env(extra={"RALPH_VALIDATOR_PRIVKEY": "x"})


def test_patched_eval_subprocess_uses_sanitized_env(monkeypatch, tmp_path):
    """End-to-end: the env handed to the eval subprocess excludes every secret."""
    import shutil

    from validator.validator import _patched_hidden_eval

    # Plant secrets in the parent environment.
    monkeypatch.setenv("RALPH_VALIDATOR_PRIVKEY", "seal_privkey_should_not_leak_123")
    monkeypatch.setenv("BT_WALLET_PASSWORD", "wallet_pw_should_not_leak")
    monkeypatch.setenv("HF_TOKEN", "hf_should_not_leak_abcdefgh")

    proof_dir = tmp_path / "proof"
    (proof_dir / "training").mkdir(parents=True)
    (proof_dir / "patch.diff").write_text("")  # empty patch → apply_patch no-ops
    ckpt = proof_dir / "training" / "checkpoint.pt"
    ckpt.write_bytes(b"")

    # Neutralize the heavy/real steps; we only care about the subprocess env.
    monkeypatch.setattr(shutil, "copytree", lambda *a, **k: None)

    captured: dict = {}

    class _FakeProc:
        returncode = 0
        stdout = (
            "RALPH_EVAL_RESULT val_bpb=1.500000 benchmark_acc=0.500000 "
            "tokens_evaluated=100 benchmark_examples=10 eval_set_hash=deadbeef\n"
        )
        stderr = ""

    def _fake_run(argv, **kwargs):
        captured["env"] = kwargs.get("env")
        return _FakeProc()

    monkeypatch.setattr(_subprocess, "run", _fake_run)

    ok, _detail, _result = _patched_hidden_eval(tmp_path, proof_dir, ckpt)
    assert ok, _detail

    env = captured["env"]
    assert env is not None, "subprocess was spawned without an explicit env"
    for name in (*_MUST_BLOCK, "BT_WALLET_PASSWORD", "HF_TOKEN"):
        assert name not in env, f"{name} leaked to the miner-code subprocess"
    leaked = {"seal_privkey_should_not_leak_123", "wallet_pw_should_not_leak", "hf_should_not_leak_abcdefgh"}
    assert not (leaked & set(env.values())), "a secret value leaked into the subprocess env"
    # PYTHONPATH is the per-bundle patched workdir (a TemporaryDirectory created
    # inside the function), so we can only assert the suffix.
    assert Path(env.get("PYTHONPATH", "")).name == "workdir"
