"""Tests for validator/sandbox.py — the hardened Docker boundary for untrusted
miner code. All run without a Docker daemon (argv builder + mocked preflight)."""
from __future__ import annotations

from pathlib import Path

import pytest

import validator.sandbox as sbx
from validator.sandbox import (
    Mount,
    SandboxConfig,
    SandboxUnavailable,
    build_container_env,
    build_docker_argv,
    parse_eval_line,
    run_in_sandbox,
)

_IMG = "ralph-eval-sandbox@sha256:" + "a" * 64


def _cfg(**kw) -> SandboxConfig:
    return SandboxConfig(image=_IMG, **kw)


def _build(cfg=None, mounts=None, out_dir="/tmp/out", env=None):
    cfg = cfg or _cfg()
    return build_docker_argv(
        cfg,
        container_argv=["python", "-m", "validator.sandbox_eval", "/in", "/out"],
        mounts=mounts or [Mount(Path("/in"), "/in", ro=True)],
        out_dir=Path(out_dir),
        name="ralph-sbx-test",
        env=env if env is not None else build_container_env({"PYTHONPATH": "/scratch/workdir"}),
    )


def test_argv_has_all_mandatory_flags():
    argv = _build()
    j = " ".join(argv)
    for required in (
        "--network=none", "--read-only", "--user=65534:65534", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--ipc=private",
        "--pids-limit=512", "--memory=16g", "--memory-swap=16g", "--cpus=8",
    ):
        assert required in argv, f"missing {required}"
    assert "--tmpfs" in argv
    assert any(t.startswith("/scratch:rw") for t in argv)
    # single pinned GPU, never "all"
    assert "--gpus" in argv and "device=0" in argv
    # image pinned by digest + writable out mount
    assert _IMG in argv
    assert any(t.endswith(":/out:rw") for t in argv)
    # nothing dangerous
    for bad in ("--privileged", "--pid=host", "--net=host"):
        assert bad not in j


def test_argv_rejects_unpinned_image():
    with pytest.raises(ValueError, match="digest"):
        build_docker_argv(
            SandboxConfig(image="ralph-eval-sandbox:latest"),
            container_argv=["true"], mounts=[], out_dir=Path("/tmp/o"),
            name="x", env={},
        )


def test_argv_rejects_gpus_all_and_forbidden_flags():
    with pytest.raises(ValueError, match="device"):
        sbx._assert_argv_safe(["docker", "run", "--gpus", "all", _IMG])
    with pytest.raises(ValueError, match="forbidden"):
        sbx._assert_argv_safe(["docker", "run", "--privileged", _IMG])


def test_argv_refuses_secret_mounts():
    for secret in ("/root/.bittensor", "/root/.ralph_validator_enc_key.json", "/root/.ssh"):
        with pytest.raises(ValueError, match="secret"):
            _build(mounts=[Mount(Path(secret), "/x", ro=True)])


def test_container_env_excludes_secrets_and_rejects_blocklist():
    env = build_container_env({"PYTHONPATH": "/scratch/workdir"})
    for secret in ("RALPH_VALIDATOR_PRIVKEY", "BT_WALLET_PASSWORD", "HF_TOKEN", "GH_TOKEN"):
        assert secret not in env
    assert env["PYTHONPATH"] == "/scratch/workdir"
    with pytest.raises(ValueError):
        build_container_env({"RALPH_VALIDATOR_PRIVKEY": "x"})


def test_preflight_fails_closed(monkeypatch):
    # Simulate a box with no docker → preflight must raise.
    monkeypatch.setattr(sbx, "_check_docker", lambda reasons: reasons.append("docker not on PATH"))
    monkeypatch.setattr(sbx, "_check_runc", lambda reasons: None)
    monkeypatch.setattr(sbx, "_check_nvidia_toolkit", lambda reasons, require_gpu: None)
    monkeypatch.setattr(sbx, "_check_image", lambda reasons, image: None)
    with pytest.raises(SandboxUnavailable, match="docker not on PATH"):
        sbx.preflight(_cfg())


def test_preflight_passes_when_all_checks_ok(monkeypatch):
    for name in ("_check_docker", "_check_runc"):
        monkeypatch.setattr(sbx, name, lambda reasons: None)
    monkeypatch.setattr(sbx, "_check_nvidia_toolkit", lambda reasons, require_gpu: None)
    monkeypatch.setattr(sbx, "_check_image", lambda reasons, image: None)
    sbx.preflight(_cfg())  # must not raise


def test_run_in_sandbox_is_fail_closed(monkeypatch):
    def _boom(cfg, **kw):
        raise SandboxUnavailable("no runtime")
    monkeypatch.setattr(sbx, "preflight", _boom)
    with pytest.raises(SandboxUnavailable):
        run_in_sandbox(_cfg(), container_argv=["true"], mounts=[], out_dir=Path("/tmp/o"), timeout_s=10)


def test_run_in_sandbox_builds_hardened_argv(monkeypatch, tmp_path):
    captured = {}

    class _Proc:
        returncode = 0
        stdout = (
            "RALPH_EVAL_RESULT val_bpb=1.234567 benchmark_acc=0.5 tokens_evaluated=100 "
            "benchmark_examples=10 eval_set_hash=" + "b" * 64 + "\n"
        )
        stderr = ""

    def _fake_run(argv, **kw):
        captured["argv"] = argv
        return _Proc()

    monkeypatch.setattr(sbx.subprocess, "run", _fake_run)
    res = run_in_sandbox(
        _cfg(), container_argv=["python", "x.py"], mounts=[Mount(tmp_path, "/in")],
        out_dir=tmp_path, timeout_s=30, skip_preflight=True,
    )
    assert "--network=none" in captured["argv"]
    assert "--cap-drop=ALL" in captured["argv"]
    assert res.returncode == 0
    assert res.eval_line is not None


def test_run_in_sandbox_timeout_is_killed(monkeypatch, tmp_path):
    import subprocess as _sp
    killed = {}

    def _fake_run(argv, **kw):
        if argv[:2] == ["docker", "kill"]:
            killed["name"] = argv[2]
            return _sp.CompletedProcess(argv, 0, "", "")
        raise _sp.TimeoutExpired(argv, kw.get("timeout", 1))

    monkeypatch.setattr(sbx.subprocess, "run", _fake_run)
    res = run_in_sandbox(
        _cfg(), container_argv=["sleep", "999"], mounts=[], out_dir=tmp_path,
        timeout_s=1, skip_preflight=True,
    )
    assert res.timed_out
    assert res.returncode == 124
    assert killed.get("name", "").startswith("ralph-sbx-")


def test_parse_eval_line_quantizes_and_drops_fields():
    line = (
        "RALPH_EVAL_RESULT val_bpb=1.2345678 benchmark_acc=0.4999999 "
        "tokens_evaluated=4096 benchmark_examples=15 eval_set_hash=" + "c" * 64 + " "
        "sealed_stream_manifest_hash=deadbeef tail_val_bpb=1.30 sneaky=exfil"
    )
    out = parse_eval_line(line)
    assert out["val_bpb"] == 1.235  # round(1.2345678, 3)
    assert out["benchmark_acc"] == 0.5
    assert set(out.keys()) == {"val_bpb", "benchmark_acc", "tokens_evaluated", "benchmark_examples", "eval_set_hash"}
    assert "sneaky" not in out
    assert out["eval_set_hash"] == "c" * 64


def test_parse_eval_line_rejects_garbage_and_bad_hash():
    with pytest.raises(ValueError):
        parse_eval_line("not a result line")
    out = parse_eval_line(
        "RALPH_EVAL_RESULT val_bpb=1.0 benchmark_acc=0.5 tokens_evaluated=1 benchmark_examples=1 eval_set_hash=tooshort"
    )
    assert out["eval_set_hash"] == ""  # invalid hash dropped, not propagated
