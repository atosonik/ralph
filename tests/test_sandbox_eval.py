"""End-to-end (CPU, no container): the sandbox entrypoint emits per-position
NLLs whose HOST-side reduction reproduces the in-process val_bpb exactly. This is
the correctness contract the op4 sandbox wiring depends on."""
from __future__ import annotations

import dataclasses
import json
import sys

import numpy as np
import pytest
import torch

import ralph_bootstrap

# The canonical model package lives under RECIPE_DIR.
RECIPE_DIR = str(ralph_bootstrap.RECIPE_DIR)
if RECIPE_DIR not in sys.path:
    sys.path.insert(0, RECIPE_DIR)

try:
    from model import RalphBase, RalphConfig  # noqa: E402
    _HAVE_MODEL = True
except Exception:  # noqa: BLE001
    _HAVE_MODEL = False

pytestmark = pytest.mark.skipif(not _HAVE_MODEL, reason="canonical model package not importable")


def _tiny_model_and_ckpt(tmp_path):
    torch.manual_seed(0)
    cfg = RalphConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2, head_dim=16,
        ffn_mult=2.0, max_seq_len=16,
    )
    model = RalphBase(cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, ckpt)
    return cfg, model, ckpt


def test_sandbox_eval_reduction_matches_in_process_val_bpb(tmp_path):
    from eval.host_reduce import expected_token_count, reduce_token_nlls
    from eval.val_bpb import compute_val_bpb
    from validator.sandbox_eval import run_sandbox_eval

    cfg, model, ckpt = _tiny_model_and_ckpt(tmp_path)
    rng = np.random.default_rng(7)
    tokens = rng.integers(0, cfg.vocab_size, size=200, dtype=np.uint16)
    eval_dir = tmp_path / "evald"
    eval_dir.mkdir()
    tokens.tofile(eval_dir / "active_tokens.bin")  # no benchmark file -> bench 0.0

    out_dir = tmp_path / "out"
    # workdir = RECIPE_DIR so the canonical `model` package resolves (no patch).
    nlls = run_sandbox_eval(RECIPE_DIR, ckpt, eval_dir, out_dir)

    # Container artifacts exist and are well-formed.
    saved = np.load(out_dir / "nlls.npy")
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["status"] == "ok"
    seq_len = cfg.max_seq_len // 2
    assert saved.shape[0] == expected_token_count(len(tokens), seq_len)
    assert manifest["tokens_emitted"] == saved.shape[0]

    # HOST reduction of the emitted NLLs == the in-process computation.
    ref = compute_val_bpb(model, tokens, seq_len, bytes_per_token=4.0)
    host = reduce_token_nlls(
        nlls, seq_len=seq_len, bytes_per_token=4.0,
        expected_tokens=expected_token_count(len(tokens), seq_len),
        eval_set_hash="x",
    )
    assert host.val_bpb == pytest.approx(ref["val_bpb"], rel=1e-5)
    assert host.tail_val_bpb == pytest.approx(ref["tail_val_bpb"], rel=1e-5)
    assert host.tokens_evaluated == ref["tokens_evaluated"]


def test_op4_routes_through_sandbox_and_host_reduces(tmp_path, monkeypatch):
    """RALPH_SANDBOX=1 -> op4 runs the model in the container (mocked) and the
    HOST reduces val_bpb from the emitted nlls — matching the in-process value."""
    import validator.sandbox as sbx
    import validator.validator as vv
    from eval.val_bpb import compute_val_bpb
    from validator.sandbox import SandboxResult

    cfg, model, _ = _tiny_model_and_ckpt(tmp_path)
    proof = tmp_path / "proof"
    (proof / "training").mkdir(parents=True)
    torch.save(
        {"model": model.state_dict(), "config": dataclasses.asdict(cfg)},
        proof / "training" / "checkpoint.pt",
    )
    (proof / "patch.diff").write_text("")  # canonical (no structural change)
    ralph_root = tmp_path / "root"
    evdir = ralph_root / "eval" / "private"
    evdir.mkdir(parents=True)
    tokens = np.random.default_rng(11).integers(0, cfg.vocab_size, size=300, dtype=np.uint16)
    tokens.tofile(evdir / "active_tokens.bin")

    monkeypatch.setenv("RALPH_SANDBOX", "1")
    monkeypatch.setenv("RALPH_SANDBOX_IMAGE", "ralph-eval-sandbox@sha256:" + "a" * 64)

    def fake_run_in_sandbox(cfg_, *, container_argv, mounts, out_dir, timeout_s, **kw):
        # Simulate the container: run the real entrypoint against the HOST paths
        # the mounts point at, writing nlls.npy + manifest.json into out_dir.
        from validator.sandbox_eval import run_prepare_and_eval
        canon = next(m.host for m in mounts if m.container == "/canon")
        indir = next(m.host for m in mounts if m.container == "/in")
        evald = next(m.host for m in mounts if m.container == "/eval-private")
        run_prepare_and_eval(canon, indir / "patch.diff", indir / "training" / "checkpoint.pt", evald, out_dir)
        return SandboxResult(returncode=0, stdout="ok", stderr="", timed_out=False)

    monkeypatch.setattr(sbx, "run_in_sandbox", fake_run_in_sandbox)

    ok, detail, result = vv.op4_hidden_eval(ralph_root, proof)
    assert ok, detail
    assert "sandboxed" in detail
    ref = compute_val_bpb(model, tokens, cfg.max_seq_len // 2, bytes_per_token=4.0)
    assert result.val_bpb == pytest.approx(ref["val_bpb"], rel=1e-4)
    assert result.tokens_evaluated == ref["tokens_evaluated"]


def test_op4_sandbox_fails_closed_when_runtime_unavailable(tmp_path, monkeypatch):
    """RALPH_SANDBOX=1 but the runtime preflight fails -> op4 REJECTS (no fallback)."""
    import validator.sandbox as sbx
    import validator.validator as vv
    from validator.sandbox import SandboxUnavailable

    cfg, model, _ = _tiny_model_and_ckpt(tmp_path)
    proof = tmp_path / "proof"
    (proof / "training").mkdir(parents=True)
    torch.save({"model": model.state_dict(), "config": dataclasses.asdict(cfg)}, proof / "training" / "checkpoint.pt")
    (proof / "patch.diff").write_text("")
    evdir = tmp_path / "root" / "eval" / "private"
    evdir.mkdir(parents=True)
    np.zeros(300, dtype=np.uint16).tofile(evdir / "active_tokens.bin")

    monkeypatch.setenv("RALPH_SANDBOX", "1")
    monkeypatch.setenv("RALPH_SANDBOX_IMAGE", "ralph-eval-sandbox@sha256:" + "a" * 64)

    def boom(*a, **k):
        raise SandboxUnavailable("docker not on PATH")
    monkeypatch.setattr(sbx, "run_in_sandbox", boom)

    ok, detail, result = vv.op4_hidden_eval(tmp_path / "root", proof)
    assert not ok and result is None and "FAIL-CLOSED" in detail


def test_prepare_workdir_copies_canon_and_applies_empty_patch(tmp_path):
    from validator.sandbox_eval import prepare_workdir

    canon = tmp_path / "canon"
    (canon / "model").mkdir(parents=True)
    (canon / "model" / "x.py").write_text("# canonical\n")
    patch = tmp_path / "patch.diff"
    patch.write_text("")  # empty patch → no-op

    wd = prepare_workdir(canon, patch, tmp_path / "workdir")
    assert (wd / "model" / "x.py").read_text() == "# canonical\n"
