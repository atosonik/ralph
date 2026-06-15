"""Tests for the patched-model fallback in op4_hidden_eval.

When a miner's structural patch adds parameters to RalphBase, the validator's
canonical model can't load the resulting checkpoint. validator.py falls back
to _patched_hidden_eval which applies the patch in a temp workdir and runs
eval_in_workdir.py as a subprocess so the patched model code scores the
checkpoint. These tests pin the recovery behaviour.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval import HiddenEvalResult
from validator import validator as vmod

# ----------------------------------------------------------------------------
# _is_state_dict_shape_mismatch
# ----------------------------------------------------------------------------


def test_shape_mismatch_detects_unexpected_key():
    err = RuntimeError(
        "Error(s) in loading state_dict for RalphBase:\n"
        '\tUnexpected key(s) in state_dict: "blocks.0.attn.q_norm.weight"'
    )
    assert vmod._is_state_dict_shape_mismatch(err) is True


def test_shape_mismatch_detects_missing_key():
    err = RuntimeError(
        'Error(s) in loading state_dict for RalphBase:\n\tMissing key(s) in state_dict: "extra.bias"'
    )
    assert vmod._is_state_dict_shape_mismatch(err) is True


def test_shape_mismatch_detects_size_mismatch():
    err = RuntimeError(
        "size mismatch for blocks.0.attn.q_proj.weight: copying a param with shape "
        "torch.Size([32, 32]) from checkpoint, the shape in current model is "
        "torch.Size([16, 16])."
    )
    assert vmod._is_state_dict_shape_mismatch(err) is True


def test_shape_mismatch_rejects_unrelated_runtime_error():
    err = RuntimeError("CUDA out of memory")
    assert vmod._is_state_dict_shape_mismatch(err) is False


# ----------------------------------------------------------------------------
# _patched_hidden_eval
# ----------------------------------------------------------------------------


def _write_minimal_proof_dir(root: Path, patch_text: str) -> Path:
    proof_dir = root / "proof"
    (proof_dir / "training").mkdir(parents=True)
    # Fake checkpoint: bytes don't matter — _patched_hidden_eval delegates the
    # actual load to a subprocess we mock. We just need the file to exist for
    # the path arg threading.
    (proof_dir / "training" / "checkpoint.pt").write_bytes(b"fake")
    (proof_dir / "patch.diff").write_text(patch_text)
    return proof_dir


def test_patched_eval_no_patch_diff_returns_error(tmp_path: Path):
    proof_dir = tmp_path / "proof"
    (proof_dir / "training").mkdir(parents=True)
    (proof_dir / "training" / "checkpoint.pt").write_bytes(b"fake")
    # NO patch.diff — fallback can't proceed.

    ok, detail, result = vmod._patched_hidden_eval(
        ralph_root=tmp_path,
        proof_dir=proof_dir,
        ckpt_path=proof_dir / "training" / "checkpoint.pt",
    )

    assert ok is False
    assert "no patch.diff" in detail
    assert result is None


def test_patched_eval_parses_subprocess_result_line(tmp_path: Path):
    proof_dir = _write_minimal_proof_dir(tmp_path, patch_text="dummy")
    ckpt_path = proof_dir / "training" / "checkpoint.pt"

    fake_stdout = (
        "[some debug noise]\n"
        "RALPH_EVAL_RESULT val_bpb=1.5419 benchmark_acc=0.2500 "
        "tokens_evaluated=4096 benchmark_examples=32 eval_set_hash=abc123\n"
        "[trailing line]\n"
    )
    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout=fake_stdout, stderr=""
    )

    with (
        mock.patch.object(vmod, "_patched_hidden_eval", wraps=vmod._patched_hidden_eval),
        mock.patch("subprocess.run", return_value=fake_completed),
        mock.patch("shutil.copytree"),
        mock.patch("validator.validator.apply_patch", create=True) if False else mock.patch("proof.runner.apply_patch"),
    ):
        ok, detail, result = vmod._patched_hidden_eval(
            ralph_root=tmp_path,
            proof_dir=proof_dir,
            ckpt_path=ckpt_path,
        )

    assert ok is True, f"expected success, got: {detail}"
    assert "val_bpb=1.5419" in detail
    assert "patched-eval" in detail
    assert isinstance(result, HiddenEvalResult)
    assert result.val_bpb == pytest.approx(1.5419)
    assert result.benchmark_accuracy == pytest.approx(0.25)


def test_patched_eval_subprocess_nonzero_exit_returns_error(tmp_path: Path):
    proof_dir = _write_minimal_proof_dir(tmp_path, patch_text="dummy")
    ckpt_path = proof_dir / "training" / "checkpoint.pt"

    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout="", stderr="ERROR: patched model state_dict load failed"
    )

    with (
        mock.patch("subprocess.run", return_value=fake_completed),
        mock.patch("shutil.copytree"),
        mock.patch("proof.runner.apply_patch"),
    ):
        ok, detail, result = vmod._patched_hidden_eval(
            ralph_root=tmp_path,
            proof_dir=proof_dir,
            ckpt_path=ckpt_path,
        )

    assert ok is False
    assert "exit=1" in detail
    assert result is None


def test_patched_eval_missing_result_line_returns_error(tmp_path: Path):
    proof_dir = _write_minimal_proof_dir(tmp_path, patch_text="dummy")
    ckpt_path = proof_dir / "training" / "checkpoint.pt"

    fake_completed = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="some output but no marker line\n", stderr=""
    )

    with (
        mock.patch("subprocess.run", return_value=fake_completed),
        mock.patch("shutil.copytree"),
        mock.patch("proof.runner.apply_patch"),
    ):
        ok, detail, result = vmod._patched_hidden_eval(
            ralph_root=tmp_path,
            proof_dir=proof_dir,
            ckpt_path=ckpt_path,
        )

    assert ok is False
    assert "RALPH_EVAL_RESULT" in detail
    assert result is None


def test_patched_eval_subprocess_timeout_returns_error(tmp_path: Path):
    proof_dir = _write_minimal_proof_dir(tmp_path, patch_text="dummy")
    ckpt_path = proof_dir / "training" / "checkpoint.pt"

    def _raise_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0] if args else "", timeout=240)

    with (
        mock.patch("subprocess.run", side_effect=_raise_timeout),
        mock.patch("shutil.copytree"),
        mock.patch("proof.runner.apply_patch"),
    ):
        ok, detail, result = vmod._patched_hidden_eval(
            ralph_root=tmp_path,
            proof_dir=proof_dir,
            ckpt_path=ckpt_path,
        )

    assert ok is False
    assert "timed out" in detail
    assert result is None


def test_patched_eval_apply_patch_failure_returns_error(tmp_path: Path):
    proof_dir = _write_minimal_proof_dir(tmp_path, patch_text="broken-patch")
    ckpt_path = proof_dir / "training" / "checkpoint.pt"

    with (
        mock.patch("shutil.copytree"),
        mock.patch("proof.runner.apply_patch", side_effect=RuntimeError("patch hunk failed")),
    ):
        ok, detail, result = vmod._patched_hidden_eval(
            ralph_root=tmp_path,
            proof_dir=proof_dir,
            ckpt_path=ckpt_path,
        )

    assert ok is False
    assert "patch apply failed" in detail
    assert result is None


# ----------------------------------------------------------------------------
# op4_hidden_eval falls back to _patched_hidden_eval on shape mismatch
# ----------------------------------------------------------------------------


def test_op4_falls_back_on_shape_mismatch(tmp_path: Path):
    """When canonical RalphBase.load_state_dict raises with shape-mismatch
    keywords, op4_hidden_eval must invoke _patched_hidden_eval. This is the
    seam that recovered Agent A's QK-Norm submission from submission_error
    to scoreable."""
    proof_dir = tmp_path / "proof"
    (proof_dir / "training").mkdir(parents=True)
    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    ckpt_path.write_bytes(b"fake")
    sidecar = ckpt_path.parent / "checkpoint_config.json"
    sidecar.write_text(json.dumps({
        "vocab_size": 64, "dim": 8, "n_layers": 1, "n_heads": 1,
        "head_dim": 8, "ffn_mult": 8 / 3, "max_seq_len": 16,
    }))

    canonical_err = RuntimeError(
        'Error(s) in loading state_dict for RalphBase:\n'
        '\tUnexpected key(s) in state_dict: "blocks.0.attn.q_norm.weight"'
    )

    fallback_result = HiddenEvalResult(
        val_bpb=1.4321,
        benchmark_accuracy=0.35,
        tokens_evaluated=4096,
        benchmark_examples=32,
        eval_set_hash="abc123",
    )

    with (
        mock.patch.object(vmod, "_safe_load_checkpoint_weights", return_value={}),
        mock.patch.object(vmod.RalphBase, "load_state_dict", side_effect=canonical_err),
        mock.patch.object(
            vmod, "_patched_hidden_eval",
            return_value=(True, "val_bpb=1.4321 bench=0.350 (patched-eval)", fallback_result),
        ) as patched_fn,
    ):
        ok, detail, result = vmod.op4_hidden_eval(tmp_path, proof_dir)

    assert ok is True
    assert "patched-eval" in detail
    assert result is fallback_result
    patched_fn.assert_called_once()
    call_args = patched_fn.call_args.args or patched_fn.call_args.kwargs
    assert proof_dir in (call_args if isinstance(call_args, tuple) else tuple(call_args.values()))


def test_op4_reraises_non_shape_mismatch_runtime_error(tmp_path: Path):
    """Unrelated RuntimeError (e.g. CUDA OOM) must propagate — falling back
    to patched-eval would mask real failures."""
    proof_dir = tmp_path / "proof"
    (proof_dir / "training").mkdir(parents=True)
    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    ckpt_path.write_bytes(b"fake")
    sidecar = ckpt_path.parent / "checkpoint_config.json"
    sidecar.write_text(json.dumps({
        "vocab_size": 64, "dim": 8, "n_layers": 1, "n_heads": 1,
        "head_dim": 8, "ffn_mult": 8 / 3, "max_seq_len": 16,
    }))

    unrelated_err = RuntimeError("CUDA out of memory")

    with (
        mock.patch.object(vmod, "_safe_load_checkpoint_weights", return_value={}),
        mock.patch.object(vmod.RalphBase, "load_state_dict", side_effect=unrelated_err),
        mock.patch.object(vmod, "_patched_hidden_eval") as patched_fn,
    ):
        with pytest.raises(RuntimeError, match="CUDA out of memory"):
            vmod.op4_hidden_eval(tmp_path, proof_dir)
        patched_fn.assert_not_called()


# ----------------------------------------------------------------------------
# Integration: canonical path still works end-to-end (no mocks)
# ----------------------------------------------------------------------------


def test_op4_canonical_path_end_to_end(tmp_path: Path):
    """Build a tiny real model + checkpoint that matches canonical RalphBase
    exactly, and verify op4_hidden_eval still runs the canonical branch
    successfully — i.e. the new try/except seam doesn't accidentally route
    healthy submissions through the slow patched-eval subprocess."""
    import torch
    from model import RalphBase, RalphConfig

    from eval import run_hidden_eval  # noqa: F401 — used to ensure import works

    # vocab_size must cover the eval token range (GPT-2 BPE, max id 50256).
    # Keep everything else tiny so the test stays fast (<2s on CPU).
    cfg = RalphConfig(
        vocab_size=50304, dim=16, n_layers=1, n_heads=2,
        head_dim=8, ffn_mult=8 / 3, max_seq_len=32,
    )
    model = RalphBase(cfg)
    proof_dir = tmp_path / "proof"
    (proof_dir / "training").mkdir(parents=True)
    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, ckpt_path)
    (ckpt_path.parent / "checkpoint_config.json").write_text(
        json.dumps({
            "vocab_size": cfg.vocab_size, "dim": cfg.dim, "n_layers": cfg.n_layers,
            "n_heads": cfg.n_heads, "head_dim": cfg.head_dim, "ffn_mult": cfg.ffn_mult,
            "max_seq_len": cfg.max_seq_len,
        })
    )

    # Sentinel: if anything tries to invoke the patched-eval subprocess, fail.
    with mock.patch.object(vmod, "_patched_hidden_eval") as patched_fn:
        ok, detail, result = vmod.op4_hidden_eval(
            ralph_root=Path(__file__).resolve().parent.parent,
            proof_dir=proof_dir,
        )
        patched_fn.assert_not_called()

    assert ok is True, f"canonical path failed: {detail}"
    assert "patched-eval" not in detail
    assert isinstance(result, HiddenEvalResult)
    assert result.val_bpb > 0  # untrained random weights produce a real bpb


# ----------------------------------------------------------------------------
# eval_in_workdir.py argument validation (script-level smoke)
# ----------------------------------------------------------------------------


def test_eval_in_workdir_script_exists_and_is_executable_python():
    """The helper script the subprocess invokes must be present, importable,
    and report bad-arg usage cleanly so the parent's stderr parser sees a
    meaningful message."""
    script = Path(__file__).resolve().parent.parent / "validator" / "eval_in_workdir.py"
    assert script.exists()
    res = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=15
    )
    assert res.returncode == 3, f"expected exit=3 on bad args, got {res.returncode}"
    assert "usage" in res.stderr.lower()
