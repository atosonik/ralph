"""Tests for the v0.10 multi-seed forward-compat scaffolding.

These pin the API surface and honestly document that:

  - derive_seeds is a deterministic function of the chain block hash.
  - Under today's deterministic op4_hidden_eval, the k=3 multi-seed wrapper
    produces byte-identical individual results and pooled_stderr == 0.

When B2 lands stochastic per-epoch eval streams, the byte-identical assertions
in this file MUST flip — that test failure is the signal that stochastic eval
went live.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from validator.multiseed import (
    MultiSeedEvalResult,
    _mean_stderr,
    derive_seeds,
    op4_hidden_eval_multiseed,
)
from validator.scoring import get_king_rule

# ----------------------------------------------------------------------------
# derive_seeds — deterministic int-seed list from a block hash
# ----------------------------------------------------------------------------


def test_derive_seeds_deterministic():
    h = "0xdeadbeef" + "00" * 30
    assert derive_seeds(h, 3) == derive_seeds(h, 3)


def test_derive_seeds_returns_k():
    h = "0xdeadbeef" + "00" * 30
    assert len(derive_seeds(h, 1)) == 1
    assert len(derive_seeds(h, 3)) == 3
    assert len(derive_seeds(h, 10)) == 10


def test_derive_seeds_bounded_int_range():
    """Every seed fits in [0, 2^31 - 1] so it works with PyTorch / NumPy /
    Python random alike."""
    h = "0x" + "a5" * 32
    for s in derive_seeds(h, 16):
        assert 0 <= s <= (1 << 31) - 1


def test_derive_seeds_different_hashes_differ():
    """Different block hashes → different seed lists. (256-bit collision
    is theoretically possible but never observed.)"""
    h1 = "0x" + "11" * 32
    h2 = "0x" + "22" * 32
    assert derive_seeds(h1, 3) != derive_seeds(h2, 3)


def test_derive_seeds_index_independence():
    """Seed[i] depends on i, not on neighbors — proves the i-th sub-hash
    derivation. Asking for 3 seeds and asking for 5 must agree on the first 3."""
    h = "0x" + "5a" * 32
    s3 = derive_seeds(h, 3)
    s5 = derive_seeds(h, 5)
    assert s3 == s5[:3]


def test_derive_seeds_tolerates_no_prefix():
    """0x prefix is optional — bittensor returns various formats; we tolerate."""
    body = "deadbeef" + "00" * 30
    assert derive_seeds("0x" + body, 3) == derive_seeds(body, 3)
    assert derive_seeds("0X" + body, 3) == derive_seeds(body, 3)


def test_derive_seeds_normalizes_case():
    """Uppercase vs lowercase hex bodies derive the same seeds — must be
    case-insensitive so two validators reading the same chain agree."""
    upper = "DEADBEEF" + "AB" * 30
    lower = upper.lower()
    assert derive_seeds("0x" + upper, 3) == derive_seeds("0x" + lower, 3)


def test_derive_seeds_rejects_k_zero():
    with pytest.raises(ValueError, match=r"k must be >= 1"):
        derive_seeds("0xabc", 0)
    with pytest.raises(ValueError, match=r"k must be >= 1"):
        derive_seeds("0xabc", -1)


def test_derive_seeds_rejects_empty_hash():
    with pytest.raises(ValueError, match=r"must not be empty"):
        derive_seeds("", 3)


# ----------------------------------------------------------------------------
# _mean_stderr helper — Bessel-corrected stderr of the mean
# ----------------------------------------------------------------------------


def test_mean_stderr_empty():
    assert _mean_stderr([]) == (0.0, 0.0)


def test_mean_stderr_single_observation_has_zero_stderr():
    """n=1 → no within-sample variance to estimate; stderr is 0 by convention."""
    mean, stderr = _mean_stderr([1.234])
    assert mean == 1.234
    assert stderr == 0.0


def test_mean_stderr_byte_identical_samples_have_zero_stderr():
    """Three identical observations → 0 variance → 0 stderr. This is the
    EXACT condition that holds for today's multi-seed wrapper because eval
    is deterministic."""
    mean, stderr = _mean_stderr([1.5, 1.5, 1.5])
    assert mean == 1.5
    assert stderr == 0.0


def test_mean_stderr_simple_case():
    """Hand-verifiable: mean(1,2,3)=2, sample-std=1, stderr=1/sqrt(3)."""
    import math
    mean, stderr = _mean_stderr([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert stderr == pytest.approx(1.0 / math.sqrt(3))


# ----------------------------------------------------------------------------
# op4_hidden_eval_multiseed — the production-shape wrapper
# ----------------------------------------------------------------------------


def _build_canonical_proof_dir(tmp_path: Path) -> Path:
    """Create a proof_dir with a real tiny-canonical-KarpaBase checkpoint so
    op4_hidden_eval has a working input. Mirrors the test_op4_canonical_path
    setup in test_validator_patched_eval.py."""
    import torch
    from model import KarpaBase, KarpaConfig

    # vocab_size matches the GPT-2 token range of active_tokens.bin (max id 50256)
    cfg = KarpaConfig(
        vocab_size=50304, dim=16, n_layers=1, n_heads=2,
        head_dim=8, ffn_mult=8 / 3, max_seq_len=32,
    )
    model = KarpaBase(cfg)
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
    return proof_dir


def test_multiseed_three_runs_produce_byte_identical_results(tmp_path: Path):
    """v0.10 LIMITATION DOCUMENTED HERE.

    With today's deterministic op4_hidden_eval, k=3 sub-runs produce
    EXACTLY the same val_bpb and benchmark_accuracy. This test asserts that
    byte-identity holds — which is the honest representation of where the
    plumbing currently sits.

    When B2 lands per-epoch sealed-stream rotation (the Cross-Scale Downstream
    Pareto build), this test MUST start failing. That failure is the signal
    the multi-seed gate is no longer inert.
    """
    karpa_root = Path(__file__).resolve().parent.parent
    proof_dir = _build_canonical_proof_dir(tmp_path)

    ok, detail, result = op4_hidden_eval_multiseed(karpa_root, proof_dir, k=3)
    assert ok, f"unexpected failure: {detail}"
    assert result is not None
    assert result.k == 3
    assert len(result.individual) == 3

    # All three runs byte-identical on val_bpb
    val_bpbs = [r.val_bpb for r in result.individual]
    assert val_bpbs[0] == val_bpbs[1] == val_bpbs[2], (
        f"v0.10 invariant broken — runs differ: {val_bpbs}. If B2 just landed, "
        "this test must be updated. Otherwise, eval has become accidentally "
        "stochastic and the multi-seed gate's behaviour is no longer pinned."
    )

    # Pooled stderr at the floating-point noise floor (mean(0.2,0.2,0.2)
    # rounds to 0.2 + 1 ULP, so stderr lands ~1e-17 not exactly 0). Anything
    # below 1e-10 means the runs are byte-identical at all practical scales.
    assert result.val_bpb_stderr < 1e-10
    assert result.benchmark_acc_stderr < 1e-10

    # Mean equals the common value (modulo floating-point rounding on benchmark)
    assert result.val_bpb_mean == val_bpbs[0]


def test_multiseed_rejects_k_zero(tmp_path: Path):
    """Defensive: k=0 is a programming error, not a no-op."""
    ok, detail, result = op4_hidden_eval_multiseed(
        tmp_path, tmp_path / "nonexistent", k=0
    )
    assert ok is False
    assert "k must be >= 1" in detail
    assert result is None


def test_multiseed_rejects_seeds_length_mismatch(tmp_path: Path):
    """If caller passes seeds, len(seeds) must equal k."""
    ok, detail, result = op4_hidden_eval_multiseed(
        tmp_path, tmp_path / "nonexistent", k=3, seeds=[1, 2]
    )
    assert ok is False
    assert "len(seeds)=2 != k=3" in detail
    assert result is None


def test_multiseed_propagates_sub_eval_failure(tmp_path: Path):
    """Missing checkpoint → op4_hidden_eval returns False → multiseed
    propagates with the underlying failure detail."""
    proof_dir = tmp_path / "no_proof"
    proof_dir.mkdir()
    karpa_root = Path(__file__).resolve().parent.parent
    ok, detail, result = op4_hidden_eval_multiseed(karpa_root, proof_dir, k=2)
    assert ok is False
    assert "sub-eval failed" in detail
    assert "missing checkpoint" in detail
    assert result is None


def test_multiseed_result_carries_seeds(tmp_path: Path):
    """The returned dataclass carries the seed list so an external observer
    can reproduce the run from chain state alone."""
    karpa_root = Path(__file__).resolve().parent.parent
    proof_dir = _build_canonical_proof_dir(tmp_path)
    seeds = [11, 22, 33]
    ok, _, result = op4_hidden_eval_multiseed(karpa_root, proof_dir, k=3, seeds=seeds)
    assert ok
    assert isinstance(result, MultiSeedEvalResult)
    assert result.seeds == seeds


# ----------------------------------------------------------------------------
# get_king_rule — env-var driven feature flag, scaffolded today for B3 later
# ----------------------------------------------------------------------------


def test_get_king_rule_default_is_legacy(monkeypatch):
    monkeypatch.delenv("KARPA_KING_RULE", raising=False)
    assert get_king_rule() == "legacy"


def test_get_king_rule_accepts_legacy(monkeypatch):
    monkeypatch.setenv("KARPA_KING_RULE", "legacy")
    assert get_king_rule() == "legacy"


def test_get_king_rule_accepts_cross_scale_v1(monkeypatch):
    """Forward-compat: B3 will swap in the new rule under this name."""
    monkeypatch.setenv("KARPA_KING_RULE", "cross_scale_v1")
    assert get_king_rule() == "cross_scale_v1"


def test_get_king_rule_falls_back_on_unknown_value(monkeypatch, capsys):
    monkeypatch.setenv("KARPA_KING_RULE", "garbage")
    assert get_king_rule() == "legacy"
    captured = capsys.readouterr()
    assert "not recognised" in captured.err
    assert "legacy" in captured.err


def test_get_king_rule_strips_whitespace(monkeypatch):
    monkeypatch.setenv("KARPA_KING_RULE", "  legacy  ")
    assert get_king_rule() == "legacy"
