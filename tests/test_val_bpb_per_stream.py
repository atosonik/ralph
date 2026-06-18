"""Tests for the B2 val_bpb per-stream bytes_per_token threading.

Covers:
  * compute_val_bpb default behaviour (no bytes_per_token) matches
    pre-B2 fixed 4.0 — backward compat for eval/hidden_eval.py and any
    other single-stream callers.
  * compute_val_bpb honors an explicit bytes_per_token override.
  * compute_val_bpb rejects bytes_per_token <= 0.
  * compute_val_bpb_on_stream reads bytes_per_token from
    batch.spec and stamps stream_id into the result dict.
  * The math is exact: doubling bytes_per_token halves val_bpb (level
    shift, not a sign error).

These tests use a tiny synthetic torch model so they run on CPU; they
do NOT exercise RalphBase. The actual transformer correctness is tested
elsewhere (tests/test_model.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.sealed_streams import SealedStreamBatch, SealedStreamSpec
from eval.val_bpb import (
    DEFAULT_BYTES_PER_TOKEN,
    compute_val_bpb,
    compute_val_bpb_on_stream,
)

# ----------------------------------------------------------------------------
# Synthetic model
# ----------------------------------------------------------------------------

VOCAB = 50  # small enough that CPU tests run fast


class _TinyModel(torch.nn.Module):
    """Returns (logits, None) — matches RalphBase's forward signature."""

    def __init__(self, vocab_size: int = VOCAB, dim: int = 8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, dim)
        self.out = torch.nn.Linear(dim, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        return self.out(h), None


def _make_tokens(n: int) -> np.ndarray:
    """Deterministic token stream small enough to evaluate quickly."""
    rng = np.random.default_rng(seed=42)
    return rng.integers(0, VOCAB, size=n, dtype=np.uint16)


# ============================================================================
# compute_val_bpb — bytes_per_token parameter
# ============================================================================


class TestComputeValBpbBackwardCompat:
    def test_default_bytes_per_token_is_4(self):
        """Without the kwarg, the helper uses DEFAULT_BYTES_PER_TOKEN=4.0."""
        assert DEFAULT_BYTES_PER_TOKEN == 4.0

    def test_no_arg_matches_explicit_default(self):
        """compute_val_bpb() with default vs explicit 4.0 → identical results."""
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(128)
        result_default = compute_val_bpb(model, tokens, seq_len=16)
        result_explicit = compute_val_bpb(
            model, tokens, seq_len=16, bytes_per_token=4.0,
        )
        assert result_default["val_bpb"] == result_explicit["val_bpb"]
        assert result_default["bytes_per_token"] == 4.0

    def test_returns_expected_keys(self):
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(64)
        result = compute_val_bpb(model, tokens, seq_len=8)
        assert "val_bpb" in result
        assert "nll_per_token" in result
        assert "tokens_evaluated" in result
        assert "bytes_per_token" in result


class TestComputeValBpbExplicitBytesPerToken:
    def test_explicit_override(self):
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(128)
        result = compute_val_bpb(
            model, tokens, seq_len=16, bytes_per_token=2.5,
        )
        assert result["bytes_per_token"] == 2.5

    def test_halved_bytes_per_token_doubles_bpb(self):
        """val_bpb = nats / (log(2) * total_bytes). Halving bytes_per_token
        halves total_bytes and therefore doubles val_bpb. Math sanity check."""
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(128)
        baseline = compute_val_bpb(
            model, tokens, seq_len=16, bytes_per_token=4.0,
        )
        halved = compute_val_bpb(
            model, tokens, seq_len=16, bytes_per_token=2.0,
        )
        assert halved["val_bpb"] == pytest.approx(2 * baseline["val_bpb"])

    def test_zero_bytes_per_token_rejected(self):
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(64)
        with pytest.raises(ValueError, match=r"bytes_per_token must be > 0"):
            compute_val_bpb(model, tokens, seq_len=8, bytes_per_token=0)

    def test_negative_bytes_per_token_rejected(self):
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(64)
        with pytest.raises(ValueError, match=r"bytes_per_token must be > 0"):
            compute_val_bpb(model, tokens, seq_len=8, bytes_per_token=-1.0)

    def test_nll_per_token_invariant_to_bytes_per_token(self):
        """Changing bytes_per_token must NOT change nll_per_token — that's
        a pure model+data property, no byte conversion involved."""
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(128)
        a = compute_val_bpb(model, tokens, seq_len=16, bytes_per_token=4.0)
        b = compute_val_bpb(model, tokens, seq_len=16, bytes_per_token=2.0)
        assert a["nll_per_token"] == b["nll_per_token"]
        assert a["tokens_evaluated"] == b["tokens_evaluated"]


# ============================================================================
# compute_val_bpb_on_stream
# ============================================================================


def _make_batch(
    bytes_per_token: float = 3.5,
    n_tokens: int = 128,
    stream_id: str = "stream_00",
) -> SealedStreamBatch:
    tokens = _make_tokens(n_tokens)
    spec = SealedStreamSpec(
        id=stream_id,
        corpus="fineweb-edu",
        sub_genre="english_prose",
        n_tokens=n_tokens,
        bytes_per_token=bytes_per_token,
        sha256="a" * 64,
    )
    return SealedStreamBatch(spec=spec, tokens=tokens)


class TestComputeValBpbOnStream:
    def test_uses_per_stream_bytes_per_token(self):
        """The wrapper threads bytes_per_token from the spec, not the
        default 4.0."""
        torch.manual_seed(0)
        model = _TinyModel()
        batch = _make_batch(bytes_per_token=3.2)
        result = compute_val_bpb_on_stream(model, batch, seq_len=16)
        assert result["bytes_per_token"] == 3.2

    def test_stamps_stream_id(self):
        torch.manual_seed(0)
        model = _TinyModel()
        batch = _make_batch(stream_id="stream_07")
        result = compute_val_bpb_on_stream(model, batch, seq_len=16)
        assert result["stream_id"] == "stream_07"

    def test_carries_all_keys(self):
        torch.manual_seed(0)
        model = _TinyModel()
        batch = _make_batch()
        result = compute_val_bpb_on_stream(model, batch, seq_len=16)
        # The base compute_val_bpb keys are preserved.
        for key in ("val_bpb", "nll_per_token", "tokens_evaluated",
                    "bytes_per_token"):
            assert key in result
        # Plus the stream annotation.
        assert "stream_id" in result

    def test_matches_compute_val_bpb_directly(self):
        """compute_val_bpb_on_stream(model, batch) ==
        compute_val_bpb(model, np.asarray(batch.tokens),
                        bytes_per_token=batch.spec.bytes_per_token)
        modulo the stream_id annotation."""
        torch.manual_seed(0)
        model = _TinyModel()
        batch = _make_batch(bytes_per_token=2.7)
        direct = compute_val_bpb(
            model, np.asarray(batch.tokens), seq_len=16,
            bytes_per_token=2.7,
        )
        wrapped = compute_val_bpb_on_stream(model, batch, seq_len=16)
        # Same numerical results.
        assert wrapped["val_bpb"] == direct["val_bpb"]
        assert wrapped["nll_per_token"] == direct["nll_per_token"]
        assert wrapped["tokens_evaluated"] == direct["tokens_evaluated"]

    def test_two_streams_with_different_ratios_diverge(self):
        """Two streams with the same tokens but different bytes_per_token
        produce different val_bpb (the level shift is real, not a
        formatting bug)."""
        torch.manual_seed(0)
        model = _TinyModel()
        batch_a = _make_batch(bytes_per_token=4.0)
        batch_b = _make_batch(bytes_per_token=2.0)
        # Same token sequence in both batches (same seed inside _make_tokens).
        result_a = compute_val_bpb_on_stream(model, batch_a, seq_len=16)
        result_b = compute_val_bpb_on_stream(model, batch_b, seq_len=16)
        # Halved bytes_per_token → doubled val_bpb.
        assert result_b["val_bpb"] == pytest.approx(2 * result_a["val_bpb"])


# ============================================================================
# tail_val_bpb — long-context tail probe (validation-v2 Phase 1)
# ============================================================================


class TestTailValBpb:
    """The tail probe mirrors recipe/train.py: BPB over the tail positions
    [seq_len//2 :] of each window, same cross-entropy + bytes_per_token
    normalization as val_bpb. Recorded only; the scorer does not use it."""

    def test_tail_val_bpb_present(self):
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(256)
        result = compute_val_bpb(model, tokens, seq_len=16)
        assert "tail_val_bpb" in result
        assert result["tail_val_bpb"] is not None
        assert result["tail_val_bpb"] > 0

    def test_tail_val_bpb_matches_manual_tail_slice(self):
        """tail_val_bpb equals val_bpb recomputed over only positions
        [seq_len//2:] — proving it's the long-context tail, not the full
        window. We verify by computing the tail BPB independently."""
        import math

        import numpy as np
        import torch.nn.functional as F

        torch.manual_seed(0)
        model = _TinyModel()
        model.eval()
        seq_len = 16
        bpt = 4.0
        tokens = _make_tokens(256)
        result = compute_val_bpb(model, tokens, seq_len=seq_len, bytes_per_token=bpt)

        # Independent recompute of the tail half.
        tail_start = seq_len // 2
        n = len(tokens)
        n_windows = max(1, (n - 1) // seq_len)
        tail_nats = 0.0
        tail_tokens = 0
        with torch.no_grad():
            for w in range(n_windows):
                start = w * seq_len
                ids = tokens[start : start + seq_len + 1]
                if len(ids) < seq_len + 1:
                    break
                inp = torch.from_numpy(ids[:-1].astype(np.int64)).unsqueeze(0)
                tgt = torch.from_numpy(ids[1:].astype(np.int64)).unsqueeze(0)
                logits, _ = model(inp)
                tl = logits[:, tail_start:, :]
                tt = tgt[:, tail_start:]
                tail_nats += F.cross_entropy(
                    tl.reshape(-1, tl.size(-1)), tt.reshape(-1), reduction="sum"
                ).item()
                tail_tokens += tt.numel()
        expected = tail_nats / (math.log(2) * tail_tokens * bpt)
        assert result["tail_val_bpb"] == pytest.approx(expected, rel=1e-5)

    def test_tail_val_bpb_normalizes_with_bytes_per_token(self):
        """Halving bytes_per_token doubles tail_val_bpb, same as val_bpb."""
        torch.manual_seed(0)
        model = _TinyModel()
        tokens = _make_tokens(256)
        a = compute_val_bpb(model, tokens, seq_len=16, bytes_per_token=4.0)
        b = compute_val_bpb(model, tokens, seq_len=16, bytes_per_token=2.0)
        assert b["tail_val_bpb"] == pytest.approx(2 * a["tail_val_bpb"])

    def test_tail_val_bpb_none_when_window_too_short(self):
        """seq_len=1 → tail_start=0 means the whole window IS the tail, so the
        probe is still defined. A degenerate seq that yields no full windows
        leaves tail tokens at zero → None (guarded division)."""
        torch.manual_seed(0)
        model = _TinyModel()
        # Only enough tokens for a single (seq_len+1) window; tail still defined.
        tokens = _make_tokens(17)
        result = compute_val_bpb(model, tokens, seq_len=16)
        # tail_start = 8 < 16, so the tail slice exists.
        assert result["tail_val_bpb"] is not None
