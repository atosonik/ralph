"""Host-side val_bpb reduction must reproduce the in-process computation exactly,
and must reject malformed/forged NLL arrays."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from eval.host_reduce import (
    expected_token_count,
    hash_target_stream,
    reduce_token_nlls,
)

LN2 = math.log(2)


def test_reduction_matches_in_process_cross_entropy():
    """reduce_token_nlls(per-position NLLs) == the bpb/tail compute_val_bpb gets
    from the same logits — the equivalence the host-side move depends on."""
    torch.manual_seed(0)
    vocab, seq_len, n_windows, bpt = 11, 8, 5, 4.0
    logits = torch.randn(n_windows, seq_len, vocab)
    targets = torch.randint(0, vocab, (n_windows, seq_len))

    # Reference: how compute_val_bpb reduces (sum over all positions).
    total_nats = F.cross_entropy(
        logits.view(-1, vocab), targets.reshape(-1), reduction="sum"
    ).item()
    tokens = n_windows * seq_len
    ref_bpb = total_nats / (LN2 * tokens * bpt)
    tail_start = seq_len // 2
    tail_nats = F.cross_entropy(
        logits[:, tail_start:, :].reshape(-1, vocab),
        targets[:, tail_start:].reshape(-1),
        reduction="sum",
    ).item()
    tail_tokens = n_windows * (seq_len - tail_start)
    ref_tail = tail_nats / (LN2 * tail_tokens * bpt)

    # The sandbox would emit these per-position NLLs (window-row-major order).
    nlls = F.cross_entropy(
        logits.view(-1, vocab), targets.reshape(-1), reduction="none"
    ).numpy()

    out = reduce_token_nlls(
        nlls, seq_len=seq_len, bytes_per_token=bpt,
        expected_tokens=tokens, eval_set_hash="deadbeef",
    )
    assert out.val_bpb == pytest.approx(ref_bpb, rel=1e-6)
    assert out.tail_val_bpb == pytest.approx(ref_tail, rel=1e-6)
    assert out.tokens_evaluated == tokens


def test_expected_token_count_matches_packing():
    # Full windows of (seq_len+1); partial last window skipped.
    assert expected_token_count(41, 8) == 40   # 5 full windows
    assert expected_token_count(40, 8) == 32    # 4 full windows
    assert expected_token_count(9, 8) == 8
    assert expected_token_count(5, 8) == 0      # no full window → 0 (bpb=inf upstream)


def test_rejects_wrong_length():
    with pytest.raises(ValueError, match="wrong number"):
        reduce_token_nlls(
            np.ones(39), seq_len=8, bytes_per_token=4.0,
            expected_tokens=40, eval_set_hash="x",
        )


def test_rejects_non_finite_and_negative():
    bad = np.ones(40)
    bad[3] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        reduce_token_nlls(bad, seq_len=8, bytes_per_token=4.0, expected_tokens=40, eval_set_hash="x")
    neg = np.ones(40)
    neg[7] = -0.5
    with pytest.raises(ValueError, match="negative"):
        reduce_token_nlls(neg, seq_len=8, bytes_per_token=4.0, expected_tokens=40, eval_set_hash="x")


def test_rejects_bad_bytes_per_token():
    with pytest.raises(ValueError, match="bytes_per_token"):
        reduce_token_nlls(np.ones(40), seq_len=8, bytes_per_token=0.0, expected_tokens=40, eval_set_hash="x")


def test_hash_binds_host_stream_and_is_deterministic():
    a = np.arange(100, dtype=np.uint16)
    b = np.arange(100, dtype=np.uint16)
    c = a.copy(); c[50] = 999
    assert hash_target_stream(a) == hash_target_stream(b)
    assert hash_target_stream(a) != hash_target_stream(c)
    assert len(hash_target_stream(a)) == 64
