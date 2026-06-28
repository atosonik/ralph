"""Host-side val_bpb reduction — make the crowning number validator-produced.

Today the crowning `val_bpb` for a STRUCTURAL patch is a float printed by the
miner's own code (`eval_in_workdir.py`), which the scorer trusts. This module
moves the reduction to the host: the sandboxed miner model emits the per-position
negative log-likelihood (nats) of each true target token, and the VALIDATOR
computes bpb from that array using the SAME formula as `eval.val_bpb`, holding
`bytes_per_token`, the expected token count, the tail mask, and the eval-set hash
itself.

Forgery scope (be honest): a miner controls the model code, so it can still emit
fake-low NLLs. Host-side reduction removes the formula / normalization / token-
count / hash forgery classes and centralizes the metric; the loss VALUES are made
trustworthy by the independent re-train audit (the paired BLOCKER — the audit
must recompute, not re-run miner code at a loose tolerance). For the CANONICAL-
architecture path the sandbox runs validator eval code, so the values are already
host-trusted; this primitive is the structural-patch counterpart.

Equivalence: `reduce_token_nlls` reproduces `eval.val_bpb.compute_val_bpb`'s
`val_bpb` / `tail_val_bpb` / `nll_per_token` exactly given the same per-position
cross-entropies, the same `seq_len`, and the same `bytes_per_token`.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

LN2 = math.log(2)


@dataclass(frozen=True)
class HostReduced:
    val_bpb: float
    tail_val_bpb: float | None
    nll_per_token: float
    tokens_evaluated: int
    bytes_per_token: float
    eval_set_hash: str  # sha256 of the HOST's target stream (not miner-supplied)


def expected_token_count(stream_len: int, seq_len: int) -> int:
    """Number of target positions `compute_val_bpb` scores over a stream of
    `stream_len` tokens: non-overlapping windows of (seq_len+1), each
    contributing `seq_len` targets. Mirrors val_bpb.compute_val_bpb's packing."""
    # Only full windows of (seq_len+1) contribute; the last partial window is
    # skipped (`if len(ids) < seq_len + 1: break`).
    full_windows = (stream_len - 1) // seq_len
    return full_windows * seq_len


def hash_target_stream(target_tokens: np.ndarray) -> str:
    """sha256 over the FULL host-held target stream — replaces the miner-printed
    100-token `eval_set_hash` that bound nothing."""
    arr = np.ascontiguousarray(np.asarray(target_tokens, dtype=np.uint16))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def reduce_token_nlls(
    nlls: np.ndarray,
    *,
    seq_len: int,
    bytes_per_token: float,
    expected_tokens: int,
    eval_set_hash: str,
) -> HostReduced:
    """Reduce a per-position NLL array (nats, canonical window order) into bpb.

    Validates the array against what the host expects before trusting it:
      - length MUST equal `expected_tokens` (host-derived from its own stream);
      - every value MUST be finite and >= 0 (cross-entropy is non-negative).
    Raises ValueError on any violation — the caller treats that as a rejected
    submission (never a silent pass).
    """
    if bytes_per_token <= 0:
        raise ValueError(f"bytes_per_token must be > 0; got {bytes_per_token}")
    nlls = np.asarray(nlls, dtype=np.float64)
    if nlls.ndim != 1:
        raise ValueError(f"nlls must be 1-D; got shape {nlls.shape}")
    if nlls.shape[0] != expected_tokens:
        raise ValueError(
            f"nll count {nlls.shape[0]} != expected {expected_tokens} "
            f"(miner returned the wrong number of scored positions)"
        )
    if not np.all(np.isfinite(nlls)):
        raise ValueError("nll array contains non-finite values")
    if np.any(nlls < 0):
        raise ValueError("nll array contains negative cross-entropy (impossible)")

    total_nats = float(nlls.sum())
    total_tokens = int(nlls.shape[0])
    total_bytes = total_tokens * bytes_per_token
    bpb = total_nats / (LN2 * total_bytes) if total_bytes > 0 else float("inf")
    nll_per_token = total_nats / max(total_tokens, 1)

    # Tail probe: positions [seq_len//2:] within each window. The host
    # reconstructs the mask from the index — it does not trust a miner tail flag.
    tail_start = seq_len // 2
    within = np.arange(total_tokens) % seq_len
    tail_mask = within >= tail_start
    tail_tokens = int(tail_mask.sum())
    if tail_tokens > 0:
        tail_nats = float(nlls[tail_mask].sum())
        tail_bytes = tail_tokens * bytes_per_token
        tail_bpb: float | None = tail_nats / (LN2 * tail_bytes)
    else:
        tail_bpb = None

    return HostReduced(
        val_bpb=bpb,
        tail_val_bpb=tail_bpb,
        nll_per_token=nll_per_token,
        tokens_evaluated=total_tokens,
        bytes_per_token=bytes_per_token,
        eval_set_hash=eval_set_hash,
    )
