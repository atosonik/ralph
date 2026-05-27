"""
val_bpb — validation bits-per-byte computation.

bpb is the per-byte negative log-likelihood under the model's predicted
distribution: bpb = (cross_entropy_in_nats * tokens_count) / (log(2) * byte_count).

It is vocabulary-independent (unlike perplexity), so architectural changes
that change tokenization are scored fairly. This is what autoresearch
optimizes by default; AutoRalph inherits the metric for the LLM
pretraining launch track.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def compute_val_bpb(
    model: torch.nn.Module,
    eval_tokens: np.ndarray,
    seq_len: int,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict:
    """
    Compute val_bpb over a held-out token stream.

    The token-to-byte ratio is recovered from the tokenizer: GPT-2 BPE
    averages roughly 4.0 bytes per token on English text. For reproducibility
    across runs we use the empirical ratio from the eval token stream's
    decoded byte length, computed once at eval-set-construction time and
    passed in here. For Phase 0 smoke tests we approximate with the typical
    GPT-2 ratio.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_nats = 0.0
    total_tokens = 0

    # Pack into non-overlapping windows of (seq_len + 1).
    n = len(eval_tokens)
    n_windows = max(1, (n - 1) // seq_len)
    with torch.no_grad():
        batch_inp = []
        batch_tgt = []
        for w in range(n_windows):
            start = w * seq_len
            ids = eval_tokens[start : start + seq_len + 1]
            if len(ids) < seq_len + 1:
                break
            batch_inp.append(torch.from_numpy(ids[:-1].astype(np.int64)))
            batch_tgt.append(torch.from_numpy(ids[1:].astype(np.int64)))
            if len(batch_inp) == batch_size or w == n_windows - 1:
                inp = torch.stack(batch_inp).to(device)
                tgt = torch.stack(batch_tgt).to(device)
                logits, _ = model(inp)
                # cross-entropy in nats, summed (not mean) so we accumulate correctly
                loss_sum = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="sum",
                )
                total_nats += loss_sum.item()
                total_tokens += tgt.numel()
                batch_inp.clear()
                batch_tgt.clear()

    # bytes per token: use a fixed ratio for Phase 0; in production this is
    # computed from the eval set's decoded byte length at construction time.
    bytes_per_token = 4.0  # typical for GPT-2 BPE on English
    total_bytes = total_tokens * bytes_per_token
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float("inf")
    nll_per_token = total_nats / max(total_tokens, 1)
    return {
        "val_bpb": bpb,
        "nll_per_token": nll_per_token,
        "tokens_evaluated": total_tokens,
        "bytes_per_token": bytes_per_token,
    }


def load_eval_tokens(path: Path | str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")
