"""
val_bpb — validation bits-per-byte computation.

bpb is the per-byte negative log-likelihood under the model's predicted
distribution: bpb = (cross_entropy_in_nats * tokens_count) / (log(2) * byte_count).

It is vocabulary-independent (unlike perplexity), so architectural changes
that change tokenization are scored fairly. This is what autoresearch
optimizes by default; Ralph inherits the metric for the LLM
pretraining launch track.

Per-stream bytes_per_token (B2):
  The token-to-byte ratio varies across the sealed pool: English prose
  hits ~4.0 under GPT-2 BPE, but code/math/multilingual streams have
  meaningfully different ratios (Python ~3.2, OpenWebMath ~3.5, FineWeb-2
  non-European ~2.0). `compute_val_bpb` now accepts the ratio as a
  parameter; `compute_val_bpb_on_stream` is a convenience wrapper for
  callers holding a `SealedStreamBatch` that reads the per-stream value
  from the manifest spec. Backward-compat: when `bytes_per_token=None`,
  the 4.0 default fires — preserves the pre-B2 behaviour for
  eval/hidden_eval.py and existing single-stream callers.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F

from .host_reduce import BlankedCell

if TYPE_CHECKING:
    from .sealed_streams import SealedStreamBatch

# Default token-to-byte ratio when caller passes bytes_per_token=None.
# Matches the pre-B2 hardcoded value; preserved for single-stream callers
# and for Phase 0 smoke tests that don't construct a sealed pool.
DEFAULT_BYTES_PER_TOKEN = 4.0

# Validator-pinned hidden-eval window. The eval MUST NOT derive its sequence
# length from the miner's checkpoint config (the old `cfg.max_seq_len // 2`): a
# miner could enlarge max_seq_len to be scored against an easier, longer-context
# eval than the king faced, and every model evaluated on a different window is
# not comparable. Pinning to a FIXED window makes all models comparable and
# removes a miner lever. This is a CANONICAL (image-baked / installed) constant —
# in the sandbox it is read from the trusted eval package, never miner-supplied.
EVAL_SEQ_LEN = 512


def pinned_eval_seq_len(model_max_seq_len: int) -> int:
    """The validator-pinned hidden-eval window: ``min(EVAL_SEQ_LEN, max_seq_len)``,
    floored at 2 (compute_val_bpb needs seq_len+1 tokens per window).

    Single source of truth so EVERY eval path (in-process, subprocess, sandbox,
    audit) pins seq_len identically and the host can independently re-derive and
    verify the value a container echoes. The cap at the model's own max_seq_len
    lets small-context models still load; the floor at EVAL_SEQ_LEN means no
    miner can choose a window LARGER than the validator's.
    """
    return max(2, min(EVAL_SEQ_LEN, int(model_max_seq_len)))


def compute_val_bpb(
    model: torch.nn.Module,
    eval_tokens: np.ndarray,
    seq_len: int,
    batch_size: int = 8,
    device: torch.device | None = None,
    *,
    bytes_per_token: float | None = None,
) -> dict:
    """
    Compute val_bpb over a held-out token stream.

    Args:
      bytes_per_token: empirical token-to-byte ratio for this stream. When
        None (default), uses `DEFAULT_BYTES_PER_TOKEN = 4.0` matching the
        pre-B2 behaviour. When set (typically by `compute_val_bpb_on_stream`
        reading from the SealedStreamManifest), uses that value. Must be
        > 0 or ValueError.

    The token-to-byte ratio is recovered from the tokenizer: GPT-2 BPE
    averages roughly 4.0 bytes per token on English text. The sealed
    pool's manifest carries per-stream ratios computed at
    construction time from each stream's decoded byte length.
    """
    if bytes_per_token is None:
        bytes_per_token = DEFAULT_BYTES_PER_TOKEN
    if bytes_per_token <= 0:
        raise ValueError(
            f"bytes_per_token must be > 0; got {bytes_per_token}"
        )
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    total_nats = 0.0
    total_tokens = 0
    # Long-context tail probe (mirrors recipe/train.py's tail_val_bpb): the SAME
    # cross-entropy, normalized by the SAME bytes_per_token, but accumulated only
    # over the tail positions [seq_len//2 :] of each window. Penalizes recipes
    # that shorten effective training context (the 250M-transfer blind spot).
    # Recorded only — not yet consumed by the scorer.
    tail_start = seq_len // 2
    tail_nats = 0.0
    tail_tokens = 0

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
                # Tail slice: positions [tail_start:] along the sequence axis.
                # logits/tgt are (batch, seq_len, vocab) / (batch, seq_len).
                if tail_start < tgt.size(1):
                    tail_logits = logits[:, tail_start:, :]
                    tail_tgt = tgt[:, tail_start:]
                    tail_loss_sum = F.cross_entropy(
                        tail_logits.reshape(-1, tail_logits.size(-1)),
                        tail_tgt.reshape(-1),
                        reduction="sum",
                    )
                    tail_nats += tail_loss_sum.item()
                    tail_tokens += tail_tgt.numel()
                batch_inp.clear()
                batch_tgt.clear()

    total_bytes = total_tokens * bytes_per_token
    bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else float("inf")
    nll_per_token = total_nats / max(total_tokens, 1)
    tail_bytes = tail_tokens * bytes_per_token
    tail_bpb = (
        tail_nats / (math.log(2) * tail_bytes) if tail_bytes > 0 else None
    )
    return {
        "val_bpb": bpb,
        "tail_val_bpb": tail_bpb,
        "nll_per_token": nll_per_token,
        "tokens_evaluated": total_tokens,
        "bytes_per_token": bytes_per_token,
    }


def per_position_nlls(
    model: torch.nn.Module,
    eval_tokens: np.ndarray,
    seq_len: int,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> np.ndarray:
    """Per-position cross-entropy (nats) over the held-out stream, in the SAME
    window-row-major order `compute_val_bpb` sums.

    This is the producer side of HOST-side reduction: the sandboxed (untrusted)
    miner model emits this array; the validator reduces it with
    `eval.host_reduce.reduce_token_nlls` instead of trusting a miner-printed
    bpb. `reduce_token_nlls(per_position_nlls(model, ...))` reproduces
    `compute_val_bpb(model, ...)["val_bpb"]` exactly (tested).

    Returns a float32 1-D array of length `n_full_windows * seq_len`.
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    n = len(eval_tokens)
    n_windows = max(1, (n - 1) // seq_len)
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        batch_inp: list[torch.Tensor] = []
        batch_tgt: list[torch.Tensor] = []
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
                nll = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    tgt.reshape(-1),
                    reduction="none",
                )
                chunks.append(nll.detach().float().cpu().numpy())
                batch_inp.clear()
                batch_tgt.clear()
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32)


# --- HOSB: Host-Owned Suffix-Blanked scoring (look-ahead useless by construction) ---
#
# The validator scores the miner's own forward() to compute val_bpb. Under the
# normal (seq_len+1) packing the target for position t is input[t+1] — it sits
# INSIDE the model input, so a non-causal forward() can read the answer and drive
# val_bpb -> 0 (a fraudulent, unbeatable king). Detecting that is structurally
# impossible (the peek is in the shared prefix; a separate probe is distinguishable
# in a co-batched forward). HOSB removes the answer instead: for each scored
# position t the HOST feeds input[0..t] REAL and input[t+1..] overwritten with
# real-text FILLER, and scores cross-entropy against the real next token it holds
# out-of-band. A causal model is invariant to input[t+1..] so its score is
# IDENTICAL to single-pass; a look-ahead model reads filler and gains nothing.


def _seed_rng(seed: bytes) -> np.random.Generator:
    """Deterministic numpy RNG from arbitrary seed bytes (block-hash-derived in
    production). blake2b folds the bytes into a 32-bit seed sequence."""
    digest = hashlib.blake2b(seed, digest_size=32).digest()
    return np.random.default_rng(np.frombuffer(digest, dtype=np.uint32))


def _draw_filler(rng: np.random.Generator, filler_tokens: np.ndarray, length: int) -> np.ndarray:
    """`length` real-text filler tokens (a contiguous slice of the host's disjoint
    filler corpus when possible, else a sample). Same distribution as the window
    so a 'detect the seam' model gains nothing — and the answer is absent anyway."""
    if length <= 0:
        return np.zeros(0, dtype=np.int64)
    n = len(filler_tokens)
    if n >= length + 1:
        start = int(rng.integers(0, n - length))
        return filler_tokens[start : start + length].astype(np.int64)
    return rng.choice(filler_tokens, size=length, replace=True).astype(np.int64)


def build_blanked_grid(
    eval_tokens: np.ndarray,
    filler_tokens: np.ndarray,
    seq_len: int,
    seed: bytes,
    *,
    n_scored_per_window: int = 16,
    tail_fraction: float = 0.4,
    witness_fraction: float = 0.125,
    wrong_target_fraction: float = 0.0625,
) -> tuple[np.ndarray, np.ndarray, list[BlankedCell]]:
    """HOST-side: build the blanked-input grid + targets + private layout.

    Packs the same non-overlapping (seq_len+1) windows as compute_val_bpb and
    scores EXACTLY ONE host-secret position `e` per window (a `tail_fraction` of
    windows score a tail position [L//2:] for long-context coverage). For that one
    position it emits:
      idx_grid[row] = real input[0..e] then FILLER on [e+1..L-1]  (answer absent)
      tgt_grid[row] = -100 everywhere except column e = the REAL next token
    plus optional witness rows for the SAME position e (a `witness_fraction` two-
    filler copy; a `wrong_target_fraction` wrong-target copy). ALL rows of a window
    blank the identical suffix [e+1:], so no sibling row exposes another's answer,
    and different windows are disjoint token slices — closing the cross-row answer
    leak (a multi-position-per-window grid let the container read window[e+1] off a
    sibling scored at e'>=e+1). `n_scored_per_window` is retained for API
    compatibility but IGNORED (the per-window count is now fixed at one).

    Returns (idx_grid (M,L) int64, tgt_grid (M,L) int64, layout). The `seed`
    (block-hash-derived in production) makes the layout unpredictable at submission
    and reproducible for audit; it is NEVER in a container-visible artifact.
    """
    eval_tokens = np.asarray(eval_tokens)
    filler_tokens = np.asarray(filler_tokens)
    L = int(seq_len)
    if L < 2:
        raise ValueError(f"seq_len must be >= 2; got {L}")
    if len(filler_tokens) == 0:
        raise ValueError("filler_tokens must be non-empty (host-held disjoint slice)")
    if not 0.0 <= tail_fraction <= 1.0:
        raise ValueError(f"tail_fraction must be in [0, 1]; got {tail_fraction}")
    if not 0.0 <= witness_fraction <= 1.0:
        raise ValueError(f"witness_fraction must be in [0, 1]; got {witness_fraction}")
    if not 0.0 <= wrong_target_fraction < 1.0:
        # < 1 so a real-target, filler-A scored cell always remains to score.
        raise ValueError(f"wrong_target_fraction must be in [0, 1); got {wrong_target_fraction}")
    rng = _seed_rng(seed)

    n = len(eval_tokens)
    n_windows = max(0, (n - 1) // L)
    if n_windows == 0:
        raise ValueError(f"eval stream too short ({n} tokens) for one (seq_len+1={L + 1}) window")
    rows_idx: list[np.ndarray] = []
    rows_tgt: list[np.ndarray] = []
    layout: list[BlankedCell] = []
    row = 0
    tail_lo = L // 2
    for w in range(n_windows):
        start = w * L
        window = eval_tokens[start : start + L + 1]
        if len(window) < L + 1:
            break
        real_input = window[:L].astype(np.int64)        # positions 0..L-1
        real_targets = window[1 : L + 1].astype(np.int64)  # target[t] = window[t+1]

        # ONE scored position per window, NEVER the last (e <= L-2). Scoring e=L-1
        # would make the answer window[L] == the NEXT window's input column 0 (real,
        # un-blanked) — a cross-window leak. Capping at L-2 keeps every answer
        # window[e+1] inside THIS window's blanked filler region.
        # tail_fraction of windows score a tail position [L//2:]; rest a head one.
        e_hi = L - 1  # exclusive: e in [.., L-2]
        if rng.random() < tail_fraction and tail_lo < e_hi:
            e = int(rng.integers(tail_lo, e_hi))
        else:
            e = int(rng.integers(0, max(1, min(tail_lo, e_hi))))

        idx_row = real_input.copy()
        idx_row[e + 1 :] = _draw_filler(rng, filler_tokens, L - 1 - e)
        tgt_row = np.full(L, -100, dtype=np.int64)
        if rng.random() < wrong_target_fraction:
            # A host-known token that differs from the real next token.
            wrong = int(rng.choice(filler_tokens))
            guard = 0
            while wrong == int(real_targets[e]) and guard < 8:
                wrong = int(rng.choice(filler_tokens))
                guard += 1
            if wrong == int(real_targets[e]):  # degenerate filler — skip the witness
                tgt_row[e] = int(real_targets[e])
                kind = "real"
            else:
                tgt_row[e] = wrong
                kind = "wrong"
        else:
            tgt_row[e] = int(real_targets[e])
            kind = "real"
        rows_idx.append(idx_row)
        rows_tgt.append(tgt_row)
        layout.append(BlankedCell(row=row, window=w, pos=e, target_kind=kind, filler_set="A"))
        row += 1

        # Two-filler witness: a second copy of a REAL cell with different filler —
        # SAME position e, SAME blanked suffix, so it exposes nothing.
        if kind == "real" and rng.random() < witness_fraction:
            idx_rowB = real_input.copy()
            idx_rowB[e + 1 :] = _draw_filler(rng, filler_tokens, L - 1 - e)
            tgt_rowB = np.full(L, -100, dtype=np.int64)
            tgt_rowB[e] = int(real_targets[e])
            rows_idx.append(idx_rowB)
            rows_tgt.append(tgt_rowB)
            layout.append(BlankedCell(row=row, window=w, pos=e, target_kind="real", filler_set="B"))
            row += 1

    # Defensive invariant (the cross-row-leak guard): every window scores exactly
    # ONE position, so no sibling row in a window can expose another's blanked
    # answer. Fail loud if a future change reintroduces multi-position windows.
    by_window: dict[int, set] = {}
    for c in layout:
        by_window.setdefault(c.window, set()).add(c.pos)
    bad = [w for w, ps in by_window.items() if len(ps) != 1]
    if bad:
        raise AssertionError(
            f"build_blanked_grid: window(s) {bad[:3]} score multiple positions — "
            "sibling rows would expose each other's answers (cross-row leak)"
        )

    idx_grid = np.stack(rows_idx) if rows_idx else np.zeros((0, L), dtype=np.int64)
    tgt_grid = np.stack(rows_tgt) if rows_tgt else np.zeros((0, L), dtype=np.int64)
    return idx_grid, tgt_grid, layout


def build_benchmark_grid(examples: list, seed: bytes):
    """HOST-side: shuffle each multiple-choice example's candidates with a PRIVATE
    per-example permutation and emit only (context, shuffled candidate token-ids)
    with NO correct-index marker. Returns:
      contexts_flat (int64), ctx_offsets (int64, len N+1)  — ragged contexts (CSR)
      cands_shuf (int64, (N, C))                            — shuffled candidates
      correct_pos (int64, (N,))  — PRIVATE: where the target landed (host-only)
    The container scores each shuffled candidate; the host un-shuffles via
    correct_pos. A container lacking the permutation forges at most chance, and
    (with a content-whitened file) the candidate token-ids reveal nothing.
    """
    rng = _seed_rng(seed)
    contexts_flat: list[int] = []
    ctx_offsets = [0]
    cands_rows: list[list[int]] = []
    correct_pos: list[int] = []
    for ex in examples:
        cands = [int(ex["target_id"])] + [int(d) for d in ex["distractors"]]  # index 0 = target
        perm = rng.permutation(len(cands))
        cands_rows.append([cands[p] for p in perm])
        correct_pos.append(int(np.where(perm == 0)[0][0]))  # where the target (orig 0) landed
        ctx = [int(t) for t in ex["context_ids"]]
        contexts_flat.extend(ctx)
        ctx_offsets.append(len(contexts_flat))
    width = len(cands_rows[0]) if cands_rows else 0
    if any(len(r) != width for r in cands_rows):
        raise ValueError("all benchmark examples must have the same candidate count")
    return (
        np.asarray(contexts_flat, dtype=np.int64),
        np.asarray(ctx_offsets, dtype=np.int64),
        np.asarray(cands_rows, dtype=np.int64) if cands_rows else np.zeros((0, 0), np.int64),
        np.asarray(correct_pos, dtype=np.int64),
    )


def per_position_nlls_blanked(
    model: torch.nn.Module,
    idx_grid: np.ndarray,
    tgt_grid: np.ndarray,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> np.ndarray:
    """Producer (runs the miner's model): per-position NLL over the HOST grid.

    Runs the SAME `logits, _ = model(idx_batch)` call as today over the host's
    blanked rows and returns cross-entropy (nats) against tgt_grid with
    reduction='none', ignore_index=-100 — so only each row's single host-chosen
    scored cell carries a value (others are 0). Knows nothing about which cell is
    scored/witness/tail (that is the host layout). Pure function of (model, grid);
    the model never sees the targets (CE is computed here, not inside forward()).
    """
    idx_grid = np.asarray(idx_grid)
    tgt_grid = np.asarray(tgt_grid)
    if idx_grid.shape != tgt_grid.shape:
        raise ValueError(f"idx/tgt grid shape mismatch: {idx_grid.shape} vs {tgt_grid.shape}")
    M, L = idx_grid.shape
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    out = np.zeros((M, L), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, M, batch_size):
            inp = torch.from_numpy(idx_grid[s : s + batch_size].astype(np.int64)).to(device)
            tgt = torch.from_numpy(tgt_grid[s : s + batch_size].astype(np.int64)).to(device)
            logits, _ = model(inp)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt.reshape(-1),
                reduction="none",
                ignore_index=-100,
            )
            out[s : s + inp.size(0)] = nll.reshape(inp.size(0), L).detach().float().cpu().numpy()
    return out


def compute_val_bpb_on_stream(
    model: torch.nn.Module,
    batch: SealedStreamBatch,
    seq_len: int,
    batch_size: int = 8,
    device: torch.device | None = None,
) -> dict:
    """Convenience wrapper: compute val_bpb on a `SealedStreamBatch`.

    Reads `batch.spec.bytes_per_token` and passes it through to
    `compute_val_bpb`. The result dict includes a `stream_id` field so
    the validator's ladder code can route per-stream results into the
    right Pareto cell.

    Behaviour-equivalent to:
        compute_val_bpb(model, np.asarray(batch.tokens), seq_len,
                        batch_size, device,
                        bytes_per_token=batch.spec.bytes_per_token)
    plus the `stream_id` annotation.
    """
    result = compute_val_bpb(
        model,
        np.asarray(batch.tokens),
        seq_len,
        batch_size,
        device,
        bytes_per_token=batch.spec.bytes_per_token,
    )
    result["stream_id"] = batch.spec.id
    return result


def load_eval_tokens(path: Path | str) -> np.ndarray:
    return np.memmap(path, dtype=np.uint16, mode="r")
