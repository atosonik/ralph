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
# Honest pre-softmax logits sit in ~[-50, 50]; anything beyond this is a forged /
# degenerate emission whose only purpose is float cancellation. Reject it.
_MAX_ABS_LOGIT = 1e4


def ce_from_topk_logits(
    topk_logits: np.ndarray,
    topk_indices: np.ndarray,
    targets: np.ndarray,
    vocab_size: int,
) -> np.ndarray:
    """Host-side cross-entropy (nats) at each scored cell from the container's
    emitted top-K logits. The container NEVER receives the targets (leak-free) AND
    never supplies the partition function: the HOST computes it.

    The (untrusted) producer emits, per scored row, only the top-K logits + their
    vocab indices. The host sets ``Z_hat = logsumexp(emitted top-K)`` — a true
    LOWER BOUND on the full-vocab partition function (a partial sum cannot exceed
    the full sum). Then:

      * target in the emitted top-K:  CE = Z_hat - logit[target].
      * target NOT in top-K (ranked below the emitted set): CE = Z_hat - min(top-K)
        — the boundary logit is an upper bound on an unranked target's logit, so
        this is the *lowest* CE the cell can honestly carry, and a degenerate
        emission (one real token + filler) makes min(top-K) tiny → a huge miss CE.

    Why this is robust (vs. trusting a container ``logsumexp``): the container
    CANNOT make Z_hat smaller than logsumexp(its true top-K) — emitting more/larger
    entries only RAISES Z_hat and thus CE; and CE is invariant to a uniform logit
    shift (Z_hat and logit[target] shift together). So the "claim tail=0 to deflate"
    and "uniformly rescale" forgeries are structurally impossible. The only residual
    lever is omitting REAL tail mass, which deflates a hit cell by exactly
    -log(1-tail_K) (tail_K = the model's true out-of-top-K mass); raising K bounds
    that below the crown margin, and dropping mid-rank tokens to enlarge the gap
    sends every non-top-1-correct cell to the huge boundary miss CE — self-defeating
    for any realistic (sub-perfect top-1) model.

    `vocab_size` is used ONLY for index-range validation. Returns float64 (M,) CE.
    """
    topk_logits = np.asarray(topk_logits, dtype=np.float64)
    topk_indices = np.asarray(topk_indices)
    targets = np.asarray(targets)
    if topk_logits.ndim != 2 or topk_logits.shape != topk_indices.shape:
        raise ValueError(
            f"topk_logits/topk_indices must match 2-D shapes; got {topk_logits.shape} {topk_indices.shape}"
        )
    m, k = topk_logits.shape
    if targets.shape != (m,):
        raise ValueError(f"targets must be ({m},); got {targets.shape}")
    if not np.all(np.isfinite(topk_logits)):
        raise ValueError("non-finite topk_logits")
    if np.any((topk_indices < 0) | (topk_indices >= vocab_size)):
        raise ValueError("topk_indices outside [0, vocab_size)")
    sorted_i = np.sort(topk_indices, axis=1)
    if k > 1 and np.any(sorted_i[:, 1:] == sorted_i[:, :-1]):
        raise ValueError("duplicate topk_indices within a row")
    # Magnitude cap: an honest softmax never needs |logit| > ~50; huge-but-finite
    # logits (e.g. 1e18) make z_hat - logit[target] catastrophically cancel to 0
    # (differences fall below the float ULP), forging CE=0. Reject them outright.
    if np.any(np.abs(topk_logits) > _MAX_ABS_LOGIT):
        raise ValueError(f"topk_logits magnitude exceeds {_MAX_ABS_LOGIT} — forged/degenerate emission")

    # Cross-entropy from MAX-SHIFTED logits so all terms are O(log K) and there is
    # no catastrophic cancellation. z_hat = logsumexp(top-K) is the host's lower
    # bound on the full-vocab Z; CE = z_hat - logit[target] = lse_shifted -
    # (logit[target] - mx).
    mx = topk_logits.max(axis=1)
    shifted = topk_logits - mx[:, None]
    lse_shifted = np.log(np.exp(shifted).sum(axis=1))   # = z_hat - mx, in [0, log K]
    match = topk_indices == targets[:, None]            # (M, K) — one True at most per row
    hit = match.any(axis=1)
    tgt_shifted = np.where(match, shifted, -np.inf).max(axis=1)  # logit[target]-mx (<=0), -inf if absent
    ce_hit = lse_shifted - tgt_shifted                  # valid where hit
    ce_miss = lse_shifted - shifted.min(axis=1)         # target below the K-th: boundary CE
    return np.maximum(np.where(hit, ce_hit, ce_miss), 0.0)


def reduce_benchmark_scores(
    bench_scores: np.ndarray,
    correct_pos: np.ndarray,
    n_witness: int = 0,
) -> tuple[float, float]:
    """HOST verdict for the multiple-choice benchmark: the container emits a score
    per SHUFFLED candidate (it never sees which slot is correct); the host argmaxes
    and compares to the PRIVATE `correct_pos`.

    `bench_scores` is (N, C). For each example the predicted slot is argmax; ties
    get FRACTIONAL credit (1/#tied if `correct_pos` is in the tie set) so an
    all-equal / blind producer scores exactly E[acc] = mean(1/C), not 1.0. Because
    argmax is permutation-equivariant, an honest deterministic model gets the SAME
    accuracy as the legacy un-shuffled scorer (king comparability). Returns
    (accuracy, stderr). `n_witness` trailing rows (if any) are excluded.
    """
    bench_scores = np.asarray(bench_scores, dtype=np.float64)
    correct_pos = np.asarray(correct_pos)
    if bench_scores.ndim != 2:
        raise ValueError(f"bench_scores must be 2-D (N,C); got {bench_scores.shape}")
    n, c = bench_scores.shape
    if correct_pos.shape != (n,):
        raise ValueError(f"correct_pos must be ({n},); got {correct_pos.shape}")
    if not np.all(np.isfinite(bench_scores)):
        raise ValueError("non-finite benchmark scores")
    keep = n - max(0, int(n_witness))
    if keep <= 0:
        return 0.0, 0.0
    credit = 0.0
    for i in range(keep):
        row = bench_scores[i]
        tied = np.flatnonzero(row >= row.max() - 1e-9)
        if correct_pos[i] in tied:
            credit += 1.0 / len(tied)
    acc = credit / keep
    stderr = math.sqrt(max(acc * (1.0 - acc), 0.0) / keep)
    return acc, stderr


@dataclass(frozen=True)
class HostReduced:
    val_bpb: float
    tail_val_bpb: float | None
    nll_per_token: float
    tokens_evaluated: int
    bytes_per_token: float
    eval_set_hash: str  # sha256 of the HOST's target stream (not miner-supplied)


class NonCausalModelError(RuntimeError):
    """The producer's per-position NLL is not consistent with an honest causal
    model under the HOST's blanked-grid scoring (HOSB) — its scored-position NLL
    moved when only the (post-position) filler changed, or it scored a
    deliberately-wrong target near zero (reading the target tensor / looking
    ahead). The caller treats this as a rejected submission, never a silent pass.
    """


@dataclass(frozen=True)
class BlankedCell:
    """Host-private record binding one emitted grid row to what it scores.

    `pos` is the single scored sequence index of `row` (the last REAL token
    before the blanked filler suffix); `target_kind` is "real" (counts toward
    val_bpb) or "wrong" (a deliberately-wrong target, witness only); `filler_set`
    is "A" (the scored copy) or "B" (a second copy with different filler, witness
    only). NEVER serialized into any container-visible artifact — it is the host
    layout the container is not told.
    """

    row: int
    window: int
    pos: int
    target_kind: str  # "real" | "wrong"
    filler_set: str   # "A" | "B"


def reduce_blanked_nlls(
    nlls_2d: np.ndarray,
    layout: list[BlankedCell],
    *,
    seq_len: int,
    bytes_per_token: float,
    eval_set_hash: str,
    tol_witness: float = 0.05,
    wrong_target_floor: float = 1.0,
    wrong_target_max_low_frac: float = 0.5,
) -> HostReduced:
    """HOST verdict over a HOSB blanked-grid NLL array. Re-runs no model.

    `nlls_2d` is the (M, L) per-position NLL the producer emitted over the host's
    blanked grid (`build_blanked_grid`). `layout` is the host-private map from row
    to its scored cell. Because every scored position's strict future was
    overwritten with filler before the producer ran, a look-ahead forward read
    filler (not the answer) and CANNOT drive the scored NLL to ~0 — so the score
    is honest BY CONSTRUCTION. Two host-owned witnesses catch residual cheats:

      * two-filler (A/B): for a causal/future-independent producer the scored NLL
        is invariant to the filler bytes; a row's "B" copy (same prefix + target,
        different filler) must match its "A" copy within `tol_witness`.
      * wrong-target: a producer reading the target tensor one-hots a
        deliberately-wrong target and scores it ~0. An HONEST model scores a
        RANDOM wrong target near-uniformly (it can't confidently predict a random
        token), so only a SMALL fraction of wrong cells land below
        `wrong_target_floor` (an occasional argmax collision). We reject only when
        MORE than `wrong_target_max_low_frac` of wrong cells are sub-floor — a
        target-reader is ~all sub-floor; a confident-but-honest model is not.

    val_bpb is a STRATUM-WEIGHTED mean over the real-target, filler-A cells: head
    [0:L//2) and tail [L//2:L) are weighted by their true sizes, so oversampling
    the tail (forced tail coverage) does NOT bias val_bpb relative to
    compute_val_bpb's full-window mean. Witness rows (B and wrong) never score.
    Raises NonCausalModelError on a witness failure; ValueError on a malformed
    array (non-finite / negative / out-of-range) or a grid with no scored cell.
    """
    if bytes_per_token <= 0:
        raise ValueError(f"bytes_per_token must be > 0; got {bytes_per_token}")
    nlls_2d = np.asarray(nlls_2d, dtype=np.float64)
    if nlls_2d.ndim != 2:
        raise ValueError(f"nlls_2d must be 2-D (M,L); got shape {nlls_2d.shape}")
    M, L = nlls_2d.shape

    # Validate every scored cell the host knows about (index bounds + sanity).
    for c in layout:
        if not (0 <= c.row < M and 0 <= c.pos < L):
            raise ValueError(f"layout cell out of bounds: row={c.row} pos={c.pos} for {nlls_2d.shape}")
        v = nlls_2d[c.row, c.pos]
        if not np.isfinite(v):
            raise ValueError("nll array has a non-finite value at a scored cell")
        if v < 0:
            raise ValueError("nll array has a negative cross-entropy at a scored cell (impossible)")

    # WRONG-TARGET WITNESS: a model that reads/one-hots the target tensor scores
    # ~0 on a deliberately-wrong target — and on (almost) ALL of them, since it
    # can't tell wrong cells from real ones. An honest model scores a RANDOM wrong
    # token near-uniformly; only an occasional argmax collision lands sub-floor.
    # Reject on the AGGREGATE (fraction), never a single cell (that false-rejects
    # confident honest models — a strictly-causal peaked LM collides sometimes).
    wrong_cells = [c for c in layout if c.target_kind == "wrong"]
    if wrong_cells:
        low = sum(1 for c in wrong_cells if nlls_2d[c.row, c.pos] < wrong_target_floor)
        frac_low = low / len(wrong_cells)
        if frac_low > wrong_target_max_low_frac:
            raise NonCausalModelError(
                f"{low}/{len(wrong_cells)} ({frac_low:.0%}) deliberately-wrong targets scored "
                f"< {wrong_target_floor} nats (> {wrong_target_max_low_frac:.0%}) — reading the "
                f"target tensor / look-ahead, rejected"
            )

    # TWO-FILLER WITNESS: scored NLL must be invariant to the filler bytes.
    a_by_cell = {
        (c.window, c.pos): c
        for c in layout
        if c.target_kind == "real" and c.filler_set == "A"
    }
    for c in layout:
        if c.target_kind == "real" and c.filler_set == "B":
            a = a_by_cell.get((c.window, c.pos))
            if a is not None and abs(nlls_2d[c.row, c.pos] - nlls_2d[a.row, a.pos]) > tol_witness:
                raise NonCausalModelError(
                    f"blanked-position NLL moved {abs(nlls_2d[c.row, c.pos] - nlls_2d[a.row, a.pos]):.4f} "
                    f"> {tol_witness} when only filler changed — future-dependent forward(), rejected"
                )

    # SCORE: real-target, filler-A cells only (B and wrong are witness-only).
    score_cells = [c for c in layout if c.target_kind == "real" and c.filler_set == "A"]
    if not score_cells:
        raise ValueError("no real-target scored cells in layout — malformed HOSB grid (fail loud, never inf)")

    # Stratum-weighted mean: head [0:tail_start) and tail [tail_start:L) are
    # weighted by their TRUE sizes, so forcing extra tail coverage does not bias
    # val_bpb relative to compute_val_bpb's uniform full-window mean. Positions
    # are sampled uniformly WITHIN each stratum, so each stratum mean is unbiased.
    tail_start = seq_len // 2
    head_vals = np.array([nlls_2d[c.row, c.pos] for c in score_cells if c.pos < tail_start], dtype=np.float64)
    tail_vals = np.array([nlls_2d[c.row, c.pos] for c in score_cells if c.pos >= tail_start], dtype=np.float64)
    if head_vals.size and tail_vals.size:
        mean_nll = (tail_start * head_vals.mean() + (seq_len - tail_start) * tail_vals.mean()) / seq_len
    else:  # only one stratum sampled (tiny n_scored) — flat mean of what's present
        mean_nll = float(np.concatenate([head_vals, tail_vals]).mean())

    bpb = mean_nll / (LN2 * bytes_per_token)
    nll_per_token = float(mean_nll)
    tail_bpb: float | None = (
        float(tail_vals.mean()) / (LN2 * bytes_per_token) if tail_vals.size else None
    )

    return HostReduced(
        val_bpb=bpb,
        tail_val_bpb=tail_bpb,
        nll_per_token=nll_per_token,
        tokens_evaluated=len(score_cells),
        bytes_per_token=bytes_per_token,
        eval_set_hash=eval_set_hash,
    )


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
