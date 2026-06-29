"""HOSB — Host-Owned Suffix-Blanked scoring (the op4 causality redesign).

The validator computes val_bpb by running the miner's own forward(); under normal
packing target[t]==input[t+1] is IN the model input, so a non-causal forward reads
the answer and collapses val_bpb to ~0. HOSB removes the answer from the input
(real prefix [0..t], filler after) and scores against the host-held real target —
look-ahead reads filler and gains nothing, BY CONSTRUCTION, while a causal model's
score is identical to single-pass. These CPU tests pin:

  * the answer is physically absent from every scored row;
  * an honest causal model's HOSB score == single-pass at the same positions;
  * a look-ahead forward gains NOTHING (HOSB stays high; single-pass collapses);
  * the host-owned witnesses (two-filler invariance, wrong-target floor) reject
    a future-dependent / target-reading producer;
  * the sampled estimator is unbiased and the tail is covered.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.host_reduce import BlankedCell, NonCausalModelError, reduce_blanked_nlls
from eval.val_bpb import (
    build_blanked_grid,
    compute_val_bpb,
    per_position_nlls,
    per_position_nlls_blanked,
)

VOCAB = 256
SEED = b"hosb-test-seed-0"


# --- stand-in models (recipe-independent; the HOSB properties are model-agnostic) ---


class CausalModel(nn.Module):
    """logits[t] = head(emb(input[t])) — depends only on input[t] (a strict subset
    of [0..t]), so genuinely causal. Stands in for any honest LM."""

    def __init__(self, vocab: int = VOCAB, dim: int = 16) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx, targets=None):
        return self.head(self.emb(idx)), None


class LookAheadModel(nn.Module):
    """Malicious: one-hots input[t+1] (== the real target under normal packing).
    Collapses single-pass val_bpb to ~0; HOSB feeds it filler at input[t+1] so it
    gains nothing. The dummy parameter stands for the trivial structural patch a
    miner adds to route op4 into the patched path where their forward() runs."""

    def __init__(self, vocab: int = VOCAB) -> None:
        super().__init__()
        self.vocab = vocab
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        logits = torch.zeros((B, T, self.vocab))  # uniform default (last position)
        if T >= 2:
            ar = torch.arange(T - 1)  # positions 0..T-2 can peek input[t+1]
            for b in range(B):
                logits[b, ar, idx[b, 1:]] = 30.0  # one-hot the peeked next token
        return logits + self.p * 0, None


def _streams(n_eval=400, n_filler=400):
    rng = np.random.default_rng(3)
    # Both in-vocab [0, VOCAB) so the stand-in models can embed them. The real
    # filler is a disjoint slice of the SAME corpus (same vocab) — disjoint here
    # means a different stream region, not a different id range.
    eval_tokens = rng.integers(0, VOCAB, size=n_eval, dtype=np.uint16)
    filler_tokens = rng.integers(0, VOCAB, size=n_filler, dtype=np.uint16)
    return eval_tokens, filler_tokens


# ---------------------------------------------------------------------------
# build_blanked_grid — structural properties (no model)
# ---------------------------------------------------------------------------


def test_blanked_grid_answer_is_physically_absent():
    eval_tokens, _ = _streams()
    # Disjoint id range for the filler so "answer absent" is checkable by exact
    # membership (no model is run here, so out-of-vocab ids are fine).
    filler = np.random.default_rng(9).integers(1000, 1000 + VOCAB, size=400, dtype=np.uint16)
    L = 16
    idx, tgt, layout = build_blanked_grid(eval_tokens, filler, L, SEED, n_scored_per_window=8)
    assert len(layout) > 0 and idx.shape[1] == L
    filler_set = set(int(x) for x in filler)
    for c in layout:
        w_start = c.window * L
        window = eval_tokens[w_start : w_start + L + 1]
        # prefix [0..pos] is the REAL input, byte-identical to single-pass
        assert np.array_equal(idx[c.row, : c.pos + 1], window[: c.pos + 1].astype(np.int64))
        # everything after the scored position is FILLER (answer overwritten)
        if c.pos + 1 < L:
            suffix = idx[c.row, c.pos + 1 :]
            assert all(int(x) in filler_set for x in suffix), "blanked suffix is not filler"
            real_target = int(window[c.pos + 1])  # the answer
            assert real_target not in set(int(x) for x in suffix), "answer leaked into the input"
        # target tensor: -100 everywhere except the scored column
        assert (tgt[c.row] == -100).sum() == L - 1
        assert tgt[c.row, c.pos] != -100


def test_grid_one_scored_position_per_window_no_cross_row_leak():
    """THE cross-row-leak fix: each window scores exactly ONE position, so no
    sibling row exposes another's blanked answer window[e+1]. (A multi-position
    grid let a container recover ~93% of answers off siblings scored at e'>=e+1.)"""
    eval_tokens, _ = _streams()
    # disjoint filler id range so "answer absent" is checkable exactly
    filler = np.random.default_rng(9).integers(1000, 1000 + VOCAB, size=400, dtype=np.uint16)
    L = 16
    idx, _tgt, layout = build_blanked_grid(eval_tokens, filler, L, SEED)

    from collections import defaultdict
    pos_by_window = defaultdict(set)
    cells_by_window = defaultdict(list)
    for c in layout:
        pos_by_window[c.window].add(c.pos)
        cells_by_window[c.window].append(c)
    assert pos_by_window  # non-empty grid
    for w, ps in pos_by_window.items():
        assert len(ps) == 1, f"window {w} scores multiple positions {ps} (cross-row leak)"
    # the scored answer is filler (absent) at its column in EVERY row of the window
    for w, cells in cells_by_window.items():
        e = cells[0].pos
        if e + 1 >= L:
            continue
        answer = int(eval_tokens[w * L : w * L + L + 1][e + 1])
        for c in cells:
            assert int(idx[c.row, e + 1]) != answer


def test_blanked_grid_covers_the_tail_and_emits_witnesses():
    eval_tokens, filler = _streams()
    L = 16
    _, _, layout = build_blanked_grid(
        eval_tokens, filler, L, SEED,
        n_scored_per_window=10, tail_fraction=0.4,
        witness_fraction=0.5, wrong_target_fraction=0.25,
    )
    real_a = [c for c in layout if c.target_kind == "real" and c.filler_set == "A"]
    tail = [c for c in real_a if c.pos >= L // 2]
    assert tail, "tail positions [L//2:] must be scored (long-context coverage)"
    assert any(c.filler_set == "B" for c in layout), "two-filler witness rows missing"
    assert any(c.target_kind == "wrong" for c in layout), "wrong-target witness rows missing"


def test_blanked_grid_seed_not_embedded_in_container_artifacts():
    eval_tokens, filler = _streams()
    idx, tgt, _ = build_blanked_grid(eval_tokens, filler, 16, SEED)
    # The seed must never reach a container-visible array.
    assert SEED not in idx.tobytes()
    assert SEED not in tgt.tobytes()


# ---------------------------------------------------------------------------
# Honest causal == single-pass (king comparability), and look-ahead gains nothing
# ---------------------------------------------------------------------------


def test_honest_causal_equals_single_pass_per_position():
    eval_tokens, filler = _streams()
    L = 16
    model = CausalModel()
    idx, tgt, layout = build_blanked_grid(eval_tokens, filler, L, SEED, n_scored_per_window=12)
    hosb = per_position_nlls_blanked(model, idx, tgt)
    single = per_position_nlls(model, eval_tokens, L)  # window-row-major: w*L + pos
    for c in layout:
        if c.target_kind == "real" and c.filler_set == "A":
            got = hosb[c.row, c.pos]
            ref = single[c.window * L + c.pos]
            # CPU float32, byte-identical real prefix → essentially exact.
            assert got == pytest.approx(ref, rel=1e-4, abs=1e-4), f"pos {c.pos}: {got} vs {ref}"


def test_lookahead_model_gains_nothing_under_hosb():
    eval_tokens, filler = _streams(n_eval=800, n_filler=800)
    L = 32
    honest, cheat = CausalModel(), LookAheadModel()
    honest_single = compute_val_bpb(honest, eval_tokens, L, bytes_per_token=4.0)["val_bpb"]
    cheat_single = compute_val_bpb(cheat, eval_tokens, L, bytes_per_token=4.0)["val_bpb"]
    # THE ATTACK: in single-pass the cheat reads input[t+1]==answer and crushes the
    # honest baseline — it would steal the crown.
    assert cheat_single < 0.25 * honest_single, f"cheat {cheat_single} vs honest {honest_single}"

    # HOSB: input[t+1] is filler at every scored position → look-ahead is useless.
    # Witnesses off here to isolate the absent-answer effect (the look-ahead model
    # is future-dependent on the filler, which the two-filler witness also catches
    # — covered separately; here we measure only the score).
    idx, tgt, layout = build_blanked_grid(
        eval_tokens, filler, L, SEED,
        n_scored_per_window=20, witness_fraction=0.0, wrong_target_fraction=0.0,
    )
    cheat_hosb = reduce_blanked_nlls(
        per_position_nlls_blanked(cheat, idx, tgt), layout,
        seq_len=L, bytes_per_token=4.0, eval_set_hash="x",
    ).val_bpb
    # The cheat no longer beats the honest baseline — the crown is not stealable.
    assert cheat_hosb > cheat_single
    assert cheat_hosb >= 0.8 * honest_single, f"HOSB cheat {cheat_hosb} should be >= honest {honest_single}"


def test_sampled_estimator_is_unbiased_for_causal_model():
    eval_tokens, filler = _streams(n_eval=600, n_filler=600)
    L = 16
    model = CausalModel()
    full = compute_val_bpb(model, eval_tokens, L, bytes_per_token=4.0)["val_bpb"]
    seeds = [f"seed-{i}".encode() for i in range(12)]
    estimates = []
    for s in seeds:
        # Defaults ON (witnesses + forced tail) — an honest causal model must pass
        # both witnesses AND remain unbiased under the stratum-weighted reduction.
        idx, tgt, layout = build_blanked_grid(eval_tokens, filler, L, s, n_scored_per_window=8)
        hosb = per_position_nlls_blanked(model, idx, tgt)
        estimates.append(reduce_blanked_nlls(
            hosb, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x",
        ).val_bpb)
    # Mean over random scored subsets converges to the full-window val_bpb.
    assert np.mean(estimates) == pytest.approx(full, rel=0.05)


def test_reduce_is_stratum_weighted_not_flat_when_tail_oversampled():
    """The estimator bug guard: with the tail OVERSAMPLED and head/tail NLL
    differing, val_bpb must equal the stratum-weighted (true full-window) mean,
    NOT the flat mean of the sampled cells (which the oversampling would skew)."""
    import math
    L = 8  # head [0:4), tail [4:8) — strata are equal size (4 each)
    # 1 head cell @ 4.0, 5 tail cells @ 2.0  → tail is heavily oversampled.
    layout = [BlankedCell(row=0, window=0, pos=1, target_kind="real", filler_set="A")]
    nlls = np.zeros((6, L), dtype=np.float32)
    nlls[0, 1] = 4.0
    for r, pos in enumerate([4, 5, 6, 7, 4], start=1):
        layout.append(BlankedCell(row=r, window=0, pos=pos, target_kind="real", filler_set="A"))
        nlls[r, pos] = 2.0
    out = reduce_blanked_nlls(nlls, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x")
    # stratum-weighted: (4*head_mean + 4*tail_mean)/8 = (4*4.0 + 4*2.0)/8 = 3.0
    # flat (biased) would be (4.0 + 5*2.0)/6 = 2.33 — must NOT be that.
    assert out.nll_per_token == pytest.approx(3.0, rel=1e-9)
    assert out.val_bpb == pytest.approx(3.0 / (math.log(2) * 4.0), rel=1e-9)
    assert out.val_bpb != pytest.approx((4.0 + 5 * 2.0) / 6 / (math.log(2) * 4.0), rel=1e-3)


# ---------------------------------------------------------------------------
# reduce_blanked_nlls — host-owned witnesses (craft NLL arrays directly)
# ---------------------------------------------------------------------------


def _grid(rows, L=8):
    return np.zeros((rows, L), dtype=np.float32)


def test_two_filler_witness_flags_future_dependence():
    L = 8
    layout = [
        BlankedCell(row=0, window=0, pos=3, target_kind="real", filler_set="A"),
        BlankedCell(row=1, window=0, pos=3, target_kind="real", filler_set="B"),
    ]
    nlls = _grid(2, L)
    nlls[0, 3] = 1.00
    nlls[1, 3] = 1.30  # moved 0.30 > tol when only filler changed → future-dependent
    with pytest.raises(NonCausalModelError, match="filler"):
        reduce_blanked_nlls(nlls, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x", tol_witness=0.05)


def test_two_filler_witness_passes_when_invariant():
    L = 8
    layout = [
        BlankedCell(row=0, window=0, pos=3, target_kind="real", filler_set="A"),
        BlankedCell(row=1, window=0, pos=3, target_kind="real", filler_set="B"),
    ]
    nlls = _grid(2, L)
    nlls[0, 3] = 1.000
    nlls[1, 3] = 1.010  # within tol → causal
    out = reduce_blanked_nlls(nlls, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x", tol_witness=0.05)
    assert out.tokens_evaluated == 1  # only the A cell scores; B is witness-only


def test_wrong_target_witness_rejects_target_reader_in_aggregate():
    """A target-tensor reader scores ~0 on (almost) ALL wrong targets — it can't
    tell them from real cells — so the sub-floor FRACTION is ~1.0 → reject."""
    L = 8
    layout = [BlankedCell(row=0, window=0, pos=2, target_kind="real", filler_set="A")]
    nlls = _grid(7, L)
    nlls[0, 2] = 3.0
    for r in range(1, 7):  # 6 wrong cells, all ~0 (reader one-hots the wrong target)
        layout.append(BlankedCell(row=r, window=0, pos=2, target_kind="wrong", filler_set="A"))
        nlls[r, 2] = 0.001
    with pytest.raises(NonCausalModelError, match="wrong"):
        reduce_blanked_nlls(nlls, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x")


def test_wrong_target_witness_tolerates_occasional_honest_collision():
    """An honest confident model occasionally argmax-collides with a random wrong
    token (one sub-floor cell), but the FRACTION stays small → must NOT reject.
    This is the false-positive the absolute per-cell floor produced."""
    L = 8
    layout = [BlankedCell(row=0, window=0, pos=2, target_kind="real", filler_set="A")]
    nlls = _grid(9, L)
    nlls[0, 2] = 3.0
    lows = {1}  # exactly one collision out of eight wrong cells (12.5% < 50%)
    for r in range(1, 9):
        layout.append(BlankedCell(row=r, window=0, pos=2, target_kind="wrong", filler_set="A"))
        nlls[r, 2] = 0.1 if r in lows else 6.0
    out = reduce_blanked_nlls(nlls, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x")
    assert out.tokens_evaluated == 1  # honest model not rejected; only the real cell scores


def test_reduce_rejects_nonfinite_and_negative_scored_cells():
    L = 8
    layout = [BlankedCell(row=0, window=0, pos=2, target_kind="real", filler_set="A")]
    bad = _grid(1, L)
    bad[0, 2] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        reduce_blanked_nlls(bad, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x")
    neg = _grid(1, L)
    neg[0, 2] = -0.5
    with pytest.raises(ValueError, match="negative"):
        reduce_blanked_nlls(neg, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x")


def test_reduce_raises_on_empty_grid_instead_of_inf():
    # A grid with no real-A scored cell must fail LOUD, never return val_bpb=inf.
    with pytest.raises(ValueError, match="no real-target scored cells"):
        reduce_blanked_nlls(np.zeros((0, 8), dtype=np.float32), [], seq_len=8, bytes_per_token=4.0, eval_set_hash="x")


def test_build_validates_fractions_and_stream_length():
    eval_tokens, filler = _streams()
    with pytest.raises(ValueError, match="wrong_target_fraction"):
        build_blanked_grid(eval_tokens, filler, 16, SEED, wrong_target_fraction=1.0)
    with pytest.raises(ValueError, match="too short"):
        build_blanked_grid(np.arange(4, dtype=np.uint16), filler, 16, SEED)


def test_tail_val_bpb_uses_exact_layout_position():
    L = 8  # tail_start = 4
    layout = [
        BlankedCell(row=0, window=0, pos=1, target_kind="real", filler_set="A"),  # head
        BlankedCell(row=1, window=0, pos=6, target_kind="real", filler_set="A"),  # tail
    ]
    nlls = _grid(2, L)
    nlls[0, 1] = 2.0
    nlls[1, 6] = 4.0
    out = reduce_blanked_nlls(nlls, layout, seq_len=L, bytes_per_token=4.0, eval_set_hash="x")
    assert out.tokens_evaluated == 2
    # tail_val_bpb is computed from the pos>=4 cell ONLY (the 4.0), not a modular mask.
    import math
    assert out.tail_val_bpb == pytest.approx(4.0 / (math.log(2) * 1 * 4.0), rel=1e-6)
