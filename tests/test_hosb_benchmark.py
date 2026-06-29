"""HOSB stage-4b — host-reduced benchmark (HRB) + content whitening.

The legacy benchmark is computed INSIDE the untrusted container from a file that
carries the correct-answer index AND (on the deployed file) a target-id
distribution distinct from distractors — so a container forges accuracy=1.0, or a
model-free "pick the smallest token-id" cheat scores far above chance. HRB moves
the comparison to the HOST (private per-example shuffle + host argmax) and the
whitened generator makes the candidate token-ids exchangeable. These tests pin:
  * the generator is content-whitened (target / distractor ids share a marginal);
  * the host shuffle carries no correct-index marker; correct_pos is exact;
  * host argmax == legacy scorer for an honest model (king comparability);
  * a blind / all-equal / smallest-id container is bounded at chance.
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
from eval.benchmark import compute_benchmark_score, make_placeholder_examples
from eval.host_reduce import reduce_benchmark_scores
from eval.val_bpb import build_benchmark_grid

VOCAB = 256


class _Model(nn.Module):
    """logits[t] = head(emb(input[t])) — deterministic, so the last-token scores
    are a fixed function of the candidate ids (a clean equivalence oracle)."""

    def __init__(self, vocab: int = VOCAB, dim: int = 16) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.emb = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, idx, targets=None):
        return self.head(self.emb(idx)), None


def _container_bench_scores(model, contexts_flat, ctx_offsets, cands_shuf):
    """Simulate the in-container benchmark scoring: per example, model(context) →
    last-token logits → score at each shuffled candidate."""
    model.eval()
    out = np.zeros(cands_shuf.shape, dtype=np.float32)
    with torch.no_grad():
        for i in range(cands_shuf.shape[0]):
            ctx = contexts_flat[ctx_offsets[i]:ctx_offsets[i + 1]]
            logits, _ = model(torch.tensor([ctx], dtype=torch.long))
            last = logits[0, -1]
            out[i] = last[torch.tensor(cands_shuf[i])].numpy()
    return out


def test_generator_is_content_whitened():
    ex = make_placeholder_examples(n=600, seed=1, vocab_size=VOCAB, n_candidates=5)
    targets = np.array([e["target_id"] for e in ex])
    distractors = np.array([d for e in ex for d in e["distractors"]])
    # Same exchangeable pool → means within sampling noise (NOT the deployed file's
    # 4550-vs-24917 skew that the smallest-id cheat exploited).
    assert abs(targets.mean() - distractors.mean()) < 0.08 * VOCAB


def test_build_benchmark_grid_has_no_marker_and_exact_correct_pos():
    ex = make_placeholder_examples(n=40, seed=2, vocab_size=VOCAB)
    _cf, _co, cands_shuf, correct_pos = build_benchmark_grid(ex, b"bench-seed")
    assert cands_shuf.shape == (40, 5)
    for i, e in enumerate(ex):
        # correct_pos points at the target token in the shuffled row, and nothing
        # in the emitted grid marks which slot that is.
        assert cands_shuf[i, correct_pos[i]] == e["target_id"]


def test_hrb_equals_legacy_for_honest_model():
    model = _Model()
    ex = make_placeholder_examples(n=120, seed=3, vocab_size=VOCAB)
    legacy = compute_benchmark_score(model, ex)["benchmark_accuracy"]

    cf, co, cands_shuf, correct_pos = build_benchmark_grid(ex, b"seed-7")
    scores = _container_bench_scores(model, cf, co, cands_shuf)
    acc, _stderr = reduce_benchmark_scores(scores, correct_pos)
    assert acc == legacy  # argmax is permutation-equivariant → bit-identical


def test_blind_container_bounded_at_chance():
    n, c = 400, 5
    correct_pos = np.random.default_rng(4).integers(0, c, size=n)
    # all-equal scores → fractional tie credit = 1/C exactly
    allequal = np.zeros((n, c), dtype=np.float32)
    acc_eq, _ = reduce_benchmark_scores(allequal, correct_pos)
    assert acc_eq == pytest.approx(1.0 / c)
    # random scores → ~1/C
    rnd = np.random.default_rng(5).standard_normal((n, c)).astype(np.float32)
    acc_rnd, _ = reduce_benchmark_scores(rnd, correct_pos)
    assert abs(acc_rnd - 1.0 / c) < 0.06


def test_smallest_id_cheat_is_chance_on_whitened_grid():
    """The model-free 'prefer the smallest candidate token-id' cheat (78.5% on the
    deployed non-whitened file) returns to chance once candidates are whitened."""
    ex = make_placeholder_examples(n=800, seed=6, vocab_size=VOCAB)
    _cf, _co, cands_shuf, correct_pos = build_benchmark_grid(ex, b"seed-9")
    cheat_scores = (-cands_shuf).astype(np.float32)  # smallest id → highest score
    acc, _ = reduce_benchmark_scores(cheat_scores, correct_pos)
    assert abs(acc - 1.0 / cands_shuf.shape[1]) < 0.05
