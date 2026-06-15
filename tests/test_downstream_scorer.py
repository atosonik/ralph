"""Tests for the score_mc / score_schema / score_lm scoring kernels.

15 cases cover the math, the position indexing, error paths, model-output
tuple handling, length normalization, deterministic batching, and the
NaN/Inf clean-logits invariant.

The forward function is mocked with a hand-built logits tensor so the tests
exercise the SCORER LOGIC, not a model. Real-model integration lives in
core22.py / private_hard.py and is tested separately.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream.scorer import (
    LMExample,
    MCExample,
    SchemaExample,
    _extract_logits,
    _logprob_of_continuation,
    score_lm,
    score_mc,
    score_schema,
)

# ----------------------------------------------------------------------------
# Helpers — hand-built logits to drive scorer functions
# ----------------------------------------------------------------------------

VOCAB = 16   # tiny vocab so we can hand-craft cases


def _make_forward(logits_by_input: dict[tuple[int, ...], list[list[float]]]):
    """Build a forward function that returns logits for specific inputs.

    `logits_by_input` maps tuple(input_ids) -> list of per-position logits
    of length T × VOCAB. The forward function looks up the input and returns
    the matching tensor. Raises if the caller passes an input not pre-registered.
    """

    def forward(input_ids: torch.Tensor) -> torch.Tensor:
        key = tuple(input_ids.flatten().tolist())
        if key not in logits_by_input:
            raise AssertionError(
                f"test bug: unregistered forward call with ids={key}"
            )
        rows = logits_by_input[key]
        return torch.tensor([rows], dtype=torch.float32)  # (1, T, V)

    return forward


def _uniform_logits(T: int) -> list[list[float]]:
    """T × VOCAB rows of uniform zero logits → uniform 1/VOCAB log-probs."""
    return [[0.0] * VOCAB for _ in range(T)]


def _logits_favoring(token: int, position: int, T: int, advantage: float = 5.0):
    """Logits that strongly prefer `token` at the given position."""
    rows = _uniform_logits(T)
    rows[position][token] = advantage
    return rows


# ----------------------------------------------------------------------------
# _logprob_of_continuation — the core math
# ----------------------------------------------------------------------------


def test_logprob_continuation_uniform_logits():
    """With uniform logits, log_prob of any token = -log(VOCAB) per token.
    A 2-token continuation has total log_prob = -2·log(VOCAB)."""
    ctx = [1, 2]
    cont = [3, 4]
    # 4 input positions → logits shape (1, 4, V); 4 rows of uniform
    fwd = _make_forward({tuple(ctx + cont): _uniform_logits(4)})
    lp = _logprob_of_continuation(fwd, ctx, cont)
    expected = -2.0 * math.log(VOCAB)
    assert lp == pytest.approx(expected, rel=1e-6)


def test_logprob_continuation_zero_for_empty_continuation():
    """No tokens to score → 0.0 by convention."""
    # forward will never be called
    fwd = _make_forward({})
    assert _logprob_of_continuation(fwd, [1, 2], []) == 0.0


def test_logprob_continuation_rejects_singleton_input():
    """Total input length < 2 → no position from which to predict the first
    continuation token. The "at least 2 tokens" branch fires before the
    position-< 0 defensive check; the latter is theoretically unreachable
    via the public API but kept as defense-in-depth in the kernel."""
    # Empty ctx + single cont = total length 1 → "at least 2 tokens"
    fwd = _make_forward({})
    with pytest.raises(ValueError, match=r"at least 2 tokens"):
        _logprob_of_continuation(fwd, [], [5])


def test_logprob_continuation_empty_both_returns_zero():
    """Empty context + empty continuation → 0 by the empty-continuation
    early return. Both empty is a degenerate input but consistent."""
    fwd = _make_forward({})
    assert _logprob_of_continuation(fwd, [], []) == 0.0


def test_logprob_continuation_position_indexing():
    """Position k+1 is predicted by logits at position k. Verify the
    off-by-one by skewing logits so only the RIGHT position gives a high
    log-prob for the target token."""
    ctx = [1, 2]
    cont = [9, 10]
    # Full input = [1, 2, 9, 10], T=4
    # Position 0 predicts token at pos 1 (= 2) — irrelevant
    # Position 1 predicts token at pos 2 (= 9) — relevant for cont[0]=9
    # Position 2 predicts token at pos 3 (= 10) — relevant for cont[1]=10
    # Position 3 predicts beyond — unused
    rows = _uniform_logits(4)
    rows[1][9] = 5.0   # strongly favor token 9 at position 1 → cont[0]=9 ✓
    rows[2][10] = 5.0  # strongly favor token 10 at position 2 → cont[1]=10 ✓

    fwd = _make_forward({tuple(ctx + cont): rows})
    lp = _logprob_of_continuation(fwd, ctx, cont)
    # Compare to uniform-logits baseline
    lp_uniform = -2.0 * math.log(VOCAB)
    assert lp > lp_uniform, "favored positions should give higher log-prob"


def test_logprob_continuation_raises_on_nan_logits():
    """NaN in logits → ValueError, not silent propagation."""
    rows = _uniform_logits(2)
    rows[0][0] = float("nan")
    fwd = _make_forward({(1, 2): rows})
    with pytest.raises(ValueError, match=r"non-finite"):
        _logprob_of_continuation(fwd, [1], [2])


def test_logprob_continuation_raises_on_inf_logits():
    rows = _uniform_logits(2)
    rows[0][3] = float("inf")
    fwd = _make_forward({(1, 2): rows})
    with pytest.raises(ValueError, match=r"non-finite"):
        _logprob_of_continuation(fwd, [1], [2])


# ----------------------------------------------------------------------------
# _extract_logits — model-output tuple handling
# ----------------------------------------------------------------------------


def test_extract_logits_passthrough_tensor():
    t = torch.zeros((1, 3, VOCAB))
    assert torch.equal(_extract_logits(t), t)


def test_extract_logits_unwraps_tuple():
    """Ralph's RalphBase returns (logits, optional_loss). The first element
    is the logits."""
    t = torch.zeros((1, 3, VOCAB))
    loss = torch.tensor(1.23)
    assert torch.equal(_extract_logits((t, loss)), t)
    # Also (logits,)-single — the unwrap should still work
    assert torch.equal(_extract_logits((t,)), t)


def test_extract_logits_rejects_empty_tuple():
    with pytest.raises(ValueError, match=r"empty tuple"):
        _extract_logits(())


# ----------------------------------------------------------------------------
# score_mc — MC kernel
# ----------------------------------------------------------------------------


def test_score_mc_picks_choice_with_higher_logprob():
    """3-choice example where choice 1 has strongly favored logits at the
    right positions. Expect prediction = 1."""
    ctx = [1, 2]
    # Each choice is one token: [3], [4], [5]
    # Full inputs:
    #   choice 0: [1, 2, 3] — pos 1 predicts pos 2 = 3
    #   choice 1: [1, 2, 4] — pos 1 predicts pos 2 = 4
    #   choice 2: [1, 2, 5] — pos 1 predicts pos 2 = 5
    rows_uniform = _uniform_logits(3)
    rows_choice1 = _uniform_logits(3)
    rows_choice1[1][4] = 10.0  # heavily favor token 4 at the relevant position
    fwd = _make_forward({
        (1, 2, 3): rows_uniform,
        (1, 2, 4): rows_choice1,
        (1, 2, 5): rows_uniform,
    })
    ex = MCExample(context_ids=ctx, choice_ids=[[3], [4], [5]], gold=1)
    pred = score_mc(fwd, [ex])
    assert pred == [1]


def test_score_mc_length_normalize_matters():
    """A longer choice with better per-token quality can win under
    length-normalize and lose under unnormalized accumulation.

    Construction (with VOCAB=16, baseline uniform logits at 0):
      Choice 0 ([7], 1 token): boost token 7 with logit 2.0
        → log p(7|1) = 2.0 - log(exp(2)+15) ≈ -1.109
        → total -1.109, per-token avg -1.109
      Choice 1 ([8,9], 2 tokens): boost both with logit 2.5
        → per-token log p ≈ 2.5 - log(exp(2.5)+15) ≈ -0.803
        → total ≈ -1.606, per-token avg ≈ -0.803

    Length-normalized: choice 1 (-0.803) > choice 0 (-1.109) → choice 1 wins.
    Unnormalized: choice 0 (-1.109) > choice 1 (-1.606) → choice 0 wins.
    """
    ctx = [1]
    rows_short = _uniform_logits(2)
    rows_short[0][7] = 2.0
    rows_long = _uniform_logits(3)
    rows_long[0][8] = 2.5
    rows_long[1][9] = 2.5
    fwd = _make_forward({
        (1, 7): rows_short,
        (1, 8, 9): rows_long,
    })
    ex = MCExample(context_ids=ctx, choice_ids=[[7], [8, 9]], gold=1)

    pred_norm = score_mc(fwd, [ex], length_normalize=True)
    assert pred_norm == [1]  # per-token avg wins

    pred_no_norm = score_mc(fwd, [ex], length_normalize=False)
    assert pred_no_norm == [0]  # shorter total wins


def test_score_mc_tie_takes_lowest_index():
    """Identical logits across choices → tie → return lowest-index choice
    (stable, deterministic)."""
    ctx = [1]
    rows = _uniform_logits(2)
    fwd = _make_forward({
        (1, 5): rows,
        (1, 6): rows,
        (1, 7): rows,
    })
    ex = MCExample(context_ids=ctx, choice_ids=[[5], [6], [7]], gold=0)
    assert score_mc(fwd, [ex]) == [0]


def test_score_mc_empty_examples_returns_empty():
    fwd = _make_forward({})
    assert score_mc(fwd, []) == []


# ----------------------------------------------------------------------------
# score_schema — schema kernel
# ----------------------------------------------------------------------------


def test_score_schema_picks_higher_logprob_variant():
    """Two variants with different (context, continuation) pairs. The one
    with sharper logits at the prediction position wins."""
    # Variant 0: context [1, 2], continuation [5]
    rows0 = _uniform_logits(3)
    # Variant 1: context [3, 4], continuation [6]
    rows1 = _uniform_logits(3)
    rows1[1][6] = 8.0
    fwd = _make_forward({
        (1, 2, 5): rows0,
        (3, 4, 6): rows1,
    })
    ex = SchemaExample(
        context_ids=[[1, 2], [3, 4]],
        continuation_ids=[[5], [6]],
        gold=1,
    )
    assert score_schema(fwd, [ex]) == [1]


def test_score_schema_rejects_length_mismatch():
    with pytest.raises(ValueError, match=r"length mismatch"):
        SchemaExample(
            context_ids=[[1], [2]],
            continuation_ids=[[3]],
            gold=0,
        )


# ----------------------------------------------------------------------------
# score_lm — LM kernel
# ----------------------------------------------------------------------------


def test_score_lm_returns_per_example_nll():
    """NLL = -sum(log p(target | prefix)). For 2 uniform-logit tokens that
    is 2·log(VOCAB)."""
    # Input [1, 2, 3, 4], context=[1], target=[2, 3, 4] — 3 target tokens
    rows = _uniform_logits(4)
    fwd = _make_forward({(1, 2, 3, 4): rows})
    ex = LMExample(context_ids=[1], target_ids=[2, 3, 4])
    nlls = score_lm(fwd, [ex])
    assert len(nlls) == 1
    assert nlls[0] == pytest.approx(3.0 * math.log(VOCAB), rel=1e-6)


def test_score_lm_zero_target_returns_zero_nll():
    """Empty target → 0 NLL by the empty-continuation convention."""
    fwd = _make_forward({})
    ex = LMExample(context_ids=[1, 2], target_ids=[])
    assert score_lm(fwd, [ex]) == [0.0]


# ----------------------------------------------------------------------------
# Determinism + format pins
# ----------------------------------------------------------------------------


def test_score_mc_deterministic_across_calls():
    """Two calls on the same examples + same forward give identical
    predictions. Required so seeded validator eval matches across re-runs."""
    ctx = [1, 2]
    rows_a = _uniform_logits(3)
    rows_b = _uniform_logits(3)
    rows_b[1][7] = 4.0
    fwd = _make_forward({
        (1, 2, 6): rows_a,
        (1, 2, 7): rows_b,
    })
    ex = MCExample(context_ids=ctx, choice_ids=[[6], [7]], gold=1)
    p1 = score_mc(fwd, [ex])
    p2 = score_mc(fwd, [ex])
    assert p1 == p2 == [1]


def test_logits_shape_validation():
    """Logits must be (1, T, V). 2D logits (missing batch dim) → ValueError."""
    def bad_forward(input_ids):
        return torch.zeros((4, VOCAB))  # missing batch dim
    with pytest.raises(ValueError, match=r"shape \(1, T, V\)"):
        _logprob_of_continuation(bad_forward, [1, 2], [3, 4])
