"""Scoring kernels for the downstream-eval harness (B1).

Three pure functions — `score_mc`, `score_schema`, `score_lm` — implement the
three task modes DCLM's CORE-22 eval bundle uses. They take a `forward_logits`
callable (a model adapter that returns `(B, T, V)` logits) and pre-tokenized
examples, and return predictions / per-example NLLs. No model loading, no
tokenizer plumbing, no I/O — those live in `core22.py` / `private_hard.py` /
`runner.py` and are layered on top.

Why pure kernels:
  * Testable with synthetic logits on CPU — no GPU, no downloaded checkpoints.
  * Reusable across CORE-22 and the private hardness subset.
  * The math of "log-prob of continuation given context" is the same in all
    three modes; only the comparison rule changes.

Forward-pass adapter:
  `_extract_logits(model_output)` tolerates models that return logits directly
  OR a `(logits, loss_or_aux)` tuple. Ralph's `RalphBase` is the latter; most
  external HF models are the former.

Position indexing (the one subtle thing):
  Given input `[t_0, t_1, ..., t_{T-1}]`, position `i` in the forward output
  predicts token `t_{i+1}`. To score the log-prob of continuation token
  `cont[k]` (the k-th continuation token), we read the model's logits at
  position `len(context) + k - 1`. The off-by-one is unit-tested.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

# ----------------------------------------------------------------------------
# Data classes — pre-tokenized examples
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class MCExample:
    """A multiple-choice example with pre-tokenized fields.

    The model scores each choice by summing per-token log-probabilities of
    the choice continuation given `context_ids` as context. The choice with
    the highest length-normalized log-probability is the prediction.

    Tokenization happens at the data-loader level (core22.py / private_hard.py);
    scorer.py only sees token IDs.
    """

    context_ids: list[int]
    choice_ids: list[list[int]]   # one list of ids per choice
    gold: int                     # index into choice_ids


@dataclass(frozen=True)
class SchemaExample:
    """A schema example — multiple (context, continuation) variants where
    BOTH the context AND the continuation may differ between variants.

    COPA is the canonical case: "I dropped the glass because ____" with two
    contexts ("because it was slippery." vs "because it was heavy.") and the
    model picks the most-likely variant.

    For winograd/winogrande the same structure applies — context and
    continuation are sliced differently per choice.
    """

    context_ids: list[list[int]]      # one per variant
    continuation_ids: list[list[int]] # one per variant — same length as context_ids
    gold: int                         # index into variants

    def __post_init__(self) -> None:
        if len(self.context_ids) != len(self.continuation_ids):
            raise ValueError(
                f"SchemaExample: context_ids ({len(self.context_ids)}) and "
                f"continuation_ids ({len(self.continuation_ids)}) length mismatch"
            )


@dataclass(frozen=True)
class LMExample:
    """A pre-tokenized language modeling example — one (context, target).

    The scoring function returns per-example NLL (negative log-likelihood
    of `target_ids` given `context_ids`). Higher-level eval code converts
    this to accuracy (e.g. LAMBADA: gold target is correct iff its NLL is
    lower than all distractor candidates).
    """

    context_ids: list[int]
    target_ids: list[int]


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _extract_logits(model_output) -> torch.Tensor:
    """Adapt to both `model(x) -> logits` and `model(x) -> (logits, _)`.

    Ralph's RalphBase returns the tuple form (logits, optional_loss); most
    HuggingFace causal-LM models return logits directly. The harness must
    handle either without forcing the caller to know which is which.
    """
    if isinstance(model_output, tuple):
        if len(model_output) == 0:
            raise ValueError("model returned an empty tuple")
        return model_output[0]
    return model_output


def _check_finite_logits(logits: torch.Tensor) -> None:
    """Raise on NaN / Inf logits.

    Silent NaN propagation through log_softmax + indexed lookup would give
    `0.0` log-probability (NaN beats anything in argmax) and silently
    poison scoring. We require the caller to ship clean logits and raise
    loudly if they don't.
    """
    if not torch.isfinite(logits).all():
        raise ValueError(
            "forward_logits returned non-finite values (NaN or Inf). "
            "Clean logits are a precondition for scorer.py."
        )


def _logprob_of_continuation(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    context_ids: list[int],
    continuation_ids: list[int],
) -> float:
    """Sum log-probabilities of `continuation_ids` tokens given `context_ids`.

    The forward pass runs over `context_ids + continuation_ids`. Position `i`
    in the resulting logits predicts token at position `i + 1`. To score
    `continuation_ids[k]` (the k-th continuation token), we read logits at
    position `len(context_ids) + k - 1`.

    Edge cases:
      * `continuation_ids == []` → returns 0.0 (the empty prefix has unit
        probability by convention; nothing to score).
      * `len(context_ids) + len(continuation_ids) < 2` → ValueError; there
        is no position from which to predict the first continuation token.
        (Equivalent: caller forgot to include a BOS / context token.)
    """
    if len(continuation_ids) == 0:
        return 0.0

    full = context_ids + continuation_ids
    if len(full) < 2:
        raise ValueError(
            f"need at least 2 tokens to score a continuation; "
            f"got context={len(context_ids)} continuation={len(continuation_ids)}"
        )

    input_ids = torch.tensor(full, dtype=torch.long).unsqueeze(0)  # (1, T)
    raw_output = forward_logits(input_ids)
    logits = _extract_logits(raw_output)
    _check_finite_logits(logits)

    if logits.dim() != 3 or logits.size(0) != 1:
        raise ValueError(
            f"expected logits of shape (1, T, V); got {tuple(logits.shape)}"
        )
    seq_len = logits.size(1)

    log_probs = torch.log_softmax(logits.float(), dim=-1)  # (1, T, V)

    context_len = len(context_ids)
    total = 0.0
    for k, target_token in enumerate(continuation_ids):
        position = context_len + k - 1
        if position < 0:
            raise ValueError(
                f"position {position} < 0 — context is empty; "
                "callers must include at least one context token"
            )
        if position >= seq_len:
            raise ValueError(
                f"position {position} >= sequence length {seq_len}; "
                "model returned fewer logits than input tokens"
            )
        total += float(log_probs[0, position, target_token].item())
    return total


# ----------------------------------------------------------------------------
# Public scorers
# ----------------------------------------------------------------------------


def score_mc(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    examples: list[MCExample],
    *,
    length_normalize: bool = True,
) -> list[int]:
    """Score multiple-choice examples.

    For each example, run the model on `context + choice` for every choice
    and pick the choice with the highest log-probability (optionally
    length-normalized). Returns one predicted-index per example.

    Length normalization (default True) matches DCLM / lm-eval-harness's
    convention for MMLU, ARC, etc. — the longer correct answer should not
    be penalized for length. The unnormalized variant matches HellaSwag's
    original protocol; pass `length_normalize=False` for that.
    """
    predictions: list[int] = []
    for ex in examples:
        scores: list[float] = []
        for choice_ids in ex.choice_ids:
            lp = _logprob_of_continuation(forward_logits, ex.context_ids, choice_ids)
            if length_normalize and len(choice_ids) > 0:
                lp = lp / len(choice_ids)
            scores.append(lp)
        # argmax with stable tie-break: take the lowest-index choice on ties.
        best = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[best]:
                best = i
        predictions.append(best)
    return predictions


def score_schema(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    examples: list[SchemaExample],
) -> list[int]:
    """Score schema examples (COPA, Winograd, Winogrande, etc.).

    Each variant has its OWN (context, continuation) pair — context differs
    per variant. Score = unnormalized log-probability of continuation given
    context. Highest-scoring variant is the prediction.

    No length normalization by default: schema tasks set continuation
    lengths to be approximately equal across variants by construction, so
    normalization can introduce noise rather than reduce it. Pass an
    externally-normalized variant if you disagree with this default.
    """
    predictions: list[int] = []
    for ex in examples:
        scores: list[float] = []
        for ctx, cont in zip(ex.context_ids, ex.continuation_ids):
            scores.append(_logprob_of_continuation(forward_logits, ctx, cont))
        best = 0
        for i in range(1, len(scores)):
            if scores[i] > scores[best]:
                best = i
        predictions.append(best)
    return predictions


def score_mc_logprobs(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    examples: list[MCExample],
    *,
    length_normalize: bool = True,
) -> list[list[float]]:
    """Like score_mc but returns the per-choice log-probabilities instead
    of the argmax index.

    Used by grader.py to compute `gold_margin_bits = log_p(gold) -
    max_{d != gold} log_p(d)` per example. Inner lists are in choice
    order; lengths match `examples[i].choice_ids` lengths.
    """
    result: list[list[float]] = []
    for ex in examples:
        scores: list[float] = []
        for choice_ids in ex.choice_ids:
            lp = _logprob_of_continuation(forward_logits, ex.context_ids, choice_ids)
            if length_normalize and len(choice_ids) > 0:
                lp = lp / len(choice_ids)
            scores.append(lp)
        result.append(scores)
    return result


def score_schema_logprobs(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    examples: list[SchemaExample],
) -> list[list[float]]:
    """Like score_schema but returns per-variant log-probabilities instead
    of the argmax index. Symmetric to score_mc_logprobs; used by grader.py."""
    result: list[list[float]] = []
    for ex in examples:
        scores: list[float] = []
        for ctx, cont in zip(ex.context_ids, ex.continuation_ids):
            scores.append(_logprob_of_continuation(forward_logits, ctx, cont))
        result.append(scores)
    return result


def score_lm(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    examples: list[LMExample],
) -> list[float]:
    """Score language-modeling examples — return per-example NLL (nats).

    Negative-log-likelihood is `-sum(log p(target_token | prefix))` over the
    target tokens. Lower is better; the caller's task-specific layer turns
    this into accuracy (e.g. LAMBADA: pick the candidate with lowest NLL).

    Returned in nats; convert with `nats / math.log(2)` for bits, or with
    the bytes-per-token ratio for bits-per-byte.
    """
    nlls: list[float] = []
    for ex in examples:
        lp = _logprob_of_continuation(forward_logits, ex.context_ids, ex.target_ids)
        nlls.append(-lp)
    return nlls
