"""
Benchmark mix scoring — placeholder for Phase 0.

The whitepaper specifies ~1500 examples drawn from a held-out benchmark mix
(MMLU/HellaSwag/ARC subsets + code/math). For Phase 0 we ship a tiny
placeholder set so the validator scoring pipeline has something to call into;
real benchmarks land in Phase 0.5 once we plug in lm-evaluation-harness or
build our own held-out mix.

The placeholder computes a stable, model-quality-correlated score on a small
synthetic completion task: given a short context, the model should rank the
correct next-token continuation above k random distractors. This isn't real
benchmark accuracy — it's a stand-in that varies with model quality so the
end-to-end pipeline has a non-trivial signal to score against.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_benchmark_score(
    model: torch.nn.Module,
    examples: list[dict],
    device: torch.device | None = None,
) -> dict:
    """
    Each example is {context_ids: list[int], target_id: int, distractors: list[int]}.
    Score = fraction where target_id has highest log-prob under model among
    (target_id + distractors).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for ex in examples:
            ctx = torch.tensor([ex["context_ids"]], dtype=torch.long, device=device)
            logits, _ = model(ctx)
            last_logits = logits[0, -1]
            candidates = [ex["target_id"]] + list(ex["distractors"])
            scores = last_logits[candidates]
            best = int(scores.argmax().item())
            if best == 0:  # index 0 == target
                correct += 1
            total += 1
    accuracy = correct / max(total, 1)
    return {"benchmark_accuracy": accuracy, "n_examples": total, "n_correct": correct}


def make_placeholder_examples(n: int = 50, seed: int = 7777, vocab_size: int = 50257) -> list[dict]:
    """Generate stable placeholder examples for the Phase 0 hidden-eval set.

    Each example is a short context drawn from a fixed token sequence; target
    is the next token; distractors are random other tokens. A model that has
    learned its training distribution will tend to put higher probability on
    targets that match adjacent-token co-occurrence statistics — coarse but
    nonzero signal.
    """
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        ctx_len = int(rng.integers(8, 24))
        context_ids = rng.integers(0, vocab_size, size=ctx_len).tolist()
        target_id = int(rng.integers(0, vocab_size))
        distractors = rng.choice(vocab_size, size=4, replace=False).tolist()
        distractors = [int(t) for t in distractors if int(t) != target_id][:4]
        examples.append({
            "context_ids": context_ids,
            "target_id": target_id,
            "distractors": distractors,
        })
    return examples
