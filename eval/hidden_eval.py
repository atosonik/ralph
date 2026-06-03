"""
Hidden-eval orchestrator — validator-side scoring of a submitted checkpoint.

The validator-owned, rotating private eval set lives at:
  eval/private/active_tokens.bin   (val_bpb stream)
  eval/private/active_benchmark.json (benchmark mix)

The active subset is drawn weekly from a 10× pool by on-chain randomness
beacon under commit-reveal (whitepaper §5.7). Phase 0 hardcodes a fixed
active subset for repeatable tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .benchmark import compute_benchmark_score, make_placeholder_examples
from .val_bpb import compute_val_bpb, load_eval_tokens


@dataclass
class HiddenEvalResult:
    val_bpb: float
    benchmark_accuracy: float
    tokens_evaluated: int
    benchmark_examples: int
    eval_set_hash: str


def _stable_hash(obj) -> str:
    import hashlib
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def run_hidden_eval(
    model: torch.nn.Module,
    eval_dir: Path | str,
    seq_len: int = 256,
    bpb_batch_size: int = 8,
) -> HiddenEvalResult:
    eval_dir = Path(eval_dir)
    tokens_path = eval_dir / "active_tokens.bin"
    if tokens_path.exists():
        eval_tokens = load_eval_tokens(tokens_path)
    else:
        # Phase 0 fallback: synthesize a small reproducible eval token stream
        # so the smoke test runs without a pre-built eval shard.
        rng = np.random.default_rng(424242)
        eval_tokens = rng.integers(0, 50257, size=4096, dtype=np.uint16)

    bpb_result = compute_val_bpb(
        model,
        np.asarray(eval_tokens),
        seq_len=seq_len,
        batch_size=bpb_batch_size,
    )

    benchmark_path = eval_dir / "active_benchmark.json"
    if benchmark_path.exists():
        examples = json.loads(benchmark_path.read_text())
    else:
        examples = make_placeholder_examples(n=50)

    bench_result = compute_benchmark_score(model, examples)

    eval_set_hash = _stable_hash({
        "tokens_sha256": _stable_hash(list(map(int, np.asarray(eval_tokens)[:100]))),
        "benchmark_sha256": _stable_hash(examples),
    })

    return HiddenEvalResult(
        val_bpb=bpb_result["val_bpb"],
        benchmark_accuracy=bench_result["benchmark_accuracy"],
        tokens_evaluated=bpb_result["tokens_evaluated"],
        benchmark_examples=bench_result["n_examples"],
        eval_set_hash=eval_set_hash,
    )
