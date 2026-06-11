"""
Multi-seed eval scaffolding (Track A / v0.10) — forward-compat plumbing.

The validator-controlled multi-seed defence is the right answer to two related
problems on the live rule:

  1. Goodhart "seed search": a miner trains the same recipe under N random init
     seeds and submits the one with the best val_bpb. Expected gain at N=200 is
     E[max of N standard normals] · σ ≈ √(2·ln N) · σ ≈ 3.26 · 0.013 ≈ 0.042 bpb
     — exactly the 3× noise-floor threshold the dominant-quality clause uses.
     A multi-seed gate averages this win away.

  2. Eval-set variance: the live op4_hidden_eval is deterministic on a given
     checkpoint (same `eval/private/active_tokens.bin`, same forward pass,
     same val_bpb every time). The Cross-Scale Downstream Pareto build (B2)
     will rotate held-out streams per epoch, making the eval genuinely
     stochastic — at which point the pooled-stderr gate this module computes
     becomes meaningful.

This module ships the plumbing today so B2's switchover is a one-line wiring
change rather than a refactor:

  - derive_seeds(block_hash_hex, k): deterministic k-tuple of int seeds from
    an on-chain block hash. Pin this signature now so callers stay stable.
  - op4_hidden_eval_multiseed(...): calls op4_hidden_eval k times, returns
    means + pooled stderrs across the runs. With today's deterministic eval
    all k runs are byte-identical and stderr is exactly 0; documented and
    tested as such in tests/test_multiseed_scaffolding.py.

Nothing in production code calls this module yet — that wiring is B3 / B7's
responsibility. Track A only ships the API surface + the Goodhart guard in
validator/scoring.py.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eval import HiddenEvalResult


# Range for derive_seeds output. NumPy's default_rng accepts any uint64, but
# Python's `random.Random.seed` and PyTorch's manual_seed are happiest with
# values that fit in a signed 64-bit integer. We bound to [0, 2^31 - 1] so the
# seeds work universally with all RNGs.
_SEED_MAX = (1 << 31) - 1


def derive_seeds(block_hash_hex: str, k: int) -> list[int]:
    """Derive k deterministic int seeds from an on-chain block-hash hex string.

    Properties (pinned by tests/test_multiseed_scaffolding.py):
      - Deterministic: same hash → same seed list, forever.
      - Independent: distinct hashes → distinct seed lists (with high probability;
        a 256-bit collision is the limit, not a regular outcome).
      - Bounded: each seed is in [0, 2^31 - 1] so it works for PyTorch /
        Python `random` / NumPy alike.
      - Order-stable: seed_i comes from the i-th sub-hash, not from
        sorted/iterated state, so reordering callers can't change the
        derivation.

    The seed at index i is the first 4 bytes of blake2b(block_hash || "/" || i)
    interpreted as a big-endian uint32, masked to fit in 31 bits.

    Args:
        block_hash_hex: A hex string from ChainInterface.get_block_hash().
            "0x" prefix is tolerated.
        k: Number of seeds to derive. Must be >= 1.

    Returns:
        A list of k seeds, each in [0, _SEED_MAX].

    Raises:
        ValueError: if k < 1 or block_hash_hex is empty.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not block_hash_hex:
        raise ValueError("block_hash_hex must not be empty")
    body = block_hash_hex[2:] if block_hash_hex[:2].lower() == "0x" else block_hash_hex
    body = body.lower()
    seeds: list[int] = []
    for i in range(k):
        h = hashlib.blake2b(f"{body}/{i}".encode("ascii"), digest_size=32).digest()
        seeds.append(int.from_bytes(h[:4], "big") & _SEED_MAX)
    return seeds


@dataclass
class MultiSeedEvalResult:
    """Aggregate of k op4_hidden_eval runs across validator-chosen seeds.

    With today's deterministic eval (B2 not yet shipped), every individual
    result is byte-identical and stderr is exactly 0. The dataclass carries
    the per-run values too so the eventual switchover to stochastic eval
    only requires the aggregation logic to start producing nonzero stderr —
    no schema change.
    """

    k: int
    seeds: list[int]
    val_bpb_mean: float
    val_bpb_stderr: float
    benchmark_acc_mean: float
    benchmark_acc_stderr: float
    individual: list[HiddenEvalResult]


def _mean_stderr(xs: list[float]) -> tuple[float, float]:
    """Mean and stderr-of-the-mean for a list of floats.

    Stderr is `sample_std / sqrt(n)` where `sample_std` uses Bessel's
    correction (ddof=1). For n=1 the stderr is 0 by convention — there is
    only one observation, no within-sample variance to estimate. For
    byte-identical samples (n>=2 but all values equal) stderr is also 0.
    """
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    mean = sum(xs) / n
    if n == 1:
        return mean, 0.0
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    sample_std = math.sqrt(var)
    return mean, sample_std / math.sqrt(n)


def op4_hidden_eval_multiseed(
    karpa_root: Path,
    proof_dir: Path,
    k: int = 3,
    *,
    seeds: list[int] | None = None,
) -> tuple[bool, str, MultiSeedEvalResult | None]:
    """Call op4_hidden_eval k times and aggregate.

    This is the forward-compat wrapper. Production code paths (judge_submission
    in validator/service.py) do NOT call this today — the v0.10 ship only
    introduces the API. B3 / B7 will wire it in.

    Args:
        karpa_root: Project root (forwarded to op4_hidden_eval).
        proof_dir: Bundle directory (forwarded to op4_hidden_eval).
        k: Number of seeds to evaluate under. Default 3. The plan's pinned
           value for v0.10 → v0.11.
        seeds: Optional explicit seed list. If None, falls back to
            derive_seeds(get_block_hash(get_current_block), k) at call time;
            callers wiring this in must pass `seeds` derived from the chain.

    Returns:
        (ok, detail, MultiSeedEvalResult | None) — mirrors op4_hidden_eval's
        triple shape so callers can pattern-match either function.
    """
    if k < 1:
        return False, f"k must be >= 1, got {k}", None
    if seeds is None:
        # The current op4_hidden_eval signature ignores seeds entirely, so
        # passing None here is fine for today's pipeline. Callers in B3+
        # must derive_seeds() from on-chain randomness and pass the result.
        seeds = list(range(k))
    if len(seeds) != k:
        return False, f"len(seeds)={len(seeds)} != k={k}", None

    # Local import to avoid a circular import in pytest collection — the
    # validator package imports `from . import scoring` which imports nothing
    # heavy, but importing `validator.validator` from within validator/* in
    # module scope creates a top-level cycle.
    from validator.validator import op4_hidden_eval

    individual: list[HiddenEvalResult] = []
    for seed in seeds:
        del seed  # forward-compat: today's eval doesn't consume the seed
        ok, detail, result = op4_hidden_eval(karpa_root, proof_dir)
        if not ok or result is None:
            return False, f"sub-eval failed: {detail}", None
        individual.append(result)

    val_bpb_mean, val_bpb_stderr = _mean_stderr([r.val_bpb for r in individual])
    bench_mean, bench_stderr = _mean_stderr([r.benchmark_accuracy for r in individual])

    return True, (
        f"k={k} val_bpb_mean={val_bpb_mean:.4f}±{val_bpb_stderr:.4f} "
        f"bench_mean={bench_mean:.3f}±{bench_stderr:.3f}"
    ), MultiSeedEvalResult(
        k=k,
        seeds=list(seeds),
        val_bpb_mean=val_bpb_mean,
        val_bpb_stderr=val_bpb_stderr,
        benchmark_acc_mean=bench_mean,
        benchmark_acc_stderr=bench_stderr,
        individual=individual,
    )
