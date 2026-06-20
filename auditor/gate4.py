"""Gate 4 — GPU re-eval (Phase 3).

Gates 1-3 (hash/sig, weight-replay, diff) prove a validator did the *scoring
math* honestly on CPU, but they TRUST the published `val_bpb` — the one number
a GPU actually produced. Gate 4 closes that: a GPU-equipped auditor re-runs the
val_bpb eval on the sealed stream the validator committed to and asserts the
recomputed number matches the claim within tolerance. Same checkpoint, same
sealed stream → the result is deterministic modulo cross-GPU floating-point
drift, which is far below the 0.0064 val_bpb noise floor.

This module is the **decision spine** — tiering, sampling, tolerance, verdict —
and is pure/CPU-importable (no torch at import time). The actual re-eval
(`reeval_val_bpb_on_sample`) lazily imports torch + the validator's OWN
`eval.val_bpb.compute_val_bpb` (fidelity by construction, exactly as Gate 2
imports the scorer constants), so it only pulls the GPU stack when invoked.

Tiering (focus the expensive re-run where it matters — see 00_v0_11_master §6):

    plain_failure      → NONE   (CPU Gates 1-3 + the recomputed king decision
                                  already cover it; its weight is in the 10% pool
                                  only and it lost — re-eval buys nothing)
    meaningful_failure → CHEAP  (modest GPU, sampled — only divides the 10% pool)
    king_change cand.  → H200   (decisive_vs_king or is_first — the 90-100% share
                                  rides on this number; full re-eval, max scrutiny)

H100_FULL is the escalation rung: a CHEAP sampled pass that lands inside an
escalation band (near tolerance) gets re-run in full on bigger iron before a
divergence is called.

Exit code (extends auditor/main.py): 4 = eval diverged (claim not reproduced).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Exit code, continuing auditor/main.py's ladder (0 clean / 1 hash-sig / 2 math
# diverge / 3 network).
EXIT_EVAL_DIVERGE = 4

# --- tiers ---------------------------------------------------------------
TIER_NONE = "none"          # no GPU re-eval; CPU gates suffice
TIER_CHEAP = "cheap_gpu"    # sampled re-eval on a modest GPU
TIER_H100_FULL = "h100_full"  # full re-eval (escalation rung)
TIER_H200_KINGCHANGE = "h200_kingchange"  # full re-eval, king-change candidates

# Fraction of eval windows to re-run per tier. King-change is load-bearing →
# full; meaningful-failure only divides the 10% pool → sampled.
_TIER_SAMPLE_FRACTION = {
    TIER_NONE: 0.0,
    TIER_CHEAP: 0.25,
    TIER_H100_FULL: 1.0,
    TIER_H200_KINGCHANGE: 1.0,
}

# Default pass band for |recomputed - claimed| val_bpb. The val_bpb noise floor
# (B5 calibration) is σ≈0.0064 and the win threshold is ≈0.013; an honest
# re-eval of the SAME checkpoint+stream drifts only by cross-GPU FP error (≪σ),
# while a validator inflating its score must move val_bpb by ≥ the win
# threshold. We default to half the noise floor — comfortably above honest drift,
# comfortably below a score-changing lie.
#
# ⚠️ CALIBRATE before trusting in production: measure real cross-GPU val_bpb
# drift (same ckpt+stream across the GPU SKUs auditors actually run) and set this
# to a few× the observed max. Passed explicitly so a calibration run can override.
VAL_BPB_NOISE_FLOOR = 0.0064
DEFAULT_GATE4_TOLERANCE = VAL_BPB_NOISE_FLOOR / 2  # 0.0032

# Re-eval landing within [tolerance, tolerance × ESCALATION_BAND] of the claim is
# "borderline" → escalate a sampled CHEAP pass to a full H100 pass before calling
# divergence (a sampled subset is noisier than the full set).
ESCALATION_BAND = 2.0


def _is_king_change_candidate(eval_output: dict[str, Any]) -> bool:
    """A submission the king decision rides on: it decisively beat the king, or
    it is the genesis (first) submission. Both are published in eval_output and
    independently recomputed by the replay gate — Gate 4 re-verifies the number
    underneath that decision."""
    return bool(
        eval_output.get("decisive_vs_king")
        or eval_output.get("is_first")
        or eval_output.get("gate") == "king_change"
    )


def select_tier(eval_output: dict[str, Any]) -> str:
    """Map one submission's published eval_output to its Gate-4 re-eval tier."""
    if _is_king_change_candidate(eval_output):
        return TIER_H200_KINGCHANGE
    if eval_output.get("gate") == "meaningful_failure":
        return TIER_CHEAP
    return TIER_NONE


def sample_fraction(tier: str) -> float:
    """Fraction of eval windows to re-run for `tier`."""
    try:
        return _TIER_SAMPLE_FRACTION[tier]
    except KeyError:
        raise ValueError(f"unknown Gate-4 tier {tier!r}") from None


@dataclass(frozen=True)
class Gate4Verdict:
    """Outcome of a Gate-4 val_bpb re-eval for one submission."""

    tier: str
    claimed_val_bpb: Optional[float]
    recomputed_val_bpb: Optional[float]
    delta: Optional[float]            # recomputed - claimed (None if not re-run)
    tolerance: float
    passed: bool
    escalate: bool                    # borderline → re-run fuller before calling it
    reason: str

    @property
    def exit_code(self) -> int:
        return EXIT_EVAL_DIVERGE if not self.passed else 0


def evaluate_val_bpb_match(
    claimed_val_bpb: Optional[float],
    recomputed_val_bpb: Optional[float],
    *,
    tier: str,
    tolerance: float = DEFAULT_GATE4_TOLERANCE,
) -> Gate4Verdict:
    """Compare a re-run val_bpb against the validator's claim.

    A TIER_NONE submission is not re-run → passes by deferral to Gates 1-3.
    Otherwise the absolute delta must be within `tolerance`; landing in the
    escalation band (≤ tolerance × ESCALATION_BAND) sets `escalate` so a sampled
    pass can be re-run in full before a hard divergence is declared.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0; got {tolerance}")

    if tier == TIER_NONE:
        return Gate4Verdict(
            tier=tier, claimed_val_bpb=claimed_val_bpb, recomputed_val_bpb=None,
            delta=None, tolerance=tolerance, passed=True, escalate=False,
            reason="tier=none: CPU gates 1-3 cover this submission; no re-eval",
        )

    if claimed_val_bpb is None or recomputed_val_bpb is None:
        return Gate4Verdict(
            tier=tier, claimed_val_bpb=claimed_val_bpb,
            recomputed_val_bpb=recomputed_val_bpb, delta=None,
            tolerance=tolerance, passed=False, escalate=False,
            reason="missing val_bpb (claimed or recomputed) — cannot verify",
        )

    delta = recomputed_val_bpb - claimed_val_bpb
    abs_delta = abs(delta)
    passed = abs_delta <= tolerance
    escalate = (
        tier in (TIER_CHEAP, TIER_H100_FULL)
        and tolerance < abs_delta <= tolerance * ESCALATION_BAND
    )
    if passed:
        reason = f"val_bpb reproduced: |Δ|={abs_delta:.6f} ≤ tol {tolerance:.6f}"
    elif escalate:
        reason = (
            f"borderline: |Δ|={abs_delta:.6f} in escalation band "
            f"({tolerance:.6f}, {tolerance * ESCALATION_BAND:.6f}] — re-run fuller"
        )
    else:
        reason = (
            f"DIVERGED: |Δ|={abs_delta:.6f} > tol {tolerance:.6f} "
            f"(claimed {claimed_val_bpb:.6f}, recomputed {recomputed_val_bpb:.6f})"
        )
    return Gate4Verdict(
        tier=tier, claimed_val_bpb=claimed_val_bpb,
        recomputed_val_bpb=recomputed_val_bpb, delta=delta, tolerance=tolerance,
        passed=passed, escalate=escalate, reason=reason,
    )


def reeval_val_bpb_on_sample(
    model: Any,
    batch: Any,
    seq_len: int,
    *,
    fraction: float,
    batch_size: int = 8,
    device: Any = None,
) -> float:
    """Re-run val_bpb on `fraction` of a sealed stream's windows (GPU path).

    Lazily imports the validator's OWN `eval.val_bpb.compute_val_bpb` so the
    recomputation is byte-for-byte the scoring code (Gate-2-style fidelity), and
    torch is only touched here. `model`/`batch` are the loaded checkpoint and a
    `SealedStreamBatch` (see eval/sealed_streams.load_stream); the caller owns
    fetching + manifest-hash verification of the bundle and stream.

    Sampling takes the first `ceil(fraction · n_windows)` windows so the subset
    is deterministic and reproducible across auditors (no RNG to agree on).
    """
    import math

    import numpy as np

    from eval.val_bpb import compute_val_bpb

    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction}")

    tokens = np.asarray(batch.tokens)
    n_windows = max(1, tokens.shape[0] // seq_len)
    keep = max(1, math.ceil(n_windows * fraction))
    sampled = tokens[: keep * seq_len]

    result = compute_val_bpb(
        model, sampled, seq_len, batch_size, device,
        bytes_per_token=batch.spec.bytes_per_token,
    )
    return float(result["val_bpb"])


__all__ = [
    "EXIT_EVAL_DIVERGE",
    "TIER_NONE",
    "TIER_CHEAP",
    "TIER_H100_FULL",
    "TIER_H200_KINGCHANGE",
    "VAL_BPB_NOISE_FLOOR",
    "DEFAULT_GATE4_TOLERANCE",
    "ESCALATION_BAND",
    "Gate4Verdict",
    "select_tier",
    "sample_fraction",
    "evaluate_val_bpb_match",
    "reeval_val_bpb_on_sample",
]
