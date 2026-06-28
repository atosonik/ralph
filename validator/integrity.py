"""Checkpoint-trainedness / log-consistency guard.

Motivation (real incident): a submission whose checkpoint was random-INITIALISED
(every weight at init std) shipped a `training_log.jsonl` claiming a full run and
got crowned — because op4 was scoring against random tokens at the time and the
re-train audit that would have caught the log/checkpoint mismatch never ran. The
checkpoint measured ~ln(vocab) nats/token (uniform output) yet the log claimed a
final loss of ~3 nats.

This is a CHEAP guard (no extra GPU work — it consumes the val_bpb op4 already
computes plus the miner's own declared `final_loss`):

  (a) UNTRAINED: the held-out loss sits within `random_fraction` of the random
      baseline ln(vocab_size) -> the checkpoint carries ~no learned signal.
  (b) LOG/CHECKPOINT MISMATCH: a declared training `final_loss` exists but the
      held-out loss is implausibly worse than it -> the scored checkpoint did
      not come from the declared training run.

Thresholds are deliberately generous so an honest model (held-out a bit worse
than training, never near random) is NEVER rejected; only garbage / fabricated
checkpoints trip it. Returns (ok, reason); ok=False means reject as fraud/broken.
"""
from __future__ import annotations

import math

# Reject if held-out loss >= this fraction of the random baseline ln(vocab).
# A real ~254M model sits at ~3-4.5 nats/token; random is ~10.8 for vocab 50257.
# 0.80 -> reject above ~8.6 nats, leaving a wide safety margin under legit models.
DEFAULT_RANDOM_FRACTION = 0.80

# Reject if held-out loss > claimed_final_loss * FACTOR + MARGIN. Generous: a
# normal train->held-out gap is well under 1.5x; this only fires on gross
# mismatch (e.g. claimed 3.0, measured 9.0).
DEFAULT_MISMATCH_FACTOR = 2.5
DEFAULT_MISMATCH_MARGIN = 1.0


def nats_per_token_from_bpb(val_bpb: float, bytes_per_token: float) -> float:
    """Invert val_bpb = nats / (ln2 * bytes_per_token)."""
    return float(val_bpb) * math.log(2) * float(bytes_per_token)


def check_checkpoint_trained(
    measured_nats_per_token: float,
    vocab_size: int,
    *,
    claimed_final_loss: float | None = None,
    random_fraction: float = DEFAULT_RANDOM_FRACTION,
    mismatch_factor: float = DEFAULT_MISMATCH_FACTOR,
    mismatch_margin: float = DEFAULT_MISMATCH_MARGIN,
) -> tuple[bool, str]:
    """Cheap guard against untrained / log-mismatched checkpoints.

    Args:
      measured_nats_per_token: held-out cross-entropy (nats/token) the validator
        actually measured for this checkpoint (e.g. from op4's val_bpb via
        `nats_per_token_from_bpb`).
      vocab_size: the checkpoint's vocab — sets the random baseline ln(vocab).
      claimed_final_loss: the miner's declared training `final_loss` (nats/token)
        from final_state.json, if present. Enables the log-mismatch check.

    Returns (ok, reason). ok=False -> reject.
    """
    if not (isinstance(measured_nats_per_token, (int, float)) and math.isfinite(measured_nats_per_token)):
        return False, f"non-finite measured loss: {measured_nats_per_token!r}"
    if not (isinstance(vocab_size, int) and vocab_size > 1):
        return False, f"invalid vocab_size: {vocab_size!r}"

    random_baseline = math.log(vocab_size)  # nats/token of a uniform predictor
    if measured_nats_per_token >= random_fraction * random_baseline:
        return False, (
            f"untrained checkpoint: held-out {measured_nats_per_token:.2f} nats/token "
            f">= {random_fraction:.0%} of random baseline {random_baseline:.2f} "
            f"(vocab {vocab_size}) — weights appear at initialization"
        )

    if claimed_final_loss is not None and isinstance(claimed_final_loss, (int, float)) and claimed_final_loss > 0:
        bound = claimed_final_loss * mismatch_factor + mismatch_margin
        if measured_nats_per_token > bound:
            return False, (
                f"log/checkpoint mismatch: held-out {measured_nats_per_token:.2f} nats/token "
                f">> declared training final_loss {claimed_final_loss:.2f} "
                f"(plausible bound {bound:.2f}) — scored checkpoint not from the declared run"
            )

    return True, "ok"
