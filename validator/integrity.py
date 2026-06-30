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


# --- Compute-plausibility (anti compute-gaming) -------------------------------
#
# `wall_clock_s` is MINER-DECLARED (and not in bundle_hash), so a miner can
# under-claim it to look efficient and win the compute-weighted crown — train a
# real model over ~30 H100h but report ~2h. The give-away is physics: the implied
# model-FLOP rate (~6*N*tok/s) cannot exceed the GPU's bf16 matmul peak, and real
# sustained TRAINING MFU is ~30-55%. An implied MFU above the ceiling means the
# wall_clock_s (hence the compute cost) is fabricated.
MAX_PLAUSIBLE_MFU = 0.7
# bf16 dense matmul peak (TFLOP/s) per GPU family — the hard physical ceiling.
_GPU_BF16_PEAK_TFLOPS = {
    "a100": 312.0, "a800": 312.0, "l4": 121.0, "l40": 362.0, "4090": 165.0,
    "h100": 989.0, "h200": 989.0, "h800": 989.0,
    "b100": 1800.0, "b200": 2250.0, "gb200": 2500.0,
}
# Unknown GPU -> assume the fastest known part, so we NEVER false-reject; the gate
# only fires when even the fastest plausible GPU cannot explain the throughput.
_DEFAULT_PEAK_TFLOPS = 2500.0


def _gpu_bf16_peak_flops(gpu_name: str | None) -> float:
    g = (gpu_name or "").lower()
    for key, tflops in _GPU_BF16_PEAK_TFLOPS.items():
        if key in g:
            return tflops * 1e12
    return _DEFAULT_PEAK_TFLOPS * 1e12


def check_compute_plausibility(
    final_state: dict,
    calibration: dict | None = None,
    *,
    max_mfu: float = MAX_PLAUSIBLE_MFU,
) -> tuple[bool, str]:
    """Reject a bundle whose declared training throughput is physically impossible.

    tokens_seen / wall_clock_s implies ~6*N FLOPs/token; over the declared GPU's
    bf16 peak that is the achieved MFU. An implied MFU > `max_mfu` means the
    wall_clock_s (and the efficiency-gate compute cost it drives) is fabricated.
    Best-effort: a missing/incomplete training_summary is skipped (deferred to the
    other gates), not rejected. Returns (ok, reason); ok=False -> reject.
    """
    fs = final_state or {}
    try:
        tokens = float(fs.get("tokens_seen", 0) or 0)
        wall = float(fs.get("wall_clock_s", 0) or 0)
        n = float(fs.get("n_params", 0) or 0)
    except (TypeError, ValueError):
        return True, "compute-plausibility: non-numeric training_summary (skipped)"
    if tokens <= 0 or wall <= 0 or n <= 0:
        return True, "compute-plausibility: incomplete training_summary (skipped)"
    gpu = (calibration or {}).get("gpu_name") or fs.get("gpu_name") or fs.get("device") or ""
    flops_per_s = 6.0 * n * tokens / wall  # 6N FLOPs/token (fwd+bwd)
    mfu = flops_per_s / _gpu_bf16_peak_flops(gpu)
    if mfu > max_mfu:
        return False, (
            f"fabricated compute: {tokens / wall:,.0f} tok/s for a {n / 1e6:.0f}M model on "
            f"'{gpu or 'unknown'}' => {mfu * 100:.0f}% MFU (> {max_mfu * 100:.0f}% physical max); "
            f"wall_clock_s={wall:.0f}s for {tokens:,.0f} tokens is not achievable"
        )
    return True, f"compute plausible: {tokens / wall:,.0f} tok/s, {mfu * 100:.0f}% MFU"


def _added_config_jsons(patch_text: str) -> list[dict]:
    """Parse every NEW/whole configs/*.json the patch adds (best-effort)."""
    import json

    out: list[dict] = []
    path: str | None = None
    buf: list[str] = []

    def _flush() -> None:
        if path and path.endswith(".json") and "config" in path and buf:
            try:
                out.append(json.loads("\n".join(buf)))
            except Exception:  # noqa: BLE001 — partial/edited config, skip
                pass

    for ln in (patch_text or "").splitlines():
        if ln.startswith("+++ b/"):
            _flush()
            path, buf = ln[6:], []
        elif ln.startswith("+") and not ln.startswith("+++"):
            buf.append(ln[1:])
    _flush()
    return out


def check_recipe_config_matches_proof(patch_text: str, final_state: dict) -> tuple[bool, str]:
    """A submitted training config (configs/*.json) must match what the proof ran.

    If the patch declares `total_steps` that differs from the steps the proof
    recorded, the crowned checkpoint was NOT produced by the declared recipe (the
    submitted config is a decoy). Best-effort: skipped when no config is added or no
    proof step count exists. Returns (ok, reason); ok=False -> reject.
    """
    fs = final_state or {}
    proof_steps = fs.get("steps")
    if proof_steps is None:
        proof_steps = (fs.get("config") or {}).get("total_steps")
    if proof_steps is None:
        return True, "config-match: no proof step count (skipped)"
    for cfg in _added_config_jsons(patch_text):
        declared = cfg.get("total_steps")
        if declared is None:
            continue
        try:
            if int(declared) != int(proof_steps):
                return False, (
                    f"declared recipe mismatch: submitted config total_steps={declared} but the "
                    f"proof ran {proof_steps} steps — crowned checkpoint not from the submitted recipe"
                )
        except (TypeError, ValueError):
            continue
    return True, "config matches proof"
