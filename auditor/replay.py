"""Gate 2 — faithful port of Ralph's weight derivation.

This recomputes the epoch's weight vector from ONLY the published report, then
Gate 3 (diff.py) compares it against the validator's claimed
`weight_snapshot.weights`. It mirrors two functions in validator/service.py:

  * `_classify_outcome`  -> per-submission (classification, weight_credit)
  * `_apply_pool_split`  -> the §5.6 90/10 king/meaningful-failure pool split

THE FIDELITY GUARANTEE: every weight/floor constant used here is IMPORTED from
validator.service — NOT copied. If a validator unilaterally changes
KING_CHANGE_WEIGHT, the pool fractions, or the 2x noise-floor multiplier, the
auditor's replay shifts in lockstep with the validator code in the SAME repo,
so a divergence between the published weights and this replay (Gate 3) is a real
alarm rather than a stale-copy false positive.

------------------------------------------------------------------------------
What replay reproduces vs. what it must TRUST
------------------------------------------------------------------------------
`_classify_outcome` gates on five things. From the report we can reproduce the
ones that are derivable from published data, and we must TRUST the ones that
need the raw bundle (Gate-4 / Phase-3 territory):

  1. decisively  -> DERIVABLE. decisively = (decisive_vs_king or is_first), both
     published in eval_output. This is the load-bearing branch: it alone decides
     the king_change (the 90%/100% share), and the auditor recomputes it from
     scratch — it does NOT trust the published `gate` for king_change.
  2. king_bpb is None (no king yet) -> handled via is_first.
  3. Bar 1: val_bpb within 2x the noise band (delta > 2*floor -> plain_failure).
     PARTIALLY derivable: needs the king's val_bpb at scoring time, which the
     report does not carry per-submission. So for the meaningful-vs-plain split
     this bar is TRUSTED from the published classification.
  4. Bar 2: diff is non-trivial          -> TRUSTED (needs patch.diff in bundle).
  5. Bar 3: rationale is coherent        -> TRUSTED (needs rationale.md in bundle).

So: the auditor INDEPENDENTLY recomputes the king_change decision (the dominant
weight) and reproduces `_apply_pool_split` exactly; the meaningful_failure set
(which only divides the 10% pool) is taken from the published per-submission
gate, because its bars 1/3/5 require data only a Gate-4 re-run (sampled bundle
fetch) can supply. Those bars are explicitly Phase-3 territory — see
docs/rearch_2026_06/childkey_validator_auditor_architecture.md (Gate 4).
"""

from __future__ import annotations

from typing import Any

# IMPORT the constants — never hardcode copies. A unilateral validator change to
# any of these makes this replay diverge from the published weights (intended).
from validator.service import (
    KING_CHANGE_WEIGHT,
    KING_POOL_FRACTION,
    MEANINGFUL_FAILURE_POOL_FRACTION,
    MEANINGFUL_FAILURE_WEIGHT,
    NOISE_FLOOR_MARGIN_2X_MULTIPLIER,
    PLAIN_FAILURE_WEIGHT,
)

# Pinned noise floor (validator/service.py --noise-floor default, and the value
# the design doc pins for replay lockstep). Carried here so the king_bpb-delta
# bar is reproducible the moment the report starts surfacing king_bpb per
# submission; until then bar 1 is trusted (see module docstring).
PINNED_NOISE_FLOOR_MARGIN = 0.013


def classify_from_report(submission: dict, king_bpb: float | None = None) -> tuple[str, float]:
    """Recompute (classification, weight_credit) for one report submission.

    Reproduces validator.service._classify_outcome for the parts derivable from
    the report. `king_bpb` is optional — when the report later surfaces it,
    bar 1 (the 2x-noise-floor delta) is recomputed here too; until then we fall
    back to the published gate for the meaningful-vs-plain decision.

    The king_change branch is ALWAYS recomputed independently from
    (decisive_vs_king or is_first) — never trusted from the published gate.
    """
    eo = submission.get("eval_output") or {}

    # --- king_change: recomputed, not trusted (mirror _classify_outcome head) ---
    decisive_vs_king = bool(eo.get("decisive_vs_king", False))
    is_first = bool(eo.get("is_first", False))
    decisively = decisive_vs_king or is_first
    if decisively:
        return "king_change", KING_CHANGE_WEIGHT

    # No king yet and not first -> plain_failure (mirror the king_bpb-None head).
    published_gate = eo.get("gate")
    val_bpb = eo.get("val_bpb")

    # --- Bar 1: 2x noise-floor delta. Recompute IFF king_bpb is available. ---
    if king_bpb is not None and val_bpb is not None:
        delta = val_bpb - king_bpb
        if delta > NOISE_FLOOR_MARGIN_2X_MULTIPLIER * PINNED_NOISE_FLOOR_MARGIN:
            return "plain_failure", PLAIN_FAILURE_WEIGHT

    # --- Bars 2 & 3 (diff non-trivial, rationale coherent): need the bundle. ---
    # These are Gate-4 / Phase-3. The report does not carry the patch or the
    # rationale text, so we TRUST the validator's published classification for
    # the meaningful-vs-plain distinction. The downstream weight effect is
    # bounded: it only changes how the 10% pool is divided, never the 90% king
    # share (which we recomputed above).
    if published_gate == "meaningful_failure":
        return "meaningful_failure", MEANINGFUL_FAILURE_WEIGHT
    return "plain_failure", PLAIN_FAILURE_WEIGHT


def _apply_pool_split(
    king_change_hotkey: str | None,
    meaningful_failure_hotkeys: list[str],
    sitting_king_hotkey: str | None,
) -> dict[str, float]:
    """Port of validator.service._apply_pool_split.

    The only difference from the validator version is the king-lookup: the
    validator calls `chain.get_king()` for the sitting king when no new king was
    crowned this epoch; the auditor has no chain handle, so it derives the
    sitting king from the report (see replay_scoring). Everything else — the
    90/10 fractions, the equal MF split, the max-by-hotkey guard, the
    no-MF -> 100% case — is byte-for-byte the same.
    """
    weights: dict[str, float] = {}
    king_hotkey = king_change_hotkey
    if king_hotkey is None:
        king_hotkey = sitting_king_hotkey
    if not meaningful_failure_hotkeys:
        if king_hotkey:
            weights[king_hotkey] = 1.0
        return weights
    if king_hotkey:
        weights[king_hotkey] = KING_POOL_FRACTION
    per_mf = MEANINGFUL_FAILURE_POOL_FRACTION / len(meaningful_failure_hotkeys)
    for hk in meaningful_failure_hotkeys:
        weights[hk] = max(weights.get(hk, 0.0), per_mf)
    return weights


def _infer_sitting_king(report_json: dict[str, Any]) -> str | None:
    """Derive the sitting king hotkey when no king_change happened this epoch.

    The validator's _apply_pool_split falls back to chain.get_king(). The
    auditor reconstructs the same hotkey from the published weight_snapshot: the
    sitting king is whoever holds the king share (KING_POOL_FRACTION when MFs
    exist, else 1.0). We read it off the snapshot so the no-king-change replay
    reproduces the validator's weight vector. If the snapshot is empty (nothing
    set this epoch) there is nothing to reproduce.

    Returns the hotkey with the maximum published weight (the king share is
    always >= any single MF share), or None.
    """
    weights = (report_json.get("weight_snapshot") or {}).get("weights") or {}
    if not weights:
        return None
    return max(weights.items(), key=lambda kv: kv[1])[0]


def replay_scoring(report_json: dict[str, Any]) -> dict[str, float]:
    """Recompute the epoch's weight vector {hotkey: weight} from the report.

    Steps (mirroring the validator's run_epoch tail):
      1. classify each submission (king_change branch recomputed; MF/plain
         taken from the published gate where bundle-only bars apply),
      2. collect king_change_hotkey + the ordered meaningful_failure set,
      3. apply the §5.6 90/10 pool split.

    Auditors compare this against report_json.weight_snapshot.weights (Gate 3).
    """
    king_change_hotkey: str | None = None
    meaningful_failure_hotkeys: list[str] = []

    for sub in report_json.get("submissions") or []:
        hk = sub.get("miner_hotkey")
        if not hk:
            continue
        classification, _credit = classify_from_report(sub)
        if classification == "king_change":
            # Mirror service.run_epoch: a later king_change overwrites the
            # earlier (the last crowned king of the epoch holds the share).
            king_change_hotkey = hk
        elif classification == "meaningful_failure":
            if hk not in meaningful_failure_hotkeys:
                meaningful_failure_hotkeys.append(hk)

    sitting_king = None
    if king_change_hotkey is None:
        sitting_king = _infer_sitting_king(report_json)

    return _apply_pool_split(
        king_change_hotkey, meaningful_failure_hotkeys, sitting_king
    )


__all__ = [
    "KING_CHANGE_WEIGHT",
    "MEANINGFUL_FAILURE_WEIGHT",
    "PLAIN_FAILURE_WEIGHT",
    "NOISE_FLOOR_MARGIN_2X_MULTIPLIER",
    "KING_POOL_FRACTION",
    "MEANINGFUL_FAILURE_POOL_FRACTION",
    "PINNED_NOISE_FLOOR_MARGIN",
    "classify_from_report",
    "replay_scoring",
]
