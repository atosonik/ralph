"""Gate 3 — diff replayed weights vs the validator's claimed weight_snapshot.

Any per-hotkey divergence beyond TOLERANCE means the published weights are not
what the published raw data scores to -> the auditor exits 2 (math diverge) and
logs claimed / replayed / Δ for each offending hotkey. Mirrors
greencompute-audit/audit/diff.py.
"""

from __future__ import annotations

# Cross-run floating-point noise is far below this; a real discrepancy is a
# changed weight, not rounding. 1e-4 per the design doc.
TOLERANCE = 1e-4


def compare_weights(
    claimed: dict[str, float],
    replayed: dict[str, float],
    tolerance: float = TOLERANCE,
) -> dict[str, dict[str, float]]:
    """Return {hotkey: {claimed, replayed, delta}} for every hotkey whose
    claimed and replayed weights diverge by more than `tolerance`. A hotkey
    present on only one side counts as 0.0 on the other (a dropped/added miner
    is itself a divergence)."""
    discrepancies: dict[str, dict[str, float]] = {}
    for hk in sorted(set(claimed) | set(replayed)):
        c = float(claimed.get(hk, 0.0))
        r = float(replayed.get(hk, 0.0))
        delta = abs(c - r)
        if delta > tolerance:
            discrepancies[hk] = {"claimed": c, "replayed": r, "delta": delta}
    return discrepancies


__all__ = ["TOLERANCE", "compare_weights"]
