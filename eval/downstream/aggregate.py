"""The Cross-Scale Downstream Pareto kernel — the core algorithm of the new
king-selection gate.

Given a `DownstreamReport` for challenger + king + per-task `NoiseFloorTable`,
the aggregator emits a `ParetoVerdict`:

  KING_CHANGE        — challenger regresses at no cell beyond the per-cell
                       threshold AND shows ≥1 statistically significant win.
  MEANINGFUL_FAILURE — no regression, but no sig win either. The challenger
                       sits inside the noise band on every cell; the validator
                       layer pairs this with the existing rationale + nontrivial
                       diff checks before crediting the 10% pool.
  PLAIN_FAILURE      — at least one cell regresses past its threshold.

Per-cell threshold is `max(2·pooled_stderr, eta_task)` where
`pooled_stderr = sqrt(challenger.stderr² + king.stderr²)`. The eta-task
floor catches the B1-era case where stderr is 0 (single-seed eval); when B2
ships multi-seed eval the pooled_stderr term takes over for cells with real
seed-to-seed variance.

Cell-key direction conventions:
  * keys ending in BPB_SUFFIX (":bpb"): lower is better — delta sign flipped
  * all other keys: higher is better — delta computed as challenger - king

NaN / Inf cells are a hard reject: any non-finite measurement returns
PLAIN_FAILURE with reason "nan_cells:<list>". Half-built reports do not crown.
"""
from __future__ import annotations

import math

from .types import (
    BPB_SUFFIX,
    DownstreamReport,
    NoiseFloorTable,
    ParetoOutcome,
    ParetoVerdict,
)

# A pooled_stderr of literally 0 would let any nonzero delta count as a sig
# win/loss, which is wrong when both arms saw the same single deterministic
# eval. We floor the pooled_stderr term at this value so the threshold is
# always at least the eta_task floor (or this min, whichever is larger).
_POOLED_STDERR_FLOOR = 1e-12


def _delta_for_cell(
    cell_key: str,
    challenger_acc: float,
    king_acc: float,
) -> float:
    """Direction-aware delta. Positive means challenger is better."""
    if cell_key.endswith(BPB_SUFFIX):
        return king_acc - challenger_acc  # lower is better
    return challenger_acc - king_acc      # higher is better


def _task_of(cell_key: str) -> str:
    """Extract the task name from "<task>:<scale>" or "<task>:bpb"."""
    return cell_key.split(":", 1)[0]


def _pooled_stderr(stderr_a: float, stderr_b: float) -> float:
    """Pooled stderr of the difference between two independent means.

    Mathematically: `sqrt(s_a² + s_b²)`. For B1's single-seed case both
    arms are 0 and this returns 0, so the per-cell threshold collapses to
    `eta_task` (which is the intended behaviour — eta_task is the only real
    signal until B2 lands stochastic eval).
    """
    return math.sqrt(stderr_a * stderr_a + stderr_b * stderr_b)


def aggregate_pareto(
    challenger: DownstreamReport,
    king: DownstreamReport,
    noise_floors: NoiseFloorTable,
    *,
    sig_multiplier: float = 2.0,
) -> ParetoVerdict:
    """Apply the Cross-Scale Downstream Pareto rule.

    Args:
        challenger: The submission's report.
        king: The reigning king's report (re-evaluated under the same eval
            surface — same harness version, same bundle SHA, same sealed-stream
            set).
        noise_floors: Per-task calibrated noise floors from B5 calibration.
        sig_multiplier: Threshold = max(sig_multiplier × pooled_stderr, eta_task).
            Default 2.0 (a 2σ gate; canonical in the recommendation). Exposed
            for unit tests.

    Returns:
        A `ParetoVerdict` carrying the outcome enum + the per-cell evidence
        the chain layer serialises into the `LadderScore` event.

    Failure modes (mapped to PLAIN_FAILURE):
        * Any NaN / Inf cell in challenger or king.
        * Any cell regresses past `max(2·pooled_stderr, eta_task)`.

    Edge cases:
        * Cells present in challenger but missing from king are skipped (a
          forward-compat scenario: B1 ships fewer cells than B2; aggregation
          on the intersection is correct).
        * Cells present in king but missing from challenger are also skipped
          for the same reason; the runner-layer schema checks catch real
          omissions before this kernel runs.
    """
    sig_wins: list[str] = []
    sig_losses: list[str] = []
    cell_deltas: dict[str, float] = {}
    nan_cells: list[str] = []

    for cell_key, challenger_cell in challenger.cells.items():
        if cell_key not in king.cells:
            continue  # forward-compat skip
        king_cell = king.cells[cell_key]

        if (not math.isfinite(challenger_cell.accuracy)
                or not math.isfinite(king_cell.accuracy)):
            nan_cells.append(cell_key)
            continue

        delta = _delta_for_cell(
            cell_key, challenger_cell.accuracy, king_cell.accuracy
        )
        cell_deltas[cell_key] = delta

        task = _task_of(cell_key)
        pooled = max(
            _pooled_stderr(challenger_cell.accuracy_stderr, king_cell.accuracy_stderr),
            _POOLED_STDERR_FLOOR,
        )
        threshold = max(sig_multiplier * pooled, noise_floors.eta_for(task))

        if delta > threshold:
            sig_wins.append(cell_key)
        elif delta < -threshold:
            sig_losses.append(cell_key)
        # |delta| <= threshold → within noise band, neither win nor loss

    # NaN cells short-circuit to PLAIN_FAILURE.
    if nan_cells:
        return ParetoVerdict(
            outcome=ParetoOutcome.PLAIN_FAILURE,
            sig_wins=[],
            sig_losses=[],
            cell_deltas=cell_deltas,
            reason=f"nan_cells:{','.join(nan_cells[:5])}",
        )

    # Any regression past threshold → PLAIN_FAILURE.
    if sig_losses:
        return ParetoVerdict(
            outcome=ParetoOutcome.PLAIN_FAILURE,
            sig_wins=sig_wins,
            sig_losses=sig_losses,
            cell_deltas=cell_deltas,
            reason=f"regression at: {', '.join(sorted(sig_losses)[:3])}",
        )

    # No regression. Any sig win → KING_CHANGE. Otherwise MEANINGFUL_FAILURE.
    if sig_wins:
        return ParetoVerdict(
            outcome=ParetoOutcome.KING_CHANGE,
            sig_wins=sig_wins,
            sig_losses=[],
            cell_deltas=cell_deltas,
            reason=f"{len(sig_wins)} sig win(s), no regressions",
        )

    return ParetoVerdict(
        outcome=ParetoOutcome.MEANINGFUL_FAILURE,
        sig_wins=[],
        sig_losses=[],
        cell_deltas=cell_deltas,
        reason="no regression, no significant win",
    )
