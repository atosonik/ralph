"""Data contract for the downstream-task eval harness (B1).

Every downstream module (scorer, core22, private_hard, calibration, runner,
aggregate) trades these dataclasses. Freezing the shape NOW unblocks B2's
validator/scoring.py rewrite + B3's ladder orchestration even before the
scorer / runner code lands — they only need the contract.

Schema versioning:
  HARNESS_VERSION is the single source of truth for "what shape of report does
  this harness emit." Every DownstreamReport carries it. When the schema
  changes (new cell key convention, new fields, semantic shift), bump it
  here and update tests/test_downstream_types.py::test_harness_version_is_pinned.

Cell-key conventions:
  Cells are keyed by either "<task>:<scale>" (the dominant convention — e.g.
  "mmlu:S3") or "<task>:bpb" for val_bpb measurements where the lower-is-better
  direction matters. The :bpb suffix is reserved; aggregate_pareto inverts
  the delta sign for these cells.

Pool conventions:
  "core22" — the 22 (or 23, see TaskSpec.pool_id) public DCLM CORE eval bundle
  "private_hard" — the 4-task hardness subset (per the v0.10 license decision
  recorded at docs/license/hardness_subset_decision.md: ARC-Challenge
  bottom-quintile + winogrande + tinyARC + tinyMMLU)
  "s2_val_bpb" — the carrier val_bpb cells at the S₂ rung
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

# Pinned schema version. Bump on any breaking change to the dataclass shapes
# or the cell-key conventions; update the pin test in lock-step.
HARNESS_VERSION = "1.0.0-b1"

# Cell-key suffix that flips the delta direction in aggregate_pareto. The
# convention is intentionally narrow — there is only ONE reserved suffix — so
# parsers don't need a registry to know whether to flip the sign.
BPB_SUFFIX = ":bpb"

# Pool identifiers as they appear in TaskSpec.pool and in calibration tables.
POOL_CORE22 = "core22"
POOL_PRIVATE_HARD = "private_hard"
POOL_S2_VAL_BPB = "s2_val_bpb"
VALID_POOLS = (POOL_CORE22, POOL_PRIVATE_HARD, POOL_S2_VAL_BPB)


class ParetoOutcome(Enum):
    """The three outcomes the Pareto kernel can emit.

    Validator layer maps these to the existing classification dispatch:
      KING_CHANGE        → crown the challenger (90% pool share)
      MEANINGFUL_FAILURE → archived for the corpus + 10% pool share (subject to
                           the existing rationale + nontrivial-diff checks)
      PLAIN_FAILURE      → 0 credit
    """

    KING_CHANGE = "king_change"
    MEANINGFUL_FAILURE = "meaningful_failure"
    PLAIN_FAILURE = "plain_failure"


@dataclass(frozen=True)
class TaskSpec:
    """Static metadata about one downstream task.

    Frozen because a TaskSpec is identity-equality across the harness — two
    references to "mmlu" must be the same object semantically. The registry
    of TaskSpec objects lives in core22.py / private_hard.py.
    """

    name: str
    mode: str             # "mc" | "schema" | "lm"
    random_baseline: float
    pool: str             # one of VALID_POOLS

    def __post_init__(self) -> None:
        if self.mode not in ("mc", "schema", "lm"):
            raise ValueError(f"TaskSpec.mode must be one of mc/schema/lm; got {self.mode!r}")
        if self.pool not in VALID_POOLS:
            raise ValueError(
                f"TaskSpec.pool must be one of {VALID_POOLS}; got {self.pool!r}"
            )
        if not 0.0 <= self.random_baseline <= 1.0:
            raise ValueError(
                f"TaskSpec.random_baseline must be in [0,1]; got {self.random_baseline}"
            )


@dataclass
class CellResult:
    """One (task, scale-or-stream) accuracy measurement on one checkpoint.

    `accuracy` is in [0, 1] for downstream tasks; for ":bpb" cells (cell-key
    ending in BPB_SUFFIX) it is the val_bpb value and lower-is-better. The
    aggregator inverts the delta direction for those cells.

    `accuracy_stderr` is the seed-pooled stderr-of-the-mean for the
    measurement. Today (single seed at B1) it is 0; B2 wires multi-seed and
    starts producing nonzero values automatically.
    """

    task: str
    accuracy: float
    accuracy_stderr: float = 0.0
    n_examples: int = 0
    seed: int = 0


@dataclass
class DownstreamReport:
    """Output of one downstream-eval run on one checkpoint.

    Cells dict is keyed by "<task>:<scale>" (e.g. "mmlu:S3") or "<task>:bpb"
    for val_bpb carrier cells (the BPB_SUFFIX convention).
    """

    harness_version: str
    bundle_sha256: str
    seed: int
    total_examples: int
    wall_clock_s: float
    cells: dict[str, CellResult] = field(default_factory=dict)


@dataclass
class NoiseFloorTable:
    """Per-task calibrated noise floors used as the per-cell threshold floor.

    Committed on chain via the forthcoming CalibrationCommit event (chain
    schema bump is a B3 deliverable, not B1's). B1's responsibility is the
    in-process dataclass + the calibration script that produces it.
    """

    floors: dict[str, float] = field(default_factory=dict)  # task name → eta_task
    harness_version: str = HARNESS_VERSION
    recipe_sha: str = ""
    n_baselines: int = 0

    def eta_for(self, task: str) -> float:
        """Per-task noise floor with a safe default of 0 (no floor)."""
        return float(self.floors.get(task, 0.0))


@dataclass
class ParetoVerdict:
    """The Pareto kernel's emit: the outcome + per-cell evidence.

    `cell_deltas` is "challenger improvement" per cell — positive means the
    challenger is better in the cell's natural direction (higher acc, or
    lower bpb for :bpb cells).
    """

    outcome: ParetoOutcome
    sig_wins: list[str] = field(default_factory=list)
    sig_losses: list[str] = field(default_factory=list)
    cell_deltas: dict[str, float] = field(default_factory=dict)
    reason: str = ""
