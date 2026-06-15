"""v0.11-lite validator ladder — acceptance preflight + S1/S2/S3 eval driver.

This module is the validator's whole-submission orchestrator. Each child
PR goes through two phases here:

  Phase 1 — GPU-free preflight (`accept_submission`): schema, branch_id,
  vocab pin, parent_csdp_cache lineage verification. Cheap. Rejects emit
  a `submission_received` chain event with the verdict.

  Phase 2 — S1/S2/S3 CSDP eval (`run_ladder_eval`): only invoked on an
  accepted submission. Dispatches `eval/downstream/runner_cli.py` once
  per rung (S1, S2, S3) via the subprocess wrapper. Merges per-rung
  `DownstreamReport`s into a single combined report and produces a
  `HiddenEvalResult` with `downstream` populated. Honors a `mode='legacy'`
  flag that skips the v0.11 downstream path entirely so the
  `to_legacy_dict()` byte-equivalence under `RALPH_KING_RULE=legacy` is
  preserved bit-for-bit.

What this module ships:

  * `Submission` + `read_submission` + `accept_submission` +
    `LadderAcceptResult` — C1-LITE preflight (unchanged).
  * `LadderRungSpec` + `LadderEvalConfig` — frozen configs for the eval
    driver. `LadderEvalConfig.standard_s1_s2_s3()` returns the canonical
    3-rung config locked in v0.10 (S1 d=256/L=4, S2 d=512/L=12, S3
    d=768/L=12 ~124M params).
  * `LadderEvalResult` — combined output: per-rung reports + merged
    report + the assembled `HiddenEvalResult`.
  * `merge_rung_reports(reports_by_scale)` — combine N rung
    `DownstreamReport`s into a single one (concatenates cells, sums
    `total_examples`, picks max `wall_clock_s`).
  * `run_ladder_eval(...)` — main orchestrator. Calls
    `run_eval_in_subprocess` once per rung, merges, and assembles the
    HiddenEvalResult. Supports `mode='v0.11'` (default) and
    `mode='legacy'` (skips downstream eval; returns a HiddenEvalResult
    with `downstream=None` that round-trips byte-equivalent via
    `to_legacy_dict`).

What this module does NOT ship (out of scope for v0.11-lite):

  * Sealed-shard accumulator integration (`--sealed-shard` CLI arg) —
    v0.11-full only.
  * Multi-branch merge window orchestration — v0.11-full only.
  * Audit / 10% sample rotation — separate `validator/audit.py` track.

Reference: docs/rearch_2026_06/00_v0_11_master.md §4.1 Validator,
§6.1 phase C2-LITE.
"""
from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from chain_layer.interface import ChainInterface
from eval.downstream.runner import RALPH_VOCAB_SIZE, EvalConfig
from eval.downstream.runner_subprocess import (
    EvalSubprocessError,
    run_eval_in_subprocess,
)
from eval.downstream.types import (
    HARNESS_VERSION,
    CellResult,
    DownstreamReport,
)
from eval.hidden_eval import HiddenEvalResult

from .lineage import (
    DEFAULT_CACHE_AGE_THRESHOLD_DAYS,
    ParentCsdpCache,
    is_valid_attestation_hash,
    verify_parent_lineage,
)

# Acceptance reasons (stable for chain event consumers + audit log).
ACCEPT_OK = "accept_ok"
REJECT_BAD_FORMAT = "bad_submission_format"
REJECT_VOCAB_MISMATCH = "vocab_mismatch"
REJECT_BAD_BRANCH_ID = "bad_branch_id"

# Supported submission schema versions. v0.11-lite accepts only "v0.11";
# v0.11-full will extend with required sealed-shard fields under a v0.11.1
# bump.
SUPPORTED_SCHEMA_VERSIONS = frozenset({"v0.11"})

# Branch id format: "main" or "branch-<int>" or "open_new_branch_<slug>".
# Slug regex is enforced separately in C5-PASS-FULL; v0.11-lite checks the
# prefix only.
_BRANCH_PREFIXES = ("main", "branch-", "open_new_branch_")


# ----------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Submission:
    """Validator's in-process view of a miner's submission bundle.

    Constructed by `read_submission` from a bundle directory. The fields
    are deliberately the v0.11-lite minimum; v0.11-full will extend with
    sealed-shard, license_spdx, dco_signed_off_by, novelty_score, etc.
    """

    schema_version: str
    parent_king_attestation_hash: Optional[str]
    branch_id: str
    bundle_hash: str
    miner_hotkey: str
    vocab_size: int
    parent_csdp_cache: Optional[ParentCsdpCache] = None

    def to_dict(self) -> dict:
        d = {
            "schema_version": self.schema_version,
            "parent_king_attestation_hash": self.parent_king_attestation_hash,
            "branch_id": self.branch_id,
            "bundle_hash": self.bundle_hash,
            "miner_hotkey": self.miner_hotkey,
            "vocab_size": self.vocab_size,
        }
        if self.parent_csdp_cache is not None:
            d["parent_csdp_cache"] = self.parent_csdp_cache.to_dict()
        return d


@dataclass(frozen=True)
class LadderAcceptResult:
    """Verdict of `accept_submission`.

    `accepted=True` means the submission cleared every GPU-free preflight
    check and is eligible for the S1/S2/S3 CSDP eval (C2-LITE). `reason`
    is the stable underscore-snake-case tag the validator emits to the
    chain event log so downstream auditors can group rejections by cause.
    """

    accepted: bool
    reason: str
    submission_dict: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------
# Submission loader
# ----------------------------------------------------------------------------


def read_submission(submission_dir: Path) -> Submission:
    """Load `submission.json` (+ optional `parent_csdp_cache.json`) from
    a miner bundle directory.

    Expected layout:
      <submission_dir>/
        submission.json           — required
        parent_csdp_cache.json    — required iff parent_king_attestation_hash is set
        recipe/                   — modified recipe tree (touched here only for
                                    docstring; the patched-eval driver consumes it)
        proof/                    — attestation quote + training_log etc.

    Raises:
      FileNotFoundError if submission.json is missing.
      ValueError on JSON parse error or missing required fields.
    """
    submission_dir = Path(submission_dir)
    sub_path = submission_dir / "submission.json"
    if not sub_path.exists():
        raise FileNotFoundError(f"submission.json not found in {submission_dir}")
    try:
        d = json.loads(sub_path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(f"submission.json invalid JSON: {e}") from e

    required = {"schema_version", "branch_id", "bundle_hash", "miner_hotkey", "vocab_size"}
    missing = required - d.keys()
    if missing:
        raise ValueError(
            f"submission.json missing required fields: {sorted(missing)}"
        )

    parent_hash = d.get("parent_king_attestation_hash")
    cache: Optional[ParentCsdpCache] = None
    cache_path = submission_dir / "parent_csdp_cache.json"
    if cache_path.exists():
        try:
            cache_dict = json.loads(cache_path.read_text())
        except json.JSONDecodeError as e:
            raise ValueError(f"parent_csdp_cache.json invalid JSON: {e}") from e
        try:
            cache = ParentCsdpCache.from_dict(cache_dict)
        except (KeyError, ValueError) as e:
            raise ValueError(
                f"parent_csdp_cache.json missing/invalid fields: {e}"
            ) from e

    return Submission(
        schema_version=str(d["schema_version"]),
        parent_king_attestation_hash=parent_hash if parent_hash else None,
        branch_id=str(d["branch_id"]),
        bundle_hash=str(d["bundle_hash"]),
        miner_hotkey=str(d["miner_hotkey"]),
        vocab_size=int(d["vocab_size"]),
        parent_csdp_cache=cache,
    )


# ----------------------------------------------------------------------------
# Acceptance preflight
# ----------------------------------------------------------------------------


def _validate_branch_id(branch_id: str) -> bool:
    """v0.11-lite branch-id check: must be one of the known prefixes.
    Full slug validation lands in C5-PASS-FULL."""
    if not branch_id:
        return False
    if branch_id == "main":
        return True
    return any(branch_id.startswith(p) for p in _BRANCH_PREFIXES if p != "main")


def accept_submission(
    submission: Submission,
    chain: ChainInterface,
    *,
    now_iso: Optional[str] = None,
    cache_age_threshold_days: int = DEFAULT_CACHE_AGE_THRESHOLD_DAYS,
    emit_chain_event: bool = True,
) -> LadderAcceptResult:
    """Run the v0.11-lite GPU-free preflight on a submission.

    Checks (in order, fail-fast):
      1. schema_version ∈ SUPPORTED_SCHEMA_VERSIONS
      2. branch_id well-formed
      3. vocab_size == RALPH_VOCAB_SIZE (50257)
      4. parent_king_attestation_hash format (if non-empty)
      5. verify_parent_lineage (signature + age + parent existence)

    Emits a `submission_received` chain event on every call (including
    rejections), with the verdict for downstream audit / reward
    accounting.

    Returns:
      LadderAcceptResult with `accepted` + a stable underscore-snake-case
      `reason` tag.
    """
    if now_iso is None:
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    result = _run_preflight(
        submission,
        chain,
        now_iso=now_iso,
        cache_age_threshold_days=cache_age_threshold_days,
    )

    if emit_chain_event:
        chain.append_event({
            "type": "submission_received",
            "schema_version": submission.schema_version,
            "branch_id": submission.branch_id,
            "bundle_hash": submission.bundle_hash,
            "miner_hotkey": submission.miner_hotkey,
            "parent_king_attestation_hash": submission.parent_king_attestation_hash,
            "accepted": result.accepted,
            "reason": result.reason,
            "now_iso": now_iso,
        })

    return result


def _run_preflight(
    submission: Submission,
    chain: ChainInterface,
    *,
    now_iso: str,
    cache_age_threshold_days: int,
) -> LadderAcceptResult:
    """Inner preflight — separated so the event emission can wrap it."""
    if submission.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        return LadderAcceptResult(
            accepted=False,
            reason=REJECT_BAD_FORMAT + f":schema_version_{submission.schema_version}",
            submission_dict=submission.to_dict(),
        )

    if not _validate_branch_id(submission.branch_id):
        return LadderAcceptResult(
            accepted=False,
            reason=REJECT_BAD_BRANCH_ID,
            submission_dict=submission.to_dict(),
        )

    if submission.vocab_size != RALPH_VOCAB_SIZE:
        return LadderAcceptResult(
            accepted=False,
            reason=REJECT_VOCAB_MISMATCH,
            submission_dict=submission.to_dict(),
        )

    parent_hash = submission.parent_king_attestation_hash
    if parent_hash and not is_valid_attestation_hash(parent_hash):
        return LadderAcceptResult(
            accepted=False,
            reason="parent_unverifiable:bad_parent_hash_format",
            submission_dict=submission.to_dict(),
        )

    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=submission.parent_csdp_cache,
        chain=chain,
        now_iso=now_iso,
        cache_age_threshold_days=cache_age_threshold_days,
    )
    if not ok:
        return LadderAcceptResult(
            accepted=False,
            reason=f"parent_unverifiable:{reason}",
            submission_dict=submission.to_dict(),
        )

    return LadderAcceptResult(
        accepted=True,
        reason=ACCEPT_OK,
        submission_dict=submission.to_dict(),
    )


# ============================================================================
# C2-LITE: S1/S2/S3 eval driver
# ============================================================================


# Canonical rung dimensions locked in v0.10. Bumping these requires a
# coordinated noise-floor recalibration; do not edit casually.
_STANDARD_RUNGS_DEFAULT: tuple[tuple[str, int, int], ...] = (
    ("S1", 256, 4),    # ~1.5M params (tunable depending on tokenizer)
    ("S2", 512, 12),   # ~18M params
    ("S3", 768, 12),   # ~124M params (the canonical S3, NanoGPT-Speedrun scale)
)

# Eval mode. v0.11 runs CSDP at S1+S2+S3 and populates HiddenEvalResult.downstream.
# Legacy mode skips the downstream eval entirely so HiddenEvalResult round-trips
# byte-equivalent to pre-v0.11 chain payloads (see HiddenEvalResult.to_legacy_dict).
EVAL_MODE_V011 = "v0.11"
EVAL_MODE_LEGACY = "legacy"
VALID_EVAL_MODES = frozenset({EVAL_MODE_V011, EVAL_MODE_LEGACY})


@dataclass(frozen=True)
class LadderRungSpec:
    """Per-rung configuration consumed by `run_ladder_eval`.

    `scale_label` is the cell-key suffix used in the per-rung
    `DownstreamReport` (e.g. `"S3"` produces cell keys like
    `"arc_easy:S3"`). The other fields are advisory — the actual
    training cost lives in the miner's recipe, not here — but the
    validator pins them so audit reproducibility is unambiguous.
    """

    scale_label: str
    dim: int
    n_layers: int
    n_examples_per_task: int = 0  # 0 = use all

    def __post_init__(self) -> None:
        if not self.scale_label:
            raise ValueError("LadderRungSpec.scale_label must be non-empty")
        if self.dim <= 0 or self.n_layers <= 0:
            raise ValueError(
                f"LadderRungSpec dim/n_layers must be > 0; "
                f"got dim={self.dim}, n_layers={self.n_layers}"
            )
        if self.n_examples_per_task < 0:
            raise ValueError(
                f"n_examples_per_task must be >= 0; got {self.n_examples_per_task}"
            )


@dataclass(frozen=True)
class LadderEvalConfig:
    """Full configuration for `run_ladder_eval`.

    Fields:
      * `rungs` — ordered tuple of `LadderRungSpec`. Standard is S1/S2/S3.
      * `tasks` — tuple of task names to evaluate (passed to `EvalConfig.tasks`
        for each rung).
      * `bundle_dir` — DCLM bundle root (contains per-task JSONLs + the
        `private_hard/` subdir for hardness tasks).
      * `bundle_sha256` — pinned bundle SHA committed on chain.
      * `hardness_index_path` — optional path to a HardnessIndex JSONL.
      * `seed` — propagates to every per-rung EvalConfig.
    """

    rungs: tuple[LadderRungSpec, ...]
    tasks: tuple[str, ...]
    bundle_dir: Path
    bundle_sha256: str
    hardness_index_path: Optional[Path] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if not self.rungs:
            raise ValueError("LadderEvalConfig.rungs must be non-empty")
        if not self.tasks:
            raise ValueError("LadderEvalConfig.tasks must be non-empty")
        labels = [r.scale_label for r in self.rungs]
        if len(set(labels)) != len(labels):
            raise ValueError(
                f"LadderEvalConfig.rungs has duplicate scale_label: {labels}"
            )

    @classmethod
    def standard_s1_s2_s3(
        cls,
        *,
        tasks: tuple[str, ...],
        bundle_dir: Path,
        bundle_sha256: str,
        hardness_index_path: Optional[Path] = None,
        seed: int = 0,
        n_examples_per_task: int = 0,
    ) -> LadderEvalConfig:
        """Return the canonical v0.10 ladder: S1 d=256/L=4 + S2 d=512/L=12 +
        S3 d=768/L=12 (~124M)."""
        rungs = tuple(
            LadderRungSpec(
                scale_label=label,
                dim=dim,
                n_layers=n_layers,
                n_examples_per_task=n_examples_per_task,
            )
            for label, dim, n_layers in _STANDARD_RUNGS_DEFAULT
        )
        return cls(
            rungs=rungs,
            tasks=tasks,
            bundle_dir=bundle_dir,
            bundle_sha256=bundle_sha256,
            hardness_index_path=hardness_index_path,
            seed=seed,
        )


@dataclass
class LadderEvalResult:
    """Output of `run_ladder_eval`.

    `per_rung_reports` is keyed by `scale_label` (e.g. `"S1"`).
    `combined_report` merges all rungs into one DownstreamReport with
    cells from every rung side-by-side (cell keys retain their
    `task:scale` suffix so no collisions).
    `hidden_eval` is the v0.11 HiddenEvalResult with `downstream`
    populated (or `None` in legacy mode).
    """

    per_rung_reports: dict[str, DownstreamReport]
    combined_report: DownstreamReport
    hidden_eval: HiddenEvalResult
    mode: str = EVAL_MODE_V011


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def merge_rung_reports(
    reports_by_scale: dict[str, DownstreamReport],
) -> DownstreamReport:
    """Combine per-rung `DownstreamReport`s into one report.

    The cell-key suffix conventions (`task:scale`) make the cells from
    different rungs non-colliding by construction, so the merged
    `cells` dict is the union of all inputs. `total_examples` sums
    across rungs; `wall_clock_s` takes the max (the validator's
    wall-clock is bounded by the slowest rung, not the sum).
    `seed` and `bundle_sha256` are taken from the first report (they're
    required to match across rungs; the caller passes the same seed
    to each per-rung EvalConfig).
    Empty input raises ValueError.
    """
    if not reports_by_scale:
        raise ValueError("merge_rung_reports requires at least one report")
    reports = list(reports_by_scale.values())
    first = reports[0]
    combined_cells: dict[str, CellResult] = {}
    total_examples = 0
    wall_clock = 0.0
    for r in reports:
        if r.bundle_sha256 != first.bundle_sha256:
            raise ValueError(
                f"bundle_sha256 mismatch across rungs: "
                f"{r.bundle_sha256!r} vs {first.bundle_sha256!r}"
            )
        if r.seed != first.seed:
            raise ValueError(
                f"seed mismatch across rungs: {r.seed} vs {first.seed}"
            )
        # Cell keys carry the scale suffix; collisions across rungs
        # would be a programmer bug in the caller.
        for key, cell in r.cells.items():
            if key in combined_cells:
                raise ValueError(
                    f"duplicate cell key {key!r} across rungs"
                )
            combined_cells[key] = cell
        total_examples += r.total_examples
        if r.wall_clock_s > wall_clock:
            wall_clock = r.wall_clock_s
    return DownstreamReport(
        harness_version=first.harness_version,
        bundle_sha256=first.bundle_sha256,
        seed=first.seed,
        total_examples=total_examples,
        wall_clock_s=wall_clock,
        cells=combined_cells,
    )


def _hidden_eval_from_report(
    report: Optional[DownstreamReport],
    *,
    legacy_val_bpb: float = 0.0,
    legacy_benchmark_accuracy: float = 0.0,
    legacy_tokens_evaluated: int = 0,
    legacy_benchmark_examples: int = 0,
    legacy_eval_set_hash: str = "",
) -> HiddenEvalResult:
    """Wrap a DownstreamReport into a HiddenEvalResult.

    Legacy fields (val_bpb, benchmark_accuracy, ...) come from the
    caller's pre-existing legacy eval pass when running in v0.11 mode.
    In legacy mode, `report=None` and the caller passes only the legacy
    fields — the resulting HiddenEvalResult is byte-equivalent to the
    pre-v0.11 shape via `to_legacy_dict`.
    """
    return HiddenEvalResult(
        val_bpb=legacy_val_bpb,
        benchmark_accuracy=legacy_benchmark_accuracy,
        tokens_evaluated=legacy_tokens_evaluated,
        benchmark_examples=legacy_benchmark_examples,
        eval_set_hash=legacy_eval_set_hash,
        downstream=report,
    )


# ----------------------------------------------------------------------------
# run_ladder_eval — the main eval driver
# ----------------------------------------------------------------------------


def run_ladder_eval(
    submission: Submission,
    config: LadderEvalConfig,
    *,
    checkpoint_path: Path,
    ralph_root: Path,
    patch_path: Optional[Path] = None,
    mode: str = EVAL_MODE_V011,
    command_prefix: Optional[Sequence[str]] = None,
    timeout_s_per_rung: float = 600.0,
    legacy_val_bpb: float = 0.0,
    legacy_benchmark_accuracy: float = 0.0,
    legacy_tokens_evaluated: int = 0,
    legacy_benchmark_examples: int = 0,
    legacy_eval_set_hash: str = "",
) -> LadderEvalResult:
    """Run the S1/S2/S3 CSDP eval for an accepted submission.

    Args:
      submission: result of an earlier `accept_submission` call. Caller
        is responsible for ensuring `accept_submission(...).accepted is
        True` before invoking this function.
      config: `LadderEvalConfig` (typically from
        `LadderEvalConfig.standard_s1_s2_s3(...)`).
      checkpoint_path: miner-submitted checkpoint to evaluate.
      ralph_root: ralph repo root for `from model import ...` resolution
        + structural-patch base.
      patch_path: optional structural patch (passed through to runner_cli).
      mode: `EVAL_MODE_V011` (default) or `EVAL_MODE_LEGACY`.
        Legacy mode SKIPS the per-rung downstream eval entirely and
        returns a HiddenEvalResult with `downstream=None` for
        byte-equivalent serialization to v0.10.
      command_prefix: forwarded to `run_eval_in_subprocess` (tests pass
        a custom prefix pointing at the synthetic CLI).
      timeout_s_per_rung: per-rung subprocess timeout.
      legacy_*: fields used by the caller's pre-existing legacy eval
        pass; in legacy mode they populate the HiddenEvalResult as-is.

    Returns:
      `LadderEvalResult`. In legacy mode `per_rung_reports == {}` and
      `combined_report` is an empty placeholder.

    Raises:
      ValueError on unknown `mode`.
      EvalSubprocessError propagated from a failing rung (caller can
      catch + emit a chain event).
    """
    if mode not in VALID_EVAL_MODES:
        raise ValueError(
            f"unknown mode {mode!r}; expected one of {sorted(VALID_EVAL_MODES)}"
        )

    if mode == EVAL_MODE_LEGACY:
        # Skip downstream eval entirely. HiddenEvalResult.downstream stays
        # None so to_legacy_dict produces v0.10 byte-equivalent output.
        empty_combined = DownstreamReport(
            harness_version=HARNESS_VERSION,
            bundle_sha256=config.bundle_sha256,
            seed=config.seed,
            total_examples=0,
            wall_clock_s=0.0,
            cells={},
        )
        hidden = _hidden_eval_from_report(
            None,
            legacy_val_bpb=legacy_val_bpb,
            legacy_benchmark_accuracy=legacy_benchmark_accuracy,
            legacy_tokens_evaluated=legacy_tokens_evaluated,
            legacy_benchmark_examples=legacy_benchmark_examples,
            legacy_eval_set_hash=legacy_eval_set_hash,
        )
        return LadderEvalResult(
            per_rung_reports={},
            combined_report=empty_combined,
            hidden_eval=hidden,
            mode=mode,
        )

    # v0.11 mode: run each rung in turn.
    per_rung: dict[str, DownstreamReport] = {}
    for rung in config.rungs:
        eval_config = EvalConfig(
            tasks=config.tasks,
            n_examples_per_task=rung.n_examples_per_task,
            seed=config.seed,
            scale_label=rung.scale_label,
        )
        report = run_eval_in_subprocess(
            checkpoint_path=checkpoint_path,
            config=eval_config,
            bundle_sha256=config.bundle_sha256,
            bundle_dir=config.bundle_dir,
            vocab_size=RALPH_VOCAB_SIZE,
            hardness_index_path=config.hardness_index_path,
            patch_path=patch_path,
            ralph_root=ralph_root,
            timeout_s=timeout_s_per_rung,
            command_prefix=command_prefix,
        )
        per_rung[rung.scale_label] = report

    combined = merge_rung_reports(per_rung)
    hidden = _hidden_eval_from_report(
        combined,
        legacy_val_bpb=legacy_val_bpb,
        legacy_benchmark_accuracy=legacy_benchmark_accuracy,
        legacy_tokens_evaluated=legacy_tokens_evaluated,
        legacy_benchmark_examples=legacy_benchmark_examples,
        legacy_eval_set_hash=legacy_eval_set_hash,
    )
    return LadderEvalResult(
        per_rung_reports=per_rung,
        combined_report=combined,
        hidden_eval=hidden,
        mode=mode,
    )


__all__ = [
    "ACCEPT_OK",
    "EVAL_MODE_LEGACY",
    "EVAL_MODE_V011",
    "EvalSubprocessError",
    "LadderAcceptResult",
    "LadderEvalConfig",
    "LadderEvalResult",
    "LadderRungSpec",
    "REJECT_BAD_BRANCH_ID",
    "REJECT_BAD_FORMAT",
    "REJECT_VOCAB_MISMATCH",
    "SUPPORTED_SCHEMA_VERSIONS",
    "Submission",
    "VALID_EVAL_MODES",
    "accept_submission",
    "merge_rung_reports",
    "read_submission",
    "run_ladder_eval",
]
