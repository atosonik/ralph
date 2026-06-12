"""v0.11-lite validator ladder — acceptance path with parent-cache verification.

C1-LITE scope: this module ships the SUBMISSION ACCEPTANCE half of the
validator ladder — the cheap, GPU-free preflight that runs on every
incoming child submission BEFORE the S1/S2/S3 CSDP eval is invoked.

What this module ships (v0.11-lite):

  * `Submission` — frozen dataclass for the in-process representation
    of a child submission's metadata + bundled parent_csdp_cache.
  * `LadderAcceptResult` — frozen dataclass for the verdict.
  * `read_submission(submission_dir)` — load submission.json +
    parent_csdp_cache.json from a miner's PR bundle.
  * `accept_submission(submission, chain, *, now_iso, ...)` — run the
    GPU-free preflight: format checks, vocab pin, parent lineage
    verification. Emits a single `submission_received` chain event on
    every call (including rejections, for audit). Returns
    `LadderAcceptResult` indicating whether the submission proceeds to
    the GPU eval stage.

What this module does NOT ship (C2-LITE):

  * The actual S1/S2/S3 dispatch — `eval/downstream/runner_cli.py` is
    invoked with the patched recipe, sealed-stream config, hardness
    index, and yields a `DownstreamReport`. The orchestration code
    that wraps that invocation lives in C2-LITE.
  * `HiddenEvalResult.downstream` population from the report — also
    C2-LITE.
  * `--legacy` mode preserving `to_legacy_dict` byte-equivalence — the
    accept_submission preflight is mode-agnostic; mode handling enters
    in C2-LITE's eval driver.

For C1-LITE, `accept_submission` is the visible API. C2-LITE's
`run_ladder_eval(submission, ladder_eval_config)` will consume an
ACCEPTED submission and produce the CSDP report.

Reference: docs/rearch_2026_06/00_v0_11_master.md §4.1 Validator.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from chain_layer.interface import ChainInterface
from eval.downstream.runner import KARPA_VOCAB_SIZE

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
      3. vocab_size == KARPA_VOCAB_SIZE (50257)
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

    if submission.vocab_size != KARPA_VOCAB_SIZE:
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
