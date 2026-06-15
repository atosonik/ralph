"""Audit scheduler — wires §5.7's deterrence math into real consequences.

Per the whitepaper, the validator should run Stage-5 full re-execution audits
on a fraction of accepted submissions. The expected value of fraud is:

    EV(fraud) = 0.9 * R - p_audit * (slash + fee + emission_loss)

where p_audit must be high enough to make EV negative. Without an audit
dispatcher (deep_review_2026-05-31 critical #13), p_audit = 0 and EV
collapses to 0.9 * R — every miner is rationally incentivized to cheat.

This module:
  - Decides whether to audit each accepted / close-margin submission
  - Enqueues audit jobs to chain_dir/audit_queue/ (don't run synchronously
    — full re-train takes ~70min)
  - Processes the queue out-of-band via run_pending_audits()
  - On audit fail: appends submission_fraud event + calls chain.blacklist()
    so subsequent set_weights zeros the cheater's hotkey

The 10% sample rate and 2x noise-floor king-margin trigger are tunable; see
RUN_PLAN_meaningful_failure.md §5.7 for the calibration math.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path


def _default_random_audit_rate() -> float:
    """Operator override via RALPH_AUDIT_RATE env var (0.0–1.0). Lets the
    operator drop the audit rate without code change — useful when running
    self-audits where audits cost without revealing fraud (no external miners
    to defend against)."""
    import os as _os
    raw = _os.environ.get("RALPH_AUDIT_RATE")
    if raw is None:
        return 0.10
    try:
        v = float(raw)
        if 0.0 <= v <= 1.0:
            return v
    except ValueError:
        pass
    return 0.10


# Tunable: probability that any accepted king-change is audited blindly.
DEFAULT_RANDOM_AUDIT_RATE = _default_random_audit_rate()
# Tunable: if a submission beats king by less than this fraction of the noise
# floor, audit regardless of the random sample. Close margins are the highest
# fraud-vs-luck-distinguishing signal.
DEFAULT_KING_MARGIN_AUDIT_MULTIPLIER = 2.0


@dataclass
class AuditJob:
    bundle_id: str
    miner_hotkey: str
    miner_github: str
    bundle_hash: str
    val_bpb: float
    king_val_bpb: float | None
    quality_gain: float
    classification: str  # "king_change" / "meaningful_failure" / "plain_failure"
    enqueued_at: float
    reason: str  # "random_sample" / "king_margin" / "manual"
    proof_dir: str

    def to_dict(self) -> dict:
        return {
            "bundle_id": self.bundle_id,
            "miner_hotkey": self.miner_hotkey,
            "miner_github": self.miner_github,
            "bundle_hash": self.bundle_hash,
            "val_bpb": self.val_bpb,
            "king_val_bpb": self.king_val_bpb,
            "quality_gain": self.quality_gain,
            "classification": self.classification,
            "enqueued_at": self.enqueued_at,
            "reason": self.reason,
            "proof_dir": self.proof_dir,
        }


def _audit_queue_dir(chain_dir: Path) -> Path:
    d = chain_dir / "audit_queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _deterministic_audit_decision(bundle_hash: str, rate: float) -> bool:
    """Decide whether to audit *this* bundle.

    Uses a deterministic hash of bundle_hash so the decision is reproducible
    (any observer can recompute) and not influenceable by a miner with a
    timing oracle on the validator's random.
    """
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    threshold = int(rate * (1 << 64))
    digest = hashlib.sha256(("karpa-audit:" + bundle_hash).encode()).digest()[:8]
    return int.from_bytes(digest, "big") < threshold


def maybe_enqueue_audit(
    chain_dir: Path,
    bundle_id: str,
    miner_hotkey: str,
    miner_github: str,
    bundle_hash: str,
    val_bpb: float,
    king_val_bpb: float | None,
    quality_gain: float,
    classification: str,
    proof_dir: Path,
    noise_floor_margin: float,
    random_audit_rate: float = DEFAULT_RANDOM_AUDIT_RATE,
    king_margin_multiplier: float = DEFAULT_KING_MARGIN_AUDIT_MULTIPLIER,
) -> AuditJob | None:
    """If this submission should be audited, write the job and return it.

    Trigger criteria:
      - random sample (deterministic 10%)
      - close-margin king change: quality_gain < king_margin_multiplier *
        noise_floor (likely lucky-seed or borderline fraud)

    Skips plain_failures (no incentive to audit; already discarded).
    """
    if classification == "plain_failure":
        return None

    triggers: list[str] = []
    if _deterministic_audit_decision(bundle_hash, random_audit_rate):
        triggers.append("random_sample")

    if (
        classification == "king_change"
        and king_val_bpb is not None
        and abs(quality_gain) < king_margin_multiplier * noise_floor_margin
    ):
        triggers.append("king_margin")

    if not triggers:
        return None

    job = AuditJob(
        bundle_id=bundle_id,
        miner_hotkey=miner_hotkey,
        miner_github=miner_github,
        bundle_hash=bundle_hash,
        val_bpb=val_bpb,
        king_val_bpb=king_val_bpb,
        quality_gain=quality_gain,
        classification=classification,
        enqueued_at=time.time(),
        reason="+".join(triggers),
        proof_dir=str(proof_dir),
    )
    queue_dir = _audit_queue_dir(chain_dir)
    (queue_dir / f"{bundle_id}.json").write_text(
        json.dumps(job.to_dict(), indent=2, sort_keys=True)
    )
    return job


def list_pending_audits(chain_dir: Path) -> list[AuditJob]:
    queue_dir = _audit_queue_dir(chain_dir)
    out: list[AuditJob] = []
    for f in sorted(queue_dir.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        out.append(AuditJob(**d))
    return out


def archive_audit_job(chain_dir: Path, bundle_id: str, verdict: str) -> None:
    """Move a processed audit job into a verdict-suffixed bin."""
    queue_dir = _audit_queue_dir(chain_dir)
    src = queue_dir / f"{bundle_id}.json"
    if not src.exists():
        return
    dest_dir = queue_dir / verdict  # "passed" / "failed"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{bundle_id}.json"
    if dest.exists():
        dest.unlink()
    src.rename(dest)


def run_pending_audits(
    chain,
    ralph_root: Path,
    chain_dir: Path,
    noise_floor_margin: float = 0.013,
    limit: int = 1,
) -> dict:
    """Process up to `limit` pending audit jobs.

    On fail: append submission_fraud event + blacklist the miner. The next
    set_weights call will zero their hotkey.

    Returns a summary dict. Intended for periodic invocation (e.g., one job
    per validator idle-epoch, or by a separate cron).
    """
    from validator.audit import run_audit

    pending = list_pending_audits(chain_dir)
    if not pending:
        return {"processed": 0, "passed": 0, "failed": 0}

    processed = 0
    passed = 0
    failed = 0

    for job in pending[:limit]:
        proof_path = Path(job.proof_dir)
        if not proof_path.exists():
            # The bundle archive may have moved between scored/ and
            # meaningful_failure/. Try a few likely locations before giving up.
            for alt_dir in ("scored", "meaningful_failure"):
                alt = chain_dir.parent / "queue" / alt_dir / job.bundle_id
                if alt.exists():
                    proof_path = alt
                    break
            else:
                archive_audit_job(chain_dir, job.bundle_id, "skipped_missing")
                continue

        audit_out = chain_dir / "audit_runs" / job.bundle_id
        audit_out.parent.mkdir(parents=True, exist_ok=True)
        # Construct the submission_dir the audit expects (proof_runner needs
        # patch.diff + proof_request.json).
        # Reuse the bundle's patch.diff; reconstruct a minimal proof_request.
        sub_dir = chain_dir / "audit_runs" / f"{job.bundle_id}_sub"
        sub_dir.mkdir(parents=True, exist_ok=True)
        patch_src = proof_path / "patch.diff"
        if patch_src.exists():
            (sub_dir / "patch.diff").write_bytes(patch_src.read_bytes())
        # proof_request.json — derive what we can from the bundle manifest.
        manifest_path = proof_path / "bundle_manifest.json"
        if manifest_path.exists():
            try:
                mf = json.loads(manifest_path.read_text())
                (sub_dir / "proof_request.json").write_text(json.dumps({
                    "handshake_nonce": mf.get("handshake_nonce", ""),
                    "seed": mf.get("declared_seed", 1337),
                    "config_path": "configs/proxy_cpu_smoke.json",
                }))
            except json.JSONDecodeError:
                archive_audit_job(chain_dir, job.bundle_id, "skipped_bad_manifest")
                continue
        else:
            archive_audit_job(chain_dir, job.bundle_id, "skipped_no_manifest")
            continue

        try:
            result = run_audit(
                ralph_root=ralph_root,
                submission_dir=sub_dir,
                miner_proof_dir=proof_path,
                audit_out_dir=audit_out,
                noise_floor_margin=noise_floor_margin,
            )
        except Exception as e:
            chain.append_event({
                "type": "audit_error",
                "timestamp": time.time(),
                "bundle_id": job.bundle_id,
                "miner_hotkey": job.miner_hotkey,
                "error": str(e),
            })
            archive_audit_job(chain_dir, job.bundle_id, "errored")
            processed += 1
            continue

        processed += 1
        if result.passed:
            passed += 1
            chain.append_event({
                "type": "audit_passed",
                "timestamp": time.time(),
                "bundle_id": job.bundle_id,
                "miner_hotkey": job.miner_hotkey,
                "miner_val_bpb": result.miner_val_bpb,
                "audit_val_bpb": result.audit_val_bpb,
                "bpb_diff": result.bpb_diff,
                "reason": job.reason,
            })
            archive_audit_job(chain_dir, job.bundle_id, "passed")
        else:
            failed += 1
            chain.append_event({
                "type": "submission_fraud",
                "timestamp": time.time(),
                "bundle_id": job.bundle_id,
                "miner_hotkey": job.miner_hotkey,
                "miner_github": job.miner_github,
                "miner_val_bpb": result.miner_val_bpb,
                "audit_val_bpb": result.audit_val_bpb,
                "bpb_diff": result.bpb_diff,
                "trajectory_max_dev": result.trajectory_max_deviation,
                "detail": result.detail,
                "reason": job.reason,
            })
            # §5.7 deterrence: blacklist + zero subsequent weights.
            chain.blacklist(
                job.miner_hotkey,
                reason=f"audit_failed: {result.detail[:200]}",
            )
            archive_audit_job(chain_dir, job.bundle_id, "failed")

    return {"processed": processed, "passed": passed, "failed": failed}
