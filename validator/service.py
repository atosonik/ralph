#!/usr/bin/env python3
"""
Continuous validator service — the 24/7 loop that makes AutoRalph a real subnet.

Each epoch (~tempo blocks):
  1. Sync metagraph
  2. Poll submission queue for new proof bundles
  3. Score each new submission (4 ops + hidden eval)
  4. Compare to current king
  5. Set weights on-chain
  6. Sleep until next epoch

Submissions arrive via the queue directory:
  queue/pending/<bundle_id>/     — new submissions waiting to be scored
  queue/scored/<bundle_id>/      — scored submissions (archived)
  queue/rejected/<bundle_id>/    — rejected submissions

A miner submits by placing their proof bundle in queue/pending/.
In production this is replaced by HuggingFace Hub polling.

Usage:
    python -m validator.service

    # Or with explicit config:
    python -m validator.service --queue-dir /path/to/queue --epoch-blocks 100
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chain_layer.config import get_chain
from validator.validator import judge_submission
from validator.scoring import score_bundle

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("validator")

AUTORALPH_ROOT = Path(__file__).resolve().parent.parent
SHUTDOWN = False


def _signal_handler(sig, frame):
    global SHUTDOWN
    log.info("shutdown signal received, finishing current epoch...")
    SHUTDOWN = True


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def poll_queue(queue_dir: Path) -> list[Path]:
    """Return paths to pending submission bundles, oldest first."""
    pending = queue_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    bundles = []
    for d in sorted(pending.iterdir()):
        if d.is_dir() and (d / "submission.json").exists():
            bundles.append(d)
    return bundles


def archive_bundle(bundle_dir: Path, queue_dir: Path, dest: str) -> None:
    """Move a processed bundle to scored/ or rejected/."""
    target = queue_dir / dest / bundle_dir.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(bundle_dir), str(target))


def score_and_decide(
    chain,
    bundle_dir: Path,
    noise_floor_margin: float,
) -> dict:
    """Score one submission against the current king. Returns a result dict."""
    result = judge_submission(AUTORALPH_ROOT, bundle_dir)

    if result.rejected:
        return {
            "status": "rejected",
            "miner_hotkey": result.miner_hotkey,
            "reason": result.rejected.reason,
            "detail": result.rejected.detail,
        }

    king = chain.get_king()
    king_bpb = king.val_bpb if king else None
    king_bench = king.benchmark_accuracy if king else None
    tier = result.operations.get("op2_attestation", {}).get("tier", "unverified")

    score = score_bundle(
        val_bpb=result.hidden_eval.val_bpb,
        benchmark_accuracy=result.hidden_eval.benchmark_accuracy,
        king_val_bpb=king_bpb,
        king_benchmark=king_bench,
        noise_floor_margin=noise_floor_margin,
        matmul_ms=result.calibration["matmul_ms"],
        wall_clock_s=result.training_summary["wall_clock_s"],
        tier=tier,
    )

    is_first = king is None
    accepted = score.decisively_beats_king or is_first

    return {
        "status": "accepted" if accepted else "below_threshold",
        "miner_hotkey": result.miner_hotkey,
        "bundle_hash": result.bundle_hash,
        "val_bpb": result.hidden_eval.val_bpb,
        "benchmark_accuracy": result.hidden_eval.benchmark_accuracy,
        "quality_gain": score.quality_gain,
        "score": score.score,
        "tier": score.tier,
        "decisive": score.decisively_beats_king,
        "accepted": accepted,
        "is_first": is_first,
        "result": result,
        "score_report": score,
    }


def run_epoch(
    chain,
    queue_dir: Path,
    noise_floor_margin: float,
) -> dict:
    """Process all pending submissions in one epoch."""
    bundles = poll_queue(queue_dir)
    if not bundles:
        return {"submissions": 0, "accepted": 0, "rejected": 0}

    log.info(f"found {len(bundles)} pending submission(s)")
    epoch_results = {"submissions": len(bundles), "accepted": 0, "rejected": 0}
    round_scores: dict[str, float] = {}

    for bundle_dir in bundles:
        bundle_id = bundle_dir.name
        log.info(f"scoring {bundle_id}...")

        try:
            result = score_and_decide(chain, bundle_dir, noise_floor_margin)
        except Exception as e:
            log.error(f"error scoring {bundle_id}: {e}")
            log.debug(traceback.format_exc())
            archive_bundle(bundle_dir, queue_dir, "rejected")
            epoch_results["rejected"] += 1
            chain.append_event({
                "type": "submission_error",
                "timestamp": time.time(),
                "bundle_id": bundle_id,
                "error": str(e),
            })
            continue

        if result["status"] == "rejected":
            log.warning(f"rejected {bundle_id}: {result['reason']}")
            archive_bundle(bundle_dir, queue_dir, "rejected")
            epoch_results["rejected"] += 1
            chain.append_event({
                "type": "submission_rejected",
                "timestamp": time.time(),
                "miner_hotkey": result["miner_hotkey"],
                "reason": result["reason"],
            })
            continue

        miner_hotkey = result["miner_hotkey"]
        round_scores[miner_hotkey] = max(round_scores.get(miner_hotkey, 0), result["score"])

        chain.append_event({
            "type": "submission_scored",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "val_bpb": result["val_bpb"],
            "quality_gain": result["quality_gain"],
            "score": result["score"],
            "tier": result["tier"],
            "decisive": result["decisive"],
            "accepted": result["accepted"],
        })

        if result["accepted"]:
            log.info(f"NEW KING: {miner_hotkey[:20]}... val_bpb={result['val_bpb']:.4f}")
            from chain_layer.interface import KingRecord
            king = chain.get_king()
            new_king = KingRecord(
                miner_hotkey=miner_hotkey,
                bundle_hash=result["bundle_hash"],
                val_bpb=result["val_bpb"],
                benchmark_accuracy=result["benchmark_accuracy"],
                compute_cost=result["score_report"].compute_cost,
                crowned_at=time.time(),
                proof_dir=str(bundle_dir),
            )
            if king:
                import dataclasses
                new_king.previous_king = dataclasses.asdict(king)
            chain.set_king(new_king)
            epoch_results["accepted"] += 1
        else:
            log.info(f"below threshold: {miner_hotkey[:20]}... gain={result['quality_gain']:+.4f}")

        archive_bundle(bundle_dir, queue_dir, "scored")

    # Set weights for all scored miners this epoch
    if round_scores:
        king = chain.get_king()
        if king:
            round_scores[king.miner_hotkey] = max(
                round_scores.get(king.miner_hotkey, 0), 1.0
            )
        log.info(f"setting weights for {len(round_scores)} miners...")
        chain.set_weights(round_scores)

    return epoch_results


def main():
    p = argparse.ArgumentParser(description="AutoRalph continuous validator service")
    p.add_argument("--queue-dir", type=Path, default=AUTORALPH_ROOT / "queue")
    p.add_argument("--epoch-seconds", type=int, default=120,
                   help="Seconds between epochs (default: 120, ~10 blocks)")
    p.add_argument("--noise-floor", type=float, default=0.013,
                   help="val_bpb margin for 'decisively beats king' (default: 0.013 from H100 calibration)")
    p.add_argument("--once", action="store_true", help="Run one epoch then exit")
    args = p.parse_args()

    log.info("=" * 60)
    log.info("  AutoRalph Validator Service")
    log.info("=" * 60)

    chain = get_chain(AUTORALPH_ROOT)
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    (args.queue_dir / "pending").mkdir(exist_ok=True)

    king = chain.get_king()
    if king:
        log.info(f"current king: {king.miner_hotkey[:20]}... val_bpb={king.val_bpb:.4f}")
    else:
        log.info("no king yet — first submission will be crowned")

    log.info(f"queue: {args.queue_dir}")
    log.info(f"epoch interval: {args.epoch_seconds}s")
    log.info(f"noise floor margin: {args.noise_floor}")
    log.info(f"submit bundles to: {args.queue_dir / 'pending' / '<bundle_id>/'}")
    log.info("")

    epoch = 0
    while not SHUTDOWN:
        epoch += 1
        log.info(f"--- epoch {epoch} ---")

        try:
            result = run_epoch(chain, args.queue_dir, args.noise_floor)
            if result["submissions"] > 0:
                log.info(f"epoch {epoch}: {result['submissions']} submissions, "
                         f"{result['accepted']} accepted, {result['rejected']} rejected")
            else:
                log.info(f"epoch {epoch}: no pending submissions")
        except Exception as e:
            log.error(f"epoch {epoch} failed: {e}")
            log.debug(traceback.format_exc())

        if args.once:
            break

        log.info(f"sleeping {args.epoch_seconds}s until next epoch...")
        for _ in range(args.epoch_seconds):
            if SHUTDOWN:
                break
            time.sleep(1)

    log.info("validator service stopped")


if __name__ == "__main__":
    main()
