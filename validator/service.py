#!/usr/bin/env python3
"""
Continuous validator service — the 24/7 loop that makes Karpa a real subnet.

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

import karpa_bootstrap  # noqa: F401  — injects KARPA_RECIPE_DIR onto sys.path

from chain_layer.config import get_chain
from validator.validator import judge_submission
from validator.scoring import score_bundle
from validator.hf_poller import poll_hub, DEFAULT_REPO as DEFAULT_HF_REPO

# Bittensor's bt.logging hijacks Python's logging module and raises the root
# level to WARNING, silencing our INFO messages. Use direct prints with
# timestamps so our output always appears regardless of what bittensor does.
def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log_info(msg: str) -> None:
    print(f"{_ts()} [INFO] {msg}", flush=True)

def log_warn(msg: str) -> None:
    print(f"{_ts()} [WARN] {msg}", flush=True)

def log_err(msg: str) -> None:
    print(f"{_ts()} [ERROR] {msg}", flush=True)

def log_debug(msg: str) -> None:
    if os.environ.get("KARPA_DEBUG"):
        print(f"{_ts()} [DEBUG] {msg}", flush=True)

KARPA_ROOT = Path(__file__).resolve().parent.parent
SHUTDOWN = False


def _signal_handler(sig, frame):
    global SHUTDOWN
    log_info("shutdown signal received, finishing current epoch...")
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


def _verify_pr_if_required(result, bundle_dir: Path) -> tuple[bool, str]:
    """If $KARPA_BOT_GH_TOKEN is set and the submission carries a pr_url,
    verify the PR's diff is byte-equal to the bundle's patch.diff.
    Returns (ok, detail). If verification isn't configured, returns (True, "").
    """
    token = os.environ.get("KARPA_BOT_GH_TOKEN", "")
    if not token or not result.pr_url:
        return True, ""
    patch_path = bundle_dir / "patch.diff"
    if not patch_path.exists():
        return True, "no patch.diff in bundle (baseline?) — skipping PR match"
    patch_text = patch_path.read_text()
    if not patch_text.strip():
        return True, "empty patch (baseline) — skipping PR match"
    from validator.github_bot import verify_pr_matches_bundle
    v = verify_pr_matches_bundle(result.pr_url, patch_text, token)
    return v.ok, v.detail


def score_and_decide(
    chain,
    bundle_dir: Path,
    noise_floor_margin: float,
) -> dict:
    """Score one submission against the current king. Returns a result dict."""
    result = judge_submission(KARPA_ROOT, bundle_dir)

    if result.rejected:
        return {
            "status": "rejected",
            "miner_hotkey": result.miner_hotkey,
            "miner_github": result.miner_github,
            "pr_url": result.pr_url,
            "reason": result.rejected.reason,
            "detail": result.rejected.detail,
        }

    pr_ok, pr_detail = _verify_pr_if_required(result, bundle_dir)
    if not pr_ok:
        return {
            "status": "rejected",
            "miner_hotkey": result.miner_hotkey,
            "miner_github": result.miner_github,
            "pr_url": result.pr_url,
            "reason": "pr_mismatch",
            "detail": pr_detail,
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
        "miner_github": result.miner_github,
        "pr_url": result.pr_url,
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
    hf_repo: str | None = None,
    hf_token: str | None = None,
    hf_limit: int = 10,
) -> dict:
    """Process all pending submissions in one epoch."""
    if hf_repo:
        try:
            new = poll_hub(queue_dir, repo_id=hf_repo, token=hf_token, limit=hf_limit)
            if new:
                log_info(f"pulled {len(new)} bundle(s) from HF Hub: {[b[:8] for b in new]}")
        except Exception as e:
            log_warn(f"HF Hub poll failed: {e}")

    bundles = poll_queue(queue_dir)
    if not bundles:
        return {"submissions": 0, "accepted": 0, "rejected": 0}

    log_info(f"found {len(bundles)} pending submission(s)")
    epoch_results = {"submissions": len(bundles), "accepted": 0, "rejected": 0}
    round_scores: dict[str, float] = {}

    for bundle_dir in bundles:
        bundle_id = bundle_dir.name
        log_info(f"scoring {bundle_id}...")

        try:
            result = score_and_decide(chain, bundle_dir, noise_floor_margin)
        except Exception as e:
            log_err(f"error scoring {bundle_id}: {e}")
            log_debug(traceback.format_exc())
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
            log_warn(f"rejected {bundle_id}: {result['reason']}")
            archive_bundle(bundle_dir, queue_dir, "rejected")
            epoch_results["rejected"] += 1
            chain.append_event({
                "type": "submission_rejected",
                "timestamp": time.time(),
                "miner_hotkey": result["miner_hotkey"],
                "miner_github": result.get("miner_github", ""),
                "reason": result["reason"],
            })
            continue

        miner_hotkey = result["miner_hotkey"]
        round_scores[miner_hotkey] = max(round_scores.get(miner_hotkey, 0), result["score"])

        chain.append_event({
            "type": "submission_scored",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "miner_github": result.get("miner_github", ""),
            "val_bpb": result["val_bpb"],
            "quality_gain": result["quality_gain"],
            "score": result["score"],
            "tier": result["tier"],
            "decisive": result["decisive"],
            "accepted": result["accepted"],
        })

        if result["accepted"]:
            gh = result.get("miner_github", "")
            who = f"{gh} ({miner_hotkey[:12]}...)" if gh else f"{miner_hotkey[:20]}..."
            log_info(f"NEW KING: {who} val_bpb={result['val_bpb']:.4f}")
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

            # Auto-merge the winning PR + tag + release on karpaai/recipe.
            # Requires KARPA_BOT_GH_TOKEN. Failures here don't unwind the king
            # — the on-chain crown already happened.
            bot_token = os.environ.get("KARPA_BOT_GH_TOKEN", "")
            pr_url = result.get("pr_url", "")
            if bot_token and pr_url:
                try:
                    from validator.github_bot import merge_and_release
                    rel = merge_and_release(
                        pr_url=pr_url,
                        metrics={
                            "val_bpb": result["val_bpb"],
                            "quality_gain": result["quality_gain"],
                            "compute_cost_h100h": result["score_report"].compute_cost,
                            "benchmark_accuracy": result["benchmark_accuracy"],
                            "miner_hotkey": miner_hotkey,
                            "miner_github": result.get("miner_github", ""),
                            "bundle_hash": result["bundle_hash"],
                            "hf_bundle_url": (
                                f"https://huggingface.co/datasets/{os.environ.get('KARPA_HF_REPO', 'karpaai/proof-bundles')}"
                                f"/tree/main/submissions/{result['bundle_hash'][:16]}"
                            ),
                        },
                        token=bot_token,
                    )
                    log_info(f"recipe released: {rel.tag} ({rel.release_url})")
                    chain.append_event({
                        "type": "recipe_released",
                        "timestamp": time.time(),
                        "tag": rel.tag,
                        "release_url": rel.release_url,
                        "merge_sha": rel.merge_sha,
                        "miner_hotkey": miner_hotkey,
                        "miner_github": result.get("miner_github", ""),
                    })
                except Exception as e:
                    log_warn(f"recipe release failed: {e}")
            elif pr_url and not bot_token:
                log_warn(f"king changed with PR {pr_url} but KARPA_BOT_GH_TOKEN unset — manual merge needed")
        else:
            log_info(f"below threshold: {miner_hotkey[:20]}... gain={result['quality_gain']:+.4f}")

        archive_bundle(bundle_dir, queue_dir, "scored")

    # Set weights for all scored miners this epoch
    if round_scores:
        king = chain.get_king()
        if king:
            round_scores[king.miner_hotkey] = max(
                round_scores.get(king.miner_hotkey, 0), 1.0
            )
        log_info(f"setting weights for {len(round_scores)} miners...")
        chain.set_weights(round_scores)

    return epoch_results


def main():
    p = argparse.ArgumentParser(description="Karpa continuous validator service")
    p.add_argument("--queue-dir", type=Path, default=KARPA_ROOT / "queue")
    p.add_argument("--epoch-seconds", type=int, default=120,
                   help="Seconds between epochs (default: 120, ~10 blocks)")
    p.add_argument("--noise-floor", type=float, default=0.013,
                   help="val_bpb margin for 'decisively beats king' (default: 0.013 from H100 calibration)")
    p.add_argument("--hf-repo", default=os.environ.get("KARPA_HF_REPO", DEFAULT_HF_REPO),
                   help=f"HuggingFace dataset repo to poll (default: {DEFAULT_HF_REPO}). Set to empty string to disable.")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HuggingFace API token (defaults to $HF_TOKEN)")
    p.add_argument("--hf-limit", type=int, default=10,
                   help="Max bundles to download per epoch (default: 10)")
    p.add_argument("--once", action="store_true", help="Run one epoch then exit")
    args = p.parse_args()

    log_info("=" * 60)
    log_info("  Karpa Validator Service")
    log_info("=" * 60)

    chain = get_chain(KARPA_ROOT)
    args.queue_dir.mkdir(parents=True, exist_ok=True)
    (args.queue_dir / "pending").mkdir(exist_ok=True)

    king = chain.get_king()
    if king:
        log_info(f"current king: {king.miner_hotkey[:20]}... val_bpb={king.val_bpb:.4f}")
    else:
        log_info("no king yet — first submission will be crowned")

    log_info(f"queue: {args.queue_dir}")
    log_info(f"epoch interval: {args.epoch_seconds}s")
    log_info(f"noise floor margin: {args.noise_floor}")
    if args.hf_repo:
        log_info(f"HF Hub poll: {args.hf_repo} (limit {args.hf_limit}/epoch)")
    else:
        log_info("HF Hub poll: disabled")
    log_info(f"submit bundles to: {args.queue_dir / 'pending' / '<bundle_id>/'}")
    log_info("")

    epoch = 0
    while not SHUTDOWN:
        epoch += 1
        log_info(f"--- epoch {epoch} ---")

        try:
            result = run_epoch(
                chain,
                args.queue_dir,
                args.noise_floor,
                hf_repo=args.hf_repo or None,
                hf_token=args.hf_token,
                hf_limit=args.hf_limit,
            )
            if result["submissions"] > 0:
                log_info(f"epoch {epoch}: {result['submissions']} submissions, "
                         f"{result['accepted']} accepted, {result['rejected']} rejected")
            else:
                log_info(f"epoch {epoch}: no pending submissions")
        except Exception as e:
            log_err(f"epoch {epoch} failed: {e}")
            log_debug(traceback.format_exc())

        if args.once:
            break

        log_info(f"sleeping {args.epoch_seconds}s until next epoch...")
        for _ in range(args.epoch_seconds):
            if SHUTDOWN:
                break
            time.sleep(1)

    log_info("validator service stopped")


if __name__ == "__main__":
    main()
