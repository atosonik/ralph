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


# Three-class submission classification — see RUN_PLAN_meaningful_failure.md.
# A submission that passes all four cheap ops is then sorted into one of:
#   - king_change       weight 1.0   (decisively beats king past noise floor)
#   - meaningful_failure weight 0.1  (informative dead-end: attested + non-
#                                     trivial diff + coherent rationale +
#                                     val_bpb within 2x the noise band)
#   - plain_failure     weight 0.0   (didn't beat king and uninformative)
KING_CHANGE_WEIGHT = 1.0
MEANINGFUL_FAILURE_WEIGHT = 0.1
PLAIN_FAILURE_WEIGHT = 0.0
NOISE_FLOOR_MARGIN_2X_MULTIPLIER = 2.0
RATIONALE_MIN_NON_WS_CHARS = 200
RATIONALE_MIN_PARAGRAPHS = 2
DIFF_MIN_CHANGED_LINES = 5

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


def _diff_is_nontrivial(patch_path: Path) -> bool:
    """A diff is non-trivial if it changes more than DIFF_MIN_CHANGED_LINES
    non-whitespace, non-comment lines AND at least one of the touched files
    looks like it actually affects training (under recipe/, training/, or
    a *.yaml / *.yml config)."""
    if not patch_path.exists():
        return False
    try:
        text = patch_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    changed = 0
    touches_training = False
    for line in text.splitlines():
        if line.startswith("diff --git ") or line.startswith("+++ ") or line.startswith("--- "):
            ll = line.lower()
            if any(t in ll for t in ("recipe/", "training", "/optim", "configs/", ".yaml", ".yml", ".json", ".toml")):
                touches_training = True
            continue
        if line.startswith("+") or line.startswith("-"):
            stripped = line[1:].strip()
            if stripped and not stripped.startswith("#"):
                changed += 1
    return changed > DIFF_MIN_CHANGED_LINES and touches_training


def _rationale_is_coherent(rationale_path: Path) -> bool:
    """Heuristic: rationale.md is coherent if it has at least
    RATIONALE_MIN_NON_WS_CHARS non-whitespace chars, at least
    RATIONALE_MIN_PARAGRAPHS non-empty paragraphs, and at least 4 distinct
    sentences (catches template / boilerplate / pure repetition).

    First cut is structural heuristic only; an LLM-judge pass is a documented
    followup — see RUN_PLAN_meaningful_failure.md."""
    if not rationale_path.exists():
        return False
    try:
        text = rationale_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    non_ws = "".join(text.split())
    if len(non_ws) < RATIONALE_MIN_NON_WS_CHARS:
        return False
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) < RATIONALE_MIN_PARAGRAPHS:
        return False
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if len(s.strip()) > 10]
    if len(sentences) < 4:
        return False
    if len(set(sentences)) < len(sentences) * 0.6:
        return False
    return True


def _classify_outcome(
    *,
    decisively: bool,
    val_bpb: float,
    king_bpb: float | None,
    noise_floor_margin: float,
    bundle_dir: Path,
) -> tuple[str, float]:
    """Classify a submission that has already passed all four cheap ops.

    Returns (classification, weight_credit) — one of:
      - ("king_change",        KING_CHANGE_WEIGHT)        — beats king past noise floor
      - ("meaningful_failure", MEANINGFUL_FAILURE_WEIGHT) — didn't beat king, but
            attested + non-trivial diff + coherent rationale + val_bpb landed
            within 2x the 2sigma noise band. Rationale ships to the corpus.
      - ("plain_failure",      PLAIN_FAILURE_WEIGHT)      — anything else
    """
    if decisively:
        return "king_change", KING_CHANGE_WEIGHT

    if king_bpb is None:
        # No king yet — meaningful_failure is only definable vs an existing king.
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    # Bar 1: val_bpb must land within 2x the noise band.
    # delta > 0 means challenger is worse than king. Inside-noise-band cases
    # (delta <= noise_floor_margin in either direction) are still candidates.
    delta = val_bpb - king_bpb
    if delta > NOISE_FLOOR_MARGIN_2X_MULTIPLIER * noise_floor_margin:
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    # Bar 2: the diff has to change a training-relevant file with more than
    # DIFF_MIN_CHANGED_LINES non-trivial lines (not whitespace, not comments).
    if not _diff_is_nontrivial(bundle_dir / "patch.diff"):
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    # Bar 3: rationale must be present and structurally coherent.
    if not _rationale_is_coherent(bundle_dir / "rationale.md"):
        return "plain_failure", PLAIN_FAILURE_WEIGHT

    return "meaningful_failure", MEANINGFUL_FAILURE_WEIGHT


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
    # v1.2: single attested-execution tier. op2 either passed (tier="verified")
    # or rejected the submission outright. Treat the field defensively for any
    # legacy bundle that slips through.
    tier = result.operations.get("op2_attestation", {}).get("tier", "verified")

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
    decisively = score.decisively_beats_king or is_first

    classification, weight_credit = _classify_outcome(
        decisively=decisively,
        val_bpb=result.hidden_eval.val_bpb,
        king_bpb=king_bpb,
        noise_floor_margin=noise_floor_margin,
        bundle_dir=bundle_dir,
    )
    accepted = classification == "king_change"
    if classification == "king_change":
        status = "accepted"
    elif classification == "meaningful_failure":
        status = "meaningful_failure"
    else:
        status = "below_threshold"

    return {
        "status": status,
        "classification": classification,
        "weight_credit": weight_credit,
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
        weight_credit = result.get("weight_credit", 0.0)
        round_scores[miner_hotkey] = max(round_scores.get(miner_hotkey, 0), weight_credit)

        chain.append_event({
            "type": "submission_scored",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "miner_github": result.get("miner_github", ""),
            "val_bpb": result["val_bpb"],
            "quality_gain": result["quality_gain"],
            "score": result["score"],
            "weight_credit": weight_credit,
            "classification": result.get("classification", "unknown"),
            "tier": result["tier"],
            "decisive": result["decisive"],
            "accepted": result["accepted"],
        })

        # Meaningful-failure branch: didn't crown a new king, but the work was
        # informative. Give 10% weight credit and archive the rationale as a
        # published negative result (HF push is a documented followup; for
        # now, local archive to queue/meaningful_failure/<bundle_id>/).
        if result["status"] == "meaningful_failure":
            gh = result.get("miner_github", "")
            who = f"{gh} ({miner_hotkey[:12]}...)" if gh else f"{miner_hotkey[:20]}..."
            king_now = chain.get_king()
            king_bpb_str = f"{king_now.val_bpb:.4f}" if king_now else "—"
            log_info(
                f"MEANINGFUL FAILURE: {who} val_bpb={result['val_bpb']:.4f} "
                f"(king {king_bpb_str}); weight={MEANINGFUL_FAILURE_WEIGHT}, "
                f"rationale archived to corpus"
            )
            archive_bundle(bundle_dir, queue_dir, "meaningful_failure")
            epoch_results.setdefault("meaningful_failures", 0)
            epoch_results["meaningful_failures"] += 1
            chain.append_event({
                "type": "meaningful_failure_archived",
                "timestamp": time.time(),
                "miner_hotkey": miner_hotkey,
                "miner_github": result.get("miner_github", ""),
                "val_bpb": result["val_bpb"],
                "bundle_hash": result["bundle_hash"],
            })
            continue

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
            # Read .hf_pr.json once up front so we can use the actual bundle
            # repo_id for the release-notes link (not just whatever the
            # validator's KARPA_HF_REPO env var happens to say). Falls back to
            # the env var if the bundle didn't ship via an HF PR.
            hf_pr_info: dict | None = None
            hf_pr_path = bundle_dir / ".hf_pr.json"
            if hf_pr_path.exists():
                try:
                    import json as _json
                    hf_pr_info = _json.loads(
                        hf_pr_path.read_text(encoding="utf-8", errors="replace")
                    )
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    log_warn(f"failed to read .hf_pr.json for {bundle_id}: {e}")
            hf_bundle_repo = (
                (hf_pr_info or {}).get("repo_id")
                or os.environ.get("KARPA_HF_REPO", "karpaai/proof-bundles")
            )
            if bot_token and pr_url:
                # Pull the hypothesis (short) from submission.json and the full
                # rationale.md (if present) so the release notes can lead with
                # the miner's reasoning.
                #
                # NOTE: `hypothesis` is currently NOT covered by the miner's
                # signature on submission.json (followup: fold it into the
                # signed payload). For now we treat it as informational; the
                # rendering side (validator/github_bot.py) prefixes it with
                # "_Miner's claim..._" so readers know it's unverified.
                hypothesis = ""
                rationale_md = ""
                sub_path = bundle_dir / "submission.json"
                try:
                    import json as _json
                    sub = _json.loads(sub_path.read_text(encoding="utf-8", errors="replace"))
                    hypothesis = sub.get("hypothesis", "")
                except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
                    log_warn(f"failed to load submission.json for {bundle_id}: {e}")
                rm = bundle_dir / "rationale.md"
                if rm.exists():
                    try:
                        # Defense-in-depth size cap: miner-side caps rationale
                        # at 64KB; we allow some slack (200KB) but anything
                        # larger is treated as adversarial and dropped.
                        if rm.stat().st_size > 200_000:
                            log_warn(
                                f"rationale.md for {bundle_id} too large "
                                f"({rm.stat().st_size} bytes) — skipping"
                            )
                        else:
                            rationale_md = rm.read_text(encoding="utf-8", errors="replace")
                    except (FileNotFoundError, UnicodeDecodeError, OSError) as e:
                        log_warn(f"failed to load rationale.md for {bundle_id}: {e}")
                # Clamp the rationale we hand off to merge_and_release to 30KB
                # to keep release notes manageable even if it slipped past
                # earlier caps.
                rationale_summary = rationale_md[:30_000] if rationale_md else ""
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
                            "hypothesis": hypothesis,
                            "rationale_md": rationale_summary,
                            "hf_bundle_url": (
                                f"https://huggingface.co/datasets/{hf_bundle_repo}"
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

            # Merge the corresponding HF PR (if the bundle came from one).
            # Reuse hf_pr_info read at the top of the king-change block.
            if hf_pr_info is not None:
                hf_pr = hf_pr_info
                hf_token = os.environ.get("KARPA_BOT_HF_TOKEN") or os.environ.get("HF_TOKEN", "")
                if hf_token:
                    from validator.hf_bot import merge_pr as hf_merge_pr
                    res = hf_merge_pr(
                        repo_id=hf_pr["repo_id"],
                        pr_num=hf_pr["pr_num"],
                        token=hf_token,
                        comment=(
                            f"Crowned king. val_bpb={result['val_bpb']:.4f}, "
                            f"quality_gain={result['quality_gain']:+.4f}, "
                            f"miner={result.get('miner_github') or miner_hotkey[:12]}."
                        ),
                    )
                    if res.merged:
                        log_info(f"HF PR #{res.pr_num} merged on {hf_pr['repo_id']}")
                    else:
                        log_warn(f"HF PR #{res.pr_num} merge failed: {res.detail}")
                    chain.append_event({
                        "type": "hf_pr_merged" if res.merged else "hf_pr_merge_failed",
                        "timestamp": time.time(),
                        "repo_id": hf_pr["repo_id"],
                        "pr_num": res.pr_num,
                        "miner_hotkey": miner_hotkey,
                        "detail": res.detail,
                    })
                else:
                    log_warn(f"king changed but no HF token to merge PR #{hf_pr['pr_num']}")
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
                mf = result.get("meaningful_failures", 0)
                mf_str = f", {mf} meaningful failures" if mf else ""
                log_info(f"epoch {epoch}: {result['submissions']} submissions, "
                         f"{result['accepted']} accepted, {result['rejected']} rejected{mf_str}")
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
