#!/usr/bin/env python3
"""
Full testnet cycle: two miners compete, validator scores, weights set on-chain.

Flow:
  1. Miner green1 submits a baseline (empty patch)
  2. Validator (green-test) scores it → sets weights on-chain → green1 becomes king
  3. Miner green2 submits a patch that improves the recipe
  4. Validator scores it → green2 beats green1 → new weights on-chain → new king

Uses CPU smoke config for fast iteration (3s per proof test).
Real ML quality doesn't matter here — testing the protocol on real Bittensor chain.

Usage:
    python scripts/testnet_cycle.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import bittensor as bt

import ralph_bootstrap  # noqa: F401  — injects RALPH_RECIPE_DIR
from chain_layer.config import get_chain
from miner.submit import sign_submission
from proof.runner import run_proof_test
from validator.scoring import score_bundle
from validator.validator import judge_submission

RALPH_ROOT = Path(__file__).resolve().parent.parent

PATCH_RAISE_LR = """\
--- a/configs/proxy_cpu_smoke.json
+++ b/configs/proxy_cpu_smoke.json
@@ -10,7 +10,7 @@
   "micro_batch_size": 4,
   "total_steps": 20,
   "warmup_steps": 5,
-  "max_lr": 0.003,
+  "max_lr": 0.005,
   "min_lr": 0.0003,
   "log_every": 2
 }
"""


def get_hotkey_ss58(wallet_name: str) -> str:
    w = bt.Wallet(name=wallet_name, hotkey="default")
    return w.hotkey.ss58_address


def submit_and_score(
    chain,
    miner_wallet: str,
    patch_text: str,
    label: str,
    tier: str = "verified",
    noise_floor_margin: float = 0.02,
) -> dict:
    """Run the full submission cycle for one miner."""
    miner_hotkey = get_hotkey_ss58(miner_wallet)
    print(f"\n{'='*60}")
    print(f"  [{label}] miner={miner_wallet} hotkey={miner_hotkey[:16]}...")
    print(f"{'='*60}")

    # 1. Prepare submission dir
    sub_dir = RALPH_ROOT / f"runs/testnet/{label}_sub"
    proof_dir = RALPH_ROOT / f"runs/testnet/{label}_proof"
    for d in [sub_dir, proof_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    patch_path = sub_dir / "patch.diff"
    patch_path.write_text(patch_text)
    patch_hash = hashlib.sha256(patch_text.encode()).hexdigest()

    # 2. Handshake: commit nonce on-chain
    print("  [handshake] committing nonce on-chain...")
    nonce = chain.request_handshake_nonce(miner_hotkey, patch_hash)
    print(f"  [handshake] nonce={nonce[:24]}...")

    (sub_dir / "proof_request.json").write_text(json.dumps({
        "handshake_nonce": nonce,
        "seed": 42 if "baseline" in label else 43,
        "config_path": "configs/proxy_cpu_smoke.json",
        "miner_hotkey": miner_hotkey,
    }))

    # 3. Proof test (CPU smoke — ~3 seconds)
    print("  [proof] running canonical training...")
    bundle = run_proof_test(
        ralph_root=RALPH_ROOT,
        submission_dir=sub_dir,
        out_dir=proof_dir,
        tier=tier,
    )
    print(f"  [proof] bundle_hash={bundle.bundle_hash[:16]}...")

    # 4. Sign the submission
    sig = sign_submission(RALPH_ROOT, miner_hotkey, bundle.bundle_hash, nonce)
    submission = {
        "miner_hotkey": miner_hotkey,
        "handshake_nonce": nonce,
        "patch_path": str(patch_path),
        "proof_dir": str(proof_dir),
        "bundle_hash": bundle.bundle_hash,
        "signature_hex": sig["signature_hex"],
        "public_key_hex": sig["public_key_hex"],
        "submitted_at": time.time(),
    }
    (proof_dir / "submission.json").write_text(json.dumps(submission, indent=2, sort_keys=True))

    # 5. Validator judges
    print("  [validator] running 4 ops + hidden eval...")
    result = judge_submission(RALPH_ROOT, proof_dir)
    if result.rejected:
        print(f"  [REJECTED] {result.rejected.reason}: {result.rejected.detail}")
        chain.append_event({
            "type": "submission_rejected",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "reason": result.rejected.reason,
        })
        return {"status": "rejected", "reason": result.rejected.reason}

    # 6. Score against current king
    king = chain.get_king()
    king_bpb = king.val_bpb if king else None
    king_bench = king.benchmark_accuracy if king else None
    tier_from_op2 = result.operations.get("op2_attestation", {}).get("tier", "unverified")

    score = score_bundle(
        val_bpb=result.hidden_eval.val_bpb,
        benchmark_accuracy=result.hidden_eval.benchmark_accuracy,
        king_val_bpb=king_bpb,
        king_benchmark=king_bench,
        noise_floor_margin=noise_floor_margin,
        matmul_ms=result.calibration["matmul_ms"],
        wall_clock_s=result.training_summary["wall_clock_s"],
        tier=tier_from_op2,
    )

    is_first = king is None
    accepted = score.decisively_beats_king or is_first

    print(f"  [score] val_bpb={result.hidden_eval.val_bpb:.4f}  "
          f"quality_gain={score.quality_gain:+.4f}  "
          f"decisive={score.decisively_beats_king}  "
          f"tier={score.tier}")

    # 7. Log event
    chain.append_event({
        "type": "submission_scored",
        "timestamp": time.time(),
        "miner_hotkey": miner_hotkey,
        "miner_wallet": miner_wallet,
        "label": label,
        "val_bpb": result.hidden_eval.val_bpb,
        "quality_gain": score.quality_gain,
        "score": score.score,
        "decisive": score.decisively_beats_king,
        "accepted": accepted,
    })

    # 8. Set weights on-chain
    if accepted:
        from chain_layer.interface import KingRecord
        new_king = KingRecord(
            miner_hotkey=miner_hotkey,
            bundle_hash=bundle.bundle_hash,
            val_bpb=result.hidden_eval.val_bpb,
            benchmark_accuracy=result.hidden_eval.benchmark_accuracy,
            compute_cost=score.compute_cost,
            crowned_at=time.time(),
            proof_dir=str(proof_dir),
        )
        if king:
            import dataclasses
            new_king.previous_king = dataclasses.asdict(king)
        chain.set_king(new_king)
        print(f"  [KING] {miner_wallet} crowned! weights set on-chain")
    else:
        # Still set weights — give the new submission some weight proportional to score
        hotkey_scores = {miner_hotkey: max(0.0, score.score)}
        if king:
            hotkey_scores[king.miner_hotkey] = 1.0  # king keeps top weight
        chain.set_weights(hotkey_scores)
        print("  [CHALLENGER] did not beat king, weights updated")

    return {
        "status": "accepted" if accepted else "below_threshold",
        "val_bpb": result.hidden_eval.val_bpb,
        "score": score.score,
        "decisive": score.decisively_beats_king,
        "miner_hotkey": miner_hotkey,
    }


def main():
    print("=" * 60)
    print("  RALPH TESTNET CYCLE — netuid 16")
    print("=" * 60)

    # Load chain (reads .env → BittensorChain)
    chain = get_chain(RALPH_ROOT)

    # Reset local chain state for clean run
    chain_dir = RALPH_ROOT / "chain"
    if chain_dir.exists():
        for f in ["king.json", "events.jsonl", "handshakes.jsonl"]:
            p = chain_dir / f
            if p.exists():
                p.unlink()

    # Prepare data if needed
    manifest = RALPH_ROOT / "data" / "data_manifest.json"
    if not manifest.exists():
        import subprocess
        print("\n[data] Generating synthetic data...")
        subprocess.run([
            sys.executable, "-m", "data.prepare",
            "--source", "synthetic",
            "--out", "data/shards",
            "--shard-tokens", "50000",
            "--total-tokens", "200000",
            "--eval-tokens", "10000",
        ], cwd=RALPH_ROOT, check=True)

    # --- Round 1: Miner green1 submits baseline ---
    res1 = submit_and_score(
        chain, "green1", "", "green1_baseline",
        tier="verified", noise_floor_margin=0.02,
    )

    # --- Round 2: Miner green2 submits a patch ---
    res2 = submit_and_score(
        chain, "green2", PATCH_RAISE_LR, "green2_raise_lr",
        tier="verified", noise_floor_margin=0.02,
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  TESTNET CYCLE COMPLETE")
    print("=" * 60)
    king = chain.get_king()
    if king:
        print(f"  King: hotkey={king.miner_hotkey[:20]}...")
        print(f"        val_bpb={king.val_bpb:.4f}")
    events = chain.get_events(limit=20)
    print(f"  Events: {len(events)}")
    for e in reversed(events):
        etype = e.get("type", "?")
        miner = e.get("miner_wallet", e.get("miner_hotkey", "")[:16])
        print(f"    {etype:25s}  {miner}")

    # Show on-chain state
    print("\n  On-chain metagraph (netuid 16):")
    chain.sync()
    for n in chain.metagraph.neurons[:5]:
        print(f"    uid={n.uid}  hotkey={n.hotkey[:16]}...  incentive={n.incentive}")


if __name__ == "__main__":
    main()
