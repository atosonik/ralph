#!/usr/bin/env python3
"""
Submit a proof bundle to the validator's queue.

Copies the completed proof bundle into queue/pending/<bundle_id>/
so the continuous validator service picks it up on the next epoch.

Usage:
    # After running the proof test:
    python -m miner.submit_to_queue --proof-dir runs/my_proof --queue-dir queue/

    # Full miner flow:
    # 1. Run proof test
    python -m proof.runner --submission submissions/my_patch --out-dir runs/my_proof --tier unverified
    # 2. Assemble + sign
    python -m miner.submit assemble --miner-hotkey <hotkey> --submission-dir submissions/my_patch --proof-dir runs/my_proof
    # 3. Submit to validator queue
    python -m miner.submit_to_queue --proof-dir runs/my_proof --queue-dir queue/
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def submit_to_queue(proof_dir: Path, queue_dir: Path) -> Path:
    """Copy a proof bundle to the validator's pending queue."""
    proof_dir = Path(proof_dir)
    queue_dir = Path(queue_dir)

    manifest_path = proof_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: no bundle_manifest.json in {proof_dir}")
        sys.exit(1)

    submission_path = proof_dir / "submission.json"
    if not submission_path.exists():
        print(f"ERROR: no submission.json in {proof_dir} — run miner.submit assemble first")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    bundle_hash = manifest["bundle_hash"]
    bundle_id = bundle_hash[:16]

    dest = queue_dir / "pending" / bundle_id
    if dest.exists():
        print(f"Bundle {bundle_id} already in queue, replacing...")
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # Copy all bundle files
    for f in proof_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dest / f.name)
    # Copy training subdir
    training_dir = proof_dir / "training"
    if training_dir.exists():
        shutil.copytree(training_dir, dest / "training")

    submission = json.loads(submission_path.read_text())
    print(f"Submitted to queue:")
    print(f"  bundle_id: {bundle_id}")
    print(f"  miner:     {submission.get('miner_hotkey', '?')[:20]}...")
    print(f"  queue:     {dest}")
    return dest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proof-dir", type=Path, required=True)
    p.add_argument("--queue-dir", type=Path, default=Path(__file__).resolve().parent.parent / "queue")
    args = p.parse_args()
    submit_to_queue(args.proof_dir, args.queue_dir)


if __name__ == "__main__":
    main()
