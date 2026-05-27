"""
End-to-end smoke test.

Runs the full AutoRalph Phase 0 flow:
  1. Reset chain + king.
  2. Miner A submits the baseline (empty patch) — auto-crowned as initial king.
  3. Miner B submits a patch that tweaks training hyperparameters.
  4. Validator scores B against A.
  5. Report whether the new king was crowned.
  6. Print the chain summary.
"""

from __future__ import annotations

import json
import secrets
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miner.submit import assemble_submission, request_handshake_nonce
from proof.runner import run_proof_test
from validator.router import process_submission


# A tiny patch that nudges max_lr in the smoke config from 3e-3 to 5e-3.
# Real autoresearch-style submissions would touch train.py or model code;
# this is the smallest patch that exercises the diff scanner + apply path.
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

PATCH_LOWER_WD = """\
--- a/configs/proxy_cpu_smoke.json
+++ b/configs/proxy_cpu_smoke.json
@@ -10,7 +10,7 @@
   "micro_batch_size": 4,
   "total_steps": 20,
   "warmup_steps": 5,
   "max_lr": 0.003,
-  "min_lr": 0.0003,
+  "min_lr": 0.001,
   "log_every": 2
 }
"""


def _reset_state(autoralph_root: Path) -> None:
    """Wipe chain + previous runs for a clean smoke test."""
    for p in ["chain", "runs/smoke_e2e"]:
        path = autoralph_root / p
        if path.exists():
            shutil.rmtree(path)
    (autoralph_root / "runs/smoke_e2e").mkdir(parents=True, exist_ok=True)


def _submit_one(
    autoralph_root: Path,
    miner_hotkey: str,
    patch_text: str,
    seed: int,
    label: str,
    tier: str = "verified",
) -> dict:
    """Run the full miner → proof → validator → router flow for one submission."""
    sub_dir = autoralph_root / "runs/smoke_e2e" / f"sub_{label}"
    proof_dir = autoralph_root / "runs/smoke_e2e" / f"proof_{label}"
    sub_dir.mkdir(parents=True, exist_ok=True)
    patch_path = sub_dir / "patch.diff"
    patch_path.write_text(patch_text)

    import hashlib
    patch_hash = hashlib.sha256(patch_text.encode()).hexdigest()

    nonce = request_handshake_nonce(autoralph_root, miner_hotkey, patch_hash)
    (sub_dir / "proof_request.json").write_text(json.dumps({
        "handshake_nonce": nonce,
        "seed": seed,
        "config_path": "configs/proxy_cpu_smoke.json",
        "miner_hotkey": miner_hotkey,
    }))

    print(f"\n--- [{label}] miner={miner_hotkey} tier={tier} ---")
    bundle = run_proof_test(
        autoralph_root=autoralph_root,
        submission_dir=sub_dir,
        out_dir=proof_dir,
        tier=tier,
    )
    assemble_submission(autoralph_root, miner_hotkey, sub_dir, proof_dir)
    return process_submission(autoralph_root, proof_dir, noise_floor_margin=0.05)


def main() -> None:
    autoralph_root = Path(__file__).resolve().parent.parent
    _reset_state(autoralph_root)

    print("=" * 60)
    print("AUTORALPH PHASE 0 — END-TO-END SMOKE TEST")
    print("=" * 60)

    # Miner A submits the baseline (empty patch).
    res_a = _submit_one(autoralph_root, "5MinerA_baseline", "", seed=42, label="A_baseline")
    print(f"\n[A] status={res_a['status']}")
    print(f"    val_bpb={res_a['result']['hidden_eval']['val_bpb']:.4f}")
    print(f"    accepted_as_king={res_a['event']['accepted_as_king']}")

    # Miner B submits a patch (verified tier).
    res_b = _submit_one(
        autoralph_root,
        "5MinerB_raise_lr",
        PATCH_RAISE_LR,
        seed=43,
        label="B_verified",
        tier="verified",
    )
    print(f"\n[B verified] status={res_b['status']}")
    print(f"    val_bpb={res_b['result']['hidden_eval']['val_bpb']:.4f}")
    print(f"    tier={res_b['score']['tier']} α={res_b['score']['alpha']}")
    print(f"    score={res_b['score']['score']:.4f}")

    # Miner C submits a different patch (UNVERIFIED tier — no attestation, α=0.5).
    res_c = _submit_one(
        autoralph_root,
        "5MinerC_unverified",
        PATCH_LOWER_WD,
        seed=44,
        label="C_unverified",
        tier="unverified",
    )
    print(f"\n[C unverified] status={res_c['status']}")
    print(f"    val_bpb={res_c['result']['hidden_eval']['val_bpb']:.4f}")
    print(f"    tier={res_c['score']['tier']} α={res_c['score']['alpha']}")
    print(f"    score={res_c['score']['score']:.4f}")
    print(f"    cost_effective={res_c['score']['compute_cost_effective']:.6f} (2× raw due to α=0.5)")

    # Print the chain state.
    king_path = autoralph_root / "chain" / "king.json"
    events_path = autoralph_root / "chain" / "events.jsonl"
    print("\n" + "=" * 60)
    print("FINAL CHAIN STATE")
    print("=" * 60)
    if king_path.exists():
        king = json.loads(king_path.read_text())
        print(f"king: {king['miner_hotkey']}  val_bpb={king['val_bpb']:.4f}  bundle={king['bundle_hash'][:12]}…")
    events = [json.loads(l) for l in events_path.read_text().splitlines() if l.strip()]
    print(f"events: {len(events)}")
    for e in events:
        print(f"  {e['type']:25s} t={e['timestamp']:.0f}")


if __name__ == "__main__":
    main()
