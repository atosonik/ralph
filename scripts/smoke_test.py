"""End-to-end CPU smoke test — mining → validating → weight-setting.

Drives the SAME `validator.service.run_epoch` the testnet sim runs, end to end,
on CPU, so you can confirm the whole loop works before spending GPU on a real
round:

  1. reset a scratch chain (the real chain/ is backed up + restored — never
     clobbered)
  2. miner A submits the baseline (empty patch) → genesis king crowned →
     weights set
  3. miner B submits an improvement patch → scored under the single
     attested-execution tier → king re-evaluated → weights set again
  4. ASSERT the plumbing ran (king crowned + weights set both epochs, B not
     rejected); REPORT the science (did B's val_bpb beat A → king change)

Single attested-execution tier (v1.2 §5.4): there is no verified/unverified
split or α discount any more. Attestation runs in TESTNET mode here
(RALPH_ALLOW_MOCK_ATTESTATION=1) — the same relaxation a non-CC Shadeform box
uses; MAINNET requires real TEE (TDX) + NVIDIA CC. CPU-only.

Run:  python scripts/smoke_test.py   (exit 0 = loop verified)
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

# Testnet relaxation: this CPU box has no TEE/CC, so the proof produces a mock
# attestation. The mainnet gate (single attested-execution tier) rejects mock;
# the sim/testnet accepts it behind this flag. Set before importing the stack.
os.environ.setdefault("RALPH_ALLOW_MOCK_ATTESTATION", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401  — injects RALPH_RECIPE_DIR
from chain_layer.local import LocalChain
from miner.submit import assemble_submission, request_handshake_nonce
from proof.runner import run_proof_test
from validator.service import run_epoch

# A tiny patch that nudges max_lr in the smoke config from 3e-3 to 5e-3 —
# the smallest change that exercises the diff scanner + apply path.
PATCH_RAISE_LR = """\
--- a/configs/proxy_cpu_smoke.json
+++ b/configs/proxy_cpu_smoke.json
@@ -11,7 +11,7 @@
   "micro_batch_size": 4,
   "total_steps": 20,
   "warmup_steps": 2,
-  "max_lr": 0.003,
+  "max_lr": 0.005,
   "min_lr": 0.0003,
   "log_every": 2
 }
"""

NOISE_FLOOR_MARGIN = 0.05
SMOKE_ROOT_REL = "runs/smoke_e2e"


def _build_bundle_into_queue(
    ralph_root: Path, queue_dir: Path, miner_hotkey: str, patch_text: str,
    seed: int, label: str,
) -> None:
    """Miner side: handshake → proof-test → assemble, landing a self-contained
    bundle (with submission.json) in queue/pending/<label> for the validator."""
    work_dir = ralph_root / SMOKE_ROOT_REL / "work" / label
    bundle_dir = queue_dir / "pending" / label
    work_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (work_dir / "patch.diff").write_text(patch_text)
    patch_hash = hashlib.sha256(patch_text.encode()).hexdigest()
    nonce = request_handshake_nonce(ralph_root, miner_hotkey, patch_hash)
    (work_dir / "proof_request.json").write_text(json.dumps({
        "handshake_nonce": nonce,
        "seed": seed,
        "config_path": "configs/proxy_cpu_smoke.json",
        "miner_hotkey": miner_hotkey,
    }))

    run_proof_test(ralph_root=ralph_root, submission_dir=work_dir, out_dir=bundle_dir)
    # op1 integrity-checks proof_dir/patch.diff against the manifest's
    # patch_sha256; run_proof_test leaves the patch in the work dir, so copy it
    # into the bundle (empty file for the baseline — sha256("") still matches).
    shutil.copyfile(work_dir / "patch.diff", bundle_dir / "patch.diff")
    assemble_submission(ralph_root, miner_hotkey, work_dir, bundle_dir)


def _weights_events(chain: LocalChain) -> list[dict]:
    # get_events() is newest-first; reverse to chronological so [-1] is latest.
    evs = [e for e in chain.get_events(limit=10000) if e.get("type") == "weights_set"]
    return list(reversed(evs))


def _run_one_epoch(chain: LocalChain, queue_dir: Path) -> dict:
    """One real validator epoch, fully offline (no HF/GH/audit)."""
    return run_epoch(
        chain, queue_dir, noise_floor_margin=NOISE_FLOOR_MARGIN,
        hf_repo=None, hf_token=None, audit_random_rate=0.0,
        audit_reports_enabled=False,
    )


def main() -> int:
    ralph_root = Path(__file__).resolve().parent.parent
    chain_dir = ralph_root / "chain"
    backup_dir = ralph_root / "chain.smoke_backup"
    queue_dir = ralph_root / SMOKE_ROOT_REL / "queue"

    print("=" * 64)
    print("RALPH E2E SMOKE — mine → validate → set-weights (single tier, CPU)")
    print("=" * 64)

    # Protect any real/sim chain state: move it aside, restore in finally.
    moved = False
    if chain_dir.exists():
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.move(str(chain_dir), str(backup_dir))
        moved = True
        print(f"(backed up existing chain/ → {backup_dir.name})")

    try:
        chain_dir.mkdir(parents=True, exist_ok=True)
        if (ralph_root / SMOKE_ROOT_REL).exists():
            shutil.rmtree(ralph_root / SMOKE_ROOT_REL)
        chain = LocalChain(chain_dir)

        # --- Epoch 1: miner A baseline -> genesis king + weights ---
        print("\n[epoch 1] miner A submits the baseline (empty patch)")
        _build_bundle_into_queue(ralph_root, queue_dir, "5MinerA_baseline", "", 42, "A_baseline")
        r1 = _run_one_epoch(chain, queue_dir)
        king1 = chain.get_king()
        w1 = _weights_events(chain)
        assert king1 is not None, "epoch 1: no king crowned"
        assert king1.miner_hotkey == "5MinerA_baseline", f"epoch 1: unexpected king {king1.miner_hotkey}"
        assert w1, "epoch 1: weights were not set"
        print(f"  king = A  val_bpb={king1.val_bpb:.4f}  | weights_set={len(w1)}  subs={r1['submissions']}")

        # --- Epoch 2: miner B improvement -> scored + weights again ---
        print("\n[epoch 2] miner B submits an improvement patch (raise LR)")
        _build_bundle_into_queue(ralph_root, queue_dir, "5MinerB_raise_lr", PATCH_RAISE_LR, 43, "B_raise_lr")
        r2 = _run_one_epoch(chain, queue_dir)
        king2 = chain.get_king()
        w2 = _weights_events(chain)
        assert r2["submissions"] >= 1, "epoch 2: B was not picked up from the queue"
        assert r2["rejected"] == 0, "epoch 2: B was rejected (expected a clean score)"
        assert king2 is not None, "epoch 2: lost the king"
        assert len(w2) > len(w1), "epoch 2: weights were not set again"
        king_changed = king2.miner_hotkey == "5MinerB_raise_lr"
        print(f"  B val_bpb={king2.val_bpb if king_changed else 'n/a'}  king_now={king2.miner_hotkey}")
        print(f"  weights_set events={len(w2)} (latest weights: {_weights_events(chain)[-1]['weights']})")

        print("\n" + "=" * 64)
        print("SMOKE PASS ✅  mine → validate → set-weights verified end to end")
        print(f"  king change (B beat A): {king_changed}"
              + ("" if king_changed else "  — B didn't clear the noise floor this run (loop still verified)"))
        print("=" * 64)
        return 0
    finally:
        if chain_dir.exists():
            shutil.rmtree(chain_dir)
        if moved:
            shutil.move(str(backup_dir), str(chain_dir))
            print(f"(restored original chain/ from {backup_dir.name})")


if __name__ == "__main__":
    sys.exit(main())
