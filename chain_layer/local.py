"""
Local JSON-file chain backend — Phase 0 / testing.

All state lives in a `chain/` directory under the karpathian root:
  chain/handshakes.jsonl   — one handshake per line
  chain/king.json          — current king
  chain/events.jsonl       — all protocol events

This is the same backend that Phase 0 used inline in miner/submit.py
and validator/router.py, now wrapped behind ChainInterface so the
Bittensor backend can slot in without changing callers.
"""

from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Optional

from .interface import ChainInterface, HandshakeRecord, KingRecord


class LocalChain(ChainInterface):

    def __init__(self, chain_dir: Path):
        self.chain_dir = Path(chain_dir)
        self.chain_dir.mkdir(parents=True, exist_ok=True)

    def request_handshake_nonce(self, miner_hotkey: str, patch_hash: str) -> str:
        nonce = "0x" + secrets.token_hex(32)
        entry = {
            "type": "proof_test_handshake",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "patch_hash": patch_hash,
            "nonce": nonce,
        }
        with (self.chain_dir / "handshakes.jsonl").open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return nonce

    def lookup_handshake(self, nonce: str) -> Optional[HandshakeRecord]:
        path = self.chain_dir / "handshakes.jsonl"
        if not path.exists():
            return None
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("nonce") == nonce:
                return HandshakeRecord(
                    nonce=entry["nonce"],
                    miner_hotkey=entry["miner_hotkey"],
                    patch_hash=entry["patch_hash"],
                    timestamp=entry["timestamp"],
                )
        return None

    def is_hotkey_registered(self, hotkey: str) -> bool:
        # Local chain: all hotkeys are "registered" (no real chain to check)
        return True

    def set_weights(self, hotkey_scores: dict[str, float]) -> bool:
        self.append_event({
            "type": "weights_set",
            "timestamp": time.time(),
            "weights": hotkey_scores,
        })
        return True

    def get_king(self) -> Optional[KingRecord]:
        path = self.chain_dir / "king.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        return KingRecord(
            miner_hotkey=d["miner_hotkey"],
            bundle_hash=d["bundle_hash"],
            val_bpb=d["val_bpb"],
            benchmark_accuracy=d.get("benchmark_accuracy", 0.0),
            compute_cost=d.get("compute_cost_h100h", 0.0),
            crowned_at=d.get("crowned_at", 0.0),
            proof_dir=d.get("proof_dir"),
            previous_king=d.get("previous_king"),
        )

    def set_king(self, king: KingRecord) -> None:
        d = {
            "miner_hotkey": king.miner_hotkey,
            "bundle_hash": king.bundle_hash,
            "val_bpb": king.val_bpb,
            "benchmark_accuracy": king.benchmark_accuracy,
            "compute_cost_h100h": king.compute_cost,
            "crowned_at": king.crowned_at,
            "proof_dir": king.proof_dir,
        }
        if king.previous_king:
            d["previous_king"] = king.previous_king
        (self.chain_dir / "king.json").write_text(json.dumps(d, indent=2, sort_keys=True))

    def append_event(self, event: dict) -> None:
        with (self.chain_dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")

    def get_events(self, limit: int = 100) -> list[dict]:
        path = self.chain_dir / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        events = [json.loads(l) for l in lines if l.strip()]
        return list(reversed(events[-limit:]))
