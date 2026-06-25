"""
Local JSON-file chain backend — Phase 0 / testing.

All state lives in a `chain/` directory under the ralph root:
  chain/handshakes.jsonl   — one handshake per line
  chain/king.json          — current king
  chain/events.jsonl       — all protocol events

This is the same backend that Phase 0 used inline in miner/submit.py
and validator/router.py, now wrapped behind ChainInterface so the
Bittensor backend can slot in without changing callers.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Optional

from .bittensor_chain import _locked_append
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
        _locked_append(self.chain_dir / "handshakes.jsonl", json.dumps(entry) + "\n")
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

    def set_burn_weights(self) -> bool:
        """Burn fallback (sim): record a 100%-to-burn-uid weight event."""
        import os as _os

        burn_uid = int(_os.environ.get("RALPH_BURN_UID", "0"))
        self.append_event({
            "type": "weights_set",
            "timestamp": time.time(),
            "weights": {f"uid:{burn_uid}": 1.0},
            "burn": True,
        })
        return True

    def get_king(self) -> Optional[KingRecord]:
        path = self.chain_dir / "king.json"
        if not path.exists():
            return None
        return KingRecord.from_dict(json.loads(path.read_text()))

    def set_king(self, king: KingRecord) -> None:
        d = {
            "miner_hotkey": king.miner_hotkey,
            "bundle_hash": king.bundle_hash,
            "val_bpb": king.val_bpb,
            "benchmark_accuracy": king.benchmark_accuracy,
            "compute_cost_h100h": king.compute_cost,
            "crowned_at": king.crowned_at,
            "crowned_at_block": king.crowned_at_block,
            "proof_dir": king.proof_dir,
        }
        if king.previous_king:
            d["previous_king"] = king.previous_king
        # v0.11-lite lineage fields. Omitted from JSON when empty/None to
        # keep legacy king.json byte-equivalent for pre-v0.11 callers.
        if king.king_attestation_hash:
            d["king_attestation_hash"] = king.king_attestation_hash
        if king.parent_king_attestation_hash is not None:
            d["parent_king_attestation_hash"] = king.parent_king_attestation_hash
        (self.chain_dir / "king.json").write_text(json.dumps(d, indent=2, sort_keys=True))

    def append_event(self, event: dict) -> None:
        _locked_append(self.chain_dir / "events.jsonl", json.dumps(event) + "\n")

    def commit_audit_root(self, sha256_hex: str) -> int:
        """File-write parity for the audit-root commitment (validation-v2 P1).

        No real chain here — record the commit as an event + write a
        last_audit_root.json so a local auditor / test can read it back.
        Returns the (event-count) block height like get_current_block().
        """
        sha = sha256_hex.lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(
                f"commit_audit_root expects 64-hex sha256, got {sha256_hex!r}"
            )
        self.append_event({
            "type": "audit_root_committed",
            "timestamp": time.time(),
            "report_sha256": sha,
        })
        block = self.get_current_block()
        (self.chain_dir / "last_audit_root.json").write_text(
            json.dumps({"report_sha256": sha, "block": block}, sort_keys=True)
        )
        return block

    def blacklist(self, hotkey: str, reason: str = "") -> None:
        path = self.chain_dir / "blacklist.json"
        current = {}
        if path.exists():
            try:
                current = json.loads(path.read_text())
            except json.JSONDecodeError:
                current = {}
        current[hotkey] = {"reason": reason, "at": time.time()}
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        self.append_event({"type": "blacklisted", "miner_hotkey": hotkey, "reason": reason, "timestamp": time.time()})

    def is_blacklisted(self, hotkey: str) -> bool:
        path = self.chain_dir / "blacklist.json"
        if not path.exists():
            return False
        try:
            return hotkey in json.loads(path.read_text())
        except json.JSONDecodeError:
            return False

    def get_events(self, limit: int = 100) -> list[dict]:
        path = self.chain_dir / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        events = [json.loads(l) for l in lines if l.strip()]
        return list(reversed(events[-limit:]))

    def get_current_block(self) -> int:
        path = self.chain_dir / "events.jsonl"
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text().splitlines() if line.strip())

    def get_block_hash(self, block: int) -> str:
        if block < 0:
            raise ValueError(f"block must be >= 0, got {block}")
        current = self.get_current_block()
        if block > current:
            raise ValueError(f"block {block} exceeds current height {current}")
        path = self.chain_dir / "events.jsonl"
        if not path.exists() or block == 0:
            payload = b""
        else:
            lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
            payload = "\n".join(lines[:block]).encode("utf-8")
        return "0x" + hashlib.blake2b(payload, digest_size=32).hexdigest()
