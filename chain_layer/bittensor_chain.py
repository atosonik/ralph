"""
Bittensor chain backend — Phase 0.5d+.

Replaces the local JSON chain with real Bittensor on-chain operations:
  - Weight setting via subtensor.set_weights()
  - Hotkey registration verification via metagraph
  - Handshake nonces committed via subtensor.commit()
  - King state derived from chain weights + off-chain metadata

Karpathian does NOT use the standard axon/dendrite query pattern. Miners
submit proof bundles asynchronously (via HuggingFace Hub); validators
download, score, and set weights on-chain. The "communication" is:

  miner → on-chain handshake commit + off-chain bundle on HF
  validator → download bundle → score → set_weights on-chain

This is the same pattern Ninja (SN66) uses: miners submit via API,
validators score and set weights.

Usage:
    chain = BittensorChain(
        network="test",
        netuid=YOUR_NETUID,
        wallet_name="validator",
        wallet_hotkey="default",
    )
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Optional

import torch

from .interface import ChainInterface, HandshakeRecord, KingRecord


class BittensorChain(ChainInterface):

    def __init__(
        self,
        network: str = "test",
        netuid: int = 1,
        wallet_name: str = "default",
        wallet_hotkey: str = "default",
        chain_dir: Path | None = None,
    ):
        import bittensor as bt

        self.network = network
        self.netuid = netuid
        self.bt = bt

        self.wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
        self.subtensor = bt.Subtensor(network=network)
        self.metagraph = self.subtensor.metagraph(netuid=netuid)

        # Off-chain storage for data that doesn't fit on-chain (handshakes,
        # king metadata, event log). In production this moves to a shared
        # database or IPFS; for Phase 0.5d it's local files augmented by
        # on-chain weight state.
        self.chain_dir = chain_dir or Path(f"chain_bt_{network}_{netuid}")
        self.chain_dir.mkdir(parents=True, exist_ok=True)

        print(f"[chain] Bittensor {network} netuid={netuid}")
        print(f"[chain] wallet: {self.wallet.name}/{self.wallet.hotkey_str}")
        print(f"[chain] block: {self.subtensor.get_current_block()}")
        print(f"[chain] neurons: {self.metagraph.n}")

    def sync(self) -> None:
        """Refresh metagraph from chain."""
        self.metagraph.sync(subtensor=self.subtensor)

    def request_handshake_nonce(self, miner_hotkey: str, patch_hash: str) -> str:
        """Commit a handshake on-chain via subtensor.commit().

        The commit binds the miner's hotkey to a nonce + patch hash, recorded
        on-chain at a specific block. Validators verify the commitment exists
        before scoring the submission.
        """
        nonce = "0x" + secrets.token_hex(32)
        commit_data = f"karpathian:handshake:{miner_hotkey}:{patch_hash}:{nonce}"
        commit_hash = hashlib.sha256(commit_data.encode()).hexdigest()

        try:
            self.subtensor.commit(
                wallet=self.wallet,
                netuid=self.netuid,
                data=commit_hash,
            )
            print(f"[chain] handshake committed on-chain: {commit_hash[:16]}...")
        except Exception as e:
            print(f"[chain] on-chain commit failed ({e}), storing locally only")

        # Also store locally for lookup (on-chain commit is a hash; we need
        # the full record to verify).
        entry = {
            "type": "proof_test_handshake",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "patch_hash": patch_hash,
            "nonce": nonce,
            "commit_hash": commit_hash,
            "block": self._current_block(),
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
        """Check if a hotkey is registered on the subnet's metagraph."""
        self.sync()
        hotkeys = [n.hotkey for n in self.metagraph.neurons]
        return hotkey in hotkeys

    def get_uid(self, hotkey: str) -> Optional[int]:
        """Get the UID for a registered hotkey."""
        for n in self.metagraph.neurons:
            if n.hotkey == hotkey:
                return n.uid
        return None

    def set_weights(self, hotkey_scores: dict[str, float]) -> bool:
        """Validator sets weights on-chain for miners.

        hotkey_scores is {miner_hotkey: score}. Scores are normalized to
        [0, 1] and mapped to UIDs before calling subtensor.set_weights().
        """
        self.sync()

        uids = []
        weights = []
        for hotkey, score in hotkey_scores.items():
            uid = self.get_uid(hotkey)
            if uid is not None:
                uids.append(uid)
                weights.append(max(0.0, score))

        if not uids:
            print("[chain] no valid UIDs to set weights for")
            return False

        # Normalize weights to sum to 1.
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]

        uid_tensor = torch.tensor(uids, dtype=torch.int64)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        try:
            success, msg = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=uid_tensor,
                weights=weight_tensor,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            print(f"[chain] set_weights: success={success} msg={msg}")
            self.append_event({
                "type": "weights_set",
                "timestamp": time.time(),
                "uids": uids,
                "weights": weights,
                "success": success,
                "block": self._current_block(),
            })
            return success
        except Exception as e:
            print(f"[chain] set_weights failed: {e}")
            return False

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

        # Also set weights: king gets weight 1.0, everyone else 0
        self.set_weights({king.miner_hotkey: 1.0})

    def append_event(self, event: dict) -> None:
        if "block" not in event:
            event["block"] = self._current_block()
        with (self.chain_dir / "events.jsonl").open("a") as f:
            f.write(json.dumps(event) + "\n")

    def get_events(self, limit: int = 100) -> list[dict]:
        path = self.chain_dir / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        events = [json.loads(l) for l in lines if l.strip()]
        return list(reversed(events[-limit:]))

    def _current_block(self) -> int:
        try:
            return self.subtensor.get_current_block()
        except Exception:
            return 0

    # ---- Subnet registration helpers (for initial setup) ----

    @staticmethod
    def register_subnet(network: str = "test", wallet_name: str = "owner") -> int:
        """Register a new subnet on testnet. Returns the assigned netuid.

        Requires the wallet to have sufficient TAO for the registration burn.
        """
        import bittensor as bt
        wallet = bt.Wallet(name=wallet_name)
        sub = bt.Subtensor(network=network)
        netuid = sub.register_subnet(wallet=wallet)
        print(f"[chain] subnet registered: netuid={netuid}")
        return netuid

    @staticmethod
    def register_hotkey(
        network: str = "test",
        netuid: int = 1,
        wallet_name: str = "default",
        wallet_hotkey: str = "default",
    ) -> bool:
        """Register a hotkey (miner or validator) on a subnet."""
        import bittensor as bt
        wallet = bt.Wallet(name=wallet_name, hotkey=wallet_hotkey)
        sub = bt.Subtensor(network=network)
        success = sub.register(wallet=wallet, netuid=netuid)
        print(f"[chain] hotkey registered: {wallet.hotkey_str} on netuid={netuid} success={success}")
        return success
