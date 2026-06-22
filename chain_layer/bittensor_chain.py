"""
Bittensor chain backend — Phase 0.5d+.

Replaces the local JSON chain with real Bittensor on-chain operations:
  - Weight setting via subtensor.set_weights()
  - Hotkey registration verification via metagraph
  - Handshake nonces committed via subtensor.commit()
  - King state derived from chain weights + off-chain metadata

Ralph does NOT use the standard axon/dendrite query pattern. Miners
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
import os
import secrets
import time
from pathlib import Path
from typing import Optional

import torch

from .interface import ChainInterface, HandshakeRecord, KingRecord


def _locked_append(path: Path, text: str) -> None:
    """Append text to a file with an advisory exclusive lock.

    Prevents interleaved bytes when multiple processes (miner + validator,
    or two validators) share the same chain_dir on a single host. fcntl is
    POSIX-only; on platforms without it we fall back to plain append (which
    is fine for the common single-writer case).
    """
    try:
        import fcntl
        with path.open("a") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(text)
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover — Windows
        with path.open("a") as f:
            f.write(text)


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
        password = os.environ.get("BT_WALLET_PASSWORD")
        if password and self.wallet.coldkey_file.is_encrypted():
            self.wallet.coldkey_file.decrypt(password)
            print("[chain] coldkey decrypted")
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
        """Commit a handshake on-chain via subtensor.set_commitment().

        The commit binds the miner's hotkey to a nonce + patch hash, recorded
        on-chain at a specific block. Validators verify the commitment exists
        before scoring the submission.

        deep_review_2026-05-31 #6: the previous code called subtensor.commit(),
        which does NOT exist on bittensor 10.4.0 — the method is
        set_commitment(). The AttributeError was being swallowed silently,
        meaning no handshake has ever made it on-chain (this is the root
        cause of RALPH_SKIP_HANDSHAKE=1 being mandatory until now).

        We now raise on commit failure so the miner gets an explicit
        rejection instead of a "successful" handshake that's actually
        invisible to the chain.
        """
        nonce = "0x" + secrets.token_hex(32)
        commit_data = f"karpa:handshake:{miner_hotkey}:{patch_hash}:{nonce}"
        commit_hash = hashlib.sha256(commit_data.encode()).hexdigest()

        try:
            self.subtensor.set_commitment(
                wallet=self.wallet,
                netuid=self.netuid,
                data=commit_hash,
            )
            print(f"[chain] handshake committed on-chain: {commit_hash[:16]}...")
        except AttributeError:
            # SDK version drift — fail loudly so we don't silently lose commits.
            raise RuntimeError(
                "bittensor SDK has no set_commitment method. "
                "Expected method on bittensor>=10.0. Got attrs containing "
                f"'commit': {[a for a in dir(self.subtensor) if 'commit' in a.lower()][:5]}"
            )
        except Exception as e:
            # Real on-chain failures (rate limit, RPC down, insufficient
            # balance, etc) must propagate so the miner can retry instead of
            # shipping a bundle the validator will then reject for missing
            # handshake.
            raise RuntimeError(f"on-chain commit failed: {type(e).__name__}: {e}")

        # Also store locally for lookup (on-chain commit is a hash; we need
        # the full record to verify). Uses fcntl.LOCK_EX so concurrent
        # miners on the same host don't interleave bytes.
        entry = {
            "type": "proof_test_handshake",
            "timestamp": time.time(),
            "miner_hotkey": miner_hotkey,
            "patch_hash": patch_hash,
            "nonce": nonce,
            "commit_hash": commit_hash,
            "block": self._current_block(),
        }
        _locked_append(self.chain_dir / "handshakes.jsonl", json.dumps(entry) + "\n")
        return nonce

    def commit_audit_root(self, sha256_hex: str) -> int:
        """Anchor a per-epoch audit-report hash on-chain (validation-v2 Phase 1).

        Commits the 64-char hex sha256 of the canonical report_json via the
        same `subtensor.set_commitment` path the handshake uses (the proven
        method on bittensor 10.x; `subtensor.commit` does NOT exist). The
        commitment overwrites each epoch, so auditors read it from an archive
        subtensor at the historical block.

        Mirrors `request_handshake_nonce`'s error discipline: RAISE on any
        failure (SDK drift, rate limit, RPC down) so the caller surfaces it —
        never silently lose a commit. The caller in service.run_epoch wraps
        this in try/except so a commit failure can't break weight-setting.

        Returns the block height the commitment landed at.
        """
        sha = sha256_hex.lower()
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError(
                f"commit_audit_root expects 64-hex sha256, got {sha256_hex!r}"
            )
        try:
            self.subtensor.set_commitment(
                wallet=self.wallet,
                netuid=self.netuid,
                data=sha,
            )
            print(f"[chain] audit root committed on-chain: {sha[:16]}...")
        except AttributeError:
            raise RuntimeError(
                "bittensor SDK has no set_commitment method. "
                "Expected method on bittensor>=10.0. Got attrs containing "
                f"'commit': {[a for a in dir(self.subtensor) if 'commit' in a.lower()][:5]}"
            )
        except Exception as e:
            raise RuntimeError(
                f"on-chain audit-root commit failed: {type(e).__name__}: {e}"
            )

        block = self._current_block()
        self.append_event({
            "type": "audit_root_committed",
            "timestamp": time.time(),
            "report_sha256": sha,
            "block": block,
        })
        return block

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
        [0, 1] and mapped to UIDs before submitting.

        Automatically uses commit-reveal if enabled on the subnet,
        otherwise falls back to direct set_weights.
        """
        self.sync()

        uids = []
        weights = []
        for hotkey, score in hotkey_scores.items():
            # Blacklisted miners get zero weight regardless of score
            # (deep_review_2026-05-31 #13: audit consequence channel).
            if self.is_blacklisted(hotkey):
                print(f"[chain] zeroing blacklisted miner {hotkey[:16]}...")
                continue
            uid = self.get_uid(hotkey)
            if uid is not None:
                uids.append(uid)
                weights.append(max(0.0, score))

        if not uids:
            print("[chain] no valid UIDs to set weights for")
            return False
        return self._submit_weight_tensors(uids, weights)

    def set_burn_weights(self) -> bool:
        """Fallback: set 100% weight to the burn UID (default 0 = subnet owner).

        Used when there is nothing real to score/audit this epoch so the
        validator (and the auditor) STILL sets weights every epoch — this keeps
        the validator's vTrust alive and burns the epoch's incentive to the
        owner uid (standard "burn to owner" pattern) instead of silently setting
        nothing. Override the target via env RALPH_BURN_UID.
        """
        import os as _os

        burn_uid = int(_os.environ.get("RALPH_BURN_UID", "0"))
        print(f"[chain] BURN fallback: 100% weight -> uid {burn_uid}")
        self.sync()
        return self._submit_weight_tensors([burn_uid], [1.0])

    def _submit_weight_tensors(self, uids: list[int], weights: list[float]) -> bool:
        """Normalize + submit one set_weights extrinsic for explicit uids/weights.

        Shared by set_weights (hotkey-mapped scores) and set_burn_weights (the
        uid-0 burn fallback) so both go through the identical rate-limit guard +
        extrinsic + event path. Caller is responsible for self.sync().
        """
        total = sum(weights) or 1.0
        weights = [w / total for w in weights]

        uid_tensor = torch.tensor(uids, dtype=torch.int64)
        weight_tensor = torch.tensor(weights, dtype=torch.float32)

        try:
            # Skip if rate limited — try again next epoch (don't block service).
            rl = self.subtensor.weights_rate_limit(netuid=self.netuid)
            my_uid = self.get_uid(self.wallet.hotkey.ss58_address)
            if my_uid is not None:
                last_update = self.metagraph.neurons[my_uid].last_update
                blocks_since = self.subtensor.get_current_block() - last_update
                if blocks_since <= rl:
                    wait_blocks = rl - blocks_since + 1
                    print(f"[chain] rate limited ({blocks_since}/{rl} blocks). "
                          f"skipping set_weights — will retry next epoch (~{wait_blocks * 12}s)")
                    return False

            # v1.2: always call set_weights — the bittensor SDK handles
            # commit-reveal internally when the subnet has it enabled, so the
            # cr_enabled branch we used to have was broken (missing salt
            # arg) and redundant. See deep_review_2026-05-31 #7.
            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.netuid,
                uids=uid_tensor,
                weights=weight_tensor,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )
            success = result.success if hasattr(result, "success") else bool(result)
            print(f"[chain] set_weights: success={success}")

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
            print(f"[chain] weight setting failed: {type(e).__name__}: {e}")
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
            king_attestation_hash=d.get("king_attestation_hash", ""),
            parent_king_attestation_hash=d.get("parent_king_attestation_hash"),
        )

    def set_king(self, king: KingRecord) -> None:
        """Persist the new king to off-chain state.

        IMPORTANT (deep_review_2026-05-31 critical #13): this used to also call
        self.set_weights({king: 1.0}) implicitly, which then collided with
        the service's end-of-epoch set_weights(round_scores), tripping the
        rate limit and silently dropping every meaningful_failure 0.1 credit.
        The service is now the single authoritative writer of weights —
        set_king ONLY updates off-chain metadata.
        """
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
        # v0.11-lite lineage fields. Omitted when empty/None for legacy
        # byte-equivalent serialization.
        if king.king_attestation_hash:
            d["king_attestation_hash"] = king.king_attestation_hash
        if king.parent_king_attestation_hash is not None:
            d["parent_king_attestation_hash"] = king.parent_king_attestation_hash
        (self.chain_dir / "king.json").write_text(json.dumps(d, indent=2, sort_keys=True))

    # ---- Audit / blacklist ----

    def blacklist(self, hotkey: str, reason: str = "") -> None:
        """Mark a miner hotkey as blacklisted. Subsequent set_weights calls
        zero its weight regardless of round_scores.

        Persisted to chain_dir/blacklist.json so the state survives validator
        restarts. The §5.7 deterrence math requires real consequences for
        audit failure — this is the consequence channel.
        """
        path = self.chain_dir / "blacklist.json"
        current = {}
        if path.exists():
            try:
                current = json.loads(path.read_text())
            except json.JSONDecodeError:
                current = {}
        current[hotkey] = {"reason": reason, "at": time.time(), "block": self._current_block()}
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        self.append_event({
            "type": "blacklisted",
            "timestamp": time.time(),
            "miner_hotkey": hotkey,
            "reason": reason,
        })

    def is_blacklisted(self, hotkey: str) -> bool:
        path = self.chain_dir / "blacklist.json"
        if not path.exists():
            return False
        try:
            current = json.loads(path.read_text())
        except json.JSONDecodeError:
            return False
        return hotkey in current

    def append_event(self, event: dict) -> None:
        if "block" not in event:
            event["block"] = self._current_block()
        _locked_append(self.chain_dir / "events.jsonl", json.dumps(event) + "\n")

    def get_events(self, limit: int = 100) -> list[dict]:
        path = self.chain_dir / "events.jsonl"
        if not path.exists():
            return []
        lines = path.read_text().splitlines()
        events = [json.loads(l) for l in lines if l.strip()]
        return list(reversed(events[-limit:]))

    def get_current_block(self) -> int:
        return int(self.subtensor.get_current_block())

    def get_block_hash(self, block: int) -> str:
        if block < 0:
            raise ValueError(f"block must be >= 0, got {block}")
        raw = self.subtensor.get_block_hash(block)
        if not isinstance(raw, str) or not raw:
            raise ValueError(f"chain returned no hash for block {block}: {raw!r}")
        body = raw[2:] if raw[:2].lower() == "0x" else raw
        body = body.lower()
        if len(body) != 64 or any(c not in "0123456789abcdef" for c in body):
            raise ValueError(f"chain returned malformed hash for block {block}: {raw!r}")
        return "0x" + body

    def _current_block(self) -> int:
        try:
            return self.get_current_block()
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
