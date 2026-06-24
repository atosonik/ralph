"""
Abstract interface for chain operations.

Every operation the Ralph protocol needs from "the chain":
  1. Handshake: miner announces proof-test intent, gets a nonce
  2. Hotkey verification: confirm a hotkey is registered on the subnet
  3. Weight setting: validator publishes scores for miners
  4. King state: who is the current canonical-baseline king
  5. Event logging: append protocol events for auditability
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class HandshakeRecord:
    nonce: str
    miner_hotkey: str
    patch_hash: str
    timestamp: float


@dataclass
class KingRecord:
    """One king's record on the chain.

    v0.11-lite lineage fields:
      * `king_attestation_hash` — the king's OWN attestation hash (64-char
        lowercase hex, sha256 of canonical attestation payload). The
        primary cryptographic identifier for lineage chaining. Defaults
        to "" for legacy / pre-v0.11 records.
      * `parent_king_attestation_hash` — the prior king's attestation
        hash that this king built on top of. None at genesis. Replaces
        the v0.10 nested `previous_king` dict with a flat 64-char hex
        pointer that's cheap to verify and chain-walk.

    `previous_king` is retained for v0.10 byte-equivalent legacy
    serialization but is NOT consulted by v0.11 lineage verification.
    """

    miner_hotkey: str
    bundle_hash: str
    val_bpb: float
    benchmark_accuracy: float
    compute_cost: float
    crowned_at: float
    proof_dir: Optional[str] = None
    # Block height at which this king was crowned. Enforces a minimum reign
    # (RALPH_KING_MIN_TENURE_BLOCKS) so a new king earns ≥1 weight cycle before
    # it can be dethroned. 0 = legacy/unknown (treated as no protection).
    crowned_at_block: int = 0
    previous_king: Optional[dict] = None
    king_attestation_hash: str = ""
    parent_king_attestation_hash: Optional[str] = None


class ChainInterface(ABC):

    @abstractmethod
    def request_handshake_nonce(self, miner_hotkey: str, patch_hash: str) -> str:
        """Commit a proof-test intent on-chain, return a fresh nonce."""

    @abstractmethod
    def lookup_handshake(self, nonce: str) -> Optional[HandshakeRecord]:
        """Verify a nonce was committed. Returns the record or None."""

    def verify_handshake_onchain(
        self, miner_hotkey: str, patch_hash: str, nonce: str
    ) -> tuple[bool, str]:
        """Verify a miner's handshake binds (hotkey, patch_hash, nonce).

        Default implementation uses the local `lookup_handshake` record;
        BittensorChain overrides this to query the live on-chain commitment.
        """
        rec = self.lookup_handshake(nonce)
        if rec is None:
            return False, "handshake nonce not found on chain"
        if rec.miner_hotkey != miner_hotkey:
            return False, "handshake nonce was committed by a different miner"
        if patch_hash and rec.patch_hash and rec.patch_hash != patch_hash:
            return False, (
                f"on-chain patch_hash mismatch: chain={rec.patch_hash[:12]}, "
                f"bundle={patch_hash[:12]}"
            )
        return True, "handshake verified"

    @abstractmethod
    def is_hotkey_registered(self, hotkey: str) -> bool:
        """Check if a hotkey is registered on the subnet."""

    @abstractmethod
    def set_weights(self, hotkey_scores: dict[str, float]) -> bool:
        """Validator publishes weights for miners. Returns success."""

    @abstractmethod
    def get_king(self) -> Optional[KingRecord]:
        """Return the current canonical-baseline king."""

    @abstractmethod
    def set_king(self, king: KingRecord) -> None:
        """Update the king after a successful merge."""

    @abstractmethod
    def append_event(self, event: dict) -> None:
        """Append an auditable event to the protocol log."""

    @abstractmethod
    def get_events(self, limit: int = 100) -> list[dict]:
        """Return recent events, newest first."""

    @abstractmethod
    def get_current_block(self) -> int:
        """Return the current chain block height — monotonically non-decreasing.

        LocalChain treats this as the number of events appended so far;
        BittensorChain wraps subtensor.get_current_block().
        """

    @abstractmethod
    def get_block_hash(self, block: int) -> str:
        """Return a deterministic hex hash for the given block height.

        Used as a randomness seed source for validator-controlled operations:
        multi-seed eval (Track A v0.10), EpochSeeds derivation for on-chain
        stream selection (Cross-Scale Downstream Pareto), and any other gate
        that must derive randomness no single miner can fake.

        Properties:
          - Deterministic: same block → same hash, forever
          - Public: derivable by any node observing the chain
          - Unfakeable: no single miner can choose the value

        Format: 64 lowercase hex chars prefixed with "0x" (66 chars total).
        Raises ValueError if `block` is negative or exceeds the current height.
        """

    def commit_audit_root(self, sha256_hex: str) -> int:
        """Anchor a per-epoch audit-report hash on-chain (validation-v2 Phase 1).

        `sha256_hex` is the 64-char lowercase hex sha256 of the canonical
        report_json. Backends MUST raise on failure (do NOT swallow) so an
        audit-report block that can't anchor surfaces loudly to the caller,
        which then logs-and-continues without breaking weight-setting.

        Returns the block height the commitment landed at.

        Default no-op for backends that don't anchor (returns the current
        block) so the interface stays incrementally adoptable.
        """
        return self.get_current_block()

    def blacklist(self, hotkey: str, reason: str = "") -> None:
        """Mark a miner-hotkey as blacklisted. Subsequent set_weights MUST
        zero its weight regardless of round_scores. Default no-op so
        backends can opt in incrementally."""
        pass

    def is_blacklisted(self, hotkey: str) -> bool:
        """Check if a miner-hotkey is blacklisted. Default False."""
        return False
