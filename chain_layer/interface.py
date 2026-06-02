"""
Abstract interface for chain operations.

Every operation the Karpa protocol needs from "the chain":
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
    miner_hotkey: str
    bundle_hash: str
    val_bpb: float
    benchmark_accuracy: float
    compute_cost: float
    crowned_at: float
    proof_dir: Optional[str] = None
    previous_king: Optional[dict] = None


class ChainInterface(ABC):

    @abstractmethod
    def request_handshake_nonce(self, miner_hotkey: str, patch_hash: str) -> str:
        """Commit a proof-test intent on-chain, return a fresh nonce."""

    @abstractmethod
    def lookup_handshake(self, nonce: str) -> Optional[HandshakeRecord]:
        """Verify a nonce was committed. Returns the record or None."""

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

    def blacklist(self, hotkey: str, reason: str = "") -> None:
        """Mark a miner-hotkey as blacklisted. Subsequent set_weights MUST
        zero its weight regardless of round_scores. Default no-op so
        backends can opt in incrementally."""
        pass

    def is_blacklisted(self, hotkey: str) -> bool:
        """Check if a miner-hotkey is blacklisted. Default False."""
        return False
