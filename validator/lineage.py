"""v0.11-lite lineage primitives — attestation hashing + parent-cache verification.

The Karpa v0.11 protocol replaces the v0.10 nested `KingRecord.previous_king`
dict with a flat cryptographic pointer `parent_king_attestation_hash`. Each
king carries its OWN `king_attestation_hash` (sha256 of canonical attestation
payload) and points back to its parent's hash via
`parent_king_attestation_hash`. Walking the chain back to genesis (where
`parent_king_attestation_hash is None`) reconstructs the full lineage.

For v0.11-lite (the B6 sprint scope), validators do NO GPU work to verify a
parent. Instead, the child miner reproduces the parent's CSDP on their own
GPU and bundles `parent_csdp_cache.json` (the parent's scores plus a quorum
signature from the parent-time validators). The validator's only work is a
SIGNATURE CHECK against the cached payload — option (c) from the protocol
design that preserves the "validators do minimal work" axiom.

What this module ships (v0.11-lite):

  * `compute_king_attestation_hash(payload)` — canonical sha256 of an
    attestation payload (JSON sort_keys, separators tight, signatures
    appended after hashing so signatures don't change the hash).
  * `ParentCsdpCache` — frozen dataclass for the cached parent payload.
    Includes the parent_king_attestation_hash, CSDP summary, age fields,
    and quorum signature list.
  * `verify_parent_lineage(parent_attestation_hash, parent_csdp_cache,
    chain, *, now_iso, cache_age_threshold_days=14)` — the main
    verification entry point.

What this module does NOT ship (v0.11-full, post-B6 PASS):

  * Real ed25519 quorum signature verification with rotating validator
    pubkey sets — for v0.11-lite, signatures are structurally validated
    only (presence + format), and the LOCAL chain backend trusts
    well-formed caches. Cryptographic verification with bonded validator
    quorums lands in C5-PASS-FULL.
  * NOW-quorum age gate (>= 14 days requires fresh 2/3 NOW-validator
    co-sign) — the age check is implemented here, but the "fresh
    signature" path raises a clean rejection rather than re-requesting
    signatures from the live quorum.
  * Sealed-shard accumulator binding (corpus_root_hash, shard_root_hash
    in the attestation payload's TDX user_data) — v0.11-full only.

Reference: docs/rearch_2026_06/00_v0_11_master.md §2.2 Lineage Chain.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from chain_layer.interface import ChainInterface

# Field that contains the quorum signatures inside an attestation payload.
# Hashing skips this field so signatures can be appended after the hash is
# computed (matching the v0.11 design where multiple validators co-sign
# the same content-addressed payload).
_QUORUM_SIG_FIELD = "validator_quorum_sig"

# Attestation hash format: lowercase hex of sha256, 64 chars total.
ATTESTATION_HASH_LEN = 64

# v0.11-lite cache age threshold. Caches older than this require a fresh
# NOW-quorum signature; under v0.11-lite that path raises a clean
# rejection (real fresh-signature negotiation is C5-PASS-FULL).
DEFAULT_CACHE_AGE_THRESHOLD_DAYS = 14


# ----------------------------------------------------------------------------
# Attestation hash
# ----------------------------------------------------------------------------


def _canonical_payload_bytes(payload: dict) -> bytes:
    """Serialize a dict for hashing — sort_keys=True, tight separators, UTF-8.

    The validator_quorum_sig field (if present) is REMOVED before hashing
    so signatures can be appended after content is content-addressed.
    """
    stripped = {k: v for k, v in payload.items() if k != _QUORUM_SIG_FIELD}
    return json.dumps(
        stripped,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def compute_king_attestation_hash(payload: dict) -> str:
    """sha256 of the canonical attestation payload (without quorum sig).

    The output is the 64-char lowercase hex digest that the protocol
    stores in `KingRecord.king_attestation_hash` and in chain events.
    Two attestation payloads with identical content (modulo their
    signature lists) produce identical hashes.
    """
    return hashlib.sha256(_canonical_payload_bytes(payload)).hexdigest()


def is_valid_attestation_hash(value: Any) -> bool:
    """True if `value` is a 64-char lowercase hex string."""
    if not isinstance(value, str) or len(value) != ATTESTATION_HASH_LEN:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


# ----------------------------------------------------------------------------
# Parent CSDP cache
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class ParentCsdpCache:
    """The cached parent-CSDP payload the child miner bundles.

    Produced when the child miner re-executes the parent recipe on its own
    GPU. The parent-time validator quorum that originally crowned the
    parent signs the resulting payload; the child includes those
    signatures in its submission so the validator that scores the child
    can verify the parent's CSDP without re-running it.

    Schema:
      * `parent_king_attestation_hash` — the parent we re-evaluated;
        must match the child's submission.parent_king_attestation_hash.
      * `csdp_summary` — dict of per-rung overall scores plus per-axis
        breakdown (e.g. `{"s1_overall", "s2_overall", "s3_overall",
        "core22_s3", "private_hard_s3", "val_bpb_s2"}`). Free-form for
        v0.11-lite; v0.11-full pins a stricter schema.
      * `streams_root_hash_at_evaluation` — the manifest root the
        parent's eval ran against; lets the child prove it used the
        same sealed-stream config as the parent.
      * `cached_at_iso` — wall-clock timestamp when the cache was
        produced. Compared to `now_iso` for the age gate.
      * `cached_at_block` — chain block at cache time; supports the
        future NOW-quorum re-signing path.
      * `quorum_signatures` — list of dicts (one per signer) with
        `{"signer": "<ss58 or pubkey>", "sig": "<hex>"}`. v0.11-lite
        validates structure only; v0.11-full ed25519-verifies each.
    """

    parent_king_attestation_hash: str
    csdp_summary: dict
    streams_root_hash_at_evaluation: str
    cached_at_iso: str
    cached_at_block: int
    quorum_signatures: list[dict] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> ParentCsdpCache:
        """Inverse of `to_dict`. Required-field violations raise KeyError;
        type violations raise ValueError at field access."""
        return cls(
            parent_king_attestation_hash=str(d["parent_king_attestation_hash"]),
            csdp_summary=dict(d["csdp_summary"]),
            streams_root_hash_at_evaluation=str(d["streams_root_hash_at_evaluation"]),
            cached_at_iso=str(d["cached_at_iso"]),
            cached_at_block=int(d["cached_at_block"]),
            quorum_signatures=list(d.get("quorum_signatures", [])),
        )

    def to_dict(self) -> dict:
        return {
            "parent_king_attestation_hash": self.parent_king_attestation_hash,
            "csdp_summary": dict(self.csdp_summary),
            "streams_root_hash_at_evaluation": self.streams_root_hash_at_evaluation,
            "cached_at_iso": self.cached_at_iso,
            "cached_at_block": self.cached_at_block,
            "quorum_signatures": list(self.quorum_signatures),
        }


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------


def _parse_iso(iso: str) -> datetime:
    """Parse ISO 8601 timestamp (UTC; trailing Z accepted).

    The chain stores cached_at_iso in `YYYY-MM-DDTHH:MM:SSZ` form; older
    payloads may use `+00:00`. Both are accepted.
    """
    s = iso.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def _age_days(cached_iso: str, now_iso: str) -> float:
    """Wall-clock age of a cache in days (floats; negative when cache is
    in the future, which raises in the caller).
    """
    return (_parse_iso(now_iso) - _parse_iso(cached_iso)).total_seconds() / 86400.0


def _structural_signature_check(sigs: list[dict]) -> tuple[bool, str]:
    """v0.11-lite structural validation of the quorum signature list.

    Each entry must be `{"signer": <non-empty str>, "sig": <hex str>}`.
    At least one signature must be present (genesis is the exception,
    handled by the caller before this is called).

    v0.11-full replaces this with ed25519 verification against a
    chain-derived validator pubkey set.
    """
    if not sigs:
        return False, "no_quorum_signatures"
    for i, entry in enumerate(sigs):
        if not isinstance(entry, dict):
            return False, f"signature_entry_{i}_not_dict"
        signer = entry.get("signer")
        sig = entry.get("sig")
        if not isinstance(signer, str) or not signer:
            return False, f"signature_entry_{i}_bad_signer"
        if not isinstance(sig, str) or not sig:
            return False, f"signature_entry_{i}_bad_sig"
        try:
            int(sig, 16)
        except ValueError:
            return False, f"signature_entry_{i}_sig_not_hex"
    return True, ""


def verify_parent_lineage(
    parent_attestation_hash: Optional[str],
    parent_csdp_cache: Optional[ParentCsdpCache],
    chain: ChainInterface,
    *,
    now_iso: str,
    cache_age_threshold_days: int = DEFAULT_CACHE_AGE_THRESHOLD_DAYS,
) -> tuple[bool, str]:
    """Validate a child submission's parent-CSDP cache.

    Returns `(ok, reason)`:
      * `(True, "")` if the lineage is valid (genesis OR cached payload
        verifies cleanly).
      * `(False, "<reason>")` on any failure; reason is a short
        underscore-snake-case tag the validator emits to the chain as
        `parent_unverifiable:<reason>`.

    Args:
      parent_attestation_hash: from `submission.parent_king_attestation_hash`.
        `None` indicates genesis; both `None` and `""` are treated as
        genesis for tolerance.
      parent_csdp_cache: the cached payload bundled by the miner. Required
        when `parent_attestation_hash` is non-empty; `None` at genesis.
      chain: the live chain interface for parent-king lookup.
      now_iso: wall-clock ISO 8601 timestamp for the age gate.
      cache_age_threshold_days: caches older than this require a fresh
        NOW-quorum signature; v0.11-lite rejects with `cache_too_old`
        (real re-signing path is C5-PASS-FULL).

    Failure tags (stable for chain event consumers):
      genesis_ok                       — no parent; submission accepted
      missing_parent_cache             — parent hash given but no cache
      bad_parent_hash_format           — hash not 64-char lowercase hex
      parent_hash_mismatch             — cache parent != submission parent
      parent_not_on_chain              — parent not found in chain history
      no_quorum_signatures             — empty signature list (post-genesis)
      signature_entry_<i>_bad_*        — structural signature issue
      cache_in_future                  — cached_at_iso > now_iso
      cache_too_old                    — beyond cache_age_threshold_days
    """
    # Genesis: no parent, no cache required. The CALLER must ensure
    # genesis submissions are otherwise valid (operator's GenesisAttestation
    # event signed separately).
    if not parent_attestation_hash:
        if parent_csdp_cache is not None:
            return False, "unexpected_parent_cache_at_genesis"
        return True, "genesis_ok"

    if not is_valid_attestation_hash(parent_attestation_hash):
        return False, "bad_parent_hash_format"

    if parent_csdp_cache is None:
        return False, "missing_parent_cache"

    if parent_csdp_cache.parent_king_attestation_hash != parent_attestation_hash:
        return False, "parent_hash_mismatch"

    # Parent must exist somewhere in chain history. v0.11-lite walks the
    # last N events; v0.11-full uses a maintained lineage index.
    if not _lookup_parent_in_chain(parent_attestation_hash, chain):
        return False, "parent_not_on_chain"

    # Cache age gate.
    try:
        age = _age_days(parent_csdp_cache.cached_at_iso, now_iso)
    except ValueError:
        return False, "bad_cached_at_iso"
    if age < 0:
        return False, "cache_in_future"
    if age > cache_age_threshold_days:
        # v0.11-full requires a fresh NOW-quorum co-sign here. v0.11-lite
        # rejects loudly so the operator sees the issue rather than
        # silently degrading trust.
        return False, "cache_too_old"

    # Structural signature validation (v0.11-lite). v0.11-full ed25519
    # verifies each signer against a chain-derived validator pubkey set.
    ok, reason = _structural_signature_check(parent_csdp_cache.quorum_signatures)
    if not ok:
        return False, reason

    return True, ""


def _lookup_parent_in_chain(
    parent_attestation_hash: str,
    chain: ChainInterface,
    *,
    event_scan_limit: int = 10_000,
) -> bool:
    """Return True iff a `king_crowned` event with this attestation hash
    has been emitted.

    v0.11-lite scans the last `event_scan_limit` events linearly. v0.11-full
    maintains an indexed lineage tree at `validator/state/lineage_state.json`
    for O(1) lookup. The linear scan is acceptable for testnet 16 where
    crowning events number in the hundreds, not the millions.

    Also recognises a current king (read via `chain.get_king()`) whose
    `king_attestation_hash` matches — useful for fresh chains where the
    crowning event may not have been re-indexed yet.
    """
    # Fast path: current king.
    current = chain.get_king()
    if current is not None and current.king_attestation_hash == parent_attestation_hash:
        return True
    # Event scan: any KingCrowned event referencing this hash.
    for event in chain.get_events(limit=event_scan_limit):
        if event.get("type") != "king_crowned":
            continue
        if event.get("king_attestation_hash") == parent_attestation_hash:
            return True
    return False
