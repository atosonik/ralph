"""
Mock attestation for Phase 0.

Stands in for the real TDX + nvtrust attestation chain we'll integrate in
Phase 0.5. The structure mirrors what the real chain will carry: a
container_measurement (= hash of the proof-test container image; here, the
hash of THIS source tree at run time), a validator-issued nonce, a
user_data field containing the bundle hash, and a signature.

The signature is HMAC-SHA256 with a key derived from a fixed "team-signing
secret" + the container_measurement. Real attestation will use TDX/SEV
quote signatures chaining to Intel/NVIDIA roots. The verification path in
validator/ is structured so that swapping HMAC for real quote verification
later is a clean substitution — nothing else changes.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

# Fixed Phase 0 "team-signing secret" used by the mock signer. In Phase 0.5+
# this is replaced by TDX/SEV-SNP hardware-root signing keys (which the
# attacker cannot forge). For now, anything holding this secret can produce
# a "valid" mock attestation — exactly the trust model of a fake signer.
_MOCK_TEAM_SECRET = b"phase0-mock-attestation-secret-DO-NOT-USE-IN-PROD"


@dataclass
class MockAttestationEpoch:
    """One epoch of a continuous-attestation chain (whitepaper §5.4)."""
    epoch: int
    timestamp: float
    rolling_log_hash: str
    nonce: str
    container_measurement: str
    signature: str  # HMAC over the above

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class MockAttestation:
    """
    The full chain: an initial handshake nonce, a sequence of per-epoch
    self-attestations during the run, and a final attestation whose
    rolling_log_hash also incorporates the submission-bundle hash.
    """
    container_measurement: str
    handshake_nonce: str
    epochs: list[MockAttestationEpoch] = field(default_factory=list)
    bundle_hash: str | None = None

    def to_dict(self) -> dict:
        return {
            "container_measurement": self.container_measurement,
            "handshake_nonce": self.handshake_nonce,
            "epochs": [e.to_dict() for e in self.epochs],
            "bundle_hash": self.bundle_hash,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "MockAttestation":
        d = json.loads(text)
        att = cls(
            container_measurement=d["container_measurement"],
            handshake_nonce=d["handshake_nonce"],
            bundle_hash=d.get("bundle_hash"),
        )
        att.epochs = [MockAttestationEpoch(**e) for e in d["epochs"]]
        return att


def _sign(payload: dict, container_measurement: str) -> str:
    """HMAC-SHA256 over a deterministic encoding of the payload."""
    key = hashlib.sha256(_MOCK_TEAM_SECRET + container_measurement.encode()).digest()
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hmac.new(key, data, hashlib.sha256).hexdigest()


def compute_container_measurement(source_files: Sequence[Path]) -> str:
    """The 'container measurement' for Phase 0: a hash of the proof-test source
    tree. In Phase 0.5+ this is replaced by the Docker image digest pinned
    on-chain.

    DEPRECATED interface — kept for backward compatibility with old call sites
    that pass a list of absolute Paths. New code should use
    proof.sources.compute_container_measurement(karpa_root, recipe_dir=...)
    which hashes repo-relative POSIX paths so different filesystem layouts
    produce the same digest. See deep_review_2026-05-31 critical #10/#11.
    """
    h = hashlib.sha256()
    # Try to make the legacy path-based hash relative when we can detect a
    # common prefix. Without a prefix we fall back to the absolute path (the
    # old behaviour), which preserves digests for any in-process caller still
    # passing pre-resolved Paths. Cross-host parity requires the new
    # proof.sources.compute_container_measurement.
    paths = [Path(p) for p in source_files]
    common = _common_parent(paths)
    for path in sorted(paths):
        key = path.relative_to(common).as_posix() if common else str(path)
        h.update(key.encode("utf-8"))
        h.update(b"\x00")
        h.update(path.read_bytes())
    return h.hexdigest()


def _common_parent(paths: Sequence[Path]) -> Path | None:
    """Return the deepest directory that is a parent of every path, or None."""
    if not paths:
        return None
    parts_list = [p.parts for p in paths]
    common: list[str] = []
    for parts in zip(*parts_list):
        if len(set(parts)) == 1:
            common.append(parts[0])
        else:
            break
    if not common:
        return None
    return Path(*common)


def attestation_epoch(
    epoch: int,
    timestamp: float,
    rolling_log_hash: str,
    nonce: str,
    container_measurement: str,
) -> MockAttestationEpoch:
    payload = {
        "epoch": epoch,
        "timestamp": timestamp,
        "rolling_log_hash": rolling_log_hash,
        "nonce": nonce,
        "container_measurement": container_measurement,
    }
    sig = _sign(payload, container_measurement)
    return MockAttestationEpoch(
        epoch=epoch,
        timestamp=timestamp,
        rolling_log_hash=rolling_log_hash,
        nonce=nonce,
        container_measurement=container_measurement,
        signature=sig,
    )


def generate_mock_attestation(
    container_measurement: str,
    handshake_nonce: str,
    epoch_records: list[tuple[int, float, str]],
    bundle_hash: str,
) -> MockAttestation:
    """
    Args:
        container_measurement: hex digest of the proof-test container/source.
        handshake_nonce: the validator-issued nonce committed on-chain at proof start.
        epoch_records: list of (epoch_index, timestamp, rolling_log_hash) tuples.
        bundle_hash: hash of the full submission bundle (patch + checkpoint + log + calib).
    """
    att = MockAttestation(
        container_measurement=container_measurement,
        handshake_nonce=handshake_nonce,
        bundle_hash=bundle_hash,
    )
    for (epoch_idx, ts, rolling_hash) in epoch_records:
        att.epochs.append(
            attestation_epoch(epoch_idx, ts, rolling_hash, handshake_nonce, container_measurement)
        )
    # Append a final epoch whose rolling-hash incorporates the bundle hash.
    if epoch_records:
        last_ts = epoch_records[-1][1] + 1.0
        last_epoch = epoch_records[-1][0] + 1
        last_rolling = hashlib.sha256(
            (epoch_records[-1][2] + bundle_hash).encode()
        ).hexdigest()
    else:
        import time
        last_ts = time.time()
        last_epoch = 0
        last_rolling = hashlib.sha256(bundle_hash.encode()).hexdigest()
    att.epochs.append(
        attestation_epoch(last_epoch, last_ts, last_rolling, handshake_nonce, container_measurement)
    )
    return att


def verify_mock_attestation(
    att: MockAttestation,
    expected_container_measurement: str,
    expected_handshake_nonce: str,
    expected_bundle_hash: str,
) -> tuple[bool, list[str]]:
    """
    Validator-side verification. Returns (ok, errors).
    Mirrors what the real TDX+nvtrust verification path will do in Phase 0.5+:
      1. Container measurement matches the on-chain pinned value.
      2. Handshake nonce matches the value committed on-chain at proof start.
      3. Every epoch's signature is valid against the container's measurement.
      4. Every epoch carries the same handshake nonce.
      5. The final epoch's rolling-hash includes the bundle hash.
    """
    errors: list[str] = []
    if att.container_measurement != expected_container_measurement:
        errors.append(
            f"container measurement mismatch (got {att.container_measurement[:16]}, "
            f"expected {expected_container_measurement[:16]})"
        )
    if att.handshake_nonce != expected_handshake_nonce:
        errors.append("handshake nonce mismatch")
    if att.bundle_hash != expected_bundle_hash:
        errors.append("bundle hash mismatch")
    if not att.epochs:
        errors.append("attestation chain has no epochs")
        return False, errors
    for i, ep in enumerate(att.epochs):
        if ep.nonce != att.handshake_nonce:
            errors.append(f"epoch {i} nonce drift")
        if ep.container_measurement != att.container_measurement:
            errors.append(f"epoch {i} container measurement drift")
        payload = {
            "epoch": ep.epoch,
            "timestamp": ep.timestamp,
            "rolling_log_hash": ep.rolling_log_hash,
            "nonce": ep.nonce,
            "container_measurement": ep.container_measurement,
        }
        expected_sig = _sign(payload, att.container_measurement)
        if not hmac.compare_digest(expected_sig, ep.signature):
            errors.append(f"epoch {i} signature invalid")
    # Final-epoch rolling hash must incorporate the bundle hash.
    final_rolling = att.epochs[-1].rolling_log_hash
    if att.bundle_hash and att.bundle_hash not in final_rolling:
        # rolling_log_hash is itself a hash; we check that recomputing matches
        # by re-deriving from the prior epoch's rolling hash (if any) + bundle hash.
        if len(att.epochs) >= 2:
            prior = att.epochs[-2].rolling_log_hash
            expected_final = hashlib.sha256((prior + att.bundle_hash).encode()).hexdigest()
            if expected_final != final_rolling:
                errors.append("final-epoch rolling hash does not include the bundle hash")
        else:
            expected_final = hashlib.sha256(att.bundle_hash.encode()).hexdigest()
            if expected_final != final_rolling:
                errors.append("final-epoch rolling hash does not match bundle hash")
    return len(errors) == 0, errors
