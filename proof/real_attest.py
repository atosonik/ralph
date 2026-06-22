"""
Real hardware attestation — TDX (CPU) + nvtrust (GPU).

Replaces mock_attest.py when running on a CC-capable H100 with Intel TDX.
Falls back to mock gracefully when the hardware or libraries aren't available.

Miner side (evidence generation):
    - NVIDIA nv-attestation-sdk: GPU attestation quote via nvtrust
    - Intel TDX: CPU attestation quote via /dev/tdx-guest or trustauthority SDK

Validator side (verification):
    - NVIDIA NRAS (Remote Attestation Service) or local GPU verifier
    - Intel Trust Authority or local TDX quote verification
    - Both produce signed JWTs that can be verified offline with cached root certs

Wire format matches mock_attest.py's RealAttestation dataclass so the
validator's verification path is a clean swap.

Dependencies (optional — graceful fallback when not installed):
    pip install nv-attestation-sdk  # NVIDIA GPU attestation
    # Intel TDX tools installed via system packages on TDX-capable hosts
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Optional

# ============================================================================
# Data structures — same shape as mock_attest.py for validator compatibility
# ============================================================================

@dataclass
class AttestationEpoch:
    epoch: int
    timestamp: float
    rolling_log_hash: str
    nonce: str
    container_measurement: str
    # Real attestation fields (None when using mock)
    gpu_evidence: Optional[str] = None
    gpu_token: Optional[str] = None
    tdx_quote: Optional[str] = None
    tdx_token: Optional[str] = None
    # Mock fallback
    mock_signature: Optional[str] = None
    attestation_type: str = "mock"  # "real" | "mock"

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RealAttestation:
    container_measurement: str
    handshake_nonce: str
    attestation_type: str  # "real_tdx_nvcc" | "real_nvcc_only" | "mock"
    epochs: list[AttestationEpoch] = field(default_factory=list)
    bundle_hash: Optional[str] = None
    gpu_name: Optional[str] = None
    tdx_available: bool = False
    nvcc_available: bool = False

    def to_dict(self) -> dict:
        return {
            "container_measurement": self.container_measurement,
            "handshake_nonce": self.handshake_nonce,
            "attestation_type": self.attestation_type,
            "epochs": [e.to_dict() for e in self.epochs],
            "bundle_hash": self.bundle_hash,
            "gpu_name": self.gpu_name,
            "tdx_available": self.tdx_available,
            "nvcc_available": self.nvcc_available,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RealAttestation":
        d = json.loads(text)
        epochs = [AttestationEpoch(**e) for e in d.pop("epochs", [])]
        att = cls(epochs=epochs, **d)
        return att


# ============================================================================
# Hardware detection
# ============================================================================

def detect_tdx() -> bool:
    """Check if Intel TDX is available (running inside a TD guest)."""
    return os.path.exists("/dev/tdx-guest") or os.path.exists("/dev/tdx_guest")


def detect_nvcc() -> bool:
    """Check if NVIDIA Confidential Computing SDK is available."""
    try:
        from nv_attestation_sdk import attestation
        return True
    except ImportError:
        return False


def detect_capabilities() -> dict:
    """Detect what attestation capabilities are available on this machine."""
    tdx = detect_tdx()
    nvcc = detect_nvcc()
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return {
        "tdx": tdx,
        "nvcc": nvcc,
        "gpu_name": gpu_name,
        "attestation_type": (
            "real_tdx_nvcc" if (tdx and nvcc)
            else "real_nvcc_only" if nvcc
            else "mock"
        ),
    }


# ============================================================================
# Evidence generation (miner side — runs inside the CVM)
# ============================================================================

def gpu_sdk_nonce(nonce: str) -> str:
    """Canonical nonce form for the NVIDIA attestation SDK / GPU token claim.

    The handshake nonce is committed on-chain as ``"0x" + 64 hex`` (66 chars),
    but nv-attestation-sdk requires exactly 64 hex chars with NO prefix
    ("Invalid Nonce Size" otherwise). Strip a leading 0x and lowercase so the
    GENERATED token's nonce claim and the VALIDATOR's comparison use the same
    form. (Issue 3 of the 2026-06-22 CC-hardware report.)

    NOTE: TDX report_data binding intentionally keeps the FULL handshake nonce
    (see ``_get_tdx_quote``) — do not route TDX through here.
    """
    n = nonce.strip()
    if n[:2].lower() == "0x":
        n = n[2:]
    return n.lower()


def _get_gpu_evidence(nonce: str) -> tuple[Optional[str], Optional[str]]:
    """Generate NVIDIA GPU attestation evidence + verify via NRAS.

    Returns (evidence_hex, jwt_token) or (None, None) if unavailable.
    """
    try:
        from nv_attestation_sdk import attestation

        # Issue 1 (2026-06-22 CC report): Attestation() with no name makes
        # nv-attestation-sdk 2.7.3 return an empty get_token(). A name is
        # required for a non-empty EAT JWT.
        client = attestation.Attestation("ralph-miner")
        # Issue 3: SDK rejects the 0x-prefixed 66-char nonce; pass 64 hex.
        client.set_nonce(gpu_sdk_nonce(nonce))
        client.add_verifier(
            attestation.Devices.GPU,
            attestation.Environment.REMOTE,
            "https://nras.attestation.nvidia.com/v4/attest/gpu",
            "",
        )
        evidence_list = client.get_evidence()
        if not evidence_list:
            return None, None
        evidence_hex = evidence_list[0].hex() if isinstance(evidence_list[0], bytes) else str(evidence_list[0])
        client.attest(evidence_list)
        token = client.get_token()
        return evidence_hex, token
    except Exception as e:
        print(f"[attest] GPU evidence generation failed: {e}")
        return None, None


def _get_tdx_quote(nonce: str, user_data: str) -> tuple[Optional[str], Optional[str]]:
    """Generate Intel TDX attestation quote.

    The quote's report_data field carries hash(nonce || user_data) so the
    validator can verify the quote is bound to this specific submission.

    Returns (quote_hex, verification_token) or (None, None) if unavailable.
    """
    if not detect_tdx():
        return None, None

    report_data = hashlib.sha256((nonce + user_data).encode()).digest()

    # Method 1: configfs-tsm interface (portable across kernel versions).
    # Issue 2 (2026-06-22 CC report): creating the report node under
    # /sys/kernel/config/tsm/report needs root, but the proof test runs as a
    # non-root user. An operator can PRE-PROVISION a writable report node and
    # point us at it via RALPH_TSM_REPORT_PATH (e.g. root pre-creates
    # /sys/kernel/config/tsm/report/ralph and chowns it to the run user before
    # the proof harness starts). If unset we still attempt makedirs (works when
    # the harness has the privilege) and emit an actionable error if not.
    preprov = os.environ.get("RALPH_TSM_REPORT_PATH", "").strip()
    tsm_path = preprov or "/sys/kernel/config/tsm/report/ralph"
    created = False
    try:
        if not preprov:
            try:
                os.makedirs(tsm_path, exist_ok=True)
                created = True
            except PermissionError:
                print(
                    "[attest] TDX quote: cannot create the configfs-tsm report node "
                    f"at {tsm_path} (needs root). Run the quote step with privilege, "
                    "or have root pre-create+chown a node and set "
                    "RALPH_TSM_REPORT_PATH to it. Falling back to trustauthority-cli."
                )
                raise
        with open(f"{tsm_path}/inblob", "wb") as f:
            f.write(report_data[:64].ljust(64, b"\0"))
        with open(f"{tsm_path}/outblob", "rb") as f:
            quote_bytes = f.read()
        quote_hex = quote_bytes.hex()
        if created:
            try:
                os.rmdir(tsm_path)
            except Exception:
                pass
        return quote_hex, None  # No JWT for local TDX; validator verifies the raw quote
    except PermissionError:
        pass  # already logged; fall through to Method 2
    except Exception as e:
        print(f"[attest] TDX quote generation (configfs) failed: {e}")

    try:
        # Method 2: Intel Trust Authority client (if installed)
        result = subprocess.run(
            ["trustauthority-cli", "quote", "--nonce", nonce, "--user-data", user_data[:64]],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip(), None
    except Exception as e:
        print(f"[attest] trustauthority-cli failed: {e}")

    return None, None


# ============================================================================
# Attestation chain generation (called by proof runner)
# ============================================================================

def generate_attestation(
    container_measurement: str,
    handshake_nonce: str,
    epoch_records: list[tuple[int, float, str]],
    bundle_hash: str,
) -> RealAttestation:
    """Generate a real or mock attestation chain depending on hardware capabilities.

    Args:
        container_measurement: hash of the proof-test container/source
        handshake_nonce: validator-issued nonce committed on-chain
        epoch_records: list of (epoch_idx, timestamp, rolling_log_hash) tuples
        bundle_hash: hash of the full submission bundle
    """
    caps = detect_capabilities()
    att_type = caps["attestation_type"]

    att = RealAttestation(
        container_measurement=container_measurement,
        handshake_nonce=handshake_nonce,
        attestation_type=att_type,
        gpu_name=caps["gpu_name"],
        tdx_available=caps["tdx"],
        nvcc_available=caps["nvcc"],
        bundle_hash=bundle_hash,
    )

    for (epoch_idx, ts, rolling_hash) in epoch_records:
        user_data = f"{container_measurement}:{rolling_hash}:{handshake_nonce}"
        epoch = AttestationEpoch(
            epoch=epoch_idx,
            timestamp=ts,
            rolling_log_hash=rolling_hash,
            nonce=handshake_nonce,
            container_measurement=container_measurement,
            attestation_type=att_type,
        )

        if caps["nvcc"]:
            gpu_ev, gpu_tok = _get_gpu_evidence(handshake_nonce)
            epoch.gpu_evidence = gpu_ev
            epoch.gpu_token = gpu_tok

        if caps["tdx"]:
            tdx_q, tdx_t = _get_tdx_quote(handshake_nonce, user_data)
            epoch.tdx_quote = tdx_q
            epoch.tdx_token = tdx_t

        if att_type == "mock":
            from .mock_attest import _sign
            payload = {
                "epoch": epoch_idx, "timestamp": ts,
                "rolling_log_hash": rolling_hash,
                "nonce": handshake_nonce,
                "container_measurement": container_measurement,
            }
            epoch.mock_signature = _sign(payload, container_measurement)

        att.epochs.append(epoch)

    # Final epoch incorporating bundle_hash
    if epoch_records:
        last_ts = epoch_records[-1][1] + 1.0
        last_epoch = epoch_records[-1][0] + 1
        last_rolling = hashlib.sha256(
            (epoch_records[-1][2] + bundle_hash).encode()
        ).hexdigest()
    else:
        last_ts = time.time()
        last_epoch = 0
        last_rolling = hashlib.sha256(bundle_hash.encode()).hexdigest()

    final_user_data = f"{container_measurement}:{last_rolling}:{handshake_nonce}:{bundle_hash}"
    final_epoch = AttestationEpoch(
        epoch=last_epoch,
        timestamp=last_ts,
        rolling_log_hash=last_rolling,
        nonce=handshake_nonce,
        container_measurement=container_measurement,
        attestation_type=att_type,
    )

    if caps["nvcc"]:
        gpu_ev, gpu_tok = _get_gpu_evidence(handshake_nonce)
        final_epoch.gpu_evidence = gpu_ev
        final_epoch.gpu_token = gpu_tok

    if caps["tdx"]:
        tdx_q, tdx_t = _get_tdx_quote(handshake_nonce, final_user_data)
        final_epoch.tdx_quote = tdx_q
        final_epoch.tdx_token = tdx_t

    if att_type == "mock":
        from .mock_attest import _sign
        payload = {
            "epoch": last_epoch, "timestamp": last_ts,
            "rolling_log_hash": last_rolling,
            "nonce": handshake_nonce,
            "container_measurement": container_measurement,
        }
        final_epoch.mock_signature = _sign(payload, container_measurement)

    att.epochs.append(final_epoch)
    return att


# ============================================================================
# Verification (validator side)
# ============================================================================

def _extract_gpu_jwt(token):
    """Extract the outer JWT string from the nv-attestation-sdk get_token() value.

    get_token() returns the detached-EAT BUNDLE, not a bare JWT:
        [["JWT", <outer_jwt>], {"GPU-0": <detached_jwt>, ...}]
    (or a JSON string of that). The stub used to feed the whole bundle to
    jwt.decode → "Invalid header string" (the 2026-06-22 CC-hardware report).
    Returns the outer JWT string; falls back to the input unchanged if it
    already looks like a bare JWT (back-compat + unit-test fixtures).

    NOTE: Part B (real NRAS verify) parses the FULL bundle — outer + each
    detached per-GPU token + the submods digest binding. This helper only
    pulls the outer JWT for the testnet stub's best-effort nonce check.
    """
    import json as _json

    val = token
    if isinstance(val, str):
        s = val.strip()
        if not s.startswith("[") and not s.startswith("{"):
            return s  # already a bare JWT
        try:
            val = _json.loads(s)
        except Exception:
            return token
    if isinstance(val, (list, tuple)) and val:
        head = val[0]
        if isinstance(head, (list, tuple)) and len(head) >= 2 and isinstance(head[1], str):
            return head[1]
        if isinstance(head, str):
            return head
    return token


def verify_gpu_token(token: str, expected_nonce: str) -> tuple[bool, str]:
    """Verify an NVIDIA GPU attestation JWT token.

    Fail-closed implementation (deep_review_2026-05-31 #3): the previous
    code returned True for any non-empty string when PyJWT was missing, and
    when PyJWT was present it decoded with options={"verify_signature": False}
    — i.e. accepted any token whose nonce claim happened to match. Either is
    a free verified-tier pass for anyone with a JSON editor.

    Until the NRAS JWKS verification is wired (TODO: real_nvcc_only on real
    H100-CC silicon), this function REFUSES verified-tier attestation. To
    allow a loud-warning mock-style acceptance on testnet, set the env var
    RALPH_ALLOW_REAL_ATTEST_STUB=1 — but never set this on mainnet.
    """
    import os as _os
    if not token:
        return False, "empty GPU token"
    if _os.environ.get("RALPH_ALLOW_REAL_ATTEST_STUB") == "1":
        # Loud-warning stub for testnet only. The NRAS signature is not
        # verified; the nonce binding is best-effort.
        import sys as _sys
        print(
            "[attest] WARNING: RALPH_ALLOW_REAL_ATTEST_STUB=1 — accepting "
            "real_* attestation without NRAS JWKS signature verification. "
            "MUST NOT BE SET ON MAINNET.",
            file=_sys.stderr,
        )
        try:
            import jwt
            # get_token() returns the detached-EAT bundle; decode the OUTER JWT,
            # not the whole bundle (fixes "Invalid header string").
            inner = _extract_gpu_jwt(token)
            claims = jwt.decode(inner, options={"verify_signature": False})
            token_nonce = claims.get("eat_nonce", claims.get("nonce", ""))
            # Compare in the SDK's 64-hex form (token claim has no 0x prefix).
            exp = gpu_sdk_nonce(expected_nonce) if expected_nonce else ""
            if exp and gpu_sdk_nonce(token_nonce) != exp:
                return False, (
                    f"nonce mismatch in GPU token (expected {exp[:16]}, "
                    f"got {gpu_sdk_nonce(token_nonce)[:16]})"
                )
            return True, "GPU token accepted (stub: signature unchecked)"
        except ImportError:
            return False, (
                "PyJWT not installed; cannot even nonce-check the GPU token. "
                "Install PyJWT or wire real NRAS JWKS verification."
            )
        except Exception as e:
            return False, f"GPU token decode failed: {e}"

    # Production path: we don't have NRAS JWKS verification wired yet.
    # Fail-closed. Wire jwt verification against NVIDIA's published JWKS
    # before flipping this to a verified result.
    return False, (
        "GPU token signature verification not implemented. "
        "Wire NRAS JWKS verification before accepting real_nvcc_only "
        "attestation, or set RALPH_ALLOW_REAL_ATTEST_STUB=1 (testnet only)."
    )


def verify_tdx_quote(quote_hex: str, expected_nonce: str, expected_measurement: str) -> tuple[bool, str]:
    """Verify an Intel TDX quote.

    Fail-closed implementation (deep_review_2026-05-31 #3): the previous
    code only checked len(quote) >= 256, never verified the Intel signature
    chain, never checked RTMRs against expected_measurement, never checked
    report_data == sha256(nonce || user_data). Any 256-byte blob passed.

    Until libtdx-attest / trustauthority-py is wired in, this function
    REFUSES TDX quote acceptance. Set RALPH_ALLOW_REAL_ATTEST_STUB=1 to
    re-enable the loose check on testnet only.
    """
    import os as _os
    if not quote_hex:
        return False, "empty TDX quote"
    if _os.environ.get("RALPH_ALLOW_REAL_ATTEST_STUB") == "1":
        import sys as _sys
        print(
            "[attest] WARNING: RALPH_ALLOW_REAL_ATTEST_STUB=1 — accepting "
            "TDX quote without Intel signature chain verification. "
            "MUST NOT BE SET ON MAINNET.",
            file=_sys.stderr,
        )
        try:
            quote_bytes = bytes.fromhex(quote_hex)
        except ValueError as e:
            return False, f"TDX quote hex decode failed: {e}"
        if len(quote_bytes) < 256:
            return False, f"TDX quote too short ({len(quote_bytes)} bytes)"
        return True, f"TDX quote accepted (stub: signature/RTMR/report_data unchecked, {len(quote_bytes)} bytes)"

    return False, (
        "TDX quote verification not implemented. Wire libtdx-attest or "
        "trustauthority-py to check Intel signature chain, RTMRs against "
        "expected_measurement, and report_data == sha256(nonce||user_data). "
        "Or set RALPH_ALLOW_REAL_ATTEST_STUB=1 (testnet only)."
    )


def _required_attest_level() -> str:
    """Minimum hardware level the subnet gates miners to
    (env RALPH_REQUIRE_ATTEST_LEVEL):

      * "tdx_nvcc" (default) — Intel TDX (TEE) **and** NVIDIA CC GPU, both
        required and verified. This is the "miners may only run in a TEE+CC
        enclave" gate.
      * "nvcc_only"          — NVIDIA CC GPU required; TDX optional. Relaxation
        for testnet / CC-GPU-without-TDX hosts.

    Unknown values fall back to the strict default.
    """
    import os as _os

    lvl = _os.environ.get("RALPH_REQUIRE_ATTEST_LEVEL", "tdx_nvcc").strip().lower()
    return lvl if lvl in {"tdx_nvcc", "nvcc_only"} else "tdx_nvcc"


def verify_attestation(
    att: RealAttestation,
    expected_container_measurement: str,
    expected_handshake_nonce: str,
    expected_bundle_hash: str,
) -> tuple[bool, list[str]]:
    """Verify an attestation chain (real or mock).

    Dispatches to the appropriate verification path based on attestation_type.
    """
    errors: list[str] = []

    if att.container_measurement != expected_container_measurement:
        errors.append("container measurement mismatch")
    if att.handshake_nonce != expected_handshake_nonce:
        errors.append("handshake nonce mismatch")
    if att.bundle_hash != expected_bundle_hash:
        errors.append("bundle hash mismatch")
    if not att.epochs:
        errors.append("no attestation epochs")
        return False, errors

    if att.attestation_type == "mock":
        from .mock_attest import MockAttestation, verify_mock_attestation
        mock = MockAttestation(
            container_measurement=att.container_measurement,
            handshake_nonce=att.handshake_nonce,
            bundle_hash=att.bundle_hash,
        )
        from .mock_attest import MockAttestationEpoch
        for ep in att.epochs:
            mock.epochs.append(MockAttestationEpoch(
                epoch=ep.epoch, timestamp=ep.timestamp,
                rolling_log_hash=ep.rolling_log_hash,
                nonce=ep.nonce, container_measurement=ep.container_measurement,
                signature=ep.mock_signature or "",
            ))
        ok, mock_errors = verify_mock_attestation(
            mock, expected_container_measurement,
            expected_handshake_nonce, expected_bundle_hash,
        )
        if not ok:
            errors.extend(mock_errors)
        return len(errors) == 0, errors

    # Real attestation verification — REQUIRE the TEE/CC evidence the subnet
    # gates on. Previously the TDX/GPU quotes were verified only "if present",
    # so a real_* attestation with EMPTY quotes passed on measurement+nonce
    # +bundle alone (all miner-derivable) — bypassing the hardware proof. We now
    # require the evidence for the configured level and verify it.
    required = _required_attest_level()
    require_tdx = required == "tdx_nvcc"
    if require_tdx and att.attestation_type != "real_tdx_nvcc":
        errors.append(
            f"attestation_type={att.attestation_type!r} below required level "
            "'tdx_nvcc' (TEE+CC); set RALPH_REQUIRE_ATTEST_LEVEL=nvcc_only to "
            "relax (testnet only)"
        )

    for i, ep in enumerate(att.epochs):
        if ep.nonce != att.handshake_nonce:
            errors.append(f"epoch {i}: nonce drift")
        if ep.container_measurement != att.container_measurement:
            errors.append(f"epoch {i}: container measurement drift")

        # NVIDIA CC GPU token — REQUIRED for every real attestation.
        if not ep.gpu_token:
            errors.append(f"epoch {i}: missing NVIDIA CC GPU attestation token (required)")
        else:
            ok, detail = verify_gpu_token(ep.gpu_token, expected_handshake_nonce)
            if not ok:
                errors.append(f"epoch {i}: {detail}")

        # Intel TDX (TEE) quote — REQUIRED at level tdx_nvcc; verified if present
        # at nvcc_only.
        if not ep.tdx_quote:
            if require_tdx:
                errors.append(
                    f"epoch {i}: missing Intel TDX (TEE) quote (required at level tdx_nvcc)"
                )
        else:
            ok, detail = verify_tdx_quote(
                ep.tdx_quote, expected_handshake_nonce,
                expected_container_measurement,
            )
            if not ok:
                errors.append(f"epoch {i}: {detail}")

    # Check final epoch includes bundle hash in rolling hash
    final = att.epochs[-1]
    if len(att.epochs) >= 2:
        prior = att.epochs[-2].rolling_log_hash
        expected_final = hashlib.sha256((prior + expected_bundle_hash).encode()).hexdigest()
        if expected_final != final.rolling_log_hash:
            errors.append("final epoch rolling hash does not include bundle hash")
    else:
        expected_final = hashlib.sha256(expected_bundle_hash.encode()).hexdigest()
        if expected_final != final.rolling_log_hash:
            errors.append("final epoch rolling hash does not match bundle hash")

    return len(errors) == 0, errors


# ============================================================================
# CLI for manual testing
# ============================================================================

if __name__ == "__main__":
    caps = detect_capabilities()
    print("Hardware capabilities:")
    print(json.dumps(caps, indent=2))
    print()
    if caps["attestation_type"] != "mock":
        print("Generating a test attestation...")
        att = generate_attestation(
            container_measurement="test_measurement_" + "0" * 48,
            handshake_nonce="test_nonce_" + "0" * 48,
            epoch_records=[(0, time.time(), "test_rolling_hash")],
            bundle_hash="test_bundle_hash",
        )
        print(att.to_json())
    else:
        print("No CC hardware detected — would use mock attestation.")
        print("Run on a CC-capable H100 with TDX to generate real attestation.")
