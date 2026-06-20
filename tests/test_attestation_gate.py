"""TEE/CC attestation-gate tests — miners may only run in a TDX(TEE)+NVIDIA-CC
enclave. Covers the required-evidence enforcement that closes the old
"verify only if present" bypass, plus the RALPH_REQUIRE_ATTEST_LEVEL policy.

The TDX/GPU quote crypto itself (verify_tdx_quote / verify_gpu_token) is
fail-closed and exercised elsewhere; here it's monkeypatched so we test the GATE
(presence + level), not the quote signature chain.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import proof.real_attest as ra
import ralph_bootstrap  # noqa: F401
from proof.real_attest import (
    AttestationEpoch,
    RealAttestation,
    _required_attest_level,
    verify_attestation,
)

_M = "measurement_abc"
_N = "nonce_xyz"
_B = "bundle_123"


def _att(att_type, *, gpu_token, tdx_quote):
    """A single-epoch real attestation whose rolling hash already commits the
    bundle (so only the TEE/CC gate is under test)."""
    rolling = hashlib.sha256(_B.encode()).hexdigest()
    ep = AttestationEpoch(
        epoch=0, timestamp=0.0, rolling_log_hash=rolling, nonce=_N,
        container_measurement=_M, gpu_token=gpu_token, tdx_quote=tdx_quote,
        attestation_type="real",
    )
    return RealAttestation(
        container_measurement=_M, handshake_nonce=_N, attestation_type=att_type,
        epochs=[ep], bundle_hash=_B,
    )


@pytest.fixture
def quotes_ok(monkeypatch):
    """Make both quote verifiers pass so only the gate logic decides."""
    monkeypatch.setattr(ra, "verify_gpu_token", lambda t, n: (True, "ok"))
    monkeypatch.setattr(ra, "verify_tdx_quote", lambda q, n, m: (True, "ok"))


def _verify(att):
    return verify_attestation(att, _M, _N, _B)


# --- _required_attest_level ----------------------------------------------
def test_level_default_is_tee_plus_cc(monkeypatch):
    monkeypatch.delenv("RALPH_REQUIRE_ATTEST_LEVEL", raising=False)
    assert _required_attest_level() == "tdx_nvcc"


def test_level_env_relax_to_nvcc_only(monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_ATTEST_LEVEL", "nvcc_only")
    assert _required_attest_level() == "nvcc_only"


def test_level_bad_value_falls_back_to_strict(monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_ATTEST_LEVEL", "lol_none")
    assert _required_attest_level() == "tdx_nvcc"


# --- the gate (default level = TEE+CC) -----------------------------------
def test_tdx_nvcc_passes_with_both_quotes(monkeypatch, quotes_ok):
    monkeypatch.delenv("RALPH_REQUIRE_ATTEST_LEVEL", raising=False)
    ok, errors = _verify(_att("real_tdx_nvcc", gpu_token="g", tdx_quote="t"))
    assert ok, errors


def test_empty_tdx_quote_rejected_at_default_level(monkeypatch, quotes_ok):
    """THE BYPASS FIX: a real_tdx_nvcc attestation with no TDX quote is rejected
    (previously it passed on measurement+nonce+bundle alone)."""
    monkeypatch.delenv("RALPH_REQUIRE_ATTEST_LEVEL", raising=False)
    ok, errors = _verify(_att("real_tdx_nvcc", gpu_token="g", tdx_quote=None))
    assert not ok
    assert any("missing Intel TDX" in e for e in errors)


def test_empty_gpu_token_rejected(monkeypatch, quotes_ok):
    monkeypatch.delenv("RALPH_REQUIRE_ATTEST_LEVEL", raising=False)
    ok, errors = _verify(_att("real_tdx_nvcc", gpu_token=None, tdx_quote="t"))
    assert not ok
    assert any("missing NVIDIA CC" in e for e in errors)


def test_nvcc_only_type_rejected_when_tee_required(monkeypatch, quotes_ok):
    monkeypatch.delenv("RALPH_REQUIRE_ATTEST_LEVEL", raising=False)
    ok, errors = _verify(_att("real_nvcc_only", gpu_token="g", tdx_quote=None))
    assert not ok
    assert any("below required level" in e for e in errors)


def test_bad_gpu_token_fails(monkeypatch):
    monkeypatch.delenv("RALPH_REQUIRE_ATTEST_LEVEL", raising=False)
    monkeypatch.setattr(ra, "verify_gpu_token", lambda t, n: (False, "bad NRAS token"))
    monkeypatch.setattr(ra, "verify_tdx_quote", lambda q, n, m: (True, "ok"))
    ok, errors = _verify(_att("real_tdx_nvcc", gpu_token="g", tdx_quote="t"))
    assert not ok
    assert any("bad NRAS token" in e for e in errors)


# --- relaxed level (nvcc_only, testnet) ----------------------------------
def test_nvcc_only_level_allows_missing_tdx(monkeypatch, quotes_ok):
    monkeypatch.setenv("RALPH_REQUIRE_ATTEST_LEVEL", "nvcc_only")
    ok, errors = _verify(_att("real_nvcc_only", gpu_token="g", tdx_quote=None))
    assert ok, errors


def test_nvcc_only_level_still_requires_gpu(monkeypatch, quotes_ok):
    monkeypatch.setenv("RALPH_REQUIRE_ATTEST_LEVEL", "nvcc_only")
    ok, errors = _verify(_att("real_nvcc_only", gpu_token=None, tdx_quote=None))
    assert not ok
    assert any("missing NVIDIA CC" in e for e in errors)
