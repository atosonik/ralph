"""Tests for the 2026-06-22 CC-hardware attestation generation fixes.

Covers the fully-CPU-testable parts: the GPU-SDK nonce normalization (Issue 3)
and that verify_gpu_token compares in the normalized form. The Attestation()
name fix (Issue 1) and the TDX privilege path (Issue 2) need real CC hardware
to exercise end-to-end and are validated there.
"""
from __future__ import annotations

import importlib.util

import pytest

from proof import real_attest as RA


# ----------------------------------------------------------- nonce normalization
def test_gpu_sdk_nonce_strips_0x_and_lowercases():
    raw = "0x" + "AB" * 32  # 66 chars, uppercase
    out = RA.gpu_sdk_nonce(raw)
    assert out == "ab" * 32
    assert len(out) == 64
    assert not out.startswith("0x")


def test_gpu_sdk_nonce_idempotent_on_bare_hex():
    bare = "cd" * 32
    assert RA.gpu_sdk_nonce(bare) == bare
    assert RA.gpu_sdk_nonce(RA.gpu_sdk_nonce("0x" + bare)) == bare


def test_gpu_sdk_nonce_handles_whitespace_and_0X():
    assert RA.gpu_sdk_nonce("  0X" + "ef" * 32 + " ") == "ef" * 32


# ------------------------------------------- verify compares in normalized form
@pytest.mark.skipif(
    importlib.util.find_spec("jwt") is None, reason="PyJWT not installed"
)
def test_verify_gpu_token_matches_across_0x_prefix(monkeypatch):
    """A token whose nonce claim is the 64-hex form must match an expected
    handshake nonce given in the on-chain 0x-prefixed form (and vice versa)."""
    import jwt

    monkeypatch.setenv("RALPH_ALLOW_REAL_ATTEST_STUB", "1")  # stub path for nonce check
    bare = "12ab" * 16  # 64 hex
    onchain = "0x" + bare
    token = jwt.encode({"nonce": bare}, "secret", algorithm="HS256")

    ok, detail = RA.verify_gpu_token(token, onchain)
    assert ok, detail

    # mismatched nonce still rejected
    token_bad = jwt.encode({"nonce": "00" * 32}, "secret", algorithm="HS256")
    ok2, _ = RA.verify_gpu_token(token_bad, onchain)
    assert not ok2


def test_verify_gpu_token_empty_is_rejected():
    ok, detail = RA.verify_gpu_token("", "0x" + "00" * 32)
    assert not ok and "empty" in detail.lower()


# ------------------------------------------------ mainnet really verifies now
def test_verify_gpu_token_rejects_junk_on_mainnet(monkeypatch):
    # Production path does real NRAS-JWKS ES384 verification; a junk token has no
    # valid NVIDIA signature → rejected (fails before any network fetch).
    monkeypatch.delenv("RALPH_ALLOW_REAL_ATTEST_STUB", raising=False)
    ok, _ = RA.verify_gpu_token("nonempty.token.value", "0x" + "00" * 32)
    assert not ok


# ----------------------------------- get_token() bundle parsing (2026-06-22 CC report)
def test_extract_gpu_jwt_from_bundle_and_bare():
    import json as _json

    # bundle as a Python list: [["JWT", outer], {detached...}]
    bundle = [["JWT", "OUTER.JWT.STR"], {"GPU-0": "DETACHED.JWT"}]
    assert RA._extract_gpu_jwt(bundle) == "OUTER.JWT.STR"
    # bundle as a JSON string
    assert RA._extract_gpu_jwt(_json.dumps(bundle)) == "OUTER.JWT.STR"
    # bare JWT passes through unchanged (back-compat)
    assert RA._extract_gpu_jwt("eyJhbG.body.sig") == "eyJhbG.body.sig"


@pytest.mark.skipif(
    importlib.util.find_spec("jwt") is None, reason="PyJWT not installed"
)
def test_stub_accepts_real_get_token_bundle(monkeypatch):
    """Regression for the miner's 'Invalid header string': the stub must parse
    the get_token() bundle and decode the inner JWT, not the whole bundle."""
    import json as _json

    import jwt

    monkeypatch.setenv("RALPH_ALLOW_REAL_ATTEST_STUB", "1")
    bare = "ab12" * 16  # 64 hex
    onchain = "0x" + bare
    outer = jwt.encode({"eat_nonce": bare}, "secret", algorithm="HS256")
    bundle = _json.dumps([["JWT", outer], {"GPU-0": outer}])  # nv-sdk shape
    ok, detail = RA.verify_gpu_token(bundle, onchain)
    assert ok, detail


def test_all_bundle_jwts_collects_nested():
    """Issue 6: the real NRAS token nests the GPU EAT under REMOTE_GPU_CLAIMS."""
    import json as _json

    nested = [["JWT", "outer.env.sig"],
              {"REMOTE_GPU_CLAIMS": [["JWT", "gpu.claims.sig"], {}]}]
    got = RA._all_bundle_jwts(nested)
    assert "outer.env.sig" in got and "gpu.claims.sig" in got
    # also works from a JSON string
    assert set(RA._all_bundle_jwts(_json.dumps(nested))) == {"outer.env.sig", "gpu.claims.sig"}


@pytest.mark.skipif(
    importlib.util.find_spec("jwt") is None, reason="PyJWT not installed"
)
def test_stub_finds_eat_nonce_on_nested_gpu_submodule(monkeypatch):
    """Issue 6 regression: eat_nonce is on the inner REMOTE_GPU_CLAIMS EAT, NOT
    the outer envelope. The stub must search the nested layer."""
    import json as _json

    import jwt

    monkeypatch.setenv("RALPH_ALLOW_REAL_ATTEST_STUB", "1")
    bare = "2fcc0dbc" * 8  # 64 hex
    onchain = "0x" + bare
    # outer envelope has NO eat_nonce (just iss/exp/jti) — like the real token
    outer = jwt.encode({"iss": "NRAS", "exp": 9999999999, "jti": "x"}, "s", algorithm="HS256")
    gpu_eat = jwt.encode({"sub": "...", "eat_nonce": bare,
                          "x-nvidia-overall-att-result": True}, "s", algorithm="HS256")
    bundle = _json.dumps([["JWT", outer], {"REMOTE_GPU_CLAIMS": [["JWT", gpu_eat], {}]}])
    ok, detail = RA.verify_gpu_token(bundle, onchain)
    assert ok, detail
