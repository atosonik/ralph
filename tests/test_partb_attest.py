"""Part B foundation tests (fixture-independent): the shared user_data binding
helper, the nested NRAS bundle parser, and the 4-arg verify_tdx_quote with the
report_data binding. The real NRAS-JWKS + Intel-DCAP signature crypto is
finalized + golden-fixture-tested against the miner's real CC-hardware bundles.
"""
from __future__ import annotations

import hashlib

from proof import real_attest as RA


# --------------------------------------------------------- build_user_data
def test_build_user_data_mid_vs_final():
    mid = RA.build_user_data("CM", "ROLL", "0xNONCE")
    assert mid == "CM:ROLL:0xNONCE"
    final = RA.build_user_data("CM", "ROLL", "0xNONCE", bundle_hash="BUN")
    assert final == "CM:ROLL:0xNONCE:BUN"
    assert final != mid  # final epoch is distinguishable


# --------------------------------------------------------- parse_nras_bundle
def test_parse_nras_bundle_nested():
    nested = [["JWT", "OUTER"], {"REMOTE_GPU_CLAIMS": [["JWT", "GPUEAT"], {}]}]
    outer, gpu = RA.parse_nras_bundle(nested)
    assert outer == "OUTER"
    assert gpu == ["GPUEAT"]


def test_parse_nras_bundle_json_string():
    import json
    nested = [["JWT", "OUTER"], {"REMOTE_GPU_CLAIMS": [["JWT", "GPUEAT"], {}]}]
    outer, gpu = RA.parse_nras_bundle(json.dumps(nested))
    assert outer == "OUTER" and gpu == ["GPUEAT"]


def test_parse_nras_bundle_bare_jwt():
    outer, gpu = RA.parse_nras_bundle("eyJh.body.sig")
    assert outer is None and gpu == ["eyJh.body.sig"]


# --------------------------------------------------- verify_tdx_quote (4-arg)
def _fake_quote_with_report_data(nonce: str, user_data: str) -> str:
    rd = hashlib.sha256((nonce + user_data).encode()).digest()  # 32 bytes
    blob = b"\x00" * 200 + rd + b"\x00" * 60  # >=256, contains rd
    return blob.hex()


def test_verify_tdx_quote_stub_binds_report_data(monkeypatch):
    monkeypatch.setenv("RALPH_ALLOW_REAL_ATTEST_STUB", "1")
    nonce = "0x" + "11" * 32
    ud = RA.build_user_data("cm", "roll", nonce)
    quote = _fake_quote_with_report_data(nonce, ud)
    ok, detail = RA.verify_tdx_quote(quote, nonce, ud)
    assert ok, detail


def test_verify_tdx_quote_stub_rejects_wrong_binding(monkeypatch):
    monkeypatch.setenv("RALPH_ALLOW_REAL_ATTEST_STUB", "1")
    nonce = "0x" + "11" * 32
    ud = RA.build_user_data("cm", "roll", nonce)
    # quote bound to a DIFFERENT user_data → expected rd absent → reject
    quote = _fake_quote_with_report_data(nonce, RA.build_user_data("cm", "roll", nonce, "OTHER"))
    ok, detail = RA.verify_tdx_quote(quote, nonce, ud)
    assert not ok and "report_data" in detail.lower()


def test_verify_tdx_quote_rejects_fake_on_mainnet(monkeypatch):
    # Production path runs dcap-qvl; a fabricated (non-Intel-signed) quote is
    # rejected at parse/verify — well before any binding check.
    monkeypatch.delenv("RALPH_ALLOW_REAL_ATTEST_STUB", raising=False)
    nonce = "0x" + "11" * 32
    ud = RA.build_user_data("cm", "roll", nonce)
    ok, _ = RA.verify_tdx_quote(_fake_quote_with_report_data(nonce, ud), nonce, ud)
    assert not ok


def test_verify_tdx_quote_empty():
    ok, detail = RA.verify_tdx_quote("", "0xn", "ud")
    assert not ok and "empty" in detail.lower()
