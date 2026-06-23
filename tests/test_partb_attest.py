"""Part B foundation tests (fixture-independent): the shared user_data binding
helper, the nested NRAS bundle parser, and the 4-arg verify_tdx_quote with the
report_data binding. The real NRAS-JWKS + Intel-DCAP signature crypto is
finalized + golden-fixture-tested against the miner's real CC-hardware bundles.
"""
from __future__ import annotations

import hashlib
import importlib.util

import pytest

from proof import real_attest as RA

_HAS_DCAP = importlib.util.find_spec("dcap_qvl") is not None


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


# ----------------------------- production path (dcap-qvl faked, no network)
# These exercise the real (non-stub) branch and pin two API contracts that a
# junk-quote test can't reach (it dies at parse_quote): get_collateral_and_verify
# is async (must be awaited) and is_tdx is a method (must be called).
class _FakeReport:
    def __init__(self, report_data):
        self.report_data = report_data
        self.mr_td = b"\xab" * 48


class _FakeQuote:
    def __init__(self, report_data, tdx=True):
        self.report = _FakeReport(report_data)
        self._tdx = tdx

    def is_tdx(self):  # real dcap-qvl exposes is_tdx as a METHOD
        return self._tdx


class _FakeVR:
    def __init__(self, status="UpToDate"):
        self.status = status
        self.advisory_ids = []


def _patch_dcap(monkeypatch, *, report_data, tdx=True, status="UpToDate"):
    import dcap_qvl
    monkeypatch.delenv("RALPH_ALLOW_REAL_ATTEST_STUB", raising=False)
    monkeypatch.setattr(dcap_qvl, "parse_quote", lambda b: _FakeQuote(report_data, tdx=tdx))

    async def _gcv(b, pccs_url=None):  # async, like the real one
        return _FakeVR(status)

    monkeypatch.setattr(dcap_qvl, "get_collateral_and_verify", _gcv)


def _bound_rd(nonce, ud):
    return hashlib.sha256((nonce + ud).encode()).digest() + b"\x00" * 32  # 32 binding + 32 zero tail


@pytest.mark.skipif(not _HAS_DCAP, reason="dcap-qvl not installed")
def test_verify_tdx_quote_production_accepts_bound_quote(monkeypatch):
    # Fails if get_collateral_and_verify is not awaited (a coroutine has no .status).
    nonce, ud = "0x" + "11" * 32, RA.build_user_data("cm", "roll", "0x" + "11" * 32)
    _patch_dcap(monkeypatch, report_data=_bound_rd(nonce, ud), tdx=True, status="UpToDate")
    ok, detail = RA.verify_tdx_quote("00" * 300, nonce, ud)
    assert ok, detail


@pytest.mark.skipif(not _HAS_DCAP, reason="dcap-qvl not installed")
def test_verify_tdx_quote_production_rejects_non_tdx(monkeypatch):
    # Fails if is_tdx is read as a bare attribute (a bound method is always truthy).
    nonce, ud = "0x" + "11" * 32, RA.build_user_data("cm", "roll", "0x" + "11" * 32)
    _patch_dcap(monkeypatch, report_data=_bound_rd(nonce, ud), tdx=False)
    ok, detail = RA.verify_tdx_quote("00" * 300, nonce, ud)
    assert not ok and "tdx" in detail.lower()


@pytest.mark.skipif(not _HAS_DCAP, reason="dcap-qvl not installed")
def test_verify_tdx_quote_production_rejects_bad_tcb(monkeypatch):
    nonce, ud = "0x" + "11" * 32, RA.build_user_data("cm", "roll", "0x" + "11" * 32)
    _patch_dcap(monkeypatch, report_data=_bound_rd(nonce, ud), status="OutOfDate")
    ok, detail = RA.verify_tdx_quote("00" * 300, nonce, ud)
    assert not ok and "tcb" in detail.lower()


@pytest.mark.skipif(not _HAS_DCAP, reason="dcap-qvl not installed")
def test_verify_tdx_quote_production_rejects_unbound_report_data(monkeypatch):
    nonce, ud = "0x" + "11" * 32, RA.build_user_data("cm", "roll", "0x" + "11" * 32)
    wrong = hashlib.sha256(b"different").digest() + b"\x00" * 32
    _patch_dcap(monkeypatch, report_data=wrong, status="UpToDate")
    ok, detail = RA.verify_tdx_quote("00" * 300, nonce, ud)
    assert not ok and "report_data" in detail.lower()
