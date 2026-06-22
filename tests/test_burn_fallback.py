"""Burn-to-uid-0 fallback: validator + auditor still set weights (keep vTrust +
burn to owner) when there is nothing real to score/audit this epoch."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path


# --------------------------------------------------------- validator flag
def test_validator_burn_fallback_default_on(monkeypatch):
    from validator.service import _burn_fallback_enabled

    monkeypatch.delenv("RALPH_BURN_FALLBACK", raising=False)
    assert _burn_fallback_enabled() is True
    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("RALPH_BURN_FALLBACK", off)
        assert _burn_fallback_enabled() is False
    for on in ("1", "true", "yes", "on"):
        monkeypatch.setenv("RALPH_BURN_FALLBACK", on)
        assert _burn_fallback_enabled() is True


# --------------------------------------------------------- LocalChain burn
def test_localchain_set_burn_weights_records_uid0(monkeypatch):
    from chain_layer.local import LocalChain

    monkeypatch.delenv("RALPH_BURN_UID", raising=False)
    d = Path(tempfile.mkdtemp())
    chain = LocalChain(d)
    assert chain.set_burn_weights() is True
    events = [json.loads(ln) for ln in (d / "events.jsonl").read_text().splitlines() if ln.strip()]
    last = events[-1]
    assert last["type"] == "weights_set"
    assert last.get("burn") is True
    assert last["weights"] == {"uid:0": 1.0}


def test_localchain_burn_uid_override(monkeypatch):
    from chain_layer.local import LocalChain

    monkeypatch.setenv("RALPH_BURN_UID", "7")
    d = Path(tempfile.mkdtemp())
    chain = LocalChain(d)
    chain.set_burn_weights()
    events = [json.loads(ln) for ln in (d / "events.jsonl").read_text().splitlines() if ln.strip()]
    assert events[-1]["weights"] == {"uid:7": 1.0}


# --------------------------------------------------------- auditor burn
def test_auditor_submit_burn_no_wallet_is_graceful(monkeypatch):
    from auditor.weights import submit_burn_weights

    monkeypatch.delenv("AUDITOR_WALLET_NAME", raising=False)
    # No wallet configured → returns False without raising (read-only).
    assert submit_burn_weights("ws://localhost:9944", 40) is False
