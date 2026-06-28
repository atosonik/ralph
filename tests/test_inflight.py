"""One-in-flight-per-hotkey: dynamic cooldown, idempotent, restart-safe."""
from __future__ import annotations

from validator.inflight import InFlightGuard

HK = "5FCDTbBDka1WxcspxAjRMUeecp3vHnYiM1nenEsKGXysYqGE"
HK2 = "5H6mytgBYJbgTRNQv11gGfhFa1Dq91rd8TiSZQnKW7cmB1jb"
A = "a" * 64
B = "b" * 64


def test_claim_then_block_then_release(tmp_path):
    g = InFlightGuard(tmp_path / "inflight.json")
    ok, _ = g.claim(HK, A)
    assert ok and g.in_flight(HK) == A
    # a DIFFERENT bundle from the same hotkey is deferred while A is in flight
    ok, reason = g.claim(HK, B)
    assert not ok and "in flight" in reason
    # once A is scored, the hotkey can submit again immediately (dynamic cooldown)
    g.release(HK, A)
    assert g.in_flight(HK) is None
    ok, _ = g.claim(HK, B)
    assert ok and g.in_flight(HK) == B


def test_claim_same_bundle_is_idempotent(tmp_path):
    g = InFlightGuard(tmp_path / "inflight.json")
    assert g.claim(HK, A)[0]
    assert g.claim(HK, A)[0]  # reprocess / restart must not be blocked


def test_other_hotkeys_are_independent(tmp_path):
    g = InFlightGuard(tmp_path / "inflight.json")
    assert g.claim(HK, A)[0]
    assert g.claim(HK2, B)[0]  # different hotkey, independent slot
    assert g.in_flight(HK) == A and g.in_flight(HK2) == B


def test_release_with_mismatched_hash_is_noop(tmp_path):
    g = InFlightGuard(tmp_path / "inflight.json")
    g.claim(HK, A)
    g.release(HK, B)              # stale release must not clear the newer claim
    assert g.in_flight(HK) == A


def test_state_persists_across_restart(tmp_path):
    p = tmp_path / "inflight.json"
    InFlightGuard(p).claim(HK, A)
    # a fresh instance (validator restart) sees the in-flight claim
    g2 = InFlightGuard(p)
    assert g2.in_flight(HK) == A
    ok, _ = g2.claim(HK, B)
    assert not ok


def test_reconcile_clears_stale_claims(tmp_path):
    g = InFlightGuard(tmp_path / "inflight.json")
    g.claim(HK, A)   # A got scored+archived but crash skipped release
    g.claim(HK2, B)  # B still pending
    g.reconcile(valid_bundle_hashes={B})  # only B is still in the queue
    assert g.in_flight(HK) is None        # stale A claim cleared
    assert g.in_flight(HK2) == B          # live claim kept
    assert g.claim(HK, A)[0]              # HK can submit again


def test_corrupt_state_does_not_wedge(tmp_path):
    p = tmp_path / "inflight.json"
    p.write_text("{ not json")
    g = InFlightGuard(p)         # must load empty, not raise
    assert g.in_flight(HK) is None
    assert g.claim(HK, A)[0]
