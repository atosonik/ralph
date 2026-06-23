"""On-chain handshake verification (the fix for external miners being rejected
at op1 because lookup_handshake read a local file the validator never had).

Covers the shared commit-hash helper, LocalChain's inherited (local) verify,
and BittensorChain's real get_commitment-based verify (with a fake subtensor).
"""
from __future__ import annotations

from chain_layer.bittensor_chain import BittensorChain, handshake_commit_hash
from chain_layer.local import LocalChain


# ------------------------------------------------- shared commit-hash helper
def test_handshake_commit_hash_binds_all_three():
    h = handshake_commit_hash("5GminerHK", "patchsha", "0xnonce")
    assert len(h) == 64
    # changing ANY of the three inputs changes the hash
    assert h != handshake_commit_hash("5GotherHK", "patchsha", "0xnonce")
    assert h != handshake_commit_hash("5GminerHK", "OTHER", "0xnonce")
    assert h != handshake_commit_hash("5GminerHK", "patchsha", "0xOTHER")
    assert h == handshake_commit_hash("5GminerHK", "patchsha", "0xnonce")  # deterministic


# ------------------------------------------------- LocalChain (inherited default)
def test_localchain_verify_handshake_matrix(tmp_path):
    lc = LocalChain(tmp_path / "chain")
    nonce = lc.request_handshake_nonce("5GminerHK", "patchsha123")
    assert lc.verify_handshake_onchain("5GminerHK", "patchsha123", nonce)[0]
    # wrong hotkey / wrong patch / wrong nonce all reject
    assert not lc.verify_handshake_onchain("5GotherHK", "patchsha123", nonce)[0]
    assert not lc.verify_handshake_onchain("5GminerHK", "DIFFERENT", nonce)[0]
    ok, detail = lc.verify_handshake_onchain("5GminerHK", "patchsha123", "0xmissing")
    assert not ok and "not found" in detail.lower()


# ------------------------------------------------- BittensorChain on-chain logic
class _FakeSub:
    def __init__(self, commitment):
        self._c = commitment

    def get_commitment(self, netuid, uid, block=None):
        return self._c


class _FakeChain:
    """Minimal stand-in exposing what verify_handshake_onchain touches."""
    netuid = 40

    def __init__(self, uid, commitment):
        self._uid = uid
        self.subtensor = _FakeSub(commitment)

    def get_uid(self, hotkey):
        return self._uid


def _verify(fc, hk, patch, nonce):
    # call the real BittensorChain method with the fake as self
    return BittensorChain.verify_handshake_onchain(fc, hk, patch, nonce)


def test_bittensorchain_verify_matches_onchain_commitment():
    hk, patch, nonce = "5Gminer", "patchabc", "0xdead"
    good = handshake_commit_hash(hk, patch, nonce)
    ok, detail = _verify(_FakeChain(7, good), hk, patch, nonce)
    assert ok, detail


def test_bittensorchain_verify_rejects_wrong_nonce_patch_hotkey():
    hk, patch, nonce = "5Gminer", "patchabc", "0xdead"
    good = handshake_commit_hash(hk, patch, nonce)
    fc = _FakeChain(7, good)
    assert not _verify(fc, hk, patch, "0xbeef")[0]      # nonce changed
    assert not _verify(fc, hk, "otherpatch", nonce)[0]  # patch changed
    assert not _verify(fc, "5Gother", patch, nonce)[0]  # different hotkey's preimage


def test_bittensorchain_verify_rejects_missing_commitment_and_unregistered():
    hk, patch, nonce = "5Gminer", "patchabc", "0xdead"
    ok1, d1 = _verify(_FakeChain(7, None), hk, patch, nonce)
    assert not ok1 and "no on-chain" in d1.lower()
    ok2, d2 = _verify(_FakeChain(None, "x"), hk, patch, nonce)
    assert not ok2 and "not registered" in d2.lower()


class _BoomSub:
    def get_commitment(self, netuid, uid, block=None):
        raise RuntimeError("rpc down")


def test_bittensorchain_verify_handles_get_commitment_error():
    fc = _FakeChain(7, None)
    fc.subtensor = _BoomSub()
    ok, detail = _verify(fc, "5Gminer", "p", "0xn")
    assert not ok and "get_commitment failed" in detail
