"""Tests for ChainInterface.get_block_hash + get_current_block.

These pin the deterministic-seed semantics that v0.10 multi-seed eval and the
forthcoming Cross-Scale Downstream Pareto EpochSeeds derivation will depend on:

  - The hash is a public function of chain state visible to any observer.
  - The same block always produces the same hash.
  - No single miner can choose the value.
  - Format is normalized to 0x + 64 lowercase hex chars across both backends.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from chain_layer.bittensor_chain import BittensorChain
from chain_layer.local import LocalChain

# ----------------------------------------------------------------------------
# LocalChain — the real path; deterministic-from-events-on-disk
# ----------------------------------------------------------------------------


def test_local_current_block_starts_at_zero(tmp_path: Path):
    chain = LocalChain(tmp_path / "chain")
    assert chain.get_current_block() == 0


def test_local_current_block_increments_with_events(tmp_path: Path):
    chain = LocalChain(tmp_path / "chain")
    chain.append_event({"type": "test", "n": 1})
    chain.append_event({"type": "test", "n": 2})
    chain.append_event({"type": "test", "n": 3})
    assert chain.get_current_block() == 3


def test_local_block_hash_format(tmp_path: Path):
    """0x prefix + exactly 64 lowercase hex chars = 66 total."""
    chain = LocalChain(tmp_path / "chain")
    h = chain.get_block_hash(0)
    assert h.startswith("0x")
    assert len(h) == 66
    body = h[2:]
    assert body == body.lower()
    assert all(c in "0123456789abcdef" for c in body)


def test_local_block_hash_genesis_is_constant_across_chains(tmp_path: Path):
    """Block 0 has empty payload → same hash everywhere. Pins the seed
    starting point so two validators agree on it without coordination."""
    c1 = LocalChain(tmp_path / "c1")
    c2 = LocalChain(tmp_path / "c2")
    c2.append_event({"type": "anything"})  # c2 has state; c1 doesn't
    assert c1.get_block_hash(0) == c2.get_block_hash(0)


def test_local_block_hash_genesis_matches_blake2b_of_empty(tmp_path: Path):
    """Spec the genesis value explicitly so any future hash-fn change is
    visible at the diff level — this is forensic evidence other validators
    can use to detect a divergent local backend."""
    chain = LocalChain(tmp_path / "chain")
    expected = "0x" + hashlib.blake2b(b"", digest_size=32).hexdigest()
    assert chain.get_block_hash(0) == expected


def test_local_block_hash_deterministic(tmp_path: Path):
    chain = LocalChain(tmp_path / "chain")
    chain.append_event({"type": "round_1", "block": 1})
    assert chain.get_block_hash(1) == chain.get_block_hash(1)


def test_local_block_hash_persistence_across_instances(tmp_path: Path):
    """A fresh LocalChain pointed at the same dir reproduces the hash —
    proves the hash is a function of disk state, not in-memory state."""
    chain_dir = tmp_path / "chain"
    c1 = LocalChain(chain_dir)
    c1.append_event({"type": "round_1"})
    c1.append_event({"type": "round_2"})
    h1 = c1.get_block_hash(2)
    c2 = LocalChain(chain_dir)
    assert c2.get_block_hash(2) == h1


def test_local_block_hash_prior_blocks_stable_after_append(tmp_path: Path):
    """Adding event N+1 doesn't change the hash at block N. Required for
    multi-seed eval: validators derive seeds from get_block_hash(epoch_block)
    and that value must NOT shift when new events arrive."""
    chain = LocalChain(tmp_path / "chain")
    chain.append_event({"type": "a"})
    h_at_1 = chain.get_block_hash(1)
    chain.append_event({"type": "b"})
    chain.append_event({"type": "c"})
    assert chain.get_block_hash(1) == h_at_1


def test_local_block_hash_differs_per_block(tmp_path: Path):
    chain = LocalChain(tmp_path / "chain")
    chain.append_event({"type": "a"})
    chain.append_event({"type": "b"})
    assert chain.get_block_hash(1) != chain.get_block_hash(2)


def test_local_block_hash_diff_histories_diverge(tmp_path: Path):
    """Two chains with one different event at the same block produce
    different hashes — proves the hash depends on event CONTENT, not just
    event COUNT."""
    c1 = LocalChain(tmp_path / "c1")
    c2 = LocalChain(tmp_path / "c2")
    c1.append_event({"type": "alpha"})
    c2.append_event({"type": "beta"})
    assert c1.get_block_hash(1) != c2.get_block_hash(1)


def test_local_block_hash_rejects_negative(tmp_path: Path):
    chain = LocalChain(tmp_path / "chain")
    with pytest.raises(ValueError, match=r"block must be >= 0"):
        chain.get_block_hash(-1)
    with pytest.raises(ValueError, match=r"block must be >= 0"):
        chain.get_block_hash(-42)


def test_local_block_hash_rejects_future_block(tmp_path: Path):
    chain = LocalChain(tmp_path / "chain")
    chain.append_event({"type": "only_one"})
    with pytest.raises(ValueError, match=r"exceeds current height"):
        chain.get_block_hash(5)
    with pytest.raises(ValueError, match=r"exceeds current height"):
        chain.get_block_hash(2)


# ----------------------------------------------------------------------------
# BittensorChain — wraps subtensor; tested with a monkeypatched subtensor so
# we don't need a live chain in CI
# ----------------------------------------------------------------------------


class _MockSubtensor:
    """A tiny fake mirroring just the two subtensor methods we depend on.

    Hash values are mixed-case + sometimes lacking 0x to verify normalization.
    """

    def __init__(self, current: int = 42, hashes: dict | None = None):
        self._current = current
        self._hashes = hashes or {}

    def get_current_block(self):
        return self._current

    def get_block_hash(self, block):
        return self._hashes.get(block)


def _make_bittensor_chain(mock: _MockSubtensor) -> BittensorChain:
    """Construct a BittensorChain without going through __init__ (which would
    try to connect to a real chain). Inject only what the tested methods need."""
    chain = BittensorChain.__new__(BittensorChain)
    chain.subtensor = mock
    return chain


def test_bittensor_current_block_passthrough():
    chain = _make_bittensor_chain(_MockSubtensor(current=12345))
    assert chain.get_current_block() == 12345


def test_bittensor_block_hash_normalizes_uppercase():
    """Mixed-case hex from chain is normalized to lowercase."""
    raw = "0x" + "ABCDEF0123456789" * 4  # 64 hex chars, mixed case
    chain = _make_bittensor_chain(_MockSubtensor(hashes={7: raw}))
    h = chain.get_block_hash(7)
    assert h.startswith("0x")
    assert len(h) == 66
    assert h == h.lower()
    assert h[2:] == raw[2:].lower()


def test_bittensor_block_hash_adds_0x_prefix_when_missing():
    """Some bittensor versions return hash without 0x prefix; we add it."""
    raw = "abcdef0123456789" * 4  # 64 chars, no prefix
    chain = _make_bittensor_chain(_MockSubtensor(hashes={9: raw}))
    h = chain.get_block_hash(9)
    assert h == "0x" + raw


def test_bittensor_block_hash_rejects_negative():
    chain = _make_bittensor_chain(_MockSubtensor())
    with pytest.raises(ValueError, match=r"block must be >= 0"):
        chain.get_block_hash(-1)


def test_bittensor_block_hash_rejects_empty_response():
    """A chain hiccup that returns None / empty string is a hard error,
    not silently zero — callers using this for seed derivation MUST know."""
    chain = _make_bittensor_chain(_MockSubtensor(hashes={5: None}))
    with pytest.raises(ValueError, match=r"no hash for block"):
        chain.get_block_hash(5)
    chain2 = _make_bittensor_chain(_MockSubtensor(hashes={5: ""}))
    with pytest.raises(ValueError, match=r"no hash for block"):
        chain2.get_block_hash(5)


def test_bittensor_block_hash_rejects_wrong_length():
    chain = _make_bittensor_chain(_MockSubtensor(hashes={1: "0xabc"}))
    with pytest.raises(ValueError, match=r"malformed hash"):
        chain.get_block_hash(1)


def test_bittensor_block_hash_rejects_non_hex():
    """Any non-hex character is rejected — prevents the seed derivation
    from silently consuming garbage."""
    raw = "0x" + "z" * 64  # 64 chars but invalid hex
    chain = _make_bittensor_chain(_MockSubtensor(hashes={3: raw}))
    with pytest.raises(ValueError, match=r"malformed hash"):
        chain.get_block_hash(3)
