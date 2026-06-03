"""Tests for the §5.6 90/10 pool split in validator.service._apply_pool_split."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from validator.service import (
    KING_POOL_FRACTION,
    MEANINGFUL_FAILURE_POOL_FRACTION,
    _apply_pool_split,
)


class _StubChain:
    """Minimal stand-in: only needs get_king() for _apply_pool_split."""
    def __init__(self, king_hotkey: str | None = None):
        if king_hotkey is None:
            self._king = None
        else:
            from chain_layer.interface import KingRecord
            self._king = KingRecord(
                miner_hotkey=king_hotkey,
                bundle_hash="bh",
                val_bpb=1.5,
                benchmark_accuracy=0.5,
                compute_cost=1.0,
                crowned_at=0.0,
            )

    def get_king(self):
        return self._king


def test_pool_fractions_sum_to_one():
    assert abs(KING_POOL_FRACTION + MEANINGFUL_FAILURE_POOL_FRACTION - 1.0) < 1e-9


def test_no_king_no_mf_returns_empty():
    assert _apply_pool_split(_StubChain(None), None, []) == {}


def test_new_king_no_mf_gets_full_pool():
    """No meaningful_failures → king gets 100% (not 90%)."""
    w = _apply_pool_split(_StubChain("old_k"), "new_k", [])
    assert w == {"new_k": 1.0}


def test_sitting_king_no_mf_gets_full_pool():
    """No king change AND no meaningful_failures → sitting king gets 100%."""
    w = _apply_pool_split(_StubChain("sitting"), None, [])
    assert w == {"sitting": 1.0}


def test_one_mf_no_king_change():
    """1 meaningful_failure + sitting king → 90/10."""
    w = _apply_pool_split(_StubChain("k"), None, ["mf1"])
    assert w == {"k": 0.9, "mf1": 0.1}


def test_one_king_change_one_mf():
    w = _apply_pool_split(_StubChain("old_k"), "new_k", ["mf1"])
    assert w == {"new_k": 0.9, "mf1": 0.1}


def test_multiple_mf_split_equally():
    """5 meaningful_failures → each gets 2%."""
    w = _apply_pool_split(_StubChain("k"), None, ["a", "b", "c", "d", "e"])
    assert abs(w["k"] - 0.9) < 1e-9
    for hk in "abcde":
        assert abs(w[hk] - 0.02) < 1e-9


def test_king_doubles_as_mf_does_not_lose_credit():
    """Defensive: if a hotkey somehow appears as both king-change AND
    meaningful_failure in the same epoch (rare; same miner submits twice),
    max(king_share, mf_share) wins so they don't lose credit."""
    w = _apply_pool_split(_StubChain(None), "new_k", ["new_k", "mf2"])
    # new_k should get the king share (0.9), not be overwritten by 0.05
    assert abs(w["new_k"] - 0.9) < 1e-9
    assert abs(w["mf2"] - 0.05) < 1e-9
