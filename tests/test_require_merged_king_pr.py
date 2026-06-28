"""Weight-time merged-PR gate + throne reinstatement (RALPH_REQUIRE_MERGED_KING_PR).

Opt-in policy: a king must carry a MERGED recipe PR to keep earning weight. The
gate (_apply_pool_split) withholds all weight from an unverified king; the
epoch-start repair (_reinstate_valid_king) demotes a throne already held by such
a king to the last merged ancestor, or clears it. Off by default = no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
import validator.service as service
from chain_layer.interface import KingRecord


def _king(hotkey, *, proof_dir="/x", previous_king=None):
    return KingRecord(
        miner_hotkey=hotkey, bundle_hash="bh", val_bpb=1.5,
        benchmark_accuracy=0.5, compute_cost=1.0, crowned_at=0.0,
        proof_dir=proof_dir, previous_king=previous_king,
    )


class _StubChain:
    """get_king/set_king/clear_king over an in-memory king, like LocalChain."""
    def __init__(self, king=None):
        self._king = king

    def get_king(self):
        return self._king

    def set_king(self, king):
        self._king = king

    def clear_king(self):
        self._king = None


# ---- KingRecord.from_dict (the get_king-divergence fix) ----

def test_from_dict_handles_compute_cost_h100h_and_unknown_keys():
    a = KingRecord.from_dict({"miner_hotkey": "x", "bundle_hash": "b", "val_bpb": 1.0,
                              "compute_cost_h100h": 2.0, "unknown_future_key": 9})
    assert a.compute_cost == 2.0  # set_king/router.py form maps; unknown key ignored (no TypeError)
    b = KingRecord.from_dict({"miner_hotkey": "x", "bundle_hash": "b", "val_bpb": 1.0,
                              "compute_cost": 3.0})
    assert b.compute_cost == 3.0  # dataclass asdict form


# ---- weight-time gate (_apply_pool_split) ----

def test_gate_off_by_default_weights_king(monkeypatch):
    monkeypatch.delenv("RALPH_REQUIRE_MERGED_KING_PR", raising=False)
    assert service._apply_pool_split(_StubChain(_king("k")), None, []) == {"k": 1.0}


def test_gate_withholds_all_weight_when_king_pr_unmerged(monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_MERGED_KING_PR", "1")
    monkeypatch.setattr(service, "_king_pr_merged", lambda chain: False)
    # gate trips -> {} so run_epoch burns instead of weighting the unverified king
    assert service._apply_pool_split(_StubChain(_king("k")), None, ["mf1"]) == {}


def test_gate_weights_king_when_pr_merged(monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_MERGED_KING_PR", "1")
    monkeypatch.setattr(service, "_king_pr_merged", lambda chain: True)
    assert service._apply_pool_split(_StubChain(_king("k")), None, []) == {"k": 1.0}


# ---- throne reinstatement (_reinstate_valid_king) ----

def test_reinstate_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("RALPH_REQUIRE_MERGED_KING_PR", raising=False)
    chain = _StubChain(_king("bad"))
    service._reinstate_valid_king(chain)
    assert chain.get_king().miner_hotkey == "bad"


def test_reinstate_demotes_to_merged_ancestor(monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_MERGED_KING_PR", "1")
    # ancestor serialized in the router.py form (compute_cost_h100h) that broke `**`
    ancestor = {"miner_hotkey": "good", "bundle_hash": "a", "val_bpb": 1.2,
                "benchmark_accuracy": 0.5, "compute_cost_h100h": 1.0,
                "crowned_at": 0.0, "proof_dir": "/good", "previous_king": None}
    chain = _StubChain(_king("bad", proof_dir="/bad", previous_king=ancestor))
    monkeypatch.setattr(service, "_king_record_pr_merged",
                        lambda k: getattr(k, "miner_hotkey", "") == "good")
    service._reinstate_valid_king(chain)
    assert chain.get_king().miner_hotkey == "good"  # walked past the unmerged king


def test_reinstate_clears_when_no_merged_ancestor(monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_MERGED_KING_PR", "1")
    chain = _StubChain(_king("bad", previous_king=None))
    monkeypatch.setattr(service, "_king_record_pr_merged", lambda k: False)
    service._reinstate_valid_king(chain)
    assert chain.get_king() is None  # throne cleared -> burns until a valid king


def test_reinstate_keeps_already_merged_king(monkeypatch):
    monkeypatch.setenv("RALPH_REQUIRE_MERGED_KING_PR", "1")
    chain = _StubChain(_king("ok"))
    monkeypatch.setattr(service, "_king_record_pr_merged", lambda k: True)
    service._reinstate_valid_king(chain)
    assert chain.get_king().miner_hotkey == "ok"


# ---- _king_record_pr_merged: empty-proof_dir guard + merged cache ----

def test_empty_proof_dir_returns_false():
    assert service._king_record_pr_merged(_king("k", proof_dir="")) is False


def test_merged_result_is_cached(tmp_path, monkeypatch):
    (tmp_path / "submission.json").write_text('{"pr_url": "https://github.com/o/r/pull/7"}')
    king = _king("k", proof_dir=str(tmp_path))
    service._MERGED_PR_URLS.discard("https://github.com/o/r/pull/7")
    calls = {"n": 0}

    def fake(url, tok):
        calls["n"] += 1
        return True

    monkeypatch.setattr("validator.github_bot.pr_is_merged", fake)
    monkeypatch.setenv("RALPH_BOT_GH_TOKEN", "t")
    assert service._king_record_pr_merged(king) is True
    assert service._king_record_pr_merged(king) is True
    assert calls["n"] == 1  # second call served from cache (no GitHub re-hit)
