"""Counter-weight cadence tests — the block-cadence weight-setting that makes
the auditor a continuous epoch-cadence validator, not a one-shot.

Pure logic only (is_weight_set_due / interval / due decision); no chain, no
wallet, no torch. The submit_weights extrinsic itself needs a live subtensor +
wallet and is exercised operationally, not here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from auditor.weights import (
    DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS,
    is_enabled,
    is_weight_set_due,
    weight_set_interval_blocks,
)


# --- is_weight_set_due ---------------------------------------------------
def test_due_when_never_set():
    # None = never set / unknown → set on the first tick
    assert is_weight_set_due(None, 300) is True


def test_not_due_before_interval():
    assert is_weight_set_due(0, 300) is False
    assert is_weight_set_due(299, 300) is False


def test_due_at_and_after_interval():
    assert is_weight_set_due(300, 300) is True
    assert is_weight_set_due(450, 300) is True


def test_due_nonpositive_interval_raises():
    with pytest.raises(ValueError, match="interval_blocks must be > 0"):
        is_weight_set_due(500, 0)


# --- weight_set_interval_blocks (env) ------------------------------------
def test_interval_default(monkeypatch):
    monkeypatch.delenv("AUDITOR_WEIGHT_INTERVAL_BLOCKS", raising=False)
    assert weight_set_interval_blocks() == DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS == 300


def test_interval_env_override(monkeypatch):
    monkeypatch.setenv("AUDITOR_WEIGHT_INTERVAL_BLOCKS", "120")
    assert weight_set_interval_blocks() == 120


@pytest.mark.parametrize("bad", ["0", "-5", "abc", ""])
def test_interval_bad_env_falls_back(monkeypatch, bad):
    monkeypatch.setenv("AUDITOR_WEIGHT_INTERVAL_BLOCKS", bad)
    assert weight_set_interval_blocks() == DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS


def test_default_interval_above_subnet_rate_limit():
    # netuid 40 weights_rate_limit is ~100 blocks; the default must clear it so
    # cadence sets are never rejected for setting too often.
    assert DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS > 100


# --- is_enabled (opt-in gate) --------------------------------------------
@pytest.mark.parametrize("val,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False),
])
def test_is_enabled(monkeypatch, val, expected):
    monkeypatch.setenv("AUDITOR_SET_WEIGHTS_ENABLED", val)
    assert is_enabled() is expected


def test_is_enabled_default_off(monkeypatch):
    monkeypatch.delenv("AUDITOR_SET_WEIGHTS_ENABLED", raising=False)
    assert is_enabled() is False


# --- maybe_counter_weight orchestration (cadence flow) -------------------
class _FakeChain:
    subtensor_url = "ws://x"
    netuid = 40

    def __init__(self, blocks_since, current=1000, tempo=None, blocks_left=None, rate_limit=None):
        self._bs = blocks_since
        self._cur = current
        self._tempo = tempo
        self._bl = blocks_left
        self._rl = rate_limit

    def blocks_since_weight_set(self, hotkey):
        return self._bs

    def get_current_block(self):
        return self._cur

    # tempo=None → maybe_counter_weight uses the flat-interval fallback (the path
    # the cadence-flow tests below cover); set tempo/blocks_left/rate_limit to
    # exercise the SN51 tempo-boundary path.
    def tempo(self):
        return self._tempo

    def blocks_until_next_epoch(self):
        return self._bl

    def weights_rate_limit(self):
        return self._rl


class _FakeApi:
    def __init__(self):
        self.fetched = []

    def get_report(self, epoch_id):
        self.fetched.append(epoch_id)
        return {"report_json": {"epoch_id": epoch_id}}


def _wire(monkeypatch, tmp_path, *, enabled, hotkey, clean_epoch):
    import auditor.main as m
    import auditor.weights as w

    monkeypatch.setattr(w, "is_enabled", lambda: enabled)
    monkeypatch.setattr(w, "auditor_hotkey_ss58", lambda: hotkey)
    submitted = {}
    monkeypatch.setattr(w, "submit_weights",
                        lambda **kw: submitted.update(kw) or True)
    monkeypatch.setattr(m, "replay_scoring", lambda rj: {"5Fminer": 1.0})
    clean = tmp_path / "clean"
    if clean_epoch is not None:
        clean.write_text(clean_epoch)
    monkeypatch.setattr(m, "LAST_CLEAN_EPOCH_FILE", clean)
    monkeypatch.setattr(m, "PUBLISHED_FILE", tmp_path / "pub")
    return m, submitted


def test_counter_weight_sets_when_due(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch="40-123")
    api = _FakeApi()
    m.maybe_counter_weight(_FakeChain(blocks_since=300, current=1000), api)
    assert api.fetched == ["40-123"]
    assert submitted["weights_by_hotkey"] == {"5Fminer": 1.0}
    assert submitted["netuid"] == 40
    assert (tmp_path / "pub").read_text() == "1000"  # records the block we set at


def test_counter_weight_sets_when_never_set(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch="40-9")
    m.maybe_counter_weight(_FakeChain(blocks_since=None), _FakeApi())
    assert submitted  # None blocks_since → due


def test_counter_weight_skips_when_not_due(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch="40-123")
    api = _FakeApi()
    m.maybe_counter_weight(_FakeChain(blocks_since=50), api)  # < 300
    assert not submitted and api.fetched == []


def test_counter_weight_noop_when_disabled(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=False,
                         hotkey="5Faudit", clean_epoch="40-123")
    m.maybe_counter_weight(_FakeChain(blocks_since=500), _FakeApi())
    assert not submitted


def test_counter_weight_skips_without_wallet(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey=None, clean_epoch="40-123")
    m.maybe_counter_weight(_FakeChain(blocks_since=500), _FakeApi())
    assert not submitted


def test_counter_weight_due_but_no_clean_epoch(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch=None)
    api = _FakeApi()
    m.maybe_counter_weight(_FakeChain(blocks_since=500), api)
    assert not submitted and api.fetched == []


# --- SN51 tempo-boundary path (tempo != None) ----------------------------
# blocks_left is computed in-code from `current`: for netuid 40, tempo 360,
# blocks_until_next_epoch(current) = 360 - (current+41) % 361. current=317 → 2
# (in the lead window); current=1000 → 41 (far from the boundary).
def test_counter_weight_sets_at_tempo_boundary(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch="40-123")
    api = _FakeApi()
    # current=317 → 2 blocks to boundary (≤ lead 2), no prior local set → sets
    m.maybe_counter_weight(
        _FakeChain(blocks_since=300, current=317, tempo=360, rate_limit=100), api)
    assert submitted and api.fetched == ["40-123"]


def test_counter_weight_double_set_guard(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch="40-123")
    # at the boundary (current=318 → 1 block left) but we set just 3 blocks ago
    # (.audit_published=315, 318-315=3 < floor 100), not stale → local guard SKIPs
    (tmp_path / "pub").write_text("315")
    m.maybe_counter_weight(
        _FakeChain(blocks_since=300, current=318, tempo=360, rate_limit=100), _FakeApi())
    assert not submitted


def test_counter_weight_skips_far_from_boundary(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch="40-123")
    # current=1000 → 41 blocks to boundary (> lead 2) → not due
    m.maybe_counter_weight(
        _FakeChain(blocks_since=200, current=1000, tempo=360, rate_limit=100), _FakeApi())
    assert not submitted


def test_counter_weight_deadman_sets_far_from_boundary(monkeypatch, tmp_path):
    m, submitted = _wire(monkeypatch, tmp_path, enabled=True,
                         hotkey="5Faudit", clean_epoch="40-123")
    # far from boundary (current=1000 → 41) but > 2 tempos since last set
    # (800 > 720) → dead-man fires and is exempt from the double-set guard
    (tmp_path / "pub").write_text("999")  # would block, but dead-man overrides
    m.maybe_counter_weight(
        _FakeChain(blocks_since=800, current=1000, tempo=360, rate_limit=100), _FakeApi())
    assert submitted
