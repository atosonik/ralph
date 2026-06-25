"""Auditor fixes: Gate-1 queries the commitment at chain_commitment_block (not
epoch_end_block), and SN51-style tempo-boundary weight timing."""

import auditor.main as am
from auditor.weights import (
    blocks_until_next_epoch,
    is_weight_set_due_tempo,
    set_lead_blocks,
)


# --------------------------------------------------------- Gate-1 block fix
class _FakeChain:
    def __init__(self):
        self.queried_block = None

    def get_commitment_hash(self, at_block, hotkey=None):
        self.queried_block = at_block
        return None  # → self-consistency-only path; verify_report still runs


class _FakeApi:
    def __init__(self, env):
        self._env = env

    def get_report(self, epoch_id):
        return self._env


def test_gate1_queries_chain_commitment_block():
    env = {
        "chain_commitment_block": 8_479_982,  # where the commit actually landed
        "signer_hotkey": "5Hsigner",
        "report_json": {"epoch_end_block": 8_479_978, "epoch_id": "40-8479978"},
    }
    chain = _FakeChain()
    am.audit_epoch("40-8479978", chain, _FakeApi(env))  # returns EXIT_HASH_OR_SIG; we only check the block
    assert chain.queried_block == 8_479_982, "must query chain_commitment_block, not epoch_end_block"


def test_gate1_falls_back_to_epoch_end_block_for_old_reports():
    env = {  # no chain_commitment_block (older report)
        "signer_hotkey": "5Hsigner",
        "report_json": {"epoch_end_block": 8_479_978, "epoch_id": "x"},
    }
    chain = _FakeChain()
    am.audit_epoch("x", chain, _FakeApi(env))
    assert chain.queried_block == 8_479_978, "fall back to epoch_end_block when chain_commitment_block absent"


# --------------------------------------------------------- tempo boundary formula
def test_blocks_until_next_epoch_formula():
    # tempo - (block + netuid + 1) % (tempo + 1); netuid 40, tempo 360
    assert blocks_until_next_epoch(317, 40, 360) == 2          # 2 blocks before boundary
    assert blocks_until_next_epoch(317 + 361, 40, 360) == 2    # periodic in tempo+1
    assert blocks_until_next_epoch(319, 40, 360) == 0          # on the boundary
    for blk in range(0, 1000, 37):
        v = blocks_until_next_epoch(blk, 40, 360)
        assert 0 <= v <= 360
    assert blocks_until_next_epoch(123, 40, 0) == 0            # degenerate tempo


# --------------------------------------------------------- due decision
def test_is_weight_set_due_tempo():
    # within lead of the boundary → due
    assert is_weight_set_due_tempo(blocks_left=2, blocks_since=50, tempo=360, lead=2) is True
    assert is_weight_set_due_tempo(blocks_left=1, blocks_since=50, tempo=360, lead=2) is True
    # far from boundary, recently set → NOT due
    assert is_weight_set_due_tempo(blocks_left=120, blocks_since=50, tempo=360, lead=2) is False
    # never set → due immediately
    assert is_weight_set_due_tempo(blocks_left=120, blocks_since=None, tempo=360, lead=2) is True
    # dead-man: > 2 tempos since last set → due regardless of phase
    assert is_weight_set_due_tempo(blocks_left=120, blocks_since=800, tempo=360, lead=2) is True


def test_set_lead_blocks(monkeypatch):
    monkeypatch.delenv("AUDITOR_SET_LEAD_BLOCKS", raising=False)
    assert set_lead_blocks() == 2
    monkeypatch.setenv("AUDITOR_SET_LEAD_BLOCKS", "5")
    assert set_lead_blocks() == 5
    monkeypatch.setenv("AUDITOR_SET_LEAD_BLOCKS", "-1")  # invalid → default
    assert set_lead_blocks() == 2
    monkeypatch.setenv("AUDITOR_SET_LEAD_BLOCKS", "x")   # invalid → default
    assert set_lead_blocks() == 2
