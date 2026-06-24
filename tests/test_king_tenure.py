"""King-tenure guard: a freshly-crowned king reigns a minimum number of blocks
(RALPH_KING_MIN_TENURE_BLOCKS) before a challenger can dethrone it, so every
king earns at least one weight cycle. Challengers inside the tenure are deferred
(left pending, not crowned, not archived)."""

from chain_layer.interface import KingRecord
from chain_layer.local import LocalChain
import validator.service as service


def _king(block, hk="5KINGold"):
    return KingRecord(
        miner_hotkey=hk, bundle_hash="kbh", val_bpb=2.0, benchmark_accuracy=0.1,
        compute_cost=0.0, crowned_at=0.0, crowned_at_block=block,
    )


class _Score:
    compute_cost = 0.0


def _accepted_result(hk="5CHALLENGER"):
    return {
        "status": "king_change", "classification": "king_change", "weight_credit": 1.0,
        "miner_hotkey": hk, "miner_github": "", "pr_url": "", "bundle_hash": "cbh",
        "val_bpb": 1.0, "benchmark_accuracy": 0.5, "quality_gain": 0.5, "score": 0.5,
        "tier": "verified", "decisive": True, "accepted": True, "is_first": False,
        "result": None, "score_report": _Score(),
    }


def _setup(tmp_path, monkeypatch, current_block, king_block):
    chain = LocalChain(tmp_path / "chain")
    chain.set_king(_king(king_block))
    monkeypatch.setattr(chain, "get_current_block", lambda: current_block)
    monkeypatch.setattr(chain, "is_hotkey_registered", lambda hk: True)
    monkeypatch.setattr(service, "score_and_decide", lambda *a, **k: _accepted_result())
    qd = tmp_path / "queue"
    b = qd / "pending" / "cbh"
    b.mkdir(parents=True)
    (b / "submission.json").write_text("{}")
    return chain, qd


def test_crowned_at_block_round_trips(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    chain.set_king(_king(8_477_000))
    assert chain.get_king().crowned_at_block == 8_477_000


def test_challenger_deferred_within_tenure(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_KING_MIN_TENURE_BLOCKS", "300")
    chain, qd = _setup(tmp_path, monkeypatch, current_block=1000, king_block=950)  # age 50 < 300
    service.run_epoch(chain, qd, noise_floor_margin=0.013, hf_repo=None, audit_reports_enabled=False)

    assert chain.get_king().miner_hotkey == "5KINGold", "king must NOT change inside tenure"
    assert (qd / "pending" / "cbh" / "submission.json").exists(), "challenger left pending for re-eval"
    assert not (qd / "scored" / "cbh").exists(), "challenger not archived while deferred"


def test_challenger_crowns_after_tenure(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_KING_MIN_TENURE_BLOCKS", "300")
    chain, qd = _setup(tmp_path, monkeypatch, current_block=1000, king_block=600)  # age 400 > 300
    service.run_epoch(chain, qd, noise_floor_margin=0.013, hf_repo=None, audit_reports_enabled=False)

    k = chain.get_king()
    assert k.miner_hotkey == "5CHALLENGER", "tenure lapsed → challenger crowns"
    assert k.crowned_at_block == 1000, "new king stamps current block"
    assert (qd / "scored" / "cbh").exists(), "crowned bundle archived to scored"
