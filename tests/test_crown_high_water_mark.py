"""High-water-mark crown gate: an empty/cleared throne must NOT hand out a
gain-0 free crown.

Background: a king record can vanish transiently — clear_king from the
no-merged-PR reinstate path, or a missing/corrupt king.json. The old logic
crowned on `accepted OR is_first` with `is_first = (king is None)`, so ANY
submission grabbed the crown while the throne was empty (quality_gain 0, no bar
to beat). The fix persists the last-crowned king's quality bar in
chain_dir/high_water.json (NOT removed by clear_king) and makes is_first true
ONLY at genuine genesis (no king ever). A challenger on an empty throne must
still decisively beat the persisted bar.

Covered here:
  * ChainInterface default get/set_high_water_mark persistence + survives
    clear_king + None at genesis + corrupt-file tolerance (LocalChain, real).
  * router.process_submission: genesis crowns once and records the bar; a
    gain-0 challenger on a CLEARED throne is rejected; a decisive challenger on
    a cleared throne is crowned (real score_bundle, faked judge_submission).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
import validator.router as router
from chain_layer.local import LocalChain

# ============================================================================
# ChainInterface default high-water-mark methods (real LocalChain)
# ============================================================================


def test_hwm_none_at_genesis(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    assert chain.get_high_water_mark() is None


def test_hwm_roundtrips(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    chain.set_high_water_mark(1.42, 0.55)
    hwm = chain.get_high_water_mark()
    assert hwm == {"val_bpb": 1.42, "benchmark_accuracy": 0.55}


def test_hwm_survives_clear_king(tmp_path):
    """The whole point: clear_king empties the throne but the bar persists."""
    from chain_layer.interface import KingRecord

    chain = LocalChain(tmp_path / "chain")
    chain.set_king(KingRecord(
        miner_hotkey="k", bundle_hash="bh", val_bpb=1.30,
        benchmark_accuracy=0.60, compute_cost=1.0, crowned_at=0.0,
    ))
    chain.set_high_water_mark(1.30, 0.60)

    chain.clear_king()
    assert chain.get_king() is None              # throne emptied
    assert chain.get_high_water_mark() == {"val_bpb": 1.30, "benchmark_accuracy": 0.60}


def test_hwm_corrupt_file_returns_none(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    (Path(chain.chain_dir) / "high_water.json").write_text("{not json")
    assert chain.get_high_water_mark() is None


def test_hwm_missing_val_bpb_returns_none(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    (Path(chain.chain_dir) / "high_water.json").write_text('{"benchmark_accuracy": 0.5}')
    assert chain.get_high_water_mark() is None


# ============================================================================
# router.process_submission crown decision (real score_bundle, faked judge)
# ============================================================================


def _fake_judge(*, val_bpb, benchmark_accuracy, tier="verified"):
    """Build a stand-in judge_submission result with just the fields
    process_submission reads. Not rejected → reaches the crown decision."""
    return types.SimpleNamespace(
        rejected=None,
        miner_hotkey="5F_challenger",
        bundle_hash="bh_challenger",
        handshake_nonce="nonce",
        hidden_eval=types.SimpleNamespace(
            val_bpb=val_bpb, benchmark_accuracy=benchmark_accuracy
        ),
        calibration={"matmul_ms": 10.0},
        training_summary={"wall_clock_s": 1.0},
        operations={"op2_attestation": {"tier": tier}},
        to_dict=lambda: {},
    )


@pytest.fixture
def patched_judge(monkeypatch):
    """Returns a setter so each test picks the challenger's metrics."""
    holder: dict = {}

    def _set(**kw):
        holder["result"] = _fake_judge(**kw)

    monkeypatch.setattr(router, "judge_submission", lambda *a, **k: holder["result"])
    return _set


def test_genesis_crowns_once_and_records_bar(tmp_path, patched_judge):
    """No king + no high-water mark → is_first → crowned, and the bar is saved."""
    patched_judge(val_bpb=2.00, benchmark_accuracy=0.40)
    out = router.process_submission(tmp_path, tmp_path / "proof_a")

    assert out["status"] == "accepted"
    king = router._load_king(tmp_path)
    assert king is not None and king["val_bpb"] == 2.00
    # The bar is now persisted for any future empty-throne window.
    assert router._load_high_water(tmp_path) == {"val_bpb": 2.00, "benchmark_accuracy": 0.40}


def test_gain_zero_challenger_on_cleared_throne_is_rejected(tmp_path, patched_judge):
    """THE GAP: genesis crowns at 2.00 → throne cleared → an identical (gain-0)
    submission must NOT free-crown. It must decisively beat the persisted bar."""
    patched_judge(val_bpb=2.00, benchmark_accuracy=0.40)
    router.process_submission(tmp_path, tmp_path / "proof_a")  # crown @ 2.00

    # Throne goes transiently empty (e.g. clear_king from the reinstate path).
    (router._chain_dir(tmp_path) / "king.json").unlink()
    assert router._load_king(tmp_path) is None
    assert router._load_high_water(tmp_path) is not None  # but the bar survives

    # Same metrics as the bar → quality_gain 0, benchmark_gain 0 → not decisive.
    patched_judge(val_bpb=2.00, benchmark_accuracy=0.40)
    out = router.process_submission(tmp_path, tmp_path / "proof_b")

    assert out["status"] == "below_threshold"          # NOT crowned
    assert router._load_king(tmp_path) is None          # throne still empty


def test_decisive_challenger_on_cleared_throne_is_crowned(tmp_path, patched_judge):
    """A genuinely better challenger on an empty throne still crowns — the bar
    gates free crowns, it does not freeze the throne."""
    patched_judge(val_bpb=2.00, benchmark_accuracy=0.40)
    router.process_submission(tmp_path, tmp_path / "proof_a")  # crown @ 2.00
    (router._chain_dir(tmp_path) / "king.json").unlink()       # empty throne, bar persists

    patched_judge(val_bpb=1.00, benchmark_accuracy=0.40)       # decisively better
    out = router.process_submission(tmp_path, tmp_path / "proof_b")

    assert out["status"] == "accepted"
    king = router._load_king(tmp_path)
    assert king is not None and king["val_bpb"] == 1.00
    assert router._load_high_water(tmp_path)["val_bpb"] == 1.00  # bar advances
