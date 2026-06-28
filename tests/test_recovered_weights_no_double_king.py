"""Regression: a rate-limited *recovered* weight must never resurrect a king the
throne has moved off of. Before the fix, the recovery merge re-added the previous
king at king-level weight on top of the new king's, so set_weights emitted to BOTH
kings every epoch until a set_weights finally landed and cleared pending_weights.
"""
from validator.service import KING_POOL_FRACTION, _merge_recovered_weights


def test_king_change_drops_stale_recovered_king():
    # NEW king crowned this epoch; the pending file still holds the OLD king at
    # king-level weight from a previous rate-limited epoch.
    round_scores = {"NEW_KING": 1.0}
    recovered = {"OLD_KING": 1.0, "MF_MINER": 0.05}
    out = _merge_recovered_weights(dict(round_scores), recovered, "NEW_KING")
    assert "OLD_KING" not in out          # dethroned king is NOT resurrected
    assert out["NEW_KING"] == 1.0         # current king intact
    assert out["MF_MINER"] == 0.05        # sub-king mf credit still recovered


def test_current_king_value_is_authoritative():
    # round_scores reflects a 90/10 split this epoch; a stale 1.0 recovered for
    # the SAME king must not override the authoritative 0.9.
    round_scores = {"KING": KING_POOL_FRACTION, "MF": 0.1}
    recovered = {"KING": 1.0}
    out = _merge_recovered_weights(dict(round_scores), recovered, "KING")
    assert out["KING"] == KING_POOL_FRACTION


def test_meaningful_failure_recovery_preserved():
    # No king change; an mf credit that never landed last epoch is still recovered.
    round_scores = {"KING": 1.0}
    recovered = {"MF": 0.1}
    out = _merge_recovered_weights(dict(round_scores), recovered, "KING")
    assert out["MF"] == 0.1
    assert out["KING"] == 1.0


def test_no_current_king_still_drops_king_level_recovered():
    # Throne cleared (genesis / post-reset): a king-level recovered weight from
    # the prior reign must not sneak back in as a phantom king.
    out = _merge_recovered_weights({}, {"OLD_KING": 1.0}, None)
    assert out == {}
