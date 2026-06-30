"""The trainedness guard must reject the random-init / log-mismatched checkpoints
that slipped through (e.g. the uid155 fraud) while never rejecting a real model."""
from __future__ import annotations

import math

import pytest

from validator.integrity import (
    check_checkpoint_trained,
    check_compute_plausibility,
    check_recipe_config_matches_proof,
    nats_per_token_from_bpb,
)

VOCAB = 50257
RANDOM_NATS = math.log(VOCAB)  # ~10.82


# --- compute-plausibility: anti compute-gaming -------------------------------
H100 = {"gpu_name": "NVIDIA H100 80GB HBM3"}


def test_rejects_fabricated_compute_the_5ctaoqf1_king():
    # 5.557B tokens in 6788s on ONE H100 => 818k tok/s => ~126% MFU = impossible.
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 6787.88, "n_params": 253_874_184}
    ok, reason = check_compute_plausibility(fs, H100)
    assert not ok and "fabricated compute" in reason and "MFU" in reason


def test_accepts_a_real_30h_run():
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 109_000, "n_params": 253_874_184}  # ~51k tok/s
    assert check_compute_plausibility(fs, H100)[0]


def test_accepts_an_optimized_run_under_the_ceiling():
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 22_000, "n_params": 253_874_184}  # ~250k tok/s, ~39% MFU
    assert check_compute_plausibility(fs, H100)[0]


def test_incomplete_training_summary_is_skipped_not_rejected():
    assert check_compute_plausibility({"tokens_seen": 0, "wall_clock_s": 0}, {})[0]
    assert check_compute_plausibility({}, None)[0]


def test_unknown_gpu_uses_fastest_peak_to_avoid_false_reject():
    fs = {"tokens_seen": 5_557_452_800, "wall_clock_s": 22_000, "n_params": 253_874_184}
    assert check_compute_plausibility(fs, {"gpu_name": "Some Future GPU"})[0]


# --- declared-recipe-matches-proof -------------------------------------------
def test_rejects_config_step_mismatch_the_5ctaoqf1_king():
    patch = '+++ b/configs/muon_wsd_qknorm_b20593.json\n+{\n+  "total_steps": 40000,\n+  "qk_norm": true\n+}\n'
    ok, reason = check_recipe_config_matches_proof(patch, {"steps": 10600})
    assert not ok and "mismatch" in reason


def test_accepts_matching_config_steps():
    patch = '+++ b/configs/run.json\n+{\n+  "total_steps": 10600\n+}\n'
    assert check_recipe_config_matches_proof(patch, {"steps": 10600})[0]


def test_config_match_skips_when_no_config_or_no_steps():
    assert check_recipe_config_matches_proof("+++ b/model/x.py\n+x = 1\n", {"steps": 10600})[0]
    assert check_recipe_config_matches_proof('+++ b/configs/c.json\n+{"total_steps": 5}\n', {})[0]


def test_rejects_the_uid155_random_king():
    # Measured in the incident: ~11.0 nats/token, log claimed final_loss 3.05.
    ok, reason = check_checkpoint_trained(11.0, VOCAB, claimed_final_loss=3.0496)
    assert not ok
    assert "untrained" in reason


def test_rejects_random_even_without_a_claimed_loss():
    ok, reason = check_checkpoint_trained(RANDOM_NATS, VOCAB)
    assert not ok and "untrained" in reason


def test_accepts_a_real_trained_model():
    # A legit king sits at val_bpb ~1.3-1.6 -> ~3.6-4.4 nats/token.
    for val_bpb in (1.306, 1.336, 1.581):
        nats = nats_per_token_from_bpb(val_bpb, bytes_per_token=4.0)
        ok, reason = check_checkpoint_trained(nats, VOCAB, claimed_final_loss=3.0496)
        assert ok, f"false-rejected a real model (val_bpb={val_bpb}, nats={nats:.2f}): {reason}"


def test_catches_subtle_log_mismatch_below_random():
    # Not fully random (7 nats), but the log claims a much better 2.0 -> the
    # scored checkpoint clearly isn't from the declared run.
    ok, reason = check_checkpoint_trained(7.0, VOCAB, claimed_final_loss=2.0)
    assert not ok and "mismatch" in reason


def test_generous_to_normal_train_test_gap():
    # Held-out modestly worse than training must NOT trip the mismatch check.
    ok, _ = check_checkpoint_trained(4.4, VOCAB, claimed_final_loss=3.05)
    assert ok


def test_bpb_inversion_roundtrips():
    nats = nats_per_token_from_bpb(1.5, 4.0)
    assert nats == pytest.approx(1.5 * math.log(2) * 4.0)


def test_rejects_non_finite_and_bad_vocab():
    assert not check_checkpoint_trained(float("nan"), VOCAB)[0]
    assert not check_checkpoint_trained(3.5, 1)[0]
