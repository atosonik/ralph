"""The trainedness guard must reject the random-init / log-mismatched checkpoints
that slipped through (e.g. the uid155 fraud) while never rejecting a real model."""
from __future__ import annotations

import math

import pytest

from validator.integrity import (
    check_checkpoint_trained,
    nats_per_token_from_bpb,
)

VOCAB = 50257
RANDOM_NATS = math.log(VOCAB)  # ~10.82


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
