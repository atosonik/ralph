"""Tests for sign_submission / verify_signature.

Covers the hypothesis-binding fix from deep_review_2026-05-31 high #2
and the file-mode hygiene from chunk 1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from miner.submit import _signed_payload, sign_submission, verify_signature


def test_signature_roundtrip(tmp_path):
    sig = sign_submission(tmp_path, "hk_alpha", "bundle_hash_xyz", "0xnonce_abc")
    assert verify_signature(
        "hk_alpha", "bundle_hash_xyz", "0xnonce_abc",
        sig["signature_hex"], sig["public_key_hex"],
    )


def test_signature_rejects_tampered_bundle_hash(tmp_path):
    sig = sign_submission(tmp_path, "hk_alpha", "bundle_hash_xyz", "0xnonce_abc")
    assert not verify_signature(
        "hk_alpha", "TAMPERED", "0xnonce_abc",
        sig["signature_hex"], sig["public_key_hex"],
    )


def test_signature_rejects_tampered_nonce(tmp_path):
    sig = sign_submission(tmp_path, "hk_alpha", "bundle_hash_xyz", "0xnonce_abc")
    assert not verify_signature(
        "hk_alpha", "bundle_hash_xyz", "0xTAMPERED",
        sig["signature_hex"], sig["public_key_hex"],
    )


def test_signature_rejects_tampered_hotkey(tmp_path):
    sig = sign_submission(tmp_path, "hk_alpha", "bundle_hash_xyz", "0xnonce_abc")
    assert not verify_signature(
        "hk_DIFFERENT", "bundle_hash_xyz", "0xnonce_abc",
        sig["signature_hex"], sig["public_key_hex"],
    )


def test_signature_rejects_malformed_pubkey(tmp_path):
    sig = sign_submission(tmp_path, "hk_alpha", "bundle_hash_xyz", "0xnonce_abc")
    assert not verify_signature(
        "hk_alpha", "bundle_hash_xyz", "0xnonce_abc",
        sig["signature_hex"], "not_hex_at_all",
    )


def test_hypothesis_in_payload(tmp_path):
    """Hypothesis hash is folded into the signed payload — same bundle but
    different hypothesis must verify under the matching hypothesis only."""
    sig = sign_submission(
        tmp_path, "hk", "bh", "0xn",
        hypothesis="We claim Lion outperforms AdamW at this scale.",
    )
    # Matching hypothesis verifies
    assert verify_signature(
        "hk", "bh", "0xn", sig["signature_hex"], sig["public_key_hex"],
        hypothesis="We claim Lion outperforms AdamW at this scale.",
    )
    # Different hypothesis rejected
    assert not verify_signature(
        "hk", "bh", "0xn", sig["signature_hex"], sig["public_key_hex"],
        hypothesis="We claim Adam outperforms Lion.",
    )


def test_back_compat_empty_hypothesis(tmp_path):
    """Pre-fix signatures (no hypothesis) verify under empty hypothesis and
    via the no-hypothesis verify path."""
    sig = sign_submission(tmp_path, "hk", "bh", "0xn")
    # Verify without hypothesis arg
    assert verify_signature("hk", "bh", "0xn", sig["signature_hex"], sig["public_key_hex"])
    # And under the hypothesis-aware path with empty string
    assert verify_signature("hk", "bh", "0xn", sig["signature_hex"], sig["public_key_hex"], hypothesis="")


def test_signed_payload_canonical():
    """The payload encoding must be deterministic — the hypothesis hash
    is the suffix only when non-empty."""
    base = _signed_payload("hk", "0xn", "bh")
    assert base == b"hk|0xn|bh"
    with_hyp = _signed_payload("hk", "0xn", "bh", hypothesis="x")
    assert with_hyp.startswith(b"hk|0xn|bh|")
    assert len(with_hyp) == len(b"hk|0xn|bh|") + 64  # sha256 hex


def test_key_file_mode_is_0600(tmp_path):
    sign_submission(tmp_path, "hk_mode_test", "bh", "0xn")
    sk_path = tmp_path / "miner" / "keys" / "hk_mode_test.sk"
    assert sk_path.exists()
    mode = sk_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0600, got {oct(mode)}"


def test_load_rejects_world_readable_key(tmp_path):
    """If a private key on disk is mode 0644, refuse to load it. Defense
    against a previously leaked key being silently re-used."""
    sign_submission(tmp_path, "hk_perm_test", "bh", "0xn")
    sk_path = tmp_path / "miner" / "keys" / "hk_perm_test.sk"
    os.chmod(sk_path, 0o644)
    with pytest.raises(RuntimeError, match="mode allows group/other"):
        sign_submission(tmp_path, "hk_perm_test", "bh", "0xn")
