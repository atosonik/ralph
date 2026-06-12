"""C1-LITE validator/ladder.py acceptance-path tests.

Covers:
  * read_submission: bundle layout + missing/bad fields + optional cache
  * accept_submission: each rejection reason path + happy path + genesis
  * Chain event emission (submission_received) on every call
  * KARPA_VOCAB_SIZE enforcement
  * Branch id format
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from chain_layer.local import LocalChain
from eval.downstream.runner import KARPA_VOCAB_SIZE
from validator.ladder import (
    ACCEPT_OK,
    SUPPORTED_SCHEMA_VERSIONS,
    Submission,
    accept_submission,
    read_submission,
)
from validator.lineage import ParentCsdpCache

# ============================================================================
# Test fixtures
# ============================================================================


def _make_cache_dict(parent_hash: str = "a" * 64) -> dict:
    return {
        "parent_king_attestation_hash": parent_hash,
        "csdp_summary": {"s3_overall": 0.7},
        "streams_root_hash_at_evaluation": "f" * 64,
        "cached_at_iso": "2026-06-12T00:00:00Z",
        "cached_at_block": 1024,
        "quorum_signatures": [
            {"signer": "validator_1", "sig": "11" * 32},
        ],
    }


def _make_submission_bundle(
    tmp_path,
    *,
    schema_version: str = "v0.11",
    parent_hash: str | None = "a" * 64,
    branch_id: str = "main",
    bundle_hash: str = "bh",
    miner_hotkey: str = "5F_miner",
    vocab_size: int = KARPA_VOCAB_SIZE,
    include_cache: bool = True,
    cache_overrides: dict | None = None,
) -> Path:
    bundle = tmp_path / "submission"
    bundle.mkdir()
    sub = {
        "schema_version": schema_version,
        "parent_king_attestation_hash": parent_hash,
        "branch_id": branch_id,
        "bundle_hash": bundle_hash,
        "miner_hotkey": miner_hotkey,
        "vocab_size": vocab_size,
    }
    (bundle / "submission.json").write_text(json.dumps(sub))
    if include_cache and parent_hash:
        cache = _make_cache_dict(parent_hash=parent_hash)
        if cache_overrides:
            cache.update(cache_overrides)
        (bundle / "parent_csdp_cache.json").write_text(json.dumps(cache))
    return bundle


@pytest.fixture
def chain(tmp_path):
    c = LocalChain(tmp_path / "chain")
    return c


@pytest.fixture
def chain_with_king(tmp_path):
    c = LocalChain(tmp_path / "chain")
    c.append_event({
        "type": "king_crowned",
        "king_attestation_hash": "a" * 64,
        "miner_hotkey": "5F_parent",
        "branch_id": "main",
    })
    return c


# ============================================================================
# read_submission
# ============================================================================


def test_read_submission_happy_path(tmp_path):
    bundle = _make_submission_bundle(tmp_path)
    sub = read_submission(bundle)
    assert sub.schema_version == "v0.11"
    assert sub.parent_king_attestation_hash == "a" * 64
    assert sub.branch_id == "main"
    assert sub.vocab_size == KARPA_VOCAB_SIZE
    assert sub.parent_csdp_cache is not None
    assert sub.parent_csdp_cache.parent_king_attestation_hash == "a" * 64


def test_read_submission_missing_file_raises(tmp_path):
    bundle = tmp_path / "empty"
    bundle.mkdir()
    with pytest.raises(FileNotFoundError, match=r"submission.json"):
        read_submission(bundle)


def test_read_submission_missing_required_field_raises(tmp_path):
    bundle = tmp_path / "bad"
    bundle.mkdir()
    (bundle / "submission.json").write_text(json.dumps({"schema_version": "v0.11"}))
    with pytest.raises(ValueError, match=r"missing required fields"):
        read_submission(bundle)


def test_read_submission_genesis_omits_cache(tmp_path):
    bundle = _make_submission_bundle(
        tmp_path, parent_hash=None, include_cache=False,
    )
    sub = read_submission(bundle)
    assert sub.parent_king_attestation_hash is None
    assert sub.parent_csdp_cache is None


def test_read_submission_bad_cache_json_raises(tmp_path):
    bundle = _make_submission_bundle(tmp_path)
    (bundle / "parent_csdp_cache.json").write_text("{this is not json")
    with pytest.raises(ValueError, match=r"parent_csdp_cache.json invalid JSON"):
        read_submission(bundle)


# ============================================================================
# accept_submission rejection paths
# ============================================================================


def test_accept_rejects_unsupported_schema_version(chain):
    sub = Submission(
        schema_version="v0.99",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
    )
    result = accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    assert not result.accepted
    assert result.reason.startswith("bad_submission_format")


def test_accept_rejects_bad_branch_id(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="random_string",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
    )
    result = accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    assert not result.accepted
    assert result.reason == "bad_branch_id"


def test_accept_rejects_vocab_mismatch(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=40000,
    )
    result = accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    assert not result.accepted
    assert result.reason == "vocab_mismatch"


def test_accept_rejects_bad_parent_hash_format(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash="not_64_chars",
        branch_id="main",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
        parent_csdp_cache=ParentCsdpCache.from_dict(_make_cache_dict()),
    )
    result = accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    assert not result.accepted
    assert result.reason == "parent_unverifiable:bad_parent_hash_format"


def test_accept_rejects_parent_not_on_chain(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash="b" * 64,
        branch_id="main",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
        parent_csdp_cache=ParentCsdpCache.from_dict(_make_cache_dict(parent_hash="b" * 64)),
    )
    result = accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    assert not result.accepted
    assert result.reason == "parent_unverifiable:parent_not_on_chain"


# ============================================================================
# accept_submission happy paths
# ============================================================================


def test_accept_genesis_happy_path(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="bh_genesis",
        miner_hotkey="5F_genesis",
        vocab_size=KARPA_VOCAB_SIZE,
    )
    result = accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    assert result.accepted
    assert result.reason == ACCEPT_OK


def test_accept_with_parent_happy_path(chain_with_king):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash="a" * 64,
        branch_id="main",
        bundle_hash="bh_child",
        miner_hotkey="5F_child",
        vocab_size=KARPA_VOCAB_SIZE,
        parent_csdp_cache=ParentCsdpCache.from_dict(_make_cache_dict()),
    )
    result = accept_submission(sub, chain_with_king, now_iso="2026-06-12T00:00:00Z")
    assert result.accepted, f"unexpected reject reason: {result.reason}"
    assert result.reason == ACCEPT_OK


def test_accept_branch_open_format(chain_with_king):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash="a" * 64,
        branch_id="open_new_branch_my_idea",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
        parent_csdp_cache=ParentCsdpCache.from_dict(_make_cache_dict()),
    )
    result = accept_submission(sub, chain_with_king, now_iso="2026-06-12T00:00:00Z")
    assert result.accepted


def test_accept_existing_branch_format(chain_with_king):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash="a" * 64,
        branch_id="branch-3",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
        parent_csdp_cache=ParentCsdpCache.from_dict(_make_cache_dict()),
    )
    result = accept_submission(sub, chain_with_king, now_iso="2026-06-12T00:00:00Z")
    assert result.accepted


# ============================================================================
# Chain event emission
# ============================================================================


def test_accept_emits_submission_received_event(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
    )
    accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    events = chain.get_events(limit=10)
    received = [e for e in events if e["type"] == "submission_received"]
    assert len(received) == 1
    assert received[0]["accepted"] is True
    assert received[0]["reason"] == ACCEPT_OK
    assert received[0]["branch_id"] == "main"


def test_accept_emits_event_on_rejection(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=12345,
    )
    accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z")
    events = chain.get_events(limit=10)
    received = [e for e in events if e["type"] == "submission_received"]
    assert received[0]["accepted"] is False
    assert received[0]["reason"] == "vocab_mismatch"


def test_accept_event_emission_can_be_disabled(chain):
    sub = Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash="bh",
        miner_hotkey="5F",
        vocab_size=KARPA_VOCAB_SIZE,
    )
    accept_submission(sub, chain, now_iso="2026-06-12T00:00:00Z", emit_chain_event=False)
    assert chain.get_events(limit=10) == []


# ============================================================================
# Module constants pinning
# ============================================================================


def test_supported_schema_versions_pinned_to_v0_11_lite():
    assert SUPPORTED_SCHEMA_VERSIONS == {"v0.11"}
