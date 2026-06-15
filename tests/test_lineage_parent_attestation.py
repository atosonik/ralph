"""C1-LITE lineage primitive tests.

Covers:
  * compute_king_attestation_hash determinism + canonical form +
    quorum-sig stripping
  * is_valid_attestation_hash + case-sensitivity
  * ParentCsdpCache round-trip via to_dict/from_dict
  * verify_parent_lineage: genesis / missing cache / hash mismatch /
    bad format / unknown parent / age gate / signature structure /
    happy path
  * KingRecord serialization preserves the new lineage fields
    (LocalChain backend)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from chain_layer.interface import KingRecord
from chain_layer.local import LocalChain
from validator.lineage import (
    ATTESTATION_HASH_LEN,
    DEFAULT_CACHE_AGE_THRESHOLD_DAYS,
    ParentCsdpCache,
    compute_king_attestation_hash,
    is_valid_attestation_hash,
    verify_parent_lineage,
)

# ============================================================================
# compute_king_attestation_hash
# ============================================================================


def test_attestation_hash_deterministic():
    p = {"miner_hotkey": "5F6W...", "csdp": {"s3": 0.7}, "branch_id": "main"}
    assert compute_king_attestation_hash(p) == compute_king_attestation_hash(p)


def test_attestation_hash_canonical_key_order():
    """Key order in input dict must not affect the hash."""
    a = {"a": 1, "b": 2, "c": 3}
    b = {"c": 3, "b": 2, "a": 1}
    assert compute_king_attestation_hash(a) == compute_king_attestation_hash(b)


def test_attestation_hash_strips_quorum_sig():
    """Adding/removing validator_quorum_sig must not affect the hash —
    signatures are appended AFTER hashing."""
    base = {"king_attestation_hash": None, "branch_id": "main", "csdp": {"s3": 0.6}}
    with_sig = dict(base)
    with_sig["validator_quorum_sig"] = {"signers": ["v1", "v2"], "signatures": ["a" * 128]}
    assert compute_king_attestation_hash(base) == compute_king_attestation_hash(with_sig)


def test_attestation_hash_changes_with_content():
    a = {"miner_hotkey": "X", "csdp": {"s3": 0.6}}
    b = {"miner_hotkey": "X", "csdp": {"s3": 0.7}}
    assert compute_king_attestation_hash(a) != compute_king_attestation_hash(b)


def test_attestation_hash_format():
    h = compute_king_attestation_hash({"x": 1})
    assert len(h) == ATTESTATION_HASH_LEN
    assert h == h.lower()
    int(h, 16)  # valid hex


# ============================================================================
# is_valid_attestation_hash
# ============================================================================


def test_is_valid_accepts_real_hash():
    assert is_valid_attestation_hash(compute_king_attestation_hash({"x": 1}))


def test_is_valid_rejects_uppercase():
    h = compute_king_attestation_hash({"x": 1}).upper()
    assert not is_valid_attestation_hash(h)


def test_is_valid_rejects_wrong_length():
    assert not is_valid_attestation_hash("a" * 63)
    assert not is_valid_attestation_hash("a" * 65)


def test_is_valid_rejects_non_hex():
    assert not is_valid_attestation_hash("z" * 64)


def test_is_valid_rejects_non_string():
    assert not is_valid_attestation_hash(12345)
    assert not is_valid_attestation_hash(None)
    assert not is_valid_attestation_hash(b"a" * 64)


# ============================================================================
# ParentCsdpCache round-trip
# ============================================================================


def _sample_cache_dict(parent_hash: str = "a" * 64) -> dict:
    return {
        "parent_king_attestation_hash": parent_hash,
        "csdp_summary": {"s1_overall": 0.4, "s2_overall": 0.55, "s3_overall": 0.7},
        "streams_root_hash_at_evaluation": "deadbeef" * 8,
        "cached_at_iso": "2026-06-12T00:00:00Z",
        "cached_at_block": 1024,
        "quorum_signatures": [
            {"signer": "validator_1", "sig": "11" * 32},
            {"signer": "validator_2", "sig": "22" * 32},
        ],
    }


def test_parent_csdp_cache_round_trip():
    d = _sample_cache_dict()
    cache = ParentCsdpCache.from_dict(d)
    assert cache.to_dict() == d


def test_parent_csdp_cache_missing_field_raises():
    d = _sample_cache_dict()
    del d["streams_root_hash_at_evaluation"]
    with pytest.raises(KeyError):
        ParentCsdpCache.from_dict(d)


def test_parent_csdp_cache_quorum_sigs_default_empty():
    d = _sample_cache_dict()
    del d["quorum_signatures"]
    cache = ParentCsdpCache.from_dict(d)
    assert cache.quorum_signatures == []


# ============================================================================
# verify_parent_lineage
# ============================================================================


@pytest.fixture
def chain_with_king(tmp_path):
    """LocalChain seeded with one crowned king."""
    chain = LocalChain(tmp_path / "chain")
    parent_hash = "a" * 64
    chain.append_event({
        "type": "king_crowned",
        "king_attestation_hash": parent_hash,
        "miner_hotkey": "5F_parent",
        "branch_id": "main",
    })
    return chain, parent_hash


def test_verify_lineage_genesis_ok(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=None,
        parent_csdp_cache=None,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert ok and reason == "genesis_ok"


def test_verify_lineage_genesis_with_cache_rejected(tmp_path):
    """A cache without a parent hash is suspicious — reject."""
    chain = LocalChain(tmp_path / "chain")
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=None,
        parent_csdp_cache=ParentCsdpCache.from_dict(_sample_cache_dict()),
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason == "unexpected_parent_cache_at_genesis"


def test_verify_lineage_missing_cache(chain_with_king):
    chain, parent_hash = chain_with_king
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=None,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason == "missing_parent_cache"


def test_verify_lineage_bad_parent_hash_format(chain_with_king):
    chain, _ = chain_with_king
    ok, reason = verify_parent_lineage(
        parent_attestation_hash="not_a_real_hash",
        parent_csdp_cache=ParentCsdpCache.from_dict(_sample_cache_dict()),
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason == "bad_parent_hash_format"


def test_verify_lineage_hash_mismatch(chain_with_king):
    chain, parent_hash = chain_with_king
    cache = ParentCsdpCache.from_dict(_sample_cache_dict(parent_hash="b" * 64))
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason == "parent_hash_mismatch"


def test_verify_lineage_parent_not_on_chain(tmp_path):
    chain = LocalChain(tmp_path / "chain")  # empty chain
    parent_hash = "c" * 64
    cache = ParentCsdpCache.from_dict(_sample_cache_dict(parent_hash=parent_hash))
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason == "parent_not_on_chain"


def test_verify_lineage_cache_in_future(chain_with_king):
    chain, parent_hash = chain_with_king
    cache_dict = _sample_cache_dict(parent_hash=parent_hash)
    cache_dict["cached_at_iso"] = "2027-01-01T00:00:00Z"
    cache = ParentCsdpCache.from_dict(cache_dict)
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason == "cache_in_future"


def test_verify_lineage_cache_too_old(chain_with_king):
    chain, parent_hash = chain_with_king
    cache_dict = _sample_cache_dict(parent_hash=parent_hash)
    cache_dict["cached_at_iso"] = "2026-01-01T00:00:00Z"
    cache = ParentCsdpCache.from_dict(cache_dict)
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
        cache_age_threshold_days=DEFAULT_CACHE_AGE_THRESHOLD_DAYS,
    )
    assert not ok and reason == "cache_too_old"


def test_verify_lineage_no_quorum_signatures(chain_with_king):
    chain, parent_hash = chain_with_king
    cache_dict = _sample_cache_dict(parent_hash=parent_hash)
    cache_dict["quorum_signatures"] = []
    cache = ParentCsdpCache.from_dict(cache_dict)
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason == "no_quorum_signatures"


def test_verify_lineage_bad_signature_structure(chain_with_king):
    chain, parent_hash = chain_with_king
    cache_dict = _sample_cache_dict(parent_hash=parent_hash)
    cache_dict["quorum_signatures"] = [{"signer": "v1"}]  # no sig field
    cache = ParentCsdpCache.from_dict(cache_dict)
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert not ok and reason.startswith("signature_entry_0_bad_sig")


def test_verify_lineage_happy_path(chain_with_king):
    chain, parent_hash = chain_with_king
    cache = ParentCsdpCache.from_dict(_sample_cache_dict(parent_hash=parent_hash))
    ok, reason = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert ok, f"expected genesis_ok or accept; got reason={reason!r}"


def test_verify_lineage_fallback_to_get_king(tmp_path):
    """A parent that's the current king (no event log entry) is also accepted."""
    chain = LocalChain(tmp_path / "chain")
    parent_hash = "d" * 64
    chain.set_king(KingRecord(
        miner_hotkey="5F_parent",
        bundle_hash="bh_parent",
        val_bpb=1.0,
        benchmark_accuracy=0.4,
        compute_cost=0.0,
        crowned_at=0.0,
        king_attestation_hash=parent_hash,
        parent_king_attestation_hash=None,
    ))
    cache = ParentCsdpCache.from_dict(_sample_cache_dict(parent_hash=parent_hash))
    ok, _ = verify_parent_lineage(
        parent_attestation_hash=parent_hash,
        parent_csdp_cache=cache,
        chain=chain,
        now_iso="2026-06-12T00:00:00Z",
    )
    assert ok


# ============================================================================
# KingRecord serialization includes the new fields
# ============================================================================


def test_king_record_round_trip_lineage_fields(tmp_path):
    chain = LocalChain(tmp_path / "chain")
    parent_hash = "e" * 64
    king_hash = "f" * 64
    chain.set_king(KingRecord(
        miner_hotkey="5F_child",
        bundle_hash="bh_child",
        val_bpb=0.9,
        benchmark_accuracy=0.5,
        compute_cost=2.0,
        crowned_at=1234.0,
        king_attestation_hash=king_hash,
        parent_king_attestation_hash=parent_hash,
    ))
    got = chain.get_king()
    assert got is not None
    assert got.king_attestation_hash == king_hash
    assert got.parent_king_attestation_hash == parent_hash


def test_king_record_legacy_byte_equivalent_when_lineage_empty(tmp_path):
    """A v0.10-shaped king (no lineage fields) round-trips without
    introducing new keys to king.json."""
    import json as _json
    chain = LocalChain(tmp_path / "chain")
    chain.set_king(KingRecord(
        miner_hotkey="5F_legacy",
        bundle_hash="bh_legacy",
        val_bpb=1.5,
        benchmark_accuracy=0.2,
        compute_cost=0.0,
        crowned_at=0.0,
    ))
    on_disk = _json.loads((tmp_path / "chain" / "king.json").read_text())
    assert "king_attestation_hash" not in on_disk
    assert "parent_king_attestation_hash" not in on_disk
