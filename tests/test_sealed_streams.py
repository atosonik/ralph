"""Tests for eval/sealed_streams.py — sampler + manifest + loader.

Covers:
  * SealedStreamSpec field validation
  * SealedStreamManifest construction + auto root-hash + duplicate-id
    rejection
  * compute_streams_root_hash determinism + order-invariance
  * select_active_streams determinism + distinctness + fairness +
    error paths
  * load_stream happy path + SHA mismatch + length mismatch + missing
    file + KeyError on unknown id
  * bytes_per_token_for lookup
  * read/write manifest JSON round-trip + atomicity + validation
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from eval.sealed_streams import (
    MANIFEST_VERSION,
    SealedStreamBatch,
    SealedStreamManifest,
    SealedStreamSpec,
    bytes_per_token_for,
    compute_streams_root_hash,
    load_stream,
    read_manifest,
    select_active_streams,
    write_manifest,
)

# ----------------------------------------------------------------------------
# Test fixtures
# ----------------------------------------------------------------------------

VALID_SHA = "a" * 64


def _spec(
    stream_id: str = "stream_00",
    corpus: str = "fineweb-edu",
    sub_genre: str = "english_prose",
    n_tokens: int = 1024,
    bytes_per_token: float = 4.0,
    sha256: str = VALID_SHA,
) -> SealedStreamSpec:
    return SealedStreamSpec(
        id=stream_id,
        corpus=corpus,
        sub_genre=sub_genre,
        n_tokens=n_tokens,
        bytes_per_token=bytes_per_token,
        sha256=sha256,
    )


def _write_stream_file(
    pool_dir: Path,
    stream_id: str,
    n_tokens: int,
    fill_value: int = 0,
) -> Path:
    """Write a synthetic stream file. Returns the path."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    arr = np.full(n_tokens, fill_value, dtype=np.uint16)
    path = pool_dir / f"{stream_id}.bin"
    arr.tofile(path)
    return path


def _real_sha_for_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ============================================================================
# SealedStreamSpec
# ============================================================================


class TestSealedStreamSpec:
    def test_minimum_valid(self):
        s = _spec()
        assert s.id == "stream_00"

    def test_frozen(self):
        s = _spec()
        with pytest.raises(Exception):
            s.id = "stream_99"  # type: ignore[misc]

    def test_empty_id_rejected(self):
        with pytest.raises(ValueError, match=r"id must be non-empty"):
            _spec(stream_id="")

    def test_zero_n_tokens_rejected(self):
        with pytest.raises(ValueError, match=r"n_tokens must be > 0"):
            _spec(n_tokens=0)

    def test_negative_bytes_per_token_rejected(self):
        with pytest.raises(ValueError, match=r"bytes_per_token must be > 0"):
            _spec(bytes_per_token=-1.0)

    def test_bad_sha256_length_rejected(self):
        with pytest.raises(ValueError, match=r"64-char hex digest"):
            _spec(sha256="short")

    def test_empty_sha256_rejected(self):
        with pytest.raises(ValueError, match=r"64-char hex digest"):
            _spec(sha256="")


# ============================================================================
# SealedStreamManifest + root hash
# ============================================================================


class TestSealedStreamManifest:
    def test_empty_streams_rejected(self):
        with pytest.raises(ValueError, match=r"streams must be non-empty"):
            SealedStreamManifest(streams=[])

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValueError, match=r"duplicate stream ids"):
            SealedStreamManifest(streams=[_spec("a"), _spec("a")])

    def test_auto_root_hash_computed(self):
        m = SealedStreamManifest(streams=[_spec("a"), _spec("b")])
        assert m.streams_root_hash != ""
        assert len(m.streams_root_hash) == 64

    def test_supplied_root_hash_preserved(self):
        explicit = "f" * 64
        m = SealedStreamManifest(
            streams=[_spec("a")],
            streams_root_hash=explicit,
        )
        # Note: __post_init__ uses the supplied value if non-empty; the
        # read_manifest path explicitly re-computes and asserts match.
        assert m.streams_root_hash == explicit

    def test_default_version(self):
        m = SealedStreamManifest(streams=[_spec()])
        assert m.version == MANIFEST_VERSION

    def test_spec_for_returns_match(self):
        m = SealedStreamManifest(streams=[_spec("a"), _spec("b")])
        s = m.spec_for("a")
        assert s.id == "a"

    def test_spec_for_unknown_raises(self):
        m = SealedStreamManifest(streams=[_spec("a")])
        with pytest.raises(KeyError):
            m.spec_for("not_a_real_id")


class TestComputeRootHash:
    def test_deterministic_same_streams(self):
        a = compute_streams_root_hash([_spec("a", sha256="b" * 64)])
        b = compute_streams_root_hash([_spec("a", sha256="b" * 64)])
        assert a == b

    def test_order_invariant(self):
        """Same streams in different order → same root hash."""
        s1, s2 = _spec("s1", sha256="b" * 64), _spec("s2", sha256="c" * 64)
        assert compute_streams_root_hash([s1, s2]) == compute_streams_root_hash([s2, s1])

    def test_different_streams_different_hash(self):
        a = compute_streams_root_hash([_spec("a", sha256="b" * 64)])
        b = compute_streams_root_hash([_spec("a", sha256="c" * 64)])
        assert a != b

    def test_returns_hex_digest(self):
        h = compute_streams_root_hash([_spec()])
        assert len(h) == 64
        int(h, 16)  # valid hex


# ============================================================================
# select_active_streams
# ============================================================================


class TestSelectActiveStreams:
    def test_deterministic(self):
        beacon = b"deadbeef"
        a = select_active_streams(beacon, n_in_pool=10, n_active=2)
        b = select_active_streams(beacon, n_in_pool=10, n_active=2)
        assert a == b

    def test_different_beacons_can_differ(self):
        """Across many beacons, selections vary (basic sanity)."""
        a = select_active_streams(b"beacon1", n_in_pool=10, n_active=2)
        # Not guaranteed for any TWO specific beacons; this asserts
        # at least one of N random pairs differs from `a`.
        all_same = all(
            select_active_streams(
                f"beacon{i}".encode(), n_in_pool=10, n_active=2,
            ) == a
            for i in range(20)
        )
        assert not all_same

    def test_distinct_indices(self):
        """The output indices are pairwise distinct."""
        for i in range(50):
            indices = select_active_streams(
                f"beacon{i}".encode(), n_in_pool=10, n_active=5,
            )
            assert len(set(indices)) == len(indices)

    def test_indices_in_range(self):
        for i in range(50):
            indices = select_active_streams(
                f"beacon{i}".encode(), n_in_pool=10, n_active=3,
            )
            for idx in indices:
                assert 0 <= idx < 10

    def test_all_streams_selectable(self):
        """Across 500 epochs, every stream appears at least once in
        the active set."""
        seen: set[int] = set()
        for i in range(500):
            indices = select_active_streams(
                f"epoch{i}".encode(), n_in_pool=10, n_active=2,
            )
            seen.update(indices)
        assert seen == set(range(10))

    def test_fair_distribution(self):
        """Across many beacons, selection frequency is approximately
        uniform — each stream appears in ~20% of epochs when n_active=2
        out of n_in_pool=10."""
        counter: collections.Counter = collections.Counter()
        n_trials = 1000
        for i in range(n_trials):
            indices = select_active_streams(
                f"beacon{i}".encode(), n_in_pool=10, n_active=2,
            )
            counter.update(indices)
        # Each stream should be active in roughly n_trials * 2 / 10 = 200 trials.
        # Tolerance ±20% (loose so it doesn't flake; actual variance is ~5%).
        expected = n_trials * 2 / 10
        for stream_id in range(10):
            count = counter[stream_id]
            assert 0.8 * expected <= count <= 1.2 * expected, (
                f"stream {stream_id} active {count}/{n_trials} ({count/n_trials:.1%}); "
                f"expected ~{expected/n_trials:.1%}"
            )

    def test_n_active_one(self):
        indices = select_active_streams(b"x", n_in_pool=10, n_active=1)
        assert len(indices) == 1

    def test_n_active_equals_pool(self):
        """When n_active == n_in_pool, the output is a permutation."""
        indices = select_active_streams(b"x", n_in_pool=5, n_active=5)
        assert set(indices) == set(range(5))

    def test_n_active_zero_rejected(self):
        with pytest.raises(ValueError, match=r"n_active must be >= 1"):
            select_active_streams(b"x", n_in_pool=10, n_active=0)

    def test_n_active_exceeds_pool_rejected(self):
        with pytest.raises(ValueError, match=r"n_active.*> n_in_pool"):
            select_active_streams(b"x", n_in_pool=5, n_active=10)

    def test_empty_beacon_rejected(self):
        with pytest.raises(ValueError, match=r"beacon must be non-empty"):
            select_active_streams(b"", n_in_pool=10, n_active=2)


# ============================================================================
# load_stream
# ============================================================================


class TestLoadStream:
    def test_happy_path(self, tmp_path):
        path = _write_stream_file(tmp_path, "stream_00", n_tokens=128)
        sha = _real_sha_for_file(path)
        manifest = SealedStreamManifest(streams=[
            _spec("stream_00", n_tokens=128, sha256=sha),
        ])
        batch = load_stream("stream_00", tmp_path, manifest)
        assert isinstance(batch, SealedStreamBatch)
        assert batch.spec.id == "stream_00"
        assert len(batch.tokens) == 128

    def test_missing_file_raises(self, tmp_path):
        manifest = SealedStreamManifest(streams=[
            _spec("stream_00", n_tokens=128, sha256=VALID_SHA),
        ])
        with pytest.raises(FileNotFoundError, match=r"sealed stream file"):
            load_stream("stream_00", tmp_path, manifest)

    def test_unknown_stream_id_raises(self, tmp_path):
        manifest = SealedStreamManifest(streams=[
            _spec("stream_00", n_tokens=128, sha256=VALID_SHA),
        ])
        with pytest.raises(KeyError):
            load_stream("stream_99", tmp_path, manifest)

    def test_sha_mismatch_raises(self, tmp_path):
        _write_stream_file(tmp_path, "stream_00", n_tokens=128)
        # Manifest claims a different SHA than the file actually has.
        manifest = SealedStreamManifest(streams=[
            _spec("stream_00", n_tokens=128, sha256="b" * 64),
        ])
        with pytest.raises(ValueError, match=r"sha mismatch"):
            load_stream("stream_00", tmp_path, manifest)

    def test_skip_sha_check(self, tmp_path):
        _write_stream_file(tmp_path, "stream_00", n_tokens=128)
        manifest = SealedStreamManifest(streams=[
            _spec("stream_00", n_tokens=128, sha256="b" * 64),
        ])
        # verify_sha=False bypasses the check.
        batch = load_stream("stream_00", tmp_path, manifest, verify_sha=False)
        assert len(batch.tokens) == 128

    def test_length_mismatch_raises(self, tmp_path):
        path = _write_stream_file(tmp_path, "stream_00", n_tokens=128)
        sha = _real_sha_for_file(path)
        # Manifest claims 256 tokens; file has 128.
        manifest = SealedStreamManifest(streams=[
            _spec("stream_00", n_tokens=256, sha256=sha),
        ])
        with pytest.raises(ValueError, match=r"length mismatch"):
            load_stream("stream_00", tmp_path, manifest)

    def test_returns_memmap_not_copy(self, tmp_path):
        path = _write_stream_file(tmp_path, "stream_00", n_tokens=128)
        sha = _real_sha_for_file(path)
        manifest = SealedStreamManifest(streams=[
            _spec("stream_00", n_tokens=128, sha256=sha),
        ])
        batch = load_stream("stream_00", tmp_path, manifest)
        assert isinstance(batch.tokens, np.memmap)


# ============================================================================
# bytes_per_token_for
# ============================================================================


class TestBytesPerTokenFor:
    def test_lookup_returns_spec_value(self):
        manifest = SealedStreamManifest(streams=[
            _spec("a", bytes_per_token=3.5),
            _spec("b", bytes_per_token=2.1),
        ])
        assert bytes_per_token_for("a", manifest) == 3.5
        assert bytes_per_token_for("b", manifest) == 2.1

    def test_unknown_id_raises(self):
        manifest = SealedStreamManifest(streams=[_spec("a")])
        with pytest.raises(KeyError):
            bytes_per_token_for("not_in_manifest", manifest)


# ============================================================================
# Manifest JSON I/O
# ============================================================================


class TestManifestJson:
    def test_round_trip(self, tmp_path):
        original = SealedStreamManifest(streams=[
            _spec("stream_00", corpus="fineweb-edu",
                  sub_genre="english_prose",
                  n_tokens=200000, bytes_per_token=4.05, sha256="a" * 64),
            _spec("stream_01", corpus="starcoderdata",
                  sub_genre="python",
                  n_tokens=200000, bytes_per_token=3.2, sha256="b" * 64),
        ])
        path = tmp_path / "manifest.json"
        write_manifest(original, path)
        restored = read_manifest(path)
        assert restored.version == original.version
        assert restored.streams_root_hash == original.streams_root_hash
        assert len(restored.streams) == 2
        assert restored.streams[0].id == "stream_00"
        assert restored.streams[1].corpus == "starcoderdata"
        assert restored.streams[0].bytes_per_token == 4.05

    def test_creates_parent_dirs(self, tmp_path):
        manifest = SealedStreamManifest(streams=[_spec()])
        path = tmp_path / "nested" / "deep" / "manifest.json"
        write_manifest(manifest, path)
        assert path.exists()

    def test_atomic_no_tmp_leftover(self, tmp_path):
        manifest = SealedStreamManifest(streams=[_spec()])
        path = tmp_path / "manifest.json"
        write_manifest(manifest, path)
        assert not (tmp_path / "manifest.json.tmp").exists()

    def test_meta_marker_present(self, tmp_path):
        manifest = SealedStreamManifest(streams=[_spec()])
        path = tmp_path / "manifest.json"
        write_manifest(manifest, path)
        loaded = json.loads(path.read_text())
        assert loaded["_meta"] == "karpa-sealed-pool-manifest"
        assert loaded["version"] == MANIFEST_VERSION

    def test_human_readable_output(self, tmp_path):
        manifest = SealedStreamManifest(streams=[_spec()])
        path = tmp_path / "manifest.json"
        write_manifest(manifest, path)
        text = path.read_text()
        # indent=2 produces multi-line output.
        assert "\n  " in text

    def test_read_rejects_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(ValueError, match=r"not valid JSON"):
            read_manifest(path)

    def test_read_rejects_missing_meta(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"version": "v1", "streams": []}))
        with pytest.raises(ValueError, match=r"_meta marker"):
            read_manifest(path)

    def test_read_rejects_wrong_version(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "_meta": "karpa-sealed-pool-manifest",
            "version": "v999",
            "streams": [],
        }))
        with pytest.raises(ValueError, match=r"version mismatch"):
            read_manifest(path)

    def test_read_rejects_missing_streams_key(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({
            "_meta": "karpa-sealed-pool-manifest",
            "version": MANIFEST_VERSION,
        }))
        with pytest.raises(ValueError, match=r"missing 'streams'"):
            read_manifest(path)

    def test_read_detects_root_hash_tampering(self, tmp_path):
        """A hand-edited manifest where a stream's sha was changed but
        the streams_root_hash wasn't updated raises ValueError."""
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({
            "_meta": "karpa-sealed-pool-manifest",
            "version": MANIFEST_VERSION,
            "streams_root_hash": "f" * 64,  # fake root hash
            "streams": [{
                "id": "stream_00",
                "corpus": "x",
                "sub_genre": "y",
                "n_tokens": 1024,
                "bytes_per_token": 4.0,
                "sha256": "a" * 64,
            }],
        }))
        with pytest.raises(ValueError, match=r"streams_root_hash mismatch"):
            read_manifest(path)
