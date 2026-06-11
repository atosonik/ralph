"""Sealed-stream pool sampler + manifest + loader (B2 foundation).

The validator owns a 10-stream sealed pool (200k uint16 tokens each) and
selects 2 active streams per epoch from an on-chain beacon. This module
ships the sampling + loading half of B2 (B2 scope §4); the build script
(`eval/build_sealed_pool.py`) that constructs the actual stream bytes
from HF corpora is a separate follow-up. CPU-testable end-to-end via
synthetic stream fixtures.

What this module ships:

  * `SealedStreamSpec` — per-stream metadata (id, corpus, sub_genre,
    n_tokens, bytes_per_token, sha256). Frozen dataclass.
  * `SealedStreamManifest` — list of specs + manifest version +
    streams_root_hash. `streams_root_hash` is `sha256(sorted_sha256s)`
    — a single fingerprint the on-chain `LadderCommitted` event will
    record, so any drift in the sealed pool is detectable from chain
    state alone.
  * `select_active_streams(beacon, *, n_in_pool=10, n_active=2)` —
    deterministic selection. For each stream index, derive a ranking
    key `sha256(beacon || index_big_endian_4byte)`, sort ascending by
    key, take the first `n_active` indices. Same beacon → same output;
    uniform-random beacons → uniform-random outputs; output indices
    are distinct by sort-uniqueness.
  * `load_stream(stream_id, pool_dir, manifest, *, verify_sha=True)` —
    np.memmap loader. Optionally hashes the file on open and rejects
    if SHA doesn't match the manifest.
  * `bytes_per_token_for(stream_id, manifest)` — per-stream
    bytes-per-token lookup so `compute_val_bpb` can stop hardcoding
    4.0 (B2 scope §"Files to MODIFY" — eval/val_bpb.py:74-77).
  * `read_manifest(path)` / `write_manifest(manifest, path)` — JSON I/O
    with `_meta` marker validation, atomic-write via .tmp + rename.

What this module does NOT ship (next B2 PRs):

  * The actual stream byte construction from HF corpora
    (`eval/build_sealed_pool.py`).
  * The public synthetic dev-stream pool + statistical-match verifier.
  * The on-chain commit helper (`scripts/commit_pool_v1.py`).
  * The TDX / LUKS / tmpfs deployment runbook.
  * `compute_val_bpb_on_stream` shim (depends on
    eval/val_bpb.py threading the per-stream bytes_per_token in;
    separate change).

Restricted-files note: this file lives under `eval/sealed_streams.py`
which is covered by the `eval/**` glob in `restricted_files.yaml` (and
explicitly via the `eval/private/streams/**` glob added in B1-D10 for
the runtime stream files themselves).

Reference scope: docs/build_scope/02_scope_B2.md §"Files to CREATE".
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

# JSON header marker; bump on any schema change to the manifest.
_MANIFEST_META = "karpa-sealed-pool-manifest"
MANIFEST_VERSION = "v1"


# ----------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class SealedStreamSpec:
    """Per-stream metadata. One entry per file in the sealed pool.

    Fields:
      * id — stable identifier, e.g. "stream_00". The file name
        convention is `{id}.bin` under the pool dir.
      * corpus — top-level corpus tag, e.g. "fineweb-edu", "starcoderdata",
        "open-web-math", "oasst2", "fineweb-2".
      * sub_genre — within-corpus tag for the composition table, e.g.
        "english_prose", "python", "math", "dialogue", "de_fr_es_ru".
        Empty string is allowed when the corpus has no sub-genre.
      * n_tokens — pinned token count. The validator asserts the
        memmap's element count equals this on load.
      * bytes_per_token — empirical ratio (decoded UTF-8 bytes /
        tokens) computed at construction time. Used by
        `compute_val_bpb` to convert NLL-per-token to NLL-per-byte.
      * sha256 — sha256 hex digest of the on-disk file bytes.
    """

    id: str
    corpus: str
    sub_genre: str
    n_tokens: int
    bytes_per_token: float
    sha256: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("SealedStreamSpec.id must be non-empty")
        if self.n_tokens <= 0:
            raise ValueError(
                f"SealedStreamSpec.n_tokens must be > 0; got {self.n_tokens}"
            )
        if self.bytes_per_token <= 0:
            raise ValueError(
                f"SealedStreamSpec.bytes_per_token must be > 0; "
                f"got {self.bytes_per_token}"
            )
        if not self.sha256 or len(self.sha256) != 64:
            raise ValueError(
                f"SealedStreamSpec.sha256 must be a 64-char hex digest; "
                f"got {self.sha256!r}"
            )


@dataclass
class SealedStreamManifest:
    """The pool's full manifest: list of specs + version + root hash.

    `streams_root_hash` is `sha256(b"".join(sorted(s.sha256.encode() for s
    in streams)))` — a single fingerprint that summarizes the pool.
    Comparing two manifests' root hashes is equivalent to comparing the
    sorted multi-set of per-stream SHAs. The `LadderCommitted` chain
    event records this single hash so any drift in the sealed pool is
    catchable from chain state alone.
    """

    streams: list[SealedStreamSpec]
    streams_root_hash: str = ""
    version: str = MANIFEST_VERSION

    def __post_init__(self) -> None:
        if not self.streams:
            raise ValueError("SealedStreamManifest.streams must be non-empty")
        # Reject duplicate IDs immediately — every spec must have a
        # distinct id so the loader's id → spec map is unambiguous.
        ids = [s.id for s in self.streams]
        if len(set(ids)) != len(ids):
            raise ValueError(
                f"duplicate stream ids in manifest: {sorted(ids)}"
            )
        # Auto-compute the root hash if the caller didn't supply one.
        # The computed hash is deterministic from the streams list.
        if not self.streams_root_hash:
            object.__setattr__(
                self, "streams_root_hash", self._compute_root_hash(),
            )

    def _compute_root_hash(self) -> str:
        return compute_streams_root_hash(self.streams)

    def spec_for(self, stream_id: str) -> SealedStreamSpec:
        """Lookup a spec by id. Raises KeyError if not present."""
        for s in self.streams:
            if s.id == stream_id:
                return s
        raise KeyError(
            f"stream id {stream_id!r} not in manifest "
            f"(have {sorted(s.id for s in self.streams)})"
        )


@dataclass(frozen=True)
class SealedStreamBatch:
    """One loaded stream + its metadata, ready for compute_val_bpb.

    Wraps the np.memmap + the spec so callers don't have to thread
    bytes_per_token + n_tokens separately. Frozen so the batch is
    hashable / safe to cache.
    """

    spec: SealedStreamSpec
    tokens: np.ndarray = field(hash=False, compare=False)


# ----------------------------------------------------------------------------
# Streams root hash
# ----------------------------------------------------------------------------


def compute_streams_root_hash(streams: list[SealedStreamSpec]) -> str:
    """`sha256(b"".join(sorted(spec.sha256.encode() for spec in streams)))`.

    Sorting before concatenation makes the root hash invariant to stream
    ordering inside the manifest — what matters is the multi-set of
    per-stream SHAs, not the order they were listed.
    """
    sorted_shas = sorted(s.sha256.encode("ascii") for s in streams)
    h = hashlib.sha256()
    for sha in sorted_shas:
        h.update(sha)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Stream selection (deterministic from beacon)
# ----------------------------------------------------------------------------


def select_active_streams(
    beacon: bytes,
    *,
    n_in_pool: int = 10,
    n_active: int = 2,
) -> tuple[int, ...]:
    """Deterministically select `n_active` distinct stream indices.

    Algorithm (constant-time, no PRNG state):
      For each stream index `i ∈ [0, n_in_pool)`, compute a ranking key
      `sha256(beacon || i_big_endian_uint32)`. Sort all indices by key
      ascending, return the first `n_active`. The 256-bit key space
      makes collisions astronomically unlikely, so the output is always
      `n_active` distinct indices.

    Properties:
      * Determinism: same beacon → same output across all callers.
      * Uniform fairness: for uniform-random beacons, each stream has
        equal probability of being in the active set (over the
        symmetry group of sha256).
      * Distinctness: sort-by-key is one-to-one in the absence of
        sha256 collisions, so the top `n_active` indices are distinct
        by construction.

    Args:
      beacon: arbitrary bytes that vary per epoch (typically the block
        hash from `chain_layer.ChainInterface.get_block_hash`).
      n_in_pool: total streams in the sealed pool. Default 10 matches
        the B2 spec; tests may pass smaller values.
      n_active: how many streams the epoch evaluates against. Default 2.

    Raises:
      ValueError if n_active > n_in_pool, n_active < 1, or beacon is
      empty (an empty beacon would make selection a constant across
      epochs, which defeats the purpose).
    """
    if n_active < 1:
        raise ValueError(f"n_active must be >= 1; got {n_active}")
    if n_active > n_in_pool:
        raise ValueError(
            f"n_active ({n_active}) > n_in_pool ({n_in_pool}); "
            "cannot select more streams than the pool contains"
        )
    if not beacon:
        raise ValueError(
            "beacon must be non-empty; an empty beacon makes selection "
            "constant across epochs"
        )

    ranked: list[tuple[bytes, int]] = []
    for i in range(n_in_pool):
        key = hashlib.sha256(beacon + i.to_bytes(4, "big")).digest()
        ranked.append((key, i))
    ranked.sort()
    return tuple(idx for _, idx in ranked[:n_active])


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """sha256 of a file's contents, chunked to keep memory bounded."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_stream(
    stream_id: str,
    pool_dir: Path,
    manifest: SealedStreamManifest,
    *,
    verify_sha: bool = True,
) -> SealedStreamBatch:
    """Load a sealed stream via `np.memmap` and return a `SealedStreamBatch`.

    Args:
      stream_id: identifier matching one of `manifest.streams[*].id`.
      pool_dir: directory containing `{id}.bin` files.
      manifest: the manifest the validator received via the out-of-band
        signed tarball; used to look up the spec and (when
        `verify_sha=True`) to validate the file bytes.
      verify_sha: if True (default), sha256 the file on open and raise
        ValueError on mismatch. Set False ONLY for tests that
        deliberately corrupt files to exercise other failure modes.

    Raises:
      KeyError if `stream_id` is not in the manifest.
      FileNotFoundError if `{pool_dir}/{stream_id}.bin` doesn't exist.
      ValueError if `verify_sha=True` and the file's sha256 doesn't
        match the manifest, OR if the loaded element count doesn't
        match the spec's `n_tokens`.
    """
    spec = manifest.spec_for(stream_id)
    path = Path(pool_dir) / f"{stream_id}.bin"
    if not path.exists():
        raise FileNotFoundError(
            f"sealed stream file not found: {path}"
        )

    if verify_sha:
        actual_sha = _sha256_file(path)
        if actual_sha != spec.sha256:
            raise ValueError(
                f"sealed stream {stream_id!r} sha mismatch:\n"
                f"  expected (from manifest): {spec.sha256}\n"
                f"  actual (from {path}):     {actual_sha}\n"
                "Either the file is corrupted or the manifest is out of date."
            )

    tokens = np.memmap(path, dtype=np.uint16, mode="r")
    if len(tokens) != spec.n_tokens:
        raise ValueError(
            f"sealed stream {stream_id!r} length mismatch: "
            f"manifest says n_tokens={spec.n_tokens}, file has "
            f"{len(tokens)} uint16 elements"
        )
    return SealedStreamBatch(spec=spec, tokens=tokens)


def bytes_per_token_for(
    stream_id: str,
    manifest: SealedStreamManifest,
) -> float:
    """Per-stream bytes-per-token lookup.

    Replaces the hardcoded `bytes_per_token = 4.0` in
    `eval/val_bpb.py:76`. The actual ratio varies by stream because
    non-English / code / math text tokenizes to different bytes-per-
    token under GPT-2 BPE.
    """
    return manifest.spec_for(stream_id).bytes_per_token


# ----------------------------------------------------------------------------
# Manifest JSON I/O
# ----------------------------------------------------------------------------


def write_manifest(manifest: SealedStreamManifest, path: Path) -> None:
    """Write a manifest to JSON via atomic .tmp + rename.

    Output schema (with `_meta` marker for the reader to validate):
      {
        "_meta": "karpa-sealed-pool-manifest",
        "version": "v1",
        "streams_root_hash": "<sha256 hex>",
        "streams": [
          {"id": "stream_00", "corpus": "...", "sub_genre": "...",
           "n_tokens": 200000, "bytes_per_token": 4.05,
           "sha256": "<hex>"},
          ...
        ]
      }

    Atomic-write so a crashed write never leaves a half-written manifest
    the validator consumes.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_meta": _MANIFEST_META,
        "version": manifest.version,
        "streams_root_hash": manifest.streams_root_hash,
        "streams": [asdict(s) for s in manifest.streams],
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2))
    tmp.replace(path)


def read_manifest(path: Path) -> SealedStreamManifest:
    """Inverse of `write_manifest`.

    Validates the `_meta` marker, the `version` matches MANIFEST_VERSION,
    and the streams list is present + non-empty. After loading, re-computes
    `streams_root_hash` from the streams list and asserts it matches the
    on-disk value — catches manifests that were hand-edited without
    updating the root hash.

    Raises ValueError on any validation failure, with a message naming
    the specific issue.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ValueError(
            f"sealed-pool manifest at {path} is not valid JSON: {e}"
        ) from e

    meta = payload.get("_meta")
    if meta != _MANIFEST_META:
        raise ValueError(
            f"unexpected _meta marker {meta!r} in {path}; "
            f"expected {_MANIFEST_META!r}"
        )
    version = payload.get("version")
    if version != MANIFEST_VERSION:
        raise ValueError(
            f"manifest version mismatch in {path}: file says {version!r}, "
            f"this reader expects {MANIFEST_VERSION!r}"
        )

    streams_in = payload.get("streams")
    if not isinstance(streams_in, list):
        raise ValueError(
            f"manifest {path} missing 'streams' list "
            f"(got {type(streams_in).__name__})"
        )

    specs: list[SealedStreamSpec] = []
    for entry in streams_in:
        specs.append(SealedStreamSpec(
            id=str(entry["id"]),
            corpus=str(entry["corpus"]),
            sub_genre=str(entry.get("sub_genre", "")),
            n_tokens=int(entry["n_tokens"]),
            bytes_per_token=float(entry["bytes_per_token"]),
            sha256=str(entry["sha256"]),
        ))

    expected_root = compute_streams_root_hash(specs)
    on_disk_root = str(payload.get("streams_root_hash", ""))
    if on_disk_root and on_disk_root != expected_root:
        raise ValueError(
            f"manifest {path} streams_root_hash mismatch:\n"
            f"  expected (recomputed): {expected_root}\n"
            f"  on disk:               {on_disk_root}\n"
            "Either the manifest was hand-edited or a stream sha was "
            "changed without updating the root hash."
        )

    # Pass the expected root explicitly so __post_init__ doesn't re-compute
    # (the value is the same, just skipping the work).
    return SealedStreamManifest(
        streams=specs,
        streams_root_hash=expected_root,
        version=version,
    )
