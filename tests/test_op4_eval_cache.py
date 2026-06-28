"""op4 hidden-eval result cache. A deferred challenger (king min-tenure guard) is
re-scored every epoch while it waits out the incumbent's tenure; the bundle and
the held-out eval shard are immutable across those epochs, so the ~90s GPU eval
is cached, keyed on the eval shard so it invalidates on eval rotation. Stored as
a per-bundle dotfile; op1 integrity is manifest-based so the extra file is ignored."""
from pathlib import Path

from eval.hidden_eval import HiddenEvalResult
from validator.validator import (
    _eval_cache_path,
    _eval_shard_fingerprint,
    _load_cached_hidden_eval,
    _save_cached_hidden_eval,
    op4_hidden_eval,
)


def _shard(eval_dir: Path, tokens: bytes = b"abc", bench: bytes = b"[]") -> None:
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "active_tokens.bin").write_bytes(tokens)
    (eval_dir / "active_benchmark.json").write_bytes(bench)


def _result(val_bpb: float = 1.5) -> HiddenEvalResult:
    return HiddenEvalResult(
        val_bpb=val_bpb, benchmark_accuracy=0.9,
        tokens_evaluated=100, benchmark_examples=10, eval_set_hash="deadbeef",
    )


def test_cache_roundtrip_and_shard_invalidation(tmp_path):
    eval_dir = tmp_path / "ralph" / "eval" / "private"
    _shard(eval_dir)
    proof = tmp_path / "queue" / "pending" / "abc123"
    proof.mkdir(parents=True)
    fp = _eval_shard_fingerprint(eval_dir)
    _save_cached_hidden_eval(proof, fp, _result(1.5))
    got = _load_cached_hidden_eval(proof, fp)
    assert got is not None and got.val_bpb == 1.5 and got.eval_set_hash == "deadbeef"
    # rotate the eval shard -> fingerprint changes -> cache miss (no stale score)
    _shard(eval_dir, tokens=b"ROTATED-SHARD")
    fp2 = _eval_shard_fingerprint(eval_dir)
    assert fp2 != fp
    assert _load_cached_hidden_eval(proof, fp2) is None


def test_op4_short_circuits_on_cache_hit(tmp_path):
    # Pre-populate the cache; op4 must return it even with NO checkpoint present,
    # proving the cache is consulted before the (expensive) checkpoint load + eval.
    ralph_root = tmp_path / "ralph"
    _shard(ralph_root / "eval" / "private")
    proof = tmp_path / "queue" / "pending" / "abc123"
    (proof / "training").mkdir(parents=True)  # deliberately NO checkpoint.pt
    fp = _eval_shard_fingerprint(ralph_root / "eval" / "private")
    _save_cached_hidden_eval(proof, fp, _result(1.42))
    ok, detail, res = op4_hidden_eval(ralph_root, proof)
    assert ok is True and res is not None and res.val_bpb == 1.42
    assert "cached" in detail


def test_op4_missing_checkpoint_still_rejects_without_cache(tmp_path):
    # No cache + no checkpoint -> normal rejection; the cache never masks errors.
    ralph_root = tmp_path / "ralph"
    _shard(ralph_root / "eval" / "private")
    proof = tmp_path / "queue" / "pending" / "def456"
    proof.mkdir(parents=True)
    ok, detail, res = op4_hidden_eval(ralph_root, proof)
    assert ok is False and res is None and "missing checkpoint" in detail


def test_cache_path_is_per_bundle_dotfile(tmp_path):
    # Inside the bundle dir as a dotfile -> per-bundle (no cross-bundle collision)
    # and ignored by op1 (manifest-based integrity verifies only declared files).
    proof_a = tmp_path / "queue" / "pending" / "aaa"
    proof_b = tmp_path / "queue" / "pending" / "bbb"
    for p in (proof_a, proof_b):
        p.mkdir(parents=True)
    assert _eval_cache_path(proof_a) == proof_a / ".hidden_eval_cache.json"
    assert _eval_cache_path(proof_a) != _eval_cache_path(proof_b)
    assert _eval_cache_path(proof_a).name.startswith(".")
