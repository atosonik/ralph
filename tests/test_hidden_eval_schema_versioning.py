"""Schema-versioning tests for HiddenEvalResult (closes B1-D12).

The v0.10 → v0.11 transition extends HiddenEvalResult with an optional
`downstream: DownstreamReport | None = None` field. These tests pin
the forward-compat invariants:

  1. An old serialized HiddenEvalResult dict (saved BEFORE the
     downstream field was added, hence with NO `downstream` key)
     deserializes cleanly via `HiddenEvalResult(**old_dict)` because
     `downstream` has a default.

  2. A new HiddenEvalResult with `downstream=None` serializes via
     `to_legacy_dict()` to a dict byte-equivalent to the pre-v0.11
     `dataclasses.asdict` shape — same keys, same values, no extra
     `downstream` key. Chain consumers reading the legacy shape
     continue to work against new validators that haven't filled in
     downstream yet.

  3. A new HiddenEvalResult with a populated downstream serializes via
     `to_legacy_dict()` to a dict that INCLUDES the downstream nested
     dict. Old consumers that don't know about `downstream` simply
     ignore the extra key.

Reference: docs/build_scope/02_scope_B1.md "DEFERRED.md B1-D12".
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream.types import (
    HARNESS_VERSION,
    CellResult,
    DownstreamReport,
)
from eval.hidden_eval import HiddenEvalResult

# ============================================================================
# Forward-compat: old serialized dict deserializes cleanly
# ============================================================================


class TestOldDictDeserializes:
    def test_old_format_dict_no_downstream_key(self):
        """Pre-v0.11 dict (no downstream field) round-trips via kwargs."""
        old_dict = {
            "val_bpb": 1.234,
            "benchmark_accuracy": 0.5,
            "tokens_evaluated": 1000,
            "benchmark_examples": 50,
            "eval_set_hash": "abc123",
        }
        result = HiddenEvalResult(**old_dict)
        assert result.val_bpb == 1.234
        assert result.benchmark_accuracy == 0.5
        assert result.tokens_evaluated == 1000
        assert result.benchmark_examples == 50
        assert result.eval_set_hash == "abc123"
        # Default fires when key is absent.
        assert result.downstream is None

    def test_old_format_with_extra_keys_rejected(self):
        """Extra keys still raise TypeError — dataclass kwargs are strict.

        The forward-compat guarantee is "missing fields use defaults",
        NOT "extra fields are silently dropped". If a future schema
        change adds a field, a deserializer reading from a yet-newer
        format would correctly fail loudly here.
        """
        old_dict = {
            "val_bpb": 1.0,
            "benchmark_accuracy": 0.5,
            "tokens_evaluated": 1000,
            "benchmark_examples": 50,
            "eval_set_hash": "abc",
            "unknown_future_field": "surprise",
        }
        with pytest.raises(TypeError, match=r"unknown_future_field"):
            HiddenEvalResult(**old_dict)

    def test_old_format_missing_required_field_raises(self):
        """Pre-existing fields (without defaults) remain required."""
        old_dict = {
            "val_bpb": 1.0,
            "benchmark_accuracy": 0.5,
            # missing tokens_evaluated
            "benchmark_examples": 50,
            "eval_set_hash": "abc",
        }
        with pytest.raises(TypeError, match=r"tokens_evaluated"):
            HiddenEvalResult(**old_dict)


# ============================================================================
# Backward-compat: new None-downstream → byte-equivalent to legacy serialization
# ============================================================================


class TestLegacyDictByteEquivalence:
    def _legacy_serialization_target(
        self, *, val_bpb=1.0, benchmark_accuracy=0.5,
        tokens_evaluated=1000, benchmark_examples=50, eval_set_hash="abc",
    ) -> dict:
        """The exact dict shape the pre-v0.11 asdict() would produce.

        This is what chain consumers reading the legacy format expect.
        """
        return {
            "val_bpb": val_bpb,
            "benchmark_accuracy": benchmark_accuracy,
            "tokens_evaluated": tokens_evaluated,
            "benchmark_examples": benchmark_examples,
            "eval_set_hash": eval_set_hash,
        }

    def test_to_legacy_dict_with_none_downstream(self):
        """downstream=None → output is byte-identical to legacy asdict."""
        result = HiddenEvalResult(
            val_bpb=1.234,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
            downstream=None,
        )
        expected = self._legacy_serialization_target(
            val_bpb=1.234,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
        )
        assert result.to_legacy_dict() == expected

    def test_to_legacy_dict_default_downstream(self):
        """Default downstream is also None → same byte-equivalence."""
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
        )
        expected = self._legacy_serialization_target()
        assert result.to_legacy_dict() == expected

    def test_to_legacy_dict_no_downstream_key_when_none(self):
        """downstream=None → output dict has NO downstream key at all."""
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
        )
        d = result.to_legacy_dict()
        assert "downstream" not in d

    def test_legacy_dict_round_trip_via_kwargs(self):
        """to_legacy_dict() → HiddenEvalResult(**) recovers the same object."""
        original = HiddenEvalResult(
            val_bpb=1.234,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
        )
        restored = HiddenEvalResult(**original.to_legacy_dict())
        assert restored == original


# ============================================================================
# Populated downstream — to_legacy_dict still includes it
# ============================================================================


class TestPopulatedDownstream:
    def _sample_downstream(self) -> DownstreamReport:
        return DownstreamReport(
            harness_version=HARNESS_VERSION,
            bundle_sha256="def456",
            seed=7,
            total_examples=10,
            wall_clock_s=1.5,
            cells={
                "arc_easy:S3": CellResult(
                    task="arc_easy",
                    accuracy=0.6,
                    accuracy_stderr=0.0,
                    n_examples=10,
                    seed=7,
                ),
            },
        )

    def test_downstream_included_when_set(self):
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
            downstream=self._sample_downstream(),
        )
        d = result.to_legacy_dict()
        assert "downstream" in d
        assert d["downstream"]["harness_version"] == HARNESS_VERSION
        assert d["downstream"]["bundle_sha256"] == "def456"

    def test_downstream_dict_shape_matches_asdict(self):
        """Nested downstream uses dataclasses.asdict conversion — full
        recursive dict; structure matches DownstreamReport's asdict."""
        downstream = self._sample_downstream()
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
            downstream=downstream,
        )
        d = result.to_legacy_dict()
        assert d["downstream"] == asdict(downstream)

    def test_cells_nested_in_downstream_dict(self):
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
            downstream=self._sample_downstream(),
        )
        d = result.to_legacy_dict()
        assert "arc_easy:S3" in d["downstream"]["cells"]
        cell = d["downstream"]["cells"]["arc_easy:S3"]
        assert cell["task"] == "arc_easy"
        assert cell["accuracy"] == 0.6


# ============================================================================
# Cross-cutting: dataclass equality + asdict semantics
# ============================================================================


class TestDataclassSemantics:
    def test_two_results_with_same_fields_equal(self):
        a = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=100,
            benchmark_examples=10,
            eval_set_hash="x",
        )
        b = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=100,
            benchmark_examples=10,
            eval_set_hash="x",
        )
        assert a == b

    def test_downstream_difference_breaks_equality(self):
        a = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=100,
            benchmark_examples=10,
            eval_set_hash="x",
        )
        b = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=100,
            benchmark_examples=10,
            eval_set_hash="x",
            downstream=DownstreamReport(
                harness_version=HARNESS_VERSION,
                bundle_sha256="any",
                seed=0,
                total_examples=0,
                wall_clock_s=0.0,
                cells={},
            ),
        )
        assert a != b

    def test_asdict_includes_downstream_none(self):
        """Raw asdict (not to_legacy_dict) keeps the downstream: None key.

        This is what legacy callers that hadn't migrated would see.
        The to_legacy_dict helper is the explicit migration tool;
        asdict() reflects the dataclass shape verbatim.
        """
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=100,
            benchmark_examples=10,
            eval_set_hash="x",
        )
        d = asdict(result)
        assert "downstream" in d
        assert d["downstream"] is None


# ============================================================================
# validation-v2 Phase 1 reproducibility fields
# (val_seq_len / sealed_stream_manifest_hash / tail_val_bpb)
# ============================================================================


class TestReproducibilityFields:
    def test_default_none_and_dropped_from_legacy_dict(self):
        """Unset → all three default None AND are dropped from to_legacy_dict
        so the pre-v0.11 byte-shape is preserved."""
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
        )
        assert result.val_seq_len is None
        assert result.sealed_stream_manifest_hash is None
        assert result.tail_val_bpb is None
        d = result.to_legacy_dict()
        for k in ("val_seq_len", "sealed_stream_manifest_hash", "tail_val_bpb"):
            assert k not in d
        # byte-shape unchanged from legacy
        assert set(d) == {
            "val_bpb", "benchmark_accuracy", "tokens_evaluated",
            "benchmark_examples", "eval_set_hash",
        }

    def test_populated_included_in_legacy_dict(self):
        result = HiddenEvalResult(
            val_bpb=1.0,
            benchmark_accuracy=0.5,
            tokens_evaluated=1000,
            benchmark_examples=50,
            eval_set_hash="abc",
            val_seq_len=512,
            sealed_stream_manifest_hash="d" * 64,
            tail_val_bpb=1.31,
        )
        d = result.to_legacy_dict()
        assert d["val_seq_len"] == 512
        assert d["sealed_stream_manifest_hash"] == "d" * 64
        assert d["tail_val_bpb"] == 1.31

    def test_old_dict_without_new_keys_deserializes(self):
        """A pre-Phase-1 serialized dict still round-trips via kwargs."""
        old = {
            "val_bpb": 1.0,
            "benchmark_accuracy": 0.5,
            "tokens_evaluated": 1000,
            "benchmark_examples": 50,
            "eval_set_hash": "abc",
        }
        r = HiddenEvalResult(**old)
        assert r.val_seq_len is None
        assert r.sealed_stream_manifest_hash is None
        assert r.tail_val_bpb is None

    def test_run_hidden_eval_surfaces_fields(self, tmp_path):
        """End-to-end: run_hidden_eval on a tiny model populates val_seq_len,
        sealed_stream_manifest_hash, and tail_val_bpb (Phase-0 fallback path,
        no on-disk shard needed)."""
        import torch

        from eval.hidden_eval import run_hidden_eval

        class _Tiny(torch.nn.Module):
            def __init__(self, vocab=50257, dim=8):
                super().__init__()
                self.embed = torch.nn.Embedding(vocab, dim)
                self.out = torch.nn.Linear(dim, vocab)

            def forward(self, x):
                return self.out(self.embed(x)), None

        torch.manual_seed(0)
        model = _Tiny()
        # tmp_path has no active_tokens.bin → Phase-0 synthesized stream fires.
        res = run_hidden_eval(model, tmp_path, seq_len=32)
        assert res.val_seq_len == 32
        assert isinstance(res.sealed_stream_manifest_hash, str)
        assert len(res.sealed_stream_manifest_hash) == 64
        assert res.tail_val_bpb is not None
        assert res.tail_val_bpb > 0
