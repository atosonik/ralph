"""Tests for eval/downstream/runner.py — in-process kernel only.

Pins the EvalConfig contract + the vocab/determinism guards + the
in-process eval driver. The subprocess wrapper (next PR) is tested in
test_downstream_runner_subprocess.py.

Covers:
  * EvalConfig: validation, JSON round-trip, defaults, frozen-ness.
  * check_vocab_compatibility: pass/fail at 50257.
  * set_eval_determinism: idempotent + seeds RNG.
  * run_downstream_eval: core22 dispatch (mc/schema/lm), private_hard
    dispatch with HardnessIndex, scale_label propagation, truncation,
    unknown-task rejection, missing-loader rejection, determinism.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream.core22 import (
    LMRawRow,
    MCRawRow,
    SchemaRawRow,
)
from eval.downstream.private_hard import (
    HardnessIndex,
    HardnessIndexRow,
)
from eval.downstream.runner import (
    RALPH_VOCAB_SIZE,
    EvalConfig,
    check_vocab_compatibility,
    run_downstream_eval,
    set_eval_determinism,
)
from eval.downstream.types import (
    HARNESS_VERSION,
    DownstreamReport,
)

# Test fixtures ---------------------------------------------------------------

VOCAB = 256  # Covers ord(c) for all ASCII chars used by _char_tokenize.


def _char_tokenize(text: str) -> list[int]:
    return [ord(c) for c in text]


def _uniform_forward(input_ids: torch.Tensor) -> torch.Tensor:
    """Uniform logits → arg-max stable tie-break picks choice 0."""
    return torch.zeros((1, input_ids.size(1), VOCAB))


def _make_loader(rows):
    return lambda: rows


# ============================================================================
# EvalConfig
# ============================================================================


class TestEvalConfig:
    def test_minimum_valid_config(self):
        cfg = EvalConfig(tasks=("arc_easy",))
        assert cfg.tasks == ("arc_easy",)
        assert cfg.n_examples_per_task == 0
        assert cfg.seed == 0
        assert cfg.scale_label == "S3"
        assert cfg.length_normalize_mc is True

    def test_empty_tasks_rejected(self):
        with pytest.raises(ValueError, match=r"tasks must be non-empty"):
            EvalConfig(tasks=())

    def test_negative_n_examples_rejected(self):
        with pytest.raises(ValueError, match=r"n_examples_per_task must be >= 0"):
            EvalConfig(tasks=("arc_easy",), n_examples_per_task=-1)

    def test_zero_n_examples_accepted(self):
        cfg = EvalConfig(tasks=("arc_easy",), n_examples_per_task=0)
        assert cfg.n_examples_per_task == 0

    def test_empty_scale_label_rejected(self):
        with pytest.raises(ValueError, match=r"scale_label must be non-empty"):
            EvalConfig(tasks=("arc_easy",), scale_label="")

    def test_to_dict_round_trip(self):
        cfg = EvalConfig(
            tasks=("arc_easy", "piqa"),
            n_examples_per_task=10,
            seed=42,
            scale_label="S2",
            length_normalize_mc=False,
        )
        roundtripped = EvalConfig.from_dict(cfg.to_dict())
        assert roundtripped == cfg

    def test_from_dict_defaults(self):
        cfg = EvalConfig.from_dict({"tasks": ["arc_easy"]})
        assert cfg.tasks == ("arc_easy",)
        assert cfg.n_examples_per_task == 0
        assert cfg.seed == 0
        assert cfg.scale_label == "S3"
        assert cfg.length_normalize_mc is True

    def test_from_dict_converts_list_to_tuple(self):
        cfg = EvalConfig.from_dict({"tasks": ["a", "b"]})
        assert isinstance(cfg.tasks, tuple)
        assert cfg.tasks == ("a", "b")

    def test_from_dict_missing_tasks_raises(self):
        with pytest.raises(KeyError):
            EvalConfig.from_dict({})

    def test_frozen(self):
        cfg = EvalConfig(tasks=("a",))
        with pytest.raises(Exception):
            cfg.seed = 5  # type: ignore[misc]

    def test_json_serializable(self):
        cfg = EvalConfig(tasks=("arc_easy",), seed=1, scale_label="X")
        encoded = json.dumps(cfg.to_dict())
        restored = EvalConfig.from_dict(json.loads(encoded))
        assert restored == cfg

    def test_hashable(self):
        """Frozen dataclass must be hashable for cache keys / chain logs."""
        cfg = EvalConfig(tasks=("a",))
        d = {cfg: 1}
        assert d[cfg] == 1


# ============================================================================
# check_vocab_compatibility (B1-D6)
# ============================================================================


class TestVocabCheck:
    def test_pass_at_50257(self):
        check_vocab_compatibility(50257)  # no raise

    def test_pass_at_constant(self):
        check_vocab_compatibility(RALPH_VOCAB_SIZE)

    def test_reject_smaller(self):
        with pytest.raises(ValueError, match=r"vocab_size mismatch"):
            check_vocab_compatibility(50000)

    def test_reject_larger(self):
        with pytest.raises(ValueError, match=r"vocab_size mismatch"):
            check_vocab_compatibility(50500)

    def test_reject_zero(self):
        with pytest.raises(ValueError, match=r"vocab_size mismatch"):
            check_vocab_compatibility(0)

    def test_message_mentions_gpt2(self):
        with pytest.raises(ValueError, match=r"GPT-2"):
            check_vocab_compatibility(40000)

    def test_message_includes_actual_value(self):
        with pytest.raises(ValueError, match=r"40000"):
            check_vocab_compatibility(40000)


# ============================================================================
# set_eval_determinism (B1-D7)
# ============================================================================


class TestDeterminism:
    def test_enables_deterministic_algorithms(self):
        set_eval_determinism(42)
        assert torch.are_deterministic_algorithms_enabled()

    def test_sets_cublas_env_var(self):
        # Clear the var first to verify the helper sets it; restore after.
        saved = os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        try:
            set_eval_determinism(42)
            assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        finally:
            if saved is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved

    def test_preserves_existing_cublas_env_var(self):
        """`setdefault` semantics: don't override if caller already set it."""
        saved = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        try:
            set_eval_determinism(0)
            assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
        finally:
            if saved is None:
                del os.environ["CUBLAS_WORKSPACE_CONFIG"]
            else:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = saved

    def test_idempotent(self):
        set_eval_determinism(42)
        set_eval_determinism(42)  # second call must not raise

    def test_seeds_same_rng_state(self):
        set_eval_determinism(7)
        x = torch.randn(3)
        set_eval_determinism(7)
        y = torch.randn(3)
        assert torch.allclose(x, y)

    def test_different_seeds_diverge(self):
        set_eval_determinism(1)
        x = torch.randn(3)
        set_eval_determinism(2)
        y = torch.randn(3)
        assert not torch.allclose(x, y)


# ============================================================================
# run_downstream_eval — core22 dispatch
# ============================================================================


class TestRunDownstreamEvalCore22:
    def test_single_mc_task(self):
        rows = [
            MCRawRow(query="q1", choices=["a", "b"], gold=0),
            MCRawRow(query="q2", choices=["a", "b"], gold=0),
            MCRawRow(query="q3", choices=["a", "b"], gold=0),
        ]
        cfg = EvalConfig(tasks=("arc_easy",))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="abc123",
            vocab_size=50257,
        )
        assert isinstance(report, DownstreamReport)
        assert report.harness_version == HARNESS_VERSION
        assert report.bundle_sha256 == "abc123"
        assert report.seed == 0
        assert report.total_examples == 3
        assert "arc_easy:S3" in report.cells
        # Uniform logits → tie → tie-break index 0 → all gold=0 → acc=1.0
        assert report.cells["arc_easy:S3"].accuracy == 1.0
        assert report.cells["arc_easy:S3"].n_examples == 3

    def test_schema_task_dispatch(self):
        rows = [
            SchemaRawRow(contexts=["x"], continuations=["y"], gold=0),
            SchemaRawRow(contexts=["a"], continuations=["b"], gold=0),
        ]
        cfg = EvalConfig(tasks=("winogrande",))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"winogrande": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert "winogrande:S3" in report.cells
        assert report.cells["winogrande:S3"].n_examples == 2

    def test_lm_task_dispatch(self):
        rows = [
            LMRawRow(context="hello", target=" world"),
            LMRawRow(context="foo", target=" bar"),
        ]
        cfg = EvalConfig(tasks=("lambada_openai",))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"lambada_openai": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert "lambada_openai:S3" in report.cells
        assert report.cells["lambada_openai:S3"].n_examples == 2
        # Uniform logits over VOCAB=256 → NLL per token ≈ log(256) ≈ 5.545 nats.
        # The reported "accuracy" field carries mean NLL for LM tasks.
        assert report.cells["lambada_openai:S3"].accuracy > 0

    def test_multi_task_dispatch(self):
        mc_rows = [MCRawRow(query="q", choices=["a", "b"], gold=0)]
        sc_rows = [SchemaRawRow(contexts=["x"], continuations=["y"], gold=0)]
        cfg = EvalConfig(tasks=("arc_easy", "winogrande"))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={
                "arc_easy": _make_loader(mc_rows),
                "winogrande": _make_loader(sc_rows),
            },
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert len(report.cells) == 2
        assert "arc_easy:S3" in report.cells
        assert "winogrande:S3" in report.cells
        assert report.total_examples == 2


# ============================================================================
# run_downstream_eval — config knobs
# ============================================================================


class TestRunDownstreamEvalConfig:
    def test_scale_label_in_cell_key(self):
        rows = [MCRawRow(query="q", choices=["a", "b"], gold=0)]
        cfg = EvalConfig(tasks=("arc_easy",), scale_label="S1")
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert "arc_easy:S1" in report.cells
        assert "arc_easy:S3" not in report.cells

    def test_scale_label_in_cell_result(self):
        rows = [MCRawRow(query="q", choices=["a", "b"], gold=0)]
        cfg = EvalConfig(tasks=("arc_easy",), scale_label="S2")
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        # Cell key uses scale label but CellResult.task stays bare.
        assert report.cells["arc_easy:S2"].task == "arc_easy"

    def test_n_examples_zero_uses_all(self):
        rows = [MCRawRow(query=f"q{i}", choices=["a", "b"], gold=0) for i in range(5)]
        cfg = EvalConfig(tasks=("arc_easy",), n_examples_per_task=0)
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert report.cells["arc_easy:S3"].n_examples == 5

    def test_n_examples_truncates(self):
        rows = [MCRawRow(query=f"q{i}", choices=["a", "b"], gold=0) for i in range(10)]
        cfg = EvalConfig(tasks=("arc_easy",), n_examples_per_task=3)
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert report.cells["arc_easy:S3"].n_examples == 3

    def test_n_examples_larger_than_dataset_uses_all(self):
        rows = [MCRawRow(query=f"q{i}", choices=["a", "b"], gold=0) for i in range(2)]
        cfg = EvalConfig(tasks=("arc_easy",), n_examples_per_task=100)
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert report.cells["arc_easy:S3"].n_examples == 2

    def test_seed_in_report_and_cell(self):
        rows = [MCRawRow(query="q", choices=["a", "b"], gold=0)]
        cfg = EvalConfig(tasks=("arc_easy",), seed=99)
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert report.seed == 99
        assert report.cells["arc_easy:S3"].seed == 99

    def test_total_examples_sums_across_cells(self):
        mc1 = [MCRawRow(query="q", choices=["a", "b"], gold=0) for _ in range(3)]
        mc2 = [MCRawRow(query="q", choices=["a", "b"], gold=0) for _ in range(5)]
        cfg = EvalConfig(tasks=("arc_easy", "piqa"))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={
                "arc_easy": _make_loader(mc1),
                "piqa": _make_loader(mc2),
            },
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert report.total_examples == 8

    def test_wall_clock_passthrough(self):
        rows = [MCRawRow(query="q", choices=["a", "b"], gold=0)]
        cfg = EvalConfig(tasks=("arc_easy",))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
            wall_clock_s=12.5,
        )
        assert report.wall_clock_s == 12.5

    def test_wall_clock_defaults_to_zero(self):
        rows = [MCRawRow(query="q", choices=["a", "b"], gold=0)]
        cfg = EvalConfig(tasks=("arc_easy",))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        assert report.wall_clock_s == 0.0


# ============================================================================
# run_downstream_eval — error paths
# ============================================================================


class TestRunDownstreamEvalErrors:
    def test_unknown_task_rejected(self):
        cfg = EvalConfig(tasks=("not_a_task",))
        with pytest.raises(ValueError, match=r"unknown task"):
            run_downstream_eval(
                _uniform_forward,
                config=cfg,
                task_loaders={"not_a_task": _make_loader([])},
                tokenize=_char_tokenize,
                bundle_sha256="x",
                vocab_size=50257,
            )

    def test_missing_loader_rejected(self):
        cfg = EvalConfig(tasks=("arc_easy",))
        with pytest.raises(ValueError, match=r"no loader registered"):
            run_downstream_eval(
                _uniform_forward,
                config=cfg,
                task_loaders={},
                tokenize=_char_tokenize,
                bundle_sha256="x",
                vocab_size=50257,
            )

    def test_vocab_mismatch_rejects_early(self):
        """Vocab check happens before any task work — empty loaders fine."""
        cfg = EvalConfig(tasks=("arc_easy",))
        with pytest.raises(ValueError, match=r"vocab_size mismatch"):
            run_downstream_eval(
                _uniform_forward,
                config=cfg,
                task_loaders={"arc_easy": _make_loader([])},
                tokenize=_char_tokenize,
                bundle_sha256="x",
                vocab_size=40000,
            )


# ============================================================================
# run_downstream_eval — private_hard dispatch
# ============================================================================


class TestRunDownstreamEvalPrivateHard:
    def test_private_hard_routing(self):
        rows = [
            ("a1", MCRawRow(query="q1", choices=["a", "b"], gold=0)),
            ("a2", MCRawRow(query="q2", choices=["a", "b"], gold=0)),
        ]
        idx = HardnessIndex(
            version="v1",
            rows=[
                HardnessIndexRow(
                    dataset="arc_challenge_hard",
                    item_id="a1",
                    gold_margin_bits=0.0,
                ),
            ],
        )
        cfg = EvalConfig(tasks=("arc_challenge_hard",))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_challenge_hard": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
            hardness_index=idx,
        )
        # Only 1 of 2 items in index → cell has n=1.
        assert report.cells["arc_challenge_hard:S3"].n_examples == 1
        assert report.total_examples == 1

    def test_private_hard_requires_index(self):
        rows = [("a1", MCRawRow(query="q", choices=["a", "b"], gold=0))]
        cfg = EvalConfig(tasks=("arc_challenge_hard",))
        with pytest.raises(ValueError, match=r"requires a hardness_index"):
            run_downstream_eval(
                _uniform_forward,
                config=cfg,
                task_loaders={"arc_challenge_hard": _make_loader(rows)},
                tokenize=_char_tokenize,
                bundle_sha256="x",
                vocab_size=50257,
                hardness_index=None,
            )

    def test_private_hard_empty_index_yields_zero_cell(self):
        rows = [("a1", MCRawRow(query="q", choices=["a", "b"], gold=0))]
        idx = HardnessIndex(version="v1", rows=[])
        cfg = EvalConfig(tasks=("arc_challenge_hard",))
        report = run_downstream_eval(
            _uniform_forward,
            config=cfg,
            task_loaders={"arc_challenge_hard": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
            hardness_index=idx,
        )
        # Empty index → filter excludes all → (0.0, 0) cell.
        assert report.cells["arc_challenge_hard:S3"].n_examples == 0
        assert report.cells["arc_challenge_hard:S3"].accuracy == 0.0


# ============================================================================
# run_downstream_eval — determinism
# ============================================================================


class TestRunDownstreamEvalDeterminism:
    def test_byte_identical_same_seed(self):
        rows = [MCRawRow(query="q", choices=["a", "b"], gold=0)]
        cfg = EvalConfig(tasks=("arc_easy",), seed=7)
        kwargs = dict(
            forward_logits=_uniform_forward,
            config=cfg,
            task_loaders={"arc_easy": _make_loader(rows)},
            tokenize=_char_tokenize,
            bundle_sha256="x",
            vocab_size=50257,
        )
        r1 = run_downstream_eval(**kwargs)
        r2 = run_downstream_eval(**kwargs)
        assert r1.cells["arc_easy:S3"].accuracy == r2.cells["arc_easy:S3"].accuracy
        assert r1.total_examples == r2.total_examples
        assert r1.seed == r2.seed
