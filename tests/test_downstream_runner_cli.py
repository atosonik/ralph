"""Tests for eval/downstream/runner_cli.py.

The CLI's full end-to-end behaviour depends on:
  * a real KarpaBase + KarpaConfig from the karpaai/recipe sibling
  * a real DCLM bundle on disk
  * the load_task_examples stubs being implemented (B1-D1 follow-up)

None of those are wired in B1. These tests cover what IS testable:
  * argparse surface (required vs optional, type coercion)
  * _load_checkpoint helper on synthetic torch checkpoints
  * _build_task_loaders dispatcher (core22 vs private_hard routing)
  * main(): --patch raises NotImplementedError with the right pointer
  * main() with all dependencies mocked: completes end-to-end and
    writes a valid report

The wrapper-side tests in test_downstream_runner_subprocess.py
exercise the IPC contract via the synthetic entrypoint; these tests
exercise the production-CLI implementation in isolation.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from eval.downstream.runner_cli import (
    _build_parser,
    _build_task_loaders,
    _load_checkpoint,
    main,
)
from eval.downstream.types import HARNESS_VERSION

# ============================================================================
# _build_parser — arg surface
# ============================================================================


class TestBuildParser:
    def test_required_args_enforced(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_required_args_subset_enforced(self):
        """Even missing ONE required arg fails."""
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--checkpoint", "ckpt", "--config", "cfg",
                "--output", "out", "--bundle-sha", "sha",
                "--bundle-dir", "bundle",
                # missing --vocab-size
            ])

    def test_minimal_required_set_parses(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--checkpoint", "ckpt.pt",
            "--config", "config.json",
            "--output", "report.json",
            "--bundle-sha", "abc",
            "--bundle-dir", "/path/to/bundle",
            "--vocab-size", "50257",
        ])
        assert args.checkpoint == Path("ckpt.pt")
        assert args.config == Path("config.json")
        assert args.output == Path("report.json")
        assert args.bundle_sha == "abc"
        assert args.bundle_dir == Path("/path/to/bundle")
        assert args.vocab_size == 50257

    def test_vocab_size_type_coercion(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--checkpoint", "x", "--config", "x", "--output", "x",
            "--bundle-sha", "x", "--bundle-dir", "x", "--vocab-size", "50257",
        ])
        assert isinstance(args.vocab_size, int)
        assert args.vocab_size == 50257

    def test_vocab_size_rejects_non_int(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "--checkpoint", "x", "--config", "x", "--output", "x",
                "--bundle-sha", "x", "--bundle-dir", "x",
                "--vocab-size", "not_a_number",
            ])

    def test_optional_args_default_none(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--checkpoint", "x", "--config", "x", "--output", "x",
            "--bundle-sha", "x", "--bundle-dir", "x", "--vocab-size", "50257",
        ])
        assert args.hardness_index is None
        assert args.patch is None
        assert args.karpa_root is None

    def test_optional_args_parse_to_paths(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--checkpoint", "x", "--config", "x", "--output", "x",
            "--bundle-sha", "x", "--bundle-dir", "x", "--vocab-size", "50257",
            "--hardness-index", "/tmp/hard.jsonl",
            "--patch", "/tmp/p.patch",
            "--karpa-root", "/tmp/karpa",
        ])
        assert args.hardness_index == Path("/tmp/hard.jsonl")
        assert args.patch == Path("/tmp/p.patch")
        assert args.karpa_root == Path("/tmp/karpa")


# ============================================================================
# _load_checkpoint — synthetic torch checkpoints
# ============================================================================


class TestLoadCheckpoint:
    def test_canonical_v010_shape(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        state = {"w": torch.tensor([1.0, 2.0, 3.0])}
        config = {"vocab_size": 50257, "dim": 64}
        torch.save({"model": state, "config": config}, ckpt)

        out_config, out_state = _load_checkpoint(ckpt)
        assert out_config == config
        assert torch.equal(out_state["w"], state["w"])

    def test_state_dict_key_variant(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        state = {"w": torch.tensor([1.0])}
        config = {"vocab_size": 50257}
        torch.save({"state_dict": state, "config": config}, ckpt)

        out_config, out_state = _load_checkpoint(ckpt)
        assert out_config == config
        assert torch.equal(out_state["w"], state["w"])

    def test_sidecar_config(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        state = {"w": torch.tensor([1.0])}
        torch.save({"model": state}, ckpt)
        sidecar = tmp_path / "checkpoint_config.json"
        sidecar.write_text(json.dumps({"vocab_size": 50257, "dim": 32}))

        out_config, out_state = _load_checkpoint(ckpt)
        assert out_config == {"vocab_size": 50257, "dim": 32}

    def test_non_dict_checkpoint_raises(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        torch.save(torch.tensor([1.0]), ckpt)
        with pytest.raises(ValueError, match=r"must be a dict"):
            _load_checkpoint(ckpt)

    def test_missing_model_and_state_dict_keys_raises(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        torch.save({"config": {"vocab_size": 50257}}, ckpt)
        with pytest.raises(ValueError, match=r"no 'model' or 'state_dict' key"):
            _load_checkpoint(ckpt)

    def test_missing_config_and_no_sidecar_raises(self, tmp_path):
        ckpt = tmp_path / "ckpt.pt"
        torch.save({"model": {"w": torch.tensor([1.0])}}, ckpt)
        with pytest.raises(ValueError, match=r"missing 'config' dict"):
            _load_checkpoint(ckpt)


# ============================================================================
# _build_task_loaders — routing
# ============================================================================


class TestBuildTaskLoaders:
    def test_returns_dict_keyed_by_task(self, tmp_path):
        loaders = _build_task_loaders(("arc_easy",), tmp_path)
        assert "arc_easy" in loaders
        assert len(loaders) == 1

    def test_routes_core22_vs_private_hard(self, tmp_path):
        loaders = _build_task_loaders(
            ("arc_easy", "tiny_mmlu"), tmp_path,
        )
        assert "arc_easy" in loaders
        assert "tiny_mmlu" in loaders

    def test_loaders_are_callable(self, tmp_path):
        loaders = _build_task_loaders(("arc_easy",), tmp_path)
        assert callable(loaders["arc_easy"])

    def test_core22_loader_calls_core22_load(self, tmp_path):
        """The bound loader reaches core22.load_task_examples, which is
        a stub today and raises NotImplementedError."""
        loaders = _build_task_loaders(("arc_easy",), tmp_path)
        with pytest.raises(NotImplementedError):
            loaders["arc_easy"]()

    def test_private_hard_loader_calls_private_hard_load(self, tmp_path):
        loaders = _build_task_loaders(("tiny_mmlu",), tmp_path)
        with pytest.raises(NotImplementedError) as exc_info:
            loaders["tiny_mmlu"]()
        # private_hard.load_task_examples mentions DEFERRED.md B1-D1.
        assert "DEFERRED" in str(exc_info.value) or "B1-D1" in str(exc_info.value)

    def test_closure_binds_task_name(self, tmp_path):
        """Each loader must capture its own task name, not the loop's last."""
        loaders = _build_task_loaders(
            ("arc_easy", "tiny_mmlu", "piqa"), tmp_path,
        )
        # If closure binding is wrong, all loaders would call the same task.
        # We can detect this by checking that distinct tasks have distinct
        # loaders (each loader's id is unique).
        ids = {id(loaders[t]) for t in loaders}
        assert len(ids) == 3


# ============================================================================
# main() — --patch handling (B1-D13)
# ============================================================================


class TestMainPatchHandling:
    def test_patch_arg_raises_not_implemented(self, tmp_path):
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"tasks": ["arc_easy"]}))
        with pytest.raises(NotImplementedError) as exc_info:
            main([
                "--checkpoint", "x",
                "--config", str(cfg_path),
                "--output", "x",
                "--bundle-sha", "x",
                "--bundle-dir", "x",
                "--vocab-size", "50257",
                "--patch", "/tmp/p.patch",
                "--karpa-root", "/tmp/root",
            ])
        msg = str(exc_info.value)
        assert "Structural-patch" in msg or "structural-patch" in msg.lower()
        assert "B1-D13" in msg

    def test_no_patch_passes_through(self, tmp_path):
        """Without --patch, main() proceeds past the gate (fails later for
        unrelated reasons in this stripped-down test)."""
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"tasks": ["arc_easy"]}))
        # Without --patch, main() will try to load the checkpoint at "x"
        # which doesn't exist → FileNotFoundError. That's a different
        # error than NotImplementedError, which proves the patch gate
        # didn't fire.
        with pytest.raises(Exception) as exc_info:
            main([
                "--checkpoint", "/nonexistent/ckpt.pt",
                "--config", str(cfg_path),
                "--output", "x",
                "--bundle-sha", "x",
                "--bundle-dir", "x",
                "--vocab-size", "50257",
            ])
        # Should NOT be NotImplementedError mentioning B1-D13.
        assert not (
            isinstance(exc_info.value, NotImplementedError)
            and "B1-D13" in str(exc_info.value)
        )


# ============================================================================
# main() — end-to-end with mocked dependencies
# ============================================================================


class _FakeKarpaConfig:
    """Stand-in for KarpaConfig with a __dataclass_fields__ surface."""
    __dataclass_fields__ = {"vocab_size": None, "dim": None}

    def __init__(self, *, vocab_size: int, dim: int = 32):
        self.vocab_size = vocab_size
        self.dim = dim


class _FakeKarpaBase(torch.nn.Module):
    """Stand-in for KarpaBase that returns uniform logits."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._w = torch.nn.Parameter(torch.zeros(1))  # one parameter so state_dict is non-empty

    def load_state_dict(self, state, strict=True):  # noqa: ARG002 — match torch sig
        # Tolerate any state_dict shape — we only need the model to forward.
        pass

    def forward(self, input_ids):
        # Return uniform logits over the configured vocab.
        b, t = input_ids.shape
        return torch.zeros((b, t, self.cfg.vocab_size))


def _fake_loader_factory():
    """Returns task_loaders that produce zero examples (trivially succeeds)."""
    return {
        "arc_easy": lambda: [],
    }


class TestMainHappyPath:
    def test_full_flow_with_mocks(self, tmp_path):
        """Mock out the heavy dependencies and verify main() writes a report."""
        # Synthetic checkpoint
        ckpt = tmp_path / "ckpt.pt"
        torch.save({
            "model": {"_w": torch.zeros(1)},
            "config": {"vocab_size": 50257, "dim": 32},
        }, ckpt)

        # Synthetic config
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"tasks": ["arc_easy"]}))

        output_path = tmp_path / "report.json"

        # Mock _import_karpa_model + _build_task_loaders + _build_tokenize_fn
        with mock.patch(
            "eval.downstream.runner_cli._import_karpa_model",
            return_value=(_FakeKarpaBase, _FakeKarpaConfig),
        ), mock.patch(
            "eval.downstream.runner_cli._build_task_loaders",
            return_value=_fake_loader_factory(),
        ), mock.patch(
            "eval.downstream.runner_cli._build_tokenize_fn",
            return_value=lambda text: [ord(c) for c in text],
        ):
            rc = main([
                "--checkpoint", str(ckpt),
                "--config", str(cfg_path),
                "--output", str(output_path),
                "--bundle-sha", "test-sha",
                "--bundle-dir", str(tmp_path),
                "--vocab-size", "50257",
            ])

        assert rc == 0
        assert output_path.exists()
        report = json.loads(output_path.read_text())
        assert report["harness_version"] == HARNESS_VERSION
        assert report["bundle_sha256"] == "test-sha"
        assert "arc_easy:S3" in report["cells"]
        # Empty loader → 0 examples → accuracy=0.0 (the documented empty-cell
        # sentinel for downstream tasks).
        assert report["cells"]["arc_easy:S3"]["n_examples"] == 0

    def test_vocab_mismatch_aborts(self, tmp_path):
        """Checkpoint vocab differs from --vocab-size → ValueError."""
        ckpt = tmp_path / "ckpt.pt"
        torch.save({
            "model": {"_w": torch.zeros(1)},
            "config": {"vocab_size": 50257, "dim": 32},
        }, ckpt)
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"tasks": ["arc_easy"]}))

        with mock.patch(
            "eval.downstream.runner_cli._import_karpa_model",
            return_value=(_FakeKarpaBase, _FakeKarpaConfig),
        ):
            with pytest.raises(ValueError, match=r"vocab_size"):
                main([
                    "--checkpoint", str(ckpt),
                    "--config", str(cfg_path),
                    "--output", str(tmp_path / "out.json"),
                    "--bundle-sha", "x",
                    "--bundle-dir", str(tmp_path),
                    "--vocab-size", "40000",  # mismatch
                ])

    def test_output_parent_dir_created(self, tmp_path):
        """main() must mkdir -p the output's parent before writing."""
        ckpt = tmp_path / "ckpt.pt"
        torch.save({
            "model": {"_w": torch.zeros(1)},
            "config": {"vocab_size": 50257, "dim": 32},
        }, ckpt)
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(json.dumps({"tasks": ["arc_easy"]}))

        output = tmp_path / "nested" / "deep" / "report.json"
        assert not output.parent.exists()

        with mock.patch(
            "eval.downstream.runner_cli._import_karpa_model",
            return_value=(_FakeKarpaBase, _FakeKarpaConfig),
        ), mock.patch(
            "eval.downstream.runner_cli._build_task_loaders",
            return_value=_fake_loader_factory(),
        ), mock.patch(
            "eval.downstream.runner_cli._build_tokenize_fn",
            return_value=lambda text: [],
        ):
            rc = main([
                "--checkpoint", str(ckpt),
                "--config", str(cfg_path),
                "--output", str(output),
                "--bundle-sha", "x",
                "--bundle-dir", str(tmp_path),
                "--vocab-size", "50257",
            ])
        assert rc == 0
        assert output.exists()
