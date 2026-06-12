"""C3-PREREG b6_run.py orchestrator tests — CPU-side via synthetic CLI.

Covers:
  * load_recipes_config: schema validation + optional patch
  * estimate_cost math
  * run_one_recipe: success on primary seed + seed-retry on first failure +
    abort log on both-seed failure
  * run_b6: full orchestrator + budget guard
  * CLI arg surface
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from scripts.b6_run import (
    DEFAULT_BUDGET_CAP_USD,
    DEFAULT_N_RECIPES,
    SEED_RETRY_OFFSET,
    RecipeSpec,
    _build_parser,
    estimate_cost,
    load_recipes_config,
    run_b6,
    run_one_recipe,
)

_TEST_ENTRY = Path(__file__).resolve().parent / "_runner_subprocess_test_entry.py"
_TEST_COMMAND_PREFIX = (sys.executable, str(_TEST_ENTRY))


@pytest.fixture(autouse=True)
def _set_test_mode(monkeypatch):
    monkeypatch.setenv("KARPA_TEST_RUNNER_MODE", "success")
    yield


def _b6_recipe_with_single_rung_only() -> tuple:
    """Override the standard rungs to a single S3 rung for tests against
    the synthetic CLI (which hard-codes its cell key).

    NOTE: run_one_recipe + run_b6 use _B6_STANDARD_RUNGS (3-rung) by
    contract. For tests we monkey-patch the constant.
    """
    from validator.ladder import LadderRungSpec
    return (LadderRungSpec(scale_label="S3", dim=768, n_layers=12),)


@pytest.fixture
def single_rung(monkeypatch):
    """Patch _B6_STANDARD_RUNGS to a single S3 rung so the synthetic CLI
    (with hard-coded `arc_easy:S3` cell key) doesn't produce duplicate
    cells across merged rungs."""
    import scripts.b6_run as b6
    monkeypatch.setattr(b6, "_B6_STANDARD_RUNGS", _b6_recipe_with_single_rung_only())


# ============================================================================
# load_recipes_config
# ============================================================================


def test_load_recipes_minimal(tmp_path):
    cfg_path = tmp_path / "recipes.json"
    cfg_path.write_text(json.dumps({
        "recipes": [
            {"id": "r1", "checkpoint": "/x/r1.pt",
             "seed_primary": 0, "tasks": ["arc_easy"]},
            {"id": "r2", "checkpoint": "/x/r2.pt",
             "seed_primary": 1, "tasks": ["arc_easy", "piqa"]},
        ]
    }))
    specs = load_recipes_config(cfg_path)
    assert len(specs) == 2
    assert specs[0].id == "r1"
    assert specs[0].seed_primary == 0
    assert specs[0].seed_retry == SEED_RETRY_OFFSET  # default
    assert specs[0].tasks == ("arc_easy",)
    assert specs[0].patch is None
    assert specs[1].tasks == ("arc_easy", "piqa")


def test_load_recipes_with_explicit_retry_seed(tmp_path):
    cfg_path = tmp_path / "recipes.json"
    cfg_path.write_text(json.dumps({
        "recipes": [
            {"id": "r1", "checkpoint": "x", "seed_primary": 5,
             "seed_retry": 9999, "tasks": ["arc_easy"]},
        ]
    }))
    specs = load_recipes_config(cfg_path)
    assert specs[0].seed_retry == 9999


def test_load_recipes_with_patch(tmp_path):
    cfg_path = tmp_path / "recipes.json"
    cfg_path.write_text(json.dumps({
        "recipes": [
            {"id": "r1", "checkpoint": "x", "patch": "/p/r1.patch",
             "seed_primary": 0, "tasks": ["arc_easy"]},
        ]
    }))
    specs = load_recipes_config(cfg_path)
    assert specs[0].patch == Path("/p/r1.patch")


def test_load_recipes_empty_list_rejected(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps({"recipes": []}))
    with pytest.raises(ValueError, match=r"non-empty 'recipes'"):
        load_recipes_config(cfg_path)


def test_load_recipes_missing_field_rejected(tmp_path):
    cfg_path = tmp_path / "bad.json"
    cfg_path.write_text(json.dumps({"recipes": [{"id": "r1"}]}))  # no checkpoint
    with pytest.raises(ValueError, match=r"recipe 0 missing"):
        load_recipes_config(cfg_path)


# ============================================================================
# estimate_cost
# ============================================================================


def test_estimate_cost_one_hour():
    assert estimate_cost(3600.0, per_rung_h100_hr_rate=2.0) == pytest.approx(2.0)


def test_estimate_cost_zero():
    assert estimate_cost(0.0) == 0.0


# ============================================================================
# run_one_recipe
# ============================================================================


def test_run_one_recipe_success(tmp_path, single_rung):
    spec = RecipeSpec(
        id="r1",
        checkpoint=tmp_path / "r1.pt",
        seed_primary=0,
        seed_retry=1000,
        tasks=("arc_easy",),
    )
    spec.checkpoint.write_bytes(b"")
    karpa_root = tmp_path / "karpa_root"
    karpa_root.mkdir()
    result = run_one_recipe(
        spec,
        karpa_root=karpa_root,
        bundle_dir=tmp_path / "bundle",
        bundle_sha256="x",
        output_dir=tmp_path / "out",
        command_prefix=_TEST_COMMAND_PREFIX,
        timeout_s_per_rung=15.0,
    )
    assert result.status == "success"
    assert result.seed_used == 0
    assert result.combined_report_path is not None
    assert Path(result.combined_report_path).exists()


def test_run_one_recipe_retry_on_first_failure(tmp_path, single_rung):
    """First seed fails (EvalSubprocessError), second seed succeeds.

    Mocks run_ladder_eval directly so we can deterministically fail the
    first call and succeed the second, regardless of subprocess env
    timing.
    """
    from unittest import mock

    from eval.downstream.runner_subprocess import EvalSubprocessError
    from eval.downstream.types import HARNESS_VERSION, CellResult, DownstreamReport
    from eval.hidden_eval import HiddenEvalResult as HER
    from validator.ladder import LadderEvalResult

    spec = RecipeSpec(
        id="r1",
        checkpoint=tmp_path / "r1.pt",
        seed_primary=0,
        seed_retry=1000,
        tasks=("arc_easy",),
    )
    spec.checkpoint.write_bytes(b"")
    (tmp_path / "karpa_root").mkdir()

    call_count = {"n": 0}

    def fake_run_ladder_eval(submission, config, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise EvalSubprocessError(
                reason="nonzero_exit", exit_code=7,
                stderr_tail="simulated", argv=[],
            )
        # Second call: return a synthetic LadderEvalResult.
        report = DownstreamReport(
            harness_version=HARNESS_VERSION,
            bundle_sha256=config.bundle_sha256,
            seed=config.seed,
            total_examples=1,
            wall_clock_s=0.5,
            cells={
                "arc_easy:S3": CellResult(
                    task="arc_easy", accuracy=0.5,
                    accuracy_stderr=0.0, n_examples=1, seed=config.seed,
                )
            },
        )
        return LadderEvalResult(
            per_rung_reports={"S3": report},
            combined_report=report,
            hidden_eval=HER(
                val_bpb=0.0, benchmark_accuracy=0.0, tokens_evaluated=0,
                benchmark_examples=0, eval_set_hash="", downstream=report,
            ),
        )

    with mock.patch("scripts.b6_run.run_ladder_eval", side_effect=fake_run_ladder_eval):
        result = run_one_recipe(
            spec,
            karpa_root=tmp_path / "karpa_root",
            bundle_dir=tmp_path / "bundle",
            bundle_sha256="x",
            output_dir=tmp_path / "out",
            command_prefix=_TEST_COMMAND_PREFIX,
            timeout_s_per_rung=15.0,
        )
    assert result.status == "success"
    assert result.seed_used == 1000  # retry seed
    assert len(result.abort_reasons) == 1
    assert "attempt_1_seed_0" in result.abort_reasons[0]
    assert call_count["n"] == 2


def test_run_one_recipe_both_seeds_fail(tmp_path, monkeypatch, single_rung):
    monkeypatch.setenv("KARPA_TEST_RUNNER_MODE", "nonzero")
    monkeypatch.setenv("KARPA_TEST_RUNNER_EXIT_CODE", "9")
    spec = RecipeSpec(
        id="r_fail",
        checkpoint=tmp_path / "r.pt",
        seed_primary=0,
        seed_retry=1000,
        tasks=("arc_easy",),
    )
    spec.checkpoint.write_bytes(b"")
    (tmp_path / "karpa_root").mkdir()
    result = run_one_recipe(
        spec,
        karpa_root=tmp_path / "karpa_root",
        bundle_dir=tmp_path / "bundle",
        bundle_sha256="x",
        output_dir=tmp_path / "out",
        command_prefix=_TEST_COMMAND_PREFIX,
        timeout_s_per_rung=15.0,
    )
    assert result.status == "aborted"
    assert result.combined_report_path is None
    assert len(result.abort_reasons) == 2
    abort_path = tmp_path / "out" / f"abort_log_{spec.id}.txt"
    assert abort_path.exists()


# ============================================================================
# run_b6
# ============================================================================


def _make_specs(n: int, tmp_path: Path) -> list[RecipeSpec]:
    specs = []
    for i in range(n):
        ckpt = tmp_path / f"r{i}.pt"
        ckpt.write_bytes(b"")
        specs.append(RecipeSpec(
            id=f"r{i}",
            checkpoint=ckpt,
            seed_primary=i,
            seed_retry=i + 1000,
            tasks=("arc_easy",),
        ))
    return specs


def test_run_b6_summary_counts_succeeded(tmp_path, single_rung):
    specs = _make_specs(3, tmp_path)
    (tmp_path / "karpa_root").mkdir()
    summary = run_b6(
        specs,
        karpa_root=tmp_path / "karpa_root",
        bundle_dir=tmp_path / "bundle",
        bundle_sha256="x",
        output_dir=tmp_path / "run_out",
        command_prefix=_TEST_COMMAND_PREFIX,
        timeout_s_per_rung=15.0,
    )
    assert summary.n_recipes_requested == 3
    assert summary.n_recipes_succeeded == 3
    assert summary.n_recipes_aborted == 0
    result_json = json.loads((tmp_path / "run_out" / "result.json").read_text())
    assert result_json["n_recipes_succeeded"] == 3


def test_run_b6_budget_guard_short_circuits_remaining(tmp_path, single_rung):
    """A $0 budget cap means even the first recipe is allowed (cumulative
    starts at 0), but all subsequent ones get budget_exhausted once the
    first one's cost is added."""
    specs = _make_specs(3, tmp_path)
    (tmp_path / "karpa_root").mkdir()
    summary = run_b6(
        specs,
        karpa_root=tmp_path / "karpa_root",
        bundle_dir=tmp_path / "bundle",
        bundle_sha256="x",
        output_dir=tmp_path / "run_out",
        budget_cap_usd=0.0001,  # essentially zero
        command_prefix=_TEST_COMMAND_PREFIX,
        timeout_s_per_rung=15.0,
    )
    # Recipe 0 runs (cumulative starts at 0 which is NOT >= cap).
    # Recipes 1 and 2 get short-circuited iff recipe 0 produced any cost.
    n_short = summary.n_recipes_budget_exhausted
    assert n_short >= 1


def test_run_b6_default_budget_cap_constant():
    assert DEFAULT_BUDGET_CAP_USD == 1200.0


def test_run_b6_default_n_recipes_constant():
    assert DEFAULT_N_RECIPES == 12


# ============================================================================
# CLI
# ============================================================================


def test_cli_required_args(tmp_path):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_minimal(tmp_path):
    parser = _build_parser()
    args = parser.parse_args([
        "--karpa-root", str(tmp_path),
        "--bundle-dir", str(tmp_path / "bundle"),
        "--bundle-sha-sha256", "x",
        "--recipes-config", str(tmp_path / "cfg.json"),
        "--output", str(tmp_path / "out"),
    ])
    assert args.budget_cap_usd == DEFAULT_BUDGET_CAP_USD
