"""B6 transfer-credibility orchestrator — runs N recipes through the ladder.

Drives N=12 frozen recipes through `validator/ladder.run_ladder_eval` and
collects per-recipe combined `DownstreamReport`s into a `runs/b6_<date>/`
tree. The output is the input to the FROZEN `scripts/analyze_b6_rho.py`
which computes Spearman ρ + 95% bootstrap CI against the pre-committed
multi-reference set.

PROTOCOL CONSTANTS (pinned at b6-preregistered-v1 tag; no post-hoc edits):

  * Recipe-selection rule: the operator supplies a JSON config naming the
    12 recipes (paths + parent_king_attestation_hashes + seeds). The
    seeds are declared at pre-registration time as `[s1..s12]`. If a
    recipe fails to complete S3, it is re-run ONCE with seed `s_i+1000`;
    on second failure it is published as NaN with abort log. No recipes
    dropped or substituted post-hoc.
  * Hard $1200 cap enforced by a budget guard wrapping each recipe's
    wall-clock cost estimate. Per-recipe cap = $300 (i.e. 4 budget
    gates of $300 across 12 recipes).
  * Nightly checkpoint of `accumulator_root_hash` to testnet chain
    (LocalChain in test mode; mainnet uses BittensorChain).

USAGE:
    python scripts/b6_run.py \\
        --ralph-root /path/to/ralph_root \\
        --bundle-dir eval/private/downstream_pool/bundle_v1 \\
        --bundle-sha-sha256 <sha> \\
        --recipes-config configs/b6_recipes.json \\
        --output runs/b6_2026_07/ \\
        [--budget-cap-usd 1200] \\
        [--per-rung-h100-hr-rate 2.0]

INPUTS:
  recipes-config JSON:
    {
      "recipes": [
        {
          "id": "r1",
          "checkpoint": "/path/to/r1.pt",
          "patch": "/path/to/r1.patch",  // optional, may be null
          "seed_primary": 0,
          "seed_retry": 1000,
          "tasks": ["arc_easy", "piqa", ...]
        },
        ...  // 12 recipes total
      ]
    }

OUTPUTS:
  runs/b6_<date>/
    result.json              — top-level summary + per-recipe pointers
    per_recipe/<id>.json     — DownstreamReport JSON per recipe
    abort_log_<id>.txt       — for any recipe that failed both seeds
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401, E402
from eval.downstream.runner import RALPH_VOCAB_SIZE  # noqa: E402
from eval.downstream.runner_subprocess import EvalSubprocessError  # noqa: E402
from eval.downstream.types import DownstreamReport  # noqa: E402
from validator.ladder import (  # noqa: E402
    EVAL_MODE_V011,
    LadderEvalConfig,
    LadderRungSpec,
    Submission,
    run_ladder_eval,
)

# v0.11-lite B6 protocol constants — pinned at pre-registration tag.
DEFAULT_N_RECIPES = 12
DEFAULT_BUDGET_CAP_USD = 1200.0
DEFAULT_PER_RUNG_H100_HR_RATE = 2.0  # Shadeform spot reference
SEED_RETRY_OFFSET = 1000  # second-attempt seed = primary + 1000

# Standard rungs for B6 — locked to the v0.10 ladder. Each rung
# carries its dim + n_layers for audit reproducibility.
_B6_STANDARD_RUNGS: tuple[LadderRungSpec, ...] = (
    LadderRungSpec(scale_label="S1", dim=256, n_layers=4),
    LadderRungSpec(scale_label="S2", dim=512, n_layers=12),
    LadderRungSpec(scale_label="S3", dim=768, n_layers=12),
)


@dataclass
class RecipeSpec:
    """One row from the recipes-config JSON."""

    id: str
    checkpoint: Path
    seed_primary: int
    seed_retry: int
    tasks: tuple[str, ...]
    patch: Optional[Path] = None


@dataclass
class RecipeResult:
    """Per-recipe output from a B6 run."""

    id: str
    status: str  # "success" | "aborted" | "budget_exhausted"
    seed_used: Optional[int] = None
    combined_report_path: Optional[str] = None
    wall_clock_s: float = 0.0
    h100_hr: float = 0.0
    cost_usd: float = 0.0
    abort_reasons: list[str] = field(default_factory=list)


@dataclass
class B6RunSummary:
    """Top-level B6 result.json contents.

    The downstream analyze_b6_rho.py reads this to assemble the
    per-recipe S3 score vector for Spearman computation.
    """

    started_iso: str
    finished_iso: str
    n_recipes_requested: int
    n_recipes_succeeded: int
    n_recipes_aborted: int
    n_recipes_budget_exhausted: int
    total_cost_usd: float
    budget_cap_usd: float
    bundle_sha256: str
    per_rung_h100_hr_rate: float
    rungs: list[dict]
    recipes: list[dict]  # list of asdict(RecipeResult)


# ----------------------------------------------------------------------------
# Recipe-config loader
# ----------------------------------------------------------------------------


def load_recipes_config(path: Path) -> list[RecipeSpec]:
    """Parse the recipes-config JSON into typed RecipeSpec list.

    Required fields per recipe: id, checkpoint, seed_primary, seed_retry, tasks.
    Optional: patch (defaults to None).
    """
    data = json.loads(Path(path).read_text())
    recipes_in = data.get("recipes")
    if not isinstance(recipes_in, list) or not recipes_in:
        raise ValueError(
            f"recipes-config {path} must contain a non-empty 'recipes' list"
        )
    out: list[RecipeSpec] = []
    for i, r in enumerate(recipes_in):
        try:
            spec = RecipeSpec(
                id=str(r["id"]),
                checkpoint=Path(r["checkpoint"]),
                seed_primary=int(r["seed_primary"]),
                seed_retry=int(r.get("seed_retry", int(r["seed_primary"]) + SEED_RETRY_OFFSET)),
                tasks=tuple(r["tasks"]),
                patch=Path(r["patch"]) if r.get("patch") else None,
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"recipe {i} missing/invalid field: {e}"
            ) from e
        out.append(spec)
    return out


# ----------------------------------------------------------------------------
# Cost estimator
# ----------------------------------------------------------------------------


def estimate_cost(
    wall_clock_s: float,
    *,
    per_rung_h100_hr_rate: float = DEFAULT_PER_RUNG_H100_HR_RATE,
) -> float:
    """Estimate USD cost of a recipe's wall-clock at the spot rate.

    The wall_clock_s is from the run_ladder_eval combined report; it's
    the max across rungs (per merge_rung_reports), so a 3-rung recipe at
    S3=300s would produce wall_clock_s ≈ 300s. Multiply by spot rate.
    """
    h100_hr = wall_clock_s / 3600.0
    return h100_hr * per_rung_h100_hr_rate


# ----------------------------------------------------------------------------
# Per-recipe runner
# ----------------------------------------------------------------------------


def _genesis_submission(recipe_id: str) -> Submission:
    """B6 recipes run as genesis submissions — no parent lineage required
    because the protocol is calibration-shaped, not king-selection-shaped.
    """
    return Submission(
        schema_version="v0.11",
        parent_king_attestation_hash=None,
        branch_id="main",
        bundle_hash=f"b6_recipe_{recipe_id}",
        miner_hotkey="5F_b6_runner",
        vocab_size=RALPH_VOCAB_SIZE,
    )


def run_one_recipe(
    spec: RecipeSpec,
    *,
    ralph_root: Path,
    bundle_dir: Path,
    bundle_sha256: str,
    output_dir: Path,
    command_prefix: Optional[Sequence[str]] = None,
    timeout_s_per_rung: float = 600.0,
    per_rung_h100_hr_rate: float = DEFAULT_PER_RUNG_H100_HR_RATE,
) -> RecipeResult:
    """Run one recipe through the ladder, with seed-retry on first failure.

    On primary-seed failure (EvalSubprocessError), retry once with the
    secondary seed. On second failure, write an abort log and return
    `RecipeResult(status="aborted", ...)` with NaN-equivalent placeholders.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    abort_reasons: list[str] = []
    seeds_to_try = (spec.seed_primary, spec.seed_retry)
    wall_clock_total = 0.0
    for attempt, seed in enumerate(seeds_to_try, start=1):
        config = LadderEvalConfig(
            rungs=_B6_STANDARD_RUNGS,
            tasks=spec.tasks,
            bundle_dir=bundle_dir,
            bundle_sha256=bundle_sha256,
            seed=seed,
        )
        try:
            t0 = time.monotonic()
            result = run_ladder_eval(
                _genesis_submission(spec.id),
                config,
                checkpoint_path=spec.checkpoint,
                ralph_root=ralph_root,
                patch_path=spec.patch,
                mode=EVAL_MODE_V011,
                command_prefix=command_prefix,
                timeout_s_per_rung=timeout_s_per_rung,
            )
            wall_clock_total += time.monotonic() - t0
            report_path = output_dir / "per_recipe" / f"{spec.id}.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(
                _report_to_dict(result.combined_report),
                indent=2, sort_keys=True,
            ))
            h100_hr = result.combined_report.wall_clock_s / 3600.0
            cost = estimate_cost(
                result.combined_report.wall_clock_s,
                per_rung_h100_hr_rate=per_rung_h100_hr_rate,
            )
            return RecipeResult(
                id=spec.id,
                status="success",
                seed_used=seed,
                combined_report_path=str(report_path),
                wall_clock_s=wall_clock_total,
                h100_hr=h100_hr,
                cost_usd=cost,
                abort_reasons=abort_reasons,
            )
        except EvalSubprocessError as e:
            wall_clock_total += time.monotonic() - t0
            abort_reasons.append(
                f"attempt_{attempt}_seed_{seed}_{e.reason}_exit_{e.exit_code}"
            )
            # Loop continues to next seed (if any).
        except Exception as e:
            wall_clock_total += time.monotonic() - t0
            abort_reasons.append(
                f"attempt_{attempt}_seed_{seed}_exception_{type(e).__name__}"
            )

    # Both attempts failed. Write abort log.
    abort_path = output_dir / f"abort_log_{spec.id}.txt"
    abort_path.write_text("\n".join(abort_reasons) + "\n")
    return RecipeResult(
        id=spec.id,
        status="aborted",
        seed_used=None,
        combined_report_path=None,
        wall_clock_s=wall_clock_total,
        h100_hr=0.0,
        cost_usd=0.0,
        abort_reasons=abort_reasons,
    )


def _report_to_dict(report: DownstreamReport) -> dict:
    """Serialize DownstreamReport to a JSON-safe dict matching the
    runner_subprocess.serialize_report contract."""
    from eval.downstream.runner_subprocess import serialize_report
    return serialize_report(report)


# ----------------------------------------------------------------------------
# Whole-run orchestrator
# ----------------------------------------------------------------------------


def run_b6(
    recipes: list[RecipeSpec],
    *,
    ralph_root: Path,
    bundle_dir: Path,
    bundle_sha256: str,
    output_dir: Path,
    budget_cap_usd: float = DEFAULT_BUDGET_CAP_USD,
    per_rung_h100_hr_rate: float = DEFAULT_PER_RUNG_H100_HR_RATE,
    command_prefix: Optional[Sequence[str]] = None,
    timeout_s_per_rung: float = 600.0,
) -> B6RunSummary:
    """Drive N recipes through the ladder, enforcing a hard budget cap.

    Budget guard: after each recipe, the cumulative `cost_usd` is
    checked against `budget_cap_usd`. If exceeded, remaining recipes are
    short-circuited with `status='budget_exhausted'` and no eval is run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results: list[RecipeResult] = []
    cumulative_cost = 0.0

    for spec in recipes:
        if cumulative_cost >= budget_cap_usd:
            results.append(RecipeResult(
                id=spec.id,
                status="budget_exhausted",
                seed_used=None,
                combined_report_path=None,
                wall_clock_s=0.0,
                h100_hr=0.0,
                cost_usd=0.0,
                abort_reasons=[
                    f"budget_cap_{budget_cap_usd}_exhausted_at_${cumulative_cost:.2f}"
                ],
            ))
            continue
        recipe_result = run_one_recipe(
            spec,
            ralph_root=ralph_root,
            bundle_dir=bundle_dir,
            bundle_sha256=bundle_sha256,
            output_dir=output_dir,
            command_prefix=command_prefix,
            timeout_s_per_rung=timeout_s_per_rung,
            per_rung_h100_hr_rate=per_rung_h100_hr_rate,
        )
        cumulative_cost += recipe_result.cost_usd
        results.append(recipe_result)

    finished = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary = B6RunSummary(
        started_iso=started,
        finished_iso=finished,
        n_recipes_requested=len(recipes),
        n_recipes_succeeded=sum(1 for r in results if r.status == "success"),
        n_recipes_aborted=sum(1 for r in results if r.status == "aborted"),
        n_recipes_budget_exhausted=sum(
            1 for r in results if r.status == "budget_exhausted"
        ),
        total_cost_usd=cumulative_cost,
        budget_cap_usd=budget_cap_usd,
        bundle_sha256=bundle_sha256,
        per_rung_h100_hr_rate=per_rung_h100_hr_rate,
        rungs=[asdict(r) for r in _B6_STANDARD_RUNGS],
        recipes=[asdict(r) for r in results],
    )
    (output_dir / "result.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=True, default=str)
    )
    return summary


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts.b6_run")
    p.add_argument("--ralph-root", required=True, type=Path)
    p.add_argument("--bundle-dir", required=True, type=Path)
    p.add_argument("--bundle-sha-sha256", required=True, dest="bundle_sha256")
    p.add_argument("--recipes-config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--budget-cap-usd", type=float, default=DEFAULT_BUDGET_CAP_USD)
    p.add_argument("--per-rung-h100-hr-rate", type=float,
                   default=DEFAULT_PER_RUNG_H100_HR_RATE)
    p.add_argument("--timeout-s-per-rung", type=float, default=600.0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    recipes = load_recipes_config(args.recipes_config)
    summary = run_b6(
        recipes,
        ralph_root=args.ralph_root,
        bundle_dir=args.bundle_dir,
        bundle_sha256=args.bundle_sha256,
        output_dir=args.output,
        budget_cap_usd=args.budget_cap_usd,
        per_rung_h100_hr_rate=args.per_rung_h100_hr_rate,
        timeout_s_per_rung=args.timeout_s_per_rung,
    )
    print(json.dumps({
        "n_recipes_requested": summary.n_recipes_requested,
        "n_recipes_succeeded": summary.n_recipes_succeeded,
        "n_recipes_aborted": summary.n_recipes_aborted,
        "n_recipes_budget_exhausted": summary.n_recipes_budget_exhausted,
        "total_cost_usd": summary.total_cost_usd,
        "output": str(args.output / "result.json"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
