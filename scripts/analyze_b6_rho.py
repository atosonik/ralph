"""B6 frozen Spearman rho + 95% bootstrap CI analysis.

THIS SCRIPT IS FROZEN AT TAG `b6-preregistered-v1`. Post-hoc edits are
forbidden by the pre-registration. Any required correction must be
published as a delta-pre-registration at a new commit tag, NOT a silent
edit here.

ALGORITHM (pinned):

  1. Read the B6 run summary (`runs/<run_id>/result.json`) and the
     reference scores (`refs.json` — operator-prepared mapping
     `recipe_id → {reference_name: float}` for each reference in the
     pre-committed set).
  2. For each reference, build paired vectors
     `(ralph_s3_overall[i], ref_score[i])` over the recipe set, dropping
     any recipe whose Ralph run aborted OR whose reference score is NaN
     (the survivor count is reported per reference).
  3. Compute the Spearman ρ via Pearson on average-ranked values
     (matches scipy.stats.spearmanr with `nan_policy='omit'` semantics
     after the manual drop step).
  4. Bootstrap 95% CI via `n_resamples=10000` paired-index resamples
     with a deterministic seed (`bootstrap_seed=0` pinned). Lower bound
     = 2.5th percentile; upper bound = 97.5th percentile. Degenerate
     resamples (constant rank vector → division by zero) are
     skipped; the effective `n_resamples_used` is reported.
  5. PASS gate (pinned in pre-registration):
       (a) Across all references, the LOWER bound of the 95% CI exceeds
           PASS_LOWER_CI_THRESHOLD = 0.5.
       (b) AND the point estimate vs OLMo-2-1B specifically exceeds
           PASS_OLMO_POINT_THRESHOLD = 0.6.

  6. Output a structured decision JSON. The operator publishes this
     verbatim to `runs/<run_id>/analysis.json` and references the SHA
     in the B6 result post.

NO POST-HOC GATING: this script does not allow ad-hoc reference
substitution, threshold tweaking, or recipe drops beyond the NaN-handling
spec above. The thresholds + reference set + bootstrap seed + n_resamples
are constants at the top of this file. Any change must move the tag.

USAGE:
    python scripts/analyze_b6_rho.py \\
        --run-result runs/b6_2026_07/result.json \\
        --refs runs/b6_2026_07/refs.json \\
        --output runs/b6_2026_07/analysis.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================================
# FROZEN PROTOCOL CONSTANTS — do not edit at this tag.
# ============================================================================

# The pre-committed multi-reference set. Order is significant for the
# survivor accounting; any change at this tag is forbidden.
PINNED_REFERENCES: tuple[str, ...] = (
    "olmo_2_1b_step_30b",
    "pythia_1_4b",
    "tinyllama_1_1b_3t",
)

# PASS thresholds.
PASS_LOWER_CI_THRESHOLD: float = 0.5
PASS_OLMO_POINT_THRESHOLD: float = 0.6
PASS_OLMO_REFERENCE_NAME: str = "olmo_2_1b_step_30b"

# Bootstrap parameters.
BOOTSTRAP_N_RESAMPLES: int = 10_000
BOOTSTRAP_CI: float = 0.95
BOOTSTRAP_SEED: int = 0

# Which Ralph axis we correlate with the reference. S3 overall is
# pinned per the pre-registration; per-axis breakdowns are diagnostic
# only, not the GATE.
RALPH_AXIS_FOR_RHO: str = "s3_overall"


# ============================================================================
# Data classes
# ============================================================================


@dataclass(frozen=True)
class ReferenceRhoResult:
    """Per-reference Spearman result + CI."""

    reference_name: str
    n_pairs_used: int
    rho_point_estimate: float
    ci_lower: float
    ci_upper: float
    n_resamples_used: int


@dataclass(frozen=True)
class B6AnalysisResult:
    """Top-level analysis output."""

    pinned_references: list[str]
    ralph_axis: str
    pass_lower_ci_threshold: float
    pass_olmo_point_threshold: float
    pass_olmo_reference_name: str
    bootstrap_n_resamples: int
    bootstrap_ci: float
    bootstrap_seed: int
    per_reference: list[ReferenceRhoResult]
    all_lower_ci_above_threshold: bool
    olmo_point_above_threshold: bool
    decision: str  # "PASS" | "FAIL"
    decision_reason: str


# ============================================================================
# Ralph S3-overall extraction
# ============================================================================


def _ralph_s3_overall_from_report(report_dict: dict) -> Optional[float]:
    """Extract a single S3-overall scalar from a DownstreamReport dict.

    v0.11-lite definition: mean accuracy across all cells with the `:S3`
    suffix (cells from CORE-22 + private_hard at S3). If no S3 cells are
    present → None (recipe will be dropped from the rho computation).
    """
    cells = report_dict.get("cells", {})
    s3_accs = [
        c["accuracy"]
        for key, c in cells.items()
        if key.endswith(":S3")
    ]
    if not s3_accs:
        return None
    return sum(s3_accs) / len(s3_accs)


def _ralph_score_for_recipe(
    recipe_record: dict,
    per_recipe_dir: Path,
) -> Optional[float]:
    """Read a recipe's per-recipe report and return its S3 overall.

    Returns None if the recipe aborted, was budget-exhausted, or had no
    S3 cells (the analyze step drops it from the rho pair set).
    """
    if recipe_record.get("status") != "success":
        return None
    path = recipe_record.get("combined_report_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        # Fall back: try per_recipe_dir
        p = per_recipe_dir / f"{recipe_record['id']}.json"
        if not p.exists():
            return None
    try:
        report_dict = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return _ralph_s3_overall_from_report(report_dict)


# ============================================================================
# Spearman rho — manual implementation (no scipy dep)
# ============================================================================


def _average_rank(values: list[float]) -> list[float]:
    """Average-rank ties (scipy.stats.rankdata 'average' method semantics).

    Ranks are 1-based per the canonical definition. Tied values share
    the average of their tied positions.
    """
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # average of 1-based positions
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation. Raises ZeroDivisionError on a constant vector."""
    n = len(x)
    if n < 2:
        raise ValueError(f"need >= 2 pairs; got {n}")
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
    sx2 = sum((v - mx) ** 2 for v in x)
    sy2 = sum((v - my) ** 2 for v in y)
    denom = math.sqrt(sx2 * sy2)
    if denom == 0.0:
        raise ZeroDivisionError("constant vector — Pearson undefined")
    return num / denom


def spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman ρ via Pearson on average-ranked values."""
    if len(x) != len(y):
        raise ValueError(
            f"x/y length mismatch: {len(x)} vs {len(y)}"
        )
    return _pearson(_average_rank(x), _average_rank(y))


# ============================================================================
# Bootstrap CI
# ============================================================================


def bootstrap_spearman_ci(
    x: list[float],
    y: list[float],
    *,
    n_resamples: int = BOOTSTRAP_N_RESAMPLES,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, int]:
    """Bootstrap 95% CI on Spearman ρ via paired-index resampling.

    Returns `(ci_lower, ci_upper, n_resamples_used)`. Degenerate
    resamples (e.g. constant rank vectors) are skipped; if FEWER than
    100 valid samples survive, raises.
    """
    rng = random.Random(seed)
    n = len(x)
    if n < 3:
        raise ValueError(f"bootstrap requires >= 3 pairs; got {n}")
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        bx = [x[i] for i in idx]
        by = [y[i] for i in idx]
        try:
            samples.append(spearman_rho(bx, by))
        except (ZeroDivisionError, ValueError):
            continue
    if len(samples) < 100:
        raise RuntimeError(
            f"bootstrap produced only {len(samples)} valid samples "
            f"(of {n_resamples} attempted); rho is too unstable to "
            "report a CI"
        )
    samples.sort()
    half = (1 - ci) / 2.0
    lo_idx = int(half * len(samples))
    hi_idx = int((1 - half) * len(samples)) - 1
    return samples[lo_idx], samples[hi_idx], len(samples)


# ============================================================================
# Main analysis
# ============================================================================


def analyze_b6(
    run_result: dict,
    refs: dict,
    per_recipe_dir: Path,
) -> B6AnalysisResult:
    """Top-level analysis. `run_result` is the parsed `result.json` dict;
    `refs` is the parsed `refs.json` dict mapping `recipe_id` to a
    `{reference_name: float}` sub-dict.

    Raises ValueError if a pinned reference is missing from ALL recipes
    (i.e. operator forgot a reference column).
    """
    recipes = run_result.get("recipes", [])
    if not recipes:
        raise ValueError("run_result.recipes is empty")

    # Build the Ralph score vector (per-recipe S3 overall) — drops
    # aborted / missing recipes.
    ralph_by_id: dict[str, float] = {}
    for rec in recipes:
        score = _ralph_score_for_recipe(rec, per_recipe_dir)
        if score is not None:
            ralph_by_id[rec["id"]] = score

    if not ralph_by_id:
        raise ValueError("no Ralph S3 scores survived; cannot compute rho")

    per_reference: list[ReferenceRhoResult] = []
    for ref_name in PINNED_REFERENCES:
        x: list[float] = []  # Ralph S3 overall
        y: list[float] = []  # reference score
        for recipe_id, ralph_score in ralph_by_id.items():
            ref_score_map = refs.get(recipe_id, {})
            ref_score = ref_score_map.get(ref_name)
            if ref_score is None:
                continue
            try:
                ref_f = float(ref_score)
            except (TypeError, ValueError):
                continue
            if math.isnan(ref_f):
                continue
            x.append(ralph_score)
            y.append(ref_f)
        if len(x) < 3:
            # Not enough pairs; CI is reported as nan and the gate fails
            # for this reference.
            per_reference.append(ReferenceRhoResult(
                reference_name=ref_name,
                n_pairs_used=len(x),
                rho_point_estimate=float("nan"),
                ci_lower=float("nan"),
                ci_upper=float("nan"),
                n_resamples_used=0,
            ))
            continue
        try:
            rho = spearman_rho(x, y)
        except (ZeroDivisionError, ValueError):
            rho = float("nan")
        try:
            lo, hi, n_used = bootstrap_spearman_ci(x, y)
        except (RuntimeError, ValueError):
            lo, hi, n_used = float("nan"), float("nan"), 0
        per_reference.append(ReferenceRhoResult(
            reference_name=ref_name,
            n_pairs_used=len(x),
            rho_point_estimate=rho,
            ci_lower=lo,
            ci_upper=hi,
            n_resamples_used=n_used,
        ))

    # Gate evaluation.
    all_lower_above = all(
        not math.isnan(r.ci_lower) and r.ci_lower > PASS_LOWER_CI_THRESHOLD
        for r in per_reference
    )
    olmo_result = next(
        (r for r in per_reference if r.reference_name == PASS_OLMO_REFERENCE_NAME),
        None,
    )
    olmo_above = (
        olmo_result is not None
        and not math.isnan(olmo_result.rho_point_estimate)
        and olmo_result.rho_point_estimate > PASS_OLMO_POINT_THRESHOLD
    )
    decision = "PASS" if (all_lower_above and olmo_above) else "FAIL"
    if decision == "PASS":
        reason = (
            f"all CI lower bounds > {PASS_LOWER_CI_THRESHOLD} "
            f"AND OLMo point > {PASS_OLMO_POINT_THRESHOLD}"
        )
    else:
        parts = []
        if not all_lower_above:
            parts.append(f"some CI lower bound <= {PASS_LOWER_CI_THRESHOLD}")
        if not olmo_above:
            parts.append(f"OLMo point estimate <= {PASS_OLMO_POINT_THRESHOLD}")
        reason = "; ".join(parts)

    return B6AnalysisResult(
        pinned_references=list(PINNED_REFERENCES),
        ralph_axis=RALPH_AXIS_FOR_RHO,
        pass_lower_ci_threshold=PASS_LOWER_CI_THRESHOLD,
        pass_olmo_point_threshold=PASS_OLMO_POINT_THRESHOLD,
        pass_olmo_reference_name=PASS_OLMO_REFERENCE_NAME,
        bootstrap_n_resamples=BOOTSTRAP_N_RESAMPLES,
        bootstrap_ci=BOOTSTRAP_CI,
        bootstrap_seed=BOOTSTRAP_SEED,
        per_reference=per_reference,
        all_lower_ci_above_threshold=all_lower_above,
        olmo_point_above_threshold=olmo_above,
        decision=decision,
        decision_reason=reason,
    )


# ============================================================================
# CLI
# ============================================================================


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts.analyze_b6_rho")
    p.add_argument("--run-result", required=True, type=Path,
                   help="path to runs/<id>/result.json")
    p.add_argument("--refs", required=True, type=Path,
                   help="path to runs/<id>/refs.json")
    p.add_argument("--output", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_result = json.loads(args.run_result.read_text())
    refs = json.loads(args.refs.read_text())
    per_recipe_dir = args.run_result.parent / "per_recipe"
    result = analyze_b6(run_result, refs, per_recipe_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True))
    print(json.dumps({
        "decision": result.decision,
        "reason": result.decision_reason,
        "output": str(args.output),
    }, indent=2, sort_keys=True))
    return 0 if result.decision == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
