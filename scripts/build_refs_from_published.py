"""Build `refs.json` for `analyze_b6_rho.py` from a published-scores file.

The B6 protocol locks the reference set to
`{olmo_2_1b_step_30b, pythia_1_4b, tinyllama_1_1b_3t}`. For each of the
N=12 recipes, we need to assemble a `{recipe_id -> {ref_name -> score}}`
mapping that `analyze_b6_rho.py` consumes.

Two paths supported, mirroring the pre-registration language:
  * **Published path (default):** the operator supplies a YAML/JSON
    `published_scores.json` mapping `recipe_id -> {ref_name -> score}`
    directly. This script just validates the shape, fills missing
    references with `NaN` (which `analyze_b6_rho` drops per-reference),
    and writes the result to `refs.json`.
  * **Internal-eval path:** the operator runs each reference checkpoint
    through their own eval tooling (lm-eval-harness, etc.) and
    aggregates per-recipe scores into the same `published_scores.json`
    shape. This script's contract is unchanged.

USAGE:
    python scripts/build_refs_from_published.py \\
        --recipes-config eval/private/b6/b6_recipes.json \\
        --published-scores eval/private/b6/published_scores.json \\
        --output runs/b6_<id>/refs.json

INPUTS:
  recipes-config: same schema as scripts/b6_run.py — used to determine
    the canonical recipe id ordering.
  published-scores: JSON of shape
    {
      "r1": {"olmo_2_1b_step_30b": 0.62, "pythia_1_4b": 0.58, ...},
      "r2": {...},
      ...
    }
    Missing entries become NaN (analyze_b6_rho drops them per-reference).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.analyze_b6_rho import PINNED_REFERENCES  # noqa: E402
from scripts.b6_run import load_recipes_config  # noqa: E402


def build_refs(
    recipe_ids: list[str],
    published_scores: dict[str, dict[str, float]],
) -> dict:
    """Assemble the refs.json content.

    For each recipe in `recipe_ids`, emit a sub-dict mapping each pinned
    reference name to its float score (or NaN if missing). Returns the
    full dict (caller writes JSON).
    """
    out: dict[str, dict[str, float]] = {}
    for recipe_id in recipe_ids:
        sub = published_scores.get(recipe_id, {})
        out[recipe_id] = {}
        for ref_name in PINNED_REFERENCES:
            val = sub.get(ref_name)
            if val is None:
                out[recipe_id][ref_name] = float("nan")
                continue
            try:
                f = float(val)
            except (TypeError, ValueError):
                out[recipe_id][ref_name] = float("nan")
                continue
            out[recipe_id][ref_name] = f
    return out


def _summary(refs: dict) -> dict:
    """Coverage report: how many recipes have each reference."""
    counts = {ref: 0 for ref in PINNED_REFERENCES}
    missing: dict[str, list[str]] = {ref: [] for ref in PINNED_REFERENCES}
    for recipe_id, ref_map in refs.items():
        for ref_name in PINNED_REFERENCES:
            v = ref_map.get(ref_name, float("nan"))
            if isinstance(v, float) and math.isnan(v):
                missing[ref_name].append(recipe_id)
            else:
                counts[ref_name] += 1
    return {
        "n_recipes": len(refs),
        "coverage_per_reference": counts,
        "missing_per_reference": missing,
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts.build_refs_from_published")
    p.add_argument("--recipes-config", required=True, type=Path)
    p.add_argument("--published-scores", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        recipes = load_recipes_config(args.recipes_config)
        published = json.loads(args.published_scores.read_text())
    except Exception as e:
        print(f"build_refs FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    recipe_ids = [r.id for r in recipes]
    refs = build_refs(recipe_ids, published)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(refs, indent=2, sort_keys=True))
    summary = _summary(refs)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
