"""Tests for scripts/build_refs_from_published.py.

Pure data transformation — no mocks. Verifies the shape of the
output refs.json + missing-reference handling + summary coverage report.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from scripts.analyze_b6_rho import PINNED_REFERENCES
from scripts.build_refs_from_published import _summary, build_refs, main


def _recipes_config_with_ids(ids: list[str]) -> dict:
    return {
        "recipes": [
            {"id": rid, "checkpoint": f"/x/{rid}.pt",
             "seed_primary": i, "tasks": ["arc_easy"]}
            for i, rid in enumerate(ids)
        ]
    }


# ============================================================================
# build_refs
# ============================================================================


def test_build_refs_happy_path():
    recipe_ids = ["r1", "r2", "r3"]
    published = {
        "r1": {ref: 0.5 for ref in PINNED_REFERENCES},
        "r2": {ref: 0.6 for ref in PINNED_REFERENCES},
        "r3": {ref: 0.7 for ref in PINNED_REFERENCES},
    }
    refs = build_refs(recipe_ids, published)
    assert set(refs.keys()) == {"r1", "r2", "r3"}
    for recipe_id in recipe_ids:
        assert set(refs[recipe_id].keys()) == set(PINNED_REFERENCES)


def test_build_refs_missing_recipe_yields_all_nan():
    refs = build_refs(["r1", "r2"], {"r1": {ref: 0.5 for ref in PINNED_REFERENCES}})
    assert refs["r1"]["olmo_2_1b_step_30b"] == 0.5
    for ref in PINNED_REFERENCES:
        assert math.isnan(refs["r2"][ref])


def test_build_refs_missing_reference_in_recipe():
    refs = build_refs(["r1"], {"r1": {"olmo_2_1b_step_30b": 0.6}})
    assert refs["r1"]["olmo_2_1b_step_30b"] == 0.6
    assert math.isnan(refs["r1"]["pythia_1_4b"])
    assert math.isnan(refs["r1"]["tinyllama_1_1b_3t"])


def test_build_refs_non_numeric_score_becomes_nan():
    refs = build_refs(["r1"], {"r1": {"olmo_2_1b_step_30b": "not_a_number"}})
    assert math.isnan(refs["r1"]["olmo_2_1b_step_30b"])


def test_build_refs_int_score_coerced_to_float():
    refs = build_refs(["r1"], {"r1": {"olmo_2_1b_step_30b": 1}})
    assert refs["r1"]["olmo_2_1b_step_30b"] == 1.0
    assert isinstance(refs["r1"]["olmo_2_1b_step_30b"], float)


def test_build_refs_preserves_recipe_order():
    recipe_ids = ["r3", "r1", "r2"]  # explicit non-sorted
    refs = build_refs(recipe_ids, {})
    assert list(refs.keys()) == recipe_ids


# ============================================================================
# _summary
# ============================================================================


def test_summary_full_coverage():
    refs = {
        "r1": {ref: 0.5 for ref in PINNED_REFERENCES},
        "r2": {ref: 0.6 for ref in PINNED_REFERENCES},
    }
    s = _summary(refs)
    assert s["n_recipes"] == 2
    for ref in PINNED_REFERENCES:
        assert s["coverage_per_reference"][ref] == 2
        assert s["missing_per_reference"][ref] == []


def test_summary_partial_coverage_lists_missing():
    refs = {
        "r1": {"olmo_2_1b_step_30b": 0.5,
               "pythia_1_4b": float("nan"),
               "tinyllama_1_1b_3t": 0.4},
        "r2": {"olmo_2_1b_step_30b": float("nan"),
               "pythia_1_4b": 0.55,
               "tinyllama_1_1b_3t": 0.45},
    }
    s = _summary(refs)
    assert s["coverage_per_reference"]["olmo_2_1b_step_30b"] == 1
    assert s["missing_per_reference"]["olmo_2_1b_step_30b"] == ["r2"]
    assert s["missing_per_reference"]["pythia_1_4b"] == ["r1"]


# ============================================================================
# main (CLI)
# ============================================================================


def test_main_happy_path(tmp_path):
    recipes_path = tmp_path / "recipes.json"
    recipes_path.write_text(json.dumps(_recipes_config_with_ids(["r1", "r2"])))
    pub_path = tmp_path / "pub.json"
    pub_path.write_text(json.dumps({
        "r1": {ref: 0.5 for ref in PINNED_REFERENCES},
        "r2": {ref: 0.6 for ref in PINNED_REFERENCES},
    }))
    out_path = tmp_path / "refs.json"
    rc = main([
        "--recipes-config", str(recipes_path),
        "--published-scores", str(pub_path),
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert set(on_disk.keys()) == {"r1", "r2"}


def test_main_creates_parent_dir(tmp_path):
    recipes_path = tmp_path / "recipes.json"
    recipes_path.write_text(json.dumps(_recipes_config_with_ids(["r1"])))
    pub_path = tmp_path / "pub.json"
    pub_path.write_text(json.dumps({"r1": {}}))
    out_path = tmp_path / "deep" / "nested" / "refs.json"
    rc = main([
        "--recipes-config", str(recipes_path),
        "--published-scores", str(pub_path),
        "--output", str(out_path),
    ])
    assert rc == 0
    assert out_path.exists()


def test_main_bad_recipes_config_returns_one(tmp_path):
    pub_path = tmp_path / "pub.json"
    pub_path.write_text("{}")
    out_path = tmp_path / "refs.json"
    rc = main([
        "--recipes-config", str(tmp_path / "nonexistent.json"),
        "--published-scores", str(pub_path),
        "--output", str(out_path),
    ])
    assert rc == 1


def test_main_bad_published_json_returns_one(tmp_path):
    recipes_path = tmp_path / "recipes.json"
    recipes_path.write_text(json.dumps(_recipes_config_with_ids(["r1"])))
    pub_path = tmp_path / "pub.json"
    pub_path.write_text("{not valid json")
    out_path = tmp_path / "refs.json"
    rc = main([
        "--recipes-config", str(recipes_path),
        "--published-scores", str(pub_path),
        "--output", str(out_path),
    ])
    assert rc == 1
