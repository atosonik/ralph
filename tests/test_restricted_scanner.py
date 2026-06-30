"""Adversarial tests for scan_diff_for_restricted.

Regression tests for the bypasses found in deep_review_2026-05-31 high #1:
  - tab-suffix on path (`--- a/eval/score.py\t2024-01-01...`)
  - rename headers (`rename from eval/score.py / rename to ...`)
  - copy headers
  - `diff --git` source/dest extraction
  - path traversal (`a/recipe/../eval/score.py`)
  - backslash separators
  - quoted paths with spaces
  - /dev/null sentinels for file create/delete
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from proof.runner import (
    _extract_diff_paths,
    scan_diff_for_exploit_patterns,
    scan_diff_for_restricted,
)

RESTRICTED = ["eval/**", "calibration/**", "validator/**", "proof/**", "restricted_files.yaml"]


# --- scan_diff_for_exploit_patterns: off-protocol input-injection (proof-forgery) ---
def _added(*lines: str) -> str:
    return "+++ b/recipe/train.py\n" + "\n".join("+" + ln for ln in lines) + "\n"


def test_warm_start_external_checkpoint_load_flagged():
    # PR#586-class: train() loads an off-protocol checkpoint from the miner's box.
    hits = scan_diff_for_exploit_patterns(
        _added('_ws = torch.load("/home/jovyan/v10sub/checkpoint.pt", weights_only=True)')
    )
    assert hits and "external/host path" in hits[0][0]


def test_noncanonical_data_path_flagged():
    # PR#344-class: a config points the data dir at a host path -> bypasses the lock.
    diff = '+++ b/configs/big.json\n+    "data_base_dir": "/mnt/scratch/SN40/data_50b",\n'
    assert scan_diff_for_exploit_patterns(diff)


def test_host_toolchain_path_flagged():
    assert scan_diff_for_exploit_patterns(_added('_GCC = "/home/root/diony/toolchain/gcc"'))


def test_abs_checkpoint_load_under_canonical_mount_flagged():
    # staging weights under a canonical mount still loads a .pt by absolute path.
    hits = scan_diff_for_exploit_patterns(_added('sd = torch.load("/data/staged/model.pt")'))
    assert hits and "absolute path" in hits[0][0]


def test_legit_lr_schedule_returns_clean():
    # the false positives a naive return-scan produced (PR#585 / #413).
    diff = _added(
        "    def lr_at(step):",
        "        return cfg.min_lr + (cfg.max_lr - cfg.min_lr) * frac",
        "        return cfg.max_lr",
    )
    assert scan_diff_for_exploit_patterns(diff) == []


def test_legit_relative_and_canonical_data_clean():
    diff = _added(
        '    shards = sorted(Path("data/shards").glob("*.bin"))',
        '    manifest = json.load(open("/data/data_manifest.json"))',
        '    ckpt = torch.load(out_dir / "checkpoint.pt")',
        '    os.environ.setdefault("TRITON_CACHE_DIR", "/tmp/ralph_triton")',
    )
    assert scan_diff_for_exploit_patterns(diff) == []


def test_removed_lines_not_scanned():
    # a removal (`-`) referencing a host path must not trip the scanner.
    diff = '+++ b/recipe/train.py\n-    x = torch.load("/home/old/ckpt.pt")\n'
    assert scan_diff_for_exploit_patterns(diff) == []


def test_clean_diff_does_not_match():
    diff = """diff --git a/configs/h100_default.json b/configs/h100_default.json
--- a/configs/h100_default.json
+++ b/configs/h100_default.json
@@ -10,7 +10,7 @@
   "max_lr": 5e-4,
-  "warmup_steps": 50,
+  "warmup_steps": 100,
"""
    assert scan_diff_for_restricted(diff, RESTRICTED) == []


def test_tab_suffixed_paths_caught():
    diff = """--- a/eval/score.py\t2024-01-01 10:00:00.000000000 +0000
+++ b/eval/score.py\t2024-01-01 10:00:01.000000000 +0000
@@ -1 +1 @@
-old
+new
"""
    assert "eval/score.py" in scan_diff_for_restricted(diff, RESTRICTED)


def test_rename_header_caught():
    diff = """diff --git a/safe.py b/eval/score.py
similarity index 100%
rename from safe.py
rename to eval/score.py
"""
    violations = scan_diff_for_restricted(diff, RESTRICTED)
    assert "eval/score.py" in violations


def test_copy_header_caught():
    diff = """diff --git a/recipe/optim.py b/calibration/optim.py
similarity index 95%
copy from recipe/optim.py
copy to calibration/optim.py
"""
    violations = scan_diff_for_restricted(diff, RESTRICTED)
    assert "calibration/optim.py" in violations


def test_path_traversal_caught():
    diff = """--- a/recipe/../eval/score.py
+++ b/recipe/../eval/score.py
+x
"""
    violations = scan_diff_for_restricted(diff, RESTRICTED)
    assert "eval/score.py" in violations


def test_backslash_normalized():
    diff = """--- a/eval\\score.py
+++ b/eval\\score.py
+x
"""
    violations = scan_diff_for_restricted(diff, RESTRICTED)
    assert "eval/score.py" in violations


def test_quoted_path_caught():
    diff = """diff --git "a/eval/score test.py" "b/eval/score test.py"
--- "a/eval/score test.py"
+++ "b/eval/score test.py"
+x
"""
    violations = scan_diff_for_restricted(diff, RESTRICTED)
    assert any("eval/score test.py" in v for v in violations)


def test_dev_null_ignored():
    """A new-file diff has --- /dev/null on the source side; not a violation."""
    diff = """--- /dev/null
+++ b/configs/new.json
+{}
"""
    assert scan_diff_for_restricted(diff, RESTRICTED) == []


def test_restricted_yaml_self_protection_with_tab():
    """The restricted_files.yaml file itself is restricted — must catch
    `--- a/restricted_files.yaml\t2024-...`."""
    diff = """--- a/restricted_files.yaml\t2024-01-01 10:00:00 +0000
+++ b/restricted_files.yaml\t2024-01-01 10:00:01 +0000
@@ -1 +1 @@
-restricted_paths:
+restricted_paths: []
"""
    assert "restricted_files.yaml" in scan_diff_for_restricted(diff, RESTRICTED)


def test_index_line_path():
    diff = """Index: validator/audit.py
===================================================================
--- a/validator/audit.py
+++ b/validator/audit.py
+x
"""
    assert "validator/audit.py" in scan_diff_for_restricted(diff, RESTRICTED)


def test_multiple_violations_each_appears_once():
    diff = """--- a/eval/a.py
+++ b/eval/a.py
+x
--- a/proof/b.py
+++ b/proof/b.py
+y
"""
    v = scan_diff_for_restricted(diff, RESTRICTED)
    assert "eval/a.py" in v
    assert "proof/b.py" in v
    # Each violation appears exactly once even though both --- and +++ name it
    assert len(set(v)) == len(v)


def test_extract_diff_paths_handles_all_forms():
    diff = """diff --git a/x.py b/y.py
--- a/x.py
+++ b/y.py
rename from x.py
rename to y.py
"""
    paths = _extract_diff_paths(diff)
    assert "x.py" in paths
    assert "y.py" in paths


# ----------------------------------------------------------------------------
# B1-D10: explicit globs for downstream-eval harness + pooled data dirs.
# Even though eval/** already covers all of these, the yaml lists them
# explicitly. These tests pin BOTH the eval/** coverage AND the
# explicit-globs behavior.
# ----------------------------------------------------------------------------

# The full live restricted-paths list from restricted_files.yaml, including
# the B1-D10 explicit globs. Hard-coded rather than parsed so the test
# fails loudly if someone removes a glob from the yaml without updating
# the test.
LIVE_RESTRICTED = [
    "eval/**",
    "eval/downstream/**",
    "eval/private/downstream_pool/**",
    "eval/private/hardness/**",
    "eval/private/calibration/**",
    "calibration/**",
    "validator/**",
    "proof/**",
    "restricted_files.yaml",
    "data_manifest.json",
]


def _diff_touching(path: str) -> str:
    return f"""diff --git a/{path} b/{path}
--- a/{path}
+++ b/{path}
@@ -1 +1 @@
-old
+new
"""


def test_eval_downstream_runner_blocked():
    """A patch touching eval/downstream/runner.py is rejected."""
    diff = _diff_touching("eval/downstream/runner.py")
    assert "eval/downstream/runner.py" in scan_diff_for_restricted(
        diff, LIVE_RESTRICTED,
    )


def test_eval_downstream_aggregate_blocked():
    """The Pareto kernel — protected against tampering."""
    diff = _diff_touching("eval/downstream/aggregate.py")
    assert "eval/downstream/aggregate.py" in scan_diff_for_restricted(
        diff, LIVE_RESTRICTED,
    )


def test_eval_downstream_calibration_blocked():
    diff = _diff_touching("eval/downstream/calibration.py")
    assert "eval/downstream/calibration.py" in scan_diff_for_restricted(
        diff, LIVE_RESTRICTED,
    )


def test_eval_private_downstream_pool_blocked():
    """The cached DCLM eval bundle — must not be modifiable."""
    diff = _diff_touching("eval/private/downstream_pool/bundle_v1/hellaswag.jsonl")
    assert (
        "eval/private/downstream_pool/bundle_v1/hellaswag.jsonl"
        in scan_diff_for_restricted(diff, LIVE_RESTRICTED)
    )


def test_eval_private_hardness_blocked():
    """The private hardness-index JSONL — the bottom-quintile selection."""
    diff = _diff_touching("eval/private/hardness/hardness_index_v1.jsonl")
    assert (
        "eval/private/hardness/hardness_index_v1.jsonl"
        in scan_diff_for_restricted(diff, LIVE_RESTRICTED)
    )


def test_eval_private_calibration_blocked():
    """noise_floors_v1.json — the king-rule threshold floors."""
    diff = _diff_touching("eval/private/calibration/noise_floors_v1.json")
    assert (
        "eval/private/calibration/noise_floors_v1.json"
        in scan_diff_for_restricted(diff, LIVE_RESTRICTED)
    )


# ----------------------------------------------------------------------------
# C1-LITE: validator/state/ + validator/cache/ globs for v0.11-lite lineage.
# ----------------------------------------------------------------------------


def test_validator_state_lineage_blocked():
    """A patch touching validator/state/lineage_state.json is rejected."""
    diff = _diff_touching("validator/state/lineage_state.json")
    assert "validator/state/lineage_state.json" in scan_diff_for_restricted(
        diff, LIVE_RESTRICTED + ["validator/state/**", "validator/cache/**"],
    )


def test_validator_cache_parent_blocked():
    """A patch touching validator/cache/parent_reproductions/ is rejected."""
    diff = _diff_touching("validator/cache/parent_reproductions/abc.json")
    assert "validator/cache/parent_reproductions/abc.json" in scan_diff_for_restricted(
        diff, LIVE_RESTRICTED + ["validator/state/**", "validator/cache/**"],
    )


def test_c1_lite_globs_present_in_live_yaml():
    """validator/state/** and validator/cache/** must be in the live yaml."""
    yaml_path = Path(__file__).resolve().parent.parent / "restricted_files.yaml"
    text = yaml_path.read_text()
    for glob in ("validator/state/**", "validator/cache/**"):
        assert f'"{glob}"' in text, (
            f"C1-LITE glob {glob!r} missing from restricted_files.yaml"
        )


def test_b1_d10_explicit_globs_present_in_live_yaml():
    """Read the live restricted_files.yaml and verify the B1-D10 globs
    are recorded. Catches accidental removals."""
    import os.path
    yaml_path = Path(__file__).resolve().parent.parent / "restricted_files.yaml"
    text = yaml_path.read_text()
    assert os.path.exists(yaml_path), f"restricted_files.yaml not found at {yaml_path}"
    for glob in (
        "eval/downstream/**",
        "eval/private/downstream_pool/**",
        "eval/private/hardness/**",
        "eval/private/calibration/**",
    ):
        assert f'"{glob}"' in text, (
            f"B1-D10 explicit glob {glob!r} missing from "
            "restricted_files.yaml — see DEFERRED.md B1-D10"
        )
