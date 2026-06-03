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

import karpa_bootstrap  # noqa: F401
from proof.runner import _extract_diff_paths, scan_diff_for_restricted

RESTRICTED = ["eval/**", "calibration/**", "validator/**", "proof/**", "restricted_files.yaml"]


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
