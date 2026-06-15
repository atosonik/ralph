"""C1-LITE runner_cli apply_patch integration tests.

The B1-D13 stub is replaced by a real implementation that:
  * Copies --ralph-root to a tmp workdir
  * Applies --patch via `patch -p1 --no-backup-if-mismatch`
  * Prepends the patched workdir to sys.path before model import

These tests exercise the patch-application path directly (without
invoking the full model import + eval chain).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from eval.downstream.runner_cli import _apply_patch_to_workdir, main

# ============================================================================
# _apply_patch_to_workdir helper
# ============================================================================


def _write_recipe(root: Path):
    """Create a tiny "ralph root" layout with a recipe file we can patch."""
    root.mkdir(parents=True)
    (root / "recipe").mkdir()
    (root / "recipe" / "config.txt").write_text("baseline_lr=1e-3\n")


def test_apply_patch_to_workdir_returns_distinct_dir(tmp_path):
    ralph = tmp_path / "ralph"
    _write_recipe(ralph)
    patch = tmp_path / "empty.patch"
    patch.write_text("")  # empty patch == no-op
    out = _apply_patch_to_workdir(patch, ralph)
    assert out != ralph
    assert (out / "recipe" / "config.txt").exists()


def test_apply_patch_empty_patch_is_noop(tmp_path):
    ralph = tmp_path / "ralph"
    _write_recipe(ralph)
    patch = tmp_path / "empty.patch"
    patch.write_text("")
    out = _apply_patch_to_workdir(patch, ralph)
    assert (out / "recipe" / "config.txt").read_text() == "baseline_lr=1e-3\n"


def test_apply_patch_real_patch_modifies_workdir(tmp_path):
    """A unified diff modifying config.txt is applied cleanly."""
    ralph = tmp_path / "ralph"
    _write_recipe(ralph)
    patch_text = (
        "--- a/recipe/config.txt\n"
        "+++ b/recipe/config.txt\n"
        "@@ -1 +1 @@\n"
        "-baseline_lr=1e-3\n"
        "+baseline_lr=5e-4\n"
    )
    patch = tmp_path / "p.patch"
    patch.write_text(patch_text)
    out = _apply_patch_to_workdir(patch, ralph)
    assert (out / "recipe" / "config.txt").read_text() == "baseline_lr=5e-4\n"
    # Original is untouched.
    assert (ralph / "recipe" / "config.txt").read_text() == "baseline_lr=1e-3\n"


def test_apply_patch_failure_raises(tmp_path):
    """A patch against missing context fails loudly."""
    ralph = tmp_path / "ralph"
    _write_recipe(ralph)
    patch_text = (
        "--- a/recipe/config.txt\n"
        "+++ b/recipe/config.txt\n"
        "@@ -1 +1 @@\n"
        "-this_line_does_not_exist\n"
        "+but_we_are_patching_it_anyway\n"
    )
    patch = tmp_path / "p.patch"
    patch.write_text(patch_text)
    with pytest.raises(RuntimeError, match=r"patch failed"):
        _apply_patch_to_workdir(patch, ralph)


def test_apply_patch_missing_ralph_root_raises(tmp_path):
    patch = tmp_path / "p.patch"
    patch.write_text("")
    with pytest.raises(FileNotFoundError, match=r"ralph_root"):
        _apply_patch_to_workdir(patch, tmp_path / "missing")


def test_apply_patch_missing_patch_raises(tmp_path):
    ralph = tmp_path / "ralph"
    _write_recipe(ralph)
    with pytest.raises(FileNotFoundError, match=r"patch"):
        _apply_patch_to_workdir(tmp_path / "missing.patch", ralph)


# ============================================================================
# main() --patch surface
# ============================================================================


def test_main_requires_ralph_root_when_patch_given(tmp_path):
    """--patch without --ralph-root must fail cleanly (no GPU work attempted)."""
    cfg = tmp_path / "cfg.json"
    cfg.write_text('{"tasks": ["arc_easy"]}')
    patch = tmp_path / "p.patch"
    patch.write_text("")
    with pytest.raises(ValueError, match=r"--patch requires --ralph-root"):
        main([
            "--checkpoint", "x",
            "--config", str(cfg),
            "--output", str(tmp_path / "out.json"),
            "--bundle-sha", "x",
            "--bundle-dir", str(tmp_path),
            "--vocab-size", "50257",
            "--patch", str(patch),
        ])
