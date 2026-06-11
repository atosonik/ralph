"""Tests for the meaningful-failure 3-class classification in validator/service.py.

Covers _diff_is_nontrivial, _rationale_is_coherent, and _classify_outcome
across the three outcome classes (king_change / meaningful_failure / plain_failure).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from validator.service import (
    KING_CHANGE_WEIGHT,
    MEANINGFUL_FAILURE_WEIGHT,
    PLAIN_FAILURE_WEIGHT,
    _classify_outcome,
    _diff_is_nontrivial,
    _rationale_is_coherent,
)

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

GOOD_DIFF = """diff --git a/recipe/training/config.yaml b/recipe/training/config.yaml
index 1111111..2222222 100644
--- a/recipe/training/config.yaml
+++ b/recipe/training/config.yaml
@@ -1,7 +1,7 @@
-lr_peak: 6e-3
-lr_min: 1e-5
-weight_decay: 0.10
-warmup_steps: 80
-cosine_decay: true
-gradient_clip: 1.0
-batch_size: 64
+lr_peak: 5e-3
+lr_min: 1e-5
+weight_decay: 0.05
+warmup_steps: 100
+cosine_decay: true
+gradient_clip: 1.0
+batch_size: 64
"""

GOOD_RATIONALE = """# Lion optimizer trial

We tested replacing AdamW with the Lion optimizer for the canonical Karpa-1
recipe. The hypothesis was that Lion's sign-based updates would converge
faster on this transformer architecture at the 254M parameter scale.

After training, we observed val_bpb=3.85, slightly above the king's 3.81.
Lion required more aggressive warmup than expected. The negative result is
informative for anyone exploring optimizer alternatives in the Karpa lineage.

Worth re-trying with a longer warmup schedule. Future agents can build on
this trajectory rather than re-discovering the same failure mode.
"""


def _write_bundle(tmp: Path, *, diff: str | None = None, rationale: str | None = None) -> Path:
    bundle = tmp / "bundle"
    bundle.mkdir()
    if diff is not None:
        (bundle / "patch.diff").write_text(diff)
    if rationale is not None:
        (bundle / "rationale.md").write_text(rationale)
    return bundle


# ----------------------------------------------------------------------------
# _diff_is_nontrivial
# ----------------------------------------------------------------------------

def test_diff_missing(tmp_path):
    assert _diff_is_nontrivial(tmp_path / "no.diff") is False


def test_diff_too_few_lines(tmp_path):
    """A single-line diff doesn't clear DIFF_MIN_CHANGED_LINES (1) — needs
    at least 2 changed lines so trivial typo/whitespace patches don't earn
    meaningful_failure credit. The cleanest legitimate hypothesis test is a
    one-scalar change which adds + removes one line each (2 total)."""
    diff = """diff --git a/recipe/config.yaml b/recipe/config.yaml
--- a/recipe/config.yaml
+++ b/recipe/config.yaml
+lr: 5e-3
"""
    p = tmp_path / "p.diff"
    p.write_text(diff)
    assert _diff_is_nontrivial(p) is False


def test_diff_single_scalar_two_lines_is_nontrivial(tmp_path):
    """The cleanest possible hypothesis test (change one scalar; old line
    out, new line in) MUST qualify as nontrivial. Round-2 round-trip:
    B's max_lr change was 4 lines (max_lr + min_lr both changed), and was
    incorrectly rejected by the old DIFF_MIN_CHANGED_LINES=5 rule despite
    delivering a 0.087 val_bpb improvement. The new floor (>1) admits any
    real scalar change."""
    diff = """diff --git a/configs/proxy_cpu_smoke.json b/configs/proxy_cpu_smoke.json
--- a/configs/proxy_cpu_smoke.json
+++ b/configs/proxy_cpu_smoke.json
@@ -12,3 +12,3 @@
   "total_steps": 20,
-  "max_lr": 0.003,
+  "max_lr": 0.0036,
   "log_every": 2
"""
    p = tmp_path / "p.diff"
    p.write_text(diff)
    assert _diff_is_nontrivial(p) is True


def test_diff_model_dir_counts_as_training_relevant(tmp_path):
    """A structural patch to model/karpa_base.py (e.g. QK-Norm) MUST count
    as touching training. The old filename whitelist omitted model/, which
    blocked attention-variant / init-scheme / structural-axis submissions
    from ever qualifying as meaningful_failure even when they beat the king
    on val_bpb. Round-2 round-trip: A's QK-Norm patch (val_bpb=1.485, beats
    king by 0.026) was incorrectly classified plain_failure for this reason."""
    diff = """diff --git a/model/karpa_base.py b/model/karpa_base.py
--- a/model/karpa_base.py
+++ b/model/karpa_base.py
@@ -38,1 +38,4 @@
     tie_embeddings: bool = True
+    use_qk_norm: bool = True
+    q_norm = RMSNorm(cfg.head_dim)
+    k_norm = RMSNorm(cfg.head_dim)
"""
    p = tmp_path / "p.diff"
    p.write_text(diff)
    assert _diff_is_nontrivial(p) is True


def test_diff_not_training_path(tmp_path):
    diff = """diff --git a/README.md b/README.md
--- a/README.md
+++ b/README.md
+line1
+line2
+line3
+line4
+line5
+line6
+line7
"""
    p = tmp_path / "p.diff"
    p.write_text(diff)
    assert _diff_is_nontrivial(p) is False


def test_diff_comments_only(tmp_path):
    diff = """diff --git a/recipe/training/config.yaml b/recipe/training/config.yaml
--- a/recipe/training/config.yaml
+++ b/recipe/training/config.yaml
+# comment 1
+# comment 2
+# comment 3
+# comment 4
+# comment 5
+# comment 6
+# comment 7
"""
    p = tmp_path / "p.diff"
    p.write_text(diff)
    assert _diff_is_nontrivial(p) is False


def test_diff_nontrivial(tmp_path):
    p = tmp_path / "p.diff"
    p.write_text(GOOD_DIFF)
    assert _diff_is_nontrivial(p) is True


def test_diff_nontrivial_configs_json(tmp_path):
    """A diff against configs/*.json (e.g. proxy_cpu_smoke.json) should count
    as touching training — config files are training-relevant."""
    diff = """diff --git a/configs/proxy_cpu_smoke.json b/configs/proxy_cpu_smoke.json
--- a/configs/proxy_cpu_smoke.json
+++ b/configs/proxy_cpu_smoke.json
@@ -9,10 +9,10 @@
   "seq_len": 128,
-  "total_steps": 20,
+  "total_steps": 25,
-  "max_lr": 0.003,
-  "min_lr": 0.0003,
+  "max_lr": 0.0035,
+  "min_lr": 0.00035,
"""
    p = tmp_path / "p.diff"
    p.write_text(diff)
    assert _diff_is_nontrivial(p) is True


# ----------------------------------------------------------------------------
# _rationale_is_coherent
# ----------------------------------------------------------------------------

def test_rationale_missing(tmp_path):
    assert _rationale_is_coherent(tmp_path / "no.md") is False


def test_rationale_too_short(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("Short note.")
    assert _rationale_is_coherent(p) is False


def test_rationale_one_paragraph_long(tmp_path):
    p = tmp_path / "r.md"
    p.write_text("a" * 300)
    assert _rationale_is_coherent(p) is False


def test_rationale_repetitive_template(tmp_path):
    p = tmp_path / "r.md"
    # 4+ identical sentences interleaved → looks like a template / repeated
    p.write_text(
        ("The same sentence repeated for padding. " * 6).strip() +
        "\n\n" +
        ("The same sentence repeated for padding. " * 6).strip()
    )
    assert _rationale_is_coherent(p) is False


def test_rationale_proper(tmp_path):
    p = tmp_path / "r.md"
    p.write_text(GOOD_RATIONALE)
    assert _rationale_is_coherent(p) is True


# ----------------------------------------------------------------------------
# _classify_outcome
# ----------------------------------------------------------------------------

def _good_bundle(tmp_path: Path) -> Path:
    return _write_bundle(tmp_path, diff=GOOD_DIFF, rationale=GOOD_RATIONALE)


def test_classify_king_change_decisive(tmp_path):
    bundle = _good_bundle(tmp_path)
    c, w = _classify_outcome(
        decisively=True,
        val_bpb=3.50,
        king_bpb=3.80,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "king_change"
    assert w == KING_CHANGE_WEIGHT


def test_classify_king_change_first_submission(tmp_path):
    """When there's no king (is_first is True), the caller passes
    decisively=True; classifier still treats it as king_change."""
    bundle = _good_bundle(tmp_path)
    c, w = _classify_outcome(
        decisively=True,
        val_bpb=3.80,
        king_bpb=None,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "king_change"
    assert w == KING_CHANGE_WEIGHT


def test_classify_meaningful_failure_inside_noise_band(tmp_path):
    """val_bpb landed ~0.005 above king — inside the 2σ noise band (0.013).
    Diff is good, rationale is good — meaningful_failure."""
    bundle = _good_bundle(tmp_path)
    c, w = _classify_outcome(
        decisively=False,
        val_bpb=3.805,
        king_bpb=3.80,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "meaningful_failure"
    assert w == MEANINGFUL_FAILURE_WEIGHT


def test_classify_meaningful_failure_within_2x_noise(tmp_path):
    """val_bpb landed 0.02 worse than king — between 1× and 2× the noise floor.
    Still meaningful_failure (the diff and rationale are good)."""
    bundle = _good_bundle(tmp_path)
    c, w = _classify_outcome(
        decisively=False,
        val_bpb=3.82,
        king_bpb=3.80,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "meaningful_failure"
    assert w == MEANINGFUL_FAILURE_WEIGHT


def test_classify_plain_failure_too_far_off(tmp_path):
    """val_bpb is 0.05 worse than king (~4× noise) — too far off the band."""
    bundle = _good_bundle(tmp_path)
    c, w = _classify_outcome(
        decisively=False,
        val_bpb=3.85,
        king_bpb=3.80,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "plain_failure"
    assert w == PLAIN_FAILURE_WEIGHT


def test_classify_plain_failure_trivial_diff(tmp_path):
    """val_bpb landed inside the band, but the diff is a README typo —
    plain_failure regardless of rationale quality."""
    bundle = _write_bundle(
        tmp_path,
        diff="diff --git a/README.md b/README.md\n+typo fix\n",
        rationale=GOOD_RATIONALE,
    )
    c, w = _classify_outcome(
        decisively=False,
        val_bpb=3.805,
        king_bpb=3.80,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "plain_failure"
    assert w == PLAIN_FAILURE_WEIGHT


def test_classify_plain_failure_missing_rationale(tmp_path):
    """Good diff, val_bpb inside band, but no rationale.md — plain_failure.
    A non-trivial training change without an explanation is uninformative."""
    bundle = _write_bundle(tmp_path, diff=GOOD_DIFF)
    c, w = _classify_outcome(
        decisively=False,
        val_bpb=3.805,
        king_bpb=3.80,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "plain_failure"
    assert w == PLAIN_FAILURE_WEIGHT


def test_classify_plain_failure_short_rationale(tmp_path):
    bundle = _write_bundle(
        tmp_path,
        diff=GOOD_DIFF,
        rationale="Tried LR=5e-3. Worse. Dunno why.",
    )
    c, w = _classify_outcome(
        decisively=False,
        val_bpb=3.805,
        king_bpb=3.80,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "plain_failure"
    assert w == PLAIN_FAILURE_WEIGHT


def test_classify_plain_failure_no_king(tmp_path):
    """Without a king to compare against and without decisively=True, we
    can't define 'meaningful failure' — fall through to plain_failure."""
    bundle = _good_bundle(tmp_path)
    c, w = _classify_outcome(
        decisively=False,
        val_bpb=3.80,
        king_bpb=None,
        noise_floor_margin=0.013,
        bundle_dir=bundle,
    )
    assert c == "plain_failure"
    assert w == PLAIN_FAILURE_WEIGHT


# ----------------------------------------------------------------------------
# Constants sanity (locks the 10% incentive ratio at module level)
# ----------------------------------------------------------------------------

def test_constants_10pct_ratio():
    """MEANINGFUL_FAILURE_WEIGHT must be exactly 10% of KING_CHANGE_WEIGHT
    so the protocol's "10% incentive" claim stays accurate."""
    assert MEANINGFUL_FAILURE_WEIGHT == 0.1 * KING_CHANGE_WEIGHT
    assert PLAIN_FAILURE_WEIGHT == 0.0
