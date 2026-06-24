"""Tests for HF-poller version-aware reprocessing + time-based queue ordering.

Covers two fairness behaviours:
  1. hf_state.json stamps each processed bundle with VALIDATOR_VERSION; entries
     judged by an older version are reprocessed on a version bump.
  2. poll_queue orders pending bundles by submission (PR creation) time, i.e.
     first-come-first-validate — NOT by bundle-hash lexical order.
"""

import json
import os
import time
from pathlib import Path

from validator import hf_poller as hp
from validator.service import poll_queue
from validator.version import VALIDATOR_VERSION


def test_legacy_list_state_is_migrated_and_reprocessed(tmp_path):
    (tmp_path / "hf_state.json").write_text(json.dumps({"processed": ["aaa", "bbb"]}))
    st = hp._load_state(tmp_path)
    assert st["processed"] == {"aaa": "legacy", "bbb": "legacy"}
    assert st["validator_version"] == "legacy"
    done = {b for b, v in st["processed"].items() if v == VALIDATOR_VERSION}
    assert done == set()  # legacy entries do not count as done → reprocessed


def test_only_current_version_counts_as_done(tmp_path):
    (tmp_path / "hf_state.json").write_text(json.dumps({
        "validator_version": VALIDATOR_VERSION,
        "processed": {"new1": VALIDATOR_VERSION, "old1": "v0"},
    }))
    st = hp._load_state(tmp_path)
    done = {b for b, v in st["processed"].items() if v == VALIDATOR_VERSION}
    assert done == {"new1"}  # old1 (older version) will be reprocessed


def test_fresh_state_defaults(tmp_path):
    st = hp._load_state(tmp_path)
    assert st == {"validator_version": VALIDATOR_VERSION, "processed": {}}


def _mk_bundle(pending: Path, name: str, created=None, pr=None, mtime=None):
    d = pending / name
    d.mkdir(parents=True)
    (d / "submission.json").write_text("{}")
    if created is not None:
        (d / ".hf_pr.json").write_text(json.dumps({"created_at": created, "pr_num": pr}))
    if mtime is not None:
        os.utime(d, (mtime, mtime))
    return d


def test_poll_queue_orders_by_submission_time_not_hash(tmp_path):
    pending = tmp_path / "pending"
    pending.mkdir()
    # Names deliberately NOT in chronological order so a hash sort would differ.
    _mk_bundle(pending, "zzz_first", created="2026-06-20T10:00:00+00:00", pr=5)
    _mk_bundle(pending, "aaa_third", created="2026-06-22T10:00:00+00:00", pr=9)
    _mk_bundle(pending, "mmm_second", created="2026-06-21T10:00:00+00:00", pr=7)
    # Legacy/local bundle with no PR metadata → falls back to (older) mtime.
    _mk_bundle(pending, "legacy_local", mtime=time.mktime((2026, 6, 19, 0, 0, 0, 0, 0, 0)))

    order = [p.name for p in poll_queue(tmp_path)]
    assert order == ["legacy_local", "zzz_first", "mmm_second", "aaa_third"]
