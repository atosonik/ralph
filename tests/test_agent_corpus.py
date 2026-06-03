"""Tests for miner.agent_corpus."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401

from miner import agent_corpus, agent_memory


@pytest.fixture
def fake_karpa_root(tmp_path, monkeypatch):
    """Redirect every path agent_corpus uses to a tmp_path tree."""
    monkeypatch.setattr(agent_corpus, "CHAIN_DIR", tmp_path / "chain")
    monkeypatch.setattr(agent_corpus, "QUEUE_DIR", tmp_path / "queue")
    monkeypatch.setattr(agent_memory, "AGENTS_DIR", tmp_path / "agents")
    (tmp_path / "chain").mkdir()
    (tmp_path / "queue" / "meaningful_failure").mkdir(parents=True)
    yield tmp_path


def test_get_king_none_when_absent(fake_karpa_root):
    assert agent_corpus.get_king() is None


def test_get_king_reads_file(fake_karpa_root):
    (fake_karpa_root / "chain" / "king.json").write_text(json.dumps({
        "miner_hotkey": "5F23x",
        "bundle_hash": "abc",
        "val_bpb": 1.5109,
        "benchmark_accuracy": 0.18,
        "compute_cost_h100h": 0.0003,
        "crowned_at": 1234567890.0,
    }))
    k = agent_corpus.get_king()
    assert k is not None
    assert k.miner_hotkey == "5F23x"
    assert k.val_bpb == 1.5109


def test_get_king_lineage_walks_chain(fake_karpa_root):
    (fake_karpa_root / "chain" / "king.json").write_text(json.dumps({
        "miner_hotkey": "k3",
        "bundle_hash": "h3",
        "val_bpb": 1.0,
        "benchmark_accuracy": 0.3,
        "compute_cost_h100h": 0.0,
        "crowned_at": 3.0,
        "previous_king": {
            "miner_hotkey": "k2", "bundle_hash": "h2", "val_bpb": 1.1,
            "benchmark_accuracy": 0.28, "compute_cost_h100h": 0.0, "crowned_at": 2.0,
            "previous_king": {
                "miner_hotkey": "k1", "bundle_hash": "h1", "val_bpb": 1.2,
                "benchmark_accuracy": 0.25, "compute_cost_h100h": 0.0, "crowned_at": 1.0,
            },
        },
    }))
    lineage = agent_corpus.get_king_lineage(max_depth=10)
    assert [k.miner_hotkey for k in lineage] == ["k3", "k2", "k1"]


def test_get_king_lineage_max_depth(fake_karpa_root):
    inner = {"miner_hotkey": "k0", "bundle_hash": "h0", "val_bpb": 9.0,
             "benchmark_accuracy": 0.0, "compute_cost_h100h": 0.0, "crowned_at": 0.0}
    chain = inner
    for i in range(20):
        chain = {**inner, "miner_hotkey": f"k{i}", "previous_king": chain}
    (fake_karpa_root / "chain" / "king.json").write_text(json.dumps(chain))
    lineage = agent_corpus.get_king_lineage(max_depth=5)
    assert len(lineage) == 5


def test_read_events_empty(fake_karpa_root):
    assert agent_corpus.read_events() == []


def test_recent_king_changes(fake_karpa_root):
    events_path = fake_karpa_root / "chain" / "events.jsonl"
    events = [
        {"type": "submission_scored", "classification": "king_change", "miner_hotkey": "x", "val_bpb": 1.5},
        {"type": "submission_scored", "classification": "plain_failure", "miner_hotkey": "y", "val_bpb": 2.0},
        {"type": "submission_scored", "classification": "king_change", "miner_hotkey": "z", "val_bpb": 1.4},
        {"type": "weights_set", "uids": [1, 2]},
    ]
    events_path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    kc = agent_corpus.recent_king_changes(limit=10)
    assert len(kc) == 2
    assert [e["miner_hotkey"] for e in kc] == ["x", "z"]


def test_recent_meaningful_failures(fake_karpa_root):
    mf_root = fake_karpa_root / "queue" / "meaningful_failure"
    for i in range(3):
        d = mf_root / f"bundle_{i:02d}"
        d.mkdir()
        (d / "rationale.md").write_text(f"# Hypothesis {i}\n\nDetailed reasoning paragraph {i}.")
        (d / "submission.json").write_text(json.dumps({"val_bpb": 1.5 + i * 0.01}))
    mfs = agent_corpus.recent_meaningful_failures(limit=5)
    assert len(mfs) == 3
    assert all("Hypothesis" in m.rationale_text for m in mfs)


def test_recent_meaningful_failures_skips_oversize(fake_karpa_root):
    mf_root = fake_karpa_root / "queue" / "meaningful_failure"
    d = mf_root / "huge"
    d.mkdir()
    (d / "rationale.md").write_text("x" * 300_000)  # > 200KB cap
    mfs = agent_corpus.recent_meaningful_failures()
    assert mfs == []


def test_recent_meaningful_failures_skips_missing_rationale(fake_karpa_root):
    mf_root = fake_karpa_root / "queue" / "meaningful_failure"
    (mf_root / "no_rationale").mkdir()
    assert agent_corpus.recent_meaningful_failures() == []


def test_list_tried_axes_recent_filters_old(fake_karpa_root):
    now = time.time()
    agent_memory.append_memory("a", {
        "ts": now - 30 * 86400, "round": 1, "axis": "old_axis", "parameter": "x",
        "hypothesis_slug": "", "bundle_hash": None, "val_bpb": None,
        "classification": "plain_failure", "rationale_summary": "", "h100_cost_usd": 0.0,
    })
    agent_memory.append_memory("a", {
        "ts": now - 1, "round": 2, "axis": "fresh_axis", "parameter": "y",
        "hypothesis_slug": "", "bundle_hash": None, "val_bpb": None,
        "classification": "king_change", "rationale_summary": "", "h100_cost_usd": 0.0,
    })
    tried = agent_corpus.list_tried_axes_recent("a", days=14)
    assert len(tried) == 1
    assert tried[0]["axis"] == "fresh_axis"


def test_noise_floor_constant():
    assert agent_corpus.get_noise_floor() == 0.013


def test_format_for_prompt_minimal(fake_karpa_root):
    # No king, no events, no failures, no memory — should still produce valid output
    p = agent_corpus.format_for_prompt("a")
    assert "No king crowned yet" in p
    assert "Noise floor" in p


def test_format_for_prompt_full(fake_karpa_root):
    (fake_karpa_root / "chain" / "king.json").write_text(json.dumps({
        "miner_hotkey": "5F23xxxxxxxxxxx",
        "bundle_hash": "abc123def456",
        "val_bpb": 1.5,
        "benchmark_accuracy": 0.2,
        "compute_cost_h100h": 0.0003,
        "crowned_at": time.time(),
    }))
    mf = fake_karpa_root / "queue" / "meaningful_failure" / "mfbundle1"
    mf.mkdir()
    (mf / "rationale.md").write_text("# Lion test\n\nLion needs more warmup.")
    agent_memory.append_memory("a", {
        "ts": time.time(), "round": 5, "axis": "optimizer", "parameter": "lion",
        "hypothesis_slug": "lion_v1", "bundle_hash": "h", "val_bpb": 1.6,
        "classification": "plain_failure", "rationale_summary": "Lion diverged",
        "h100_cost_usd": 2.4,
    })
    p = agent_corpus.format_for_prompt("a")
    assert "val_bpb" in p
    assert "1.5000" in p
    assert "Noise floor" in p
    assert "Lion needs more warmup" in p or "Lion test" in p  # rationale preview shown
    assert "optimizer" in p
    assert "lion" in p
