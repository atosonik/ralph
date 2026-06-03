"""Tests for miner.agent_memory."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401
from miner import agent_memory


@pytest.fixture
def isolated_agents_dir(tmp_path, monkeypatch):
    """Redirect AGENTS_DIR to a tmp_path so tests don't touch real agents/."""
    monkeypatch.setattr(agent_memory, "AGENTS_DIR", tmp_path / "agents")
    yield tmp_path / "agents"


def test_agent_root_creates_subdirs(isolated_agents_dir):
    p = agent_memory.agent_root("a")
    assert p.exists()
    assert (p / "prompts").exists()
    assert (p / "runs").exists()


def test_agent_id_validation(isolated_agents_dir):
    with pytest.raises(ValueError):
        agent_memory.agent_root("../etc/passwd")
    with pytest.raises(ValueError):
        agent_memory.agent_root("agent with spaces")
    with pytest.raises(ValueError):
        agent_memory.agent_root("")


def test_state_roundtrip(isolated_agents_dir):
    s = agent_memory.read_state("a")
    assert s["phase"] == "IDLE"
    assert s["round"] == 0
    agent_memory.write_state("a", {**s, "phase": "DECIDING", "round": 7})
    s2 = agent_memory.read_state("a")
    assert s2["phase"] == "DECIDING"
    assert s2["round"] == 7


def test_state_atomic_write(isolated_agents_dir):
    """The temp-file dance must leave no .tmp behind on success."""
    agent_memory.write_state("a", {"phase": "TRAINING"})
    state_dir = isolated_agents_dir / "a"
    assert (state_dir / "state.json").exists()
    assert not (state_dir / "state.json.tmp").exists()


def test_state_corrupt_fallback(isolated_agents_dir):
    p = isolated_agents_dir / "a"
    p.mkdir(parents=True)
    (p / "prompts").mkdir()
    (p / "runs").mkdir()
    (p / "state.json").write_text("{ this is not json }")
    s = agent_memory.read_state("a")
    # Should fall back to defaults, NOT overwrite the corrupt file
    assert s["phase"] == "IDLE"
    assert (p / "state.json").read_text() == "{ this is not json }"


def test_memory_append_and_read(isolated_agents_dir):
    e = agent_memory.MemoryEntry(
        ts=time.time(),
        round=1,
        axis="optimizer",
        parameter="lion",
        hypothesis_slug="lion_lr0.3king",
        bundle_hash="abc" * 16,
        val_bpb=1.512,
        classification="meaningful_failure",
        rationale_summary="Lion needs longer warmup",
        h100_cost_usd=2.42,
        king_val_bpb_at_time=1.5,
    )
    agent_memory.append_memory("a", e)
    mem = agent_memory.read_memory("a")
    assert len(mem) == 1
    assert mem[0]["axis"] == "optimizer"
    assert mem[0]["val_bpb"] == 1.512


def test_memory_last_n(isolated_agents_dir):
    for i in range(5):
        agent_memory.append_memory("a", {
            "ts": time.time() + i, "round": i, "axis": "lr_peak",
            "parameter": f"v{i}", "hypothesis_slug": f"r{i}",
            "bundle_hash": None, "val_bpb": 1.5 + i * 0.01,
            "classification": "plain_failure", "rationale_summary": "",
            "h100_cost_usd": 2.0,
        })
    last3 = agent_memory.read_memory("a", last_n=3)
    assert len(last3) == 3
    assert [m["round"] for m in last3] == [2, 3, 4]


def test_memory_dict_or_entry(isolated_agents_dir):
    agent_memory.append_memory("a", {"axis": "lr_peak", "round": 1, "ts": 0,
        "parameter": "5e-4", "hypothesis_slug": "x", "bundle_hash": None,
        "val_bpb": None, "classification": "aborted", "rationale_summary": "",
        "h100_cost_usd": 0.0})
    mem = agent_memory.read_memory("a")
    assert mem[0]["axis"] == "lr_peak"


def test_memory_corrupt_line_skipped(isolated_agents_dir):
    p = agent_memory.agent_root("a") / "memory.jsonl"
    p.write_text('{"axis":"lr","round":1,"ts":0}\nthis is not json\n{"axis":"wd","round":2,"ts":0}\n')
    mem = agent_memory.read_memory("a")
    assert len(mem) == 2
    assert mem[0]["axis"] == "lr"
    assert mem[1]["axis"] == "wd"


def test_lock_acquire_release(isolated_agents_dir):
    assert agent_memory.acquire_lock("a")
    assert agent_memory.is_locked("a")
    # Second acquire should fail (not stale)
    assert not agent_memory.acquire_lock("a")
    agent_memory.release_lock("a")
    assert not agent_memory.is_locked("a")


def test_stale_lock_can_be_stolen(isolated_agents_dir):
    agent_memory.acquire_lock("a")
    # Backdate the mtime so it looks stale
    lock_path = agent_memory.agent_root("a") / "lock"
    import os
    backdate = time.time() - 24 * 3600
    os.utime(lock_path, (backdate, backdate))
    assert agent_memory.acquire_lock("a")  # steals the stale lock


def test_lock_release_idempotent(isolated_agents_dir):
    agent_memory.release_lock("a")  # no error even if no lock
    agent_memory.release_lock("a")


def test_round_artifact_dir(isolated_agents_dir):
    d = agent_memory.round_artifact_dir("a", 7)
    assert d.exists()
    assert d.name == "round_0007"


def test_save_prompt(isolated_agents_dir):
    p = agent_memory.save_prompt("a", 3, "system prompt + context for round 3")
    assert p.exists()
    assert p.name == "round_0003.txt"
    assert "round 3" in p.read_text()
