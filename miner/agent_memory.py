"""Per-agent state + memory file helpers.

Each agent owns a directory under `agents/<agent_id>/` that holds:
  - memory.jsonl      append-only attempt log (one JSON object per line)
  - state.json        current state-machine state + cooldowns + counters
  - lock              pidfile preventing two concurrent rounds for this agent
  - prompts/          one .txt per round (the exact prompt assembled and sent)
  - runs/round_<N>/   per-round artifacts staged before submission

This module is pure file-system bookkeeping — no chain access, no network.
The agent_corpus module is the read-only counterpart for chain + public corpus
data; together they're the data layer the per-round orchestrator stitches on
top of when composing an LLM hypothesis.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


KARPA_ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = KARPA_ROOT / "agents"


def agent_root(agent_id: str) -> Path:
    """Return the per-agent directory; create on first access."""
    _validate_agent_id(agent_id)
    p = AGENTS_DIR / agent_id
    p.mkdir(parents=True, exist_ok=True)
    (p / "prompts").mkdir(exist_ok=True)
    (p / "runs").mkdir(exist_ok=True)
    return p


def _validate_agent_id(agent_id: str) -> None:
    """Refuse weird agent_ids — path traversal or shell metacharacters."""
    if not agent_id or not all(c.isalnum() or c in "-_" for c in agent_id):
        raise ValueError(f"invalid agent_id {agent_id!r} (alphanumeric, -, _ only)")


# ----------------------------------------------------------------------------
# state.json — the per-agent state machine + counters
# ----------------------------------------------------------------------------

_DEFAULT_STATE = {
    "phase": "IDLE",
    "round": 0,
    "last_run_at": None,
    "last_run_outcome": None,
    "consecutive_failures": 0,
    "cooldown_until": None,
    # rolling daily counters reset by the orchestrator at session start
    "runs_today": 0,
    "spend_today_usd": 0.0,
    # the active GPU rental, if any — kept here so a crashed orchestrator
    # can recover the instance reference for teardown
    "current_instance_name": None,
    "current_instance_kill_at": None,
}


def read_state(agent_id: str) -> dict:
    p = agent_root(agent_id) / "state.json"
    if not p.exists():
        return dict(_DEFAULT_STATE)
    try:
        cur = json.loads(p.read_text())
    except json.JSONDecodeError:
        # Corrupt state.json → return defaults; do not overwrite the
        # corrupt file (operator should inspect before clearing).
        return dict(_DEFAULT_STATE)
    # Fill in missing defaults so older state files keep working.
    out = dict(_DEFAULT_STATE)
    out.update(cur)
    return out


def write_state(agent_id: str, state: dict) -> None:
    """Atomic-write the state.json — write to temp + rename so a crash
    mid-write doesn't leave a partial JSON document on disk."""
    p = agent_root(agent_id) / "state.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(p)


def update_state(agent_id: str, **patch) -> dict:
    """Read-modify-write helper. Returns the updated state."""
    cur = read_state(agent_id)
    cur.update(patch)
    write_state(agent_id, cur)
    return cur


# ----------------------------------------------------------------------------
# memory.jsonl — append-only attempt log
# ----------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    ts: float
    round: int
    axis: str  # the search axis the LLM picked (optimizer / lr_peak / ...)
    parameter: str  # short slug for the specific parameter or value
    hypothesis_slug: str
    bundle_hash: Optional[str]
    val_bpb: Optional[float]
    classification: str  # king_change / meaningful_failure / plain_failure / aborted
    rationale_summary: str
    h100_cost_usd: float
    failure_reason: Optional[str] = None
    king_val_bpb_at_time: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "round": self.round,
            "axis": self.axis,
            "parameter": self.parameter,
            "hypothesis_slug": self.hypothesis_slug,
            "bundle_hash": self.bundle_hash,
            "val_bpb": self.val_bpb,
            "classification": self.classification,
            "rationale_summary": self.rationale_summary,
            "h100_cost_usd": self.h100_cost_usd,
            "failure_reason": self.failure_reason,
            "king_val_bpb_at_time": self.king_val_bpb_at_time,
        }


def append_memory(agent_id: str, entry: MemoryEntry | dict) -> None:
    p = agent_root(agent_id) / "memory.jsonl"
    if isinstance(entry, MemoryEntry):
        entry = entry.to_dict()
    # Locked append so two orchestrator processes don't interleave bytes.
    try:
        import fcntl
        with p.open("a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
                f.flush()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:  # pragma: no cover — Windows
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")


def read_memory(agent_id: str, last_n: int | None = None) -> list[dict]:
    p = agent_root(agent_id) / "memory.jsonl"
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if last_n is not None and last_n > 0:
        out = out[-last_n:]
    return out


# ----------------------------------------------------------------------------
# Lockfile — prevents two concurrent rounds for the same agent
# ----------------------------------------------------------------------------

def acquire_lock(agent_id: str, stale_after_s: int = 6 * 3600) -> bool:
    """Atomic create-if-not-exists pidfile. Returns True if acquired.

    Stale-lock guard: if the lock is older than `stale_after_s` (default 6 hr,
    longer than the absolute MAX_H100_WALL_CLOCK_PER_RUN safety cap), treat
    it as crashed and steal it.
    """
    p = agent_root(agent_id) / "lock"
    now = time.time()
    if p.exists():
        try:
            age = now - p.stat().st_mtime
        except OSError:
            age = 0
        if age < stale_after_s:
            return False
        # Stale — overwrite
    p.write_text(f"{os.getpid()}\n{now}\n")
    return True


def release_lock(agent_id: str) -> None:
    p = agent_root(agent_id) / "lock"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


def is_locked(agent_id: str) -> bool:
    return (agent_root(agent_id) / "lock").exists()


# ----------------------------------------------------------------------------
# Per-round prompt + artifact archive
# ----------------------------------------------------------------------------

def save_prompt(agent_id: str, round_n: int, prompt_text: str) -> Path:
    """Persist the exact assembled prompt for post-hoc audit."""
    p = agent_root(agent_id) / "prompts" / f"round_{round_n:04d}.txt"
    p.write_text(prompt_text, encoding="utf-8")
    return p


def round_artifact_dir(agent_id: str, round_n: int) -> Path:
    p = agent_root(agent_id) / "runs" / f"round_{round_n:04d}"
    p.mkdir(parents=True, exist_ok=True)
    return p
