"""Read-only helpers for an agent to consume the public corpus.

What the agent's hypothesis-generation prompt needs to see, distilled into a
few function calls:
  - Current king (recipe tag, val_bpb, hypothesis text, hotkey, when crowned).
  - The lineage of prior kings (val_bpb trajectory, diffs).
  - Recent meaningful_failure rationales (the public negative-result corpus).
  - Recent submission_scored events (rejected, accepted, classified).
  - The validator's noise-floor margin (so the agent knows the bar).
  - The axes the THIS agent has already tried recently (avoid-list).

This module is read-only. It does NOT touch the chain RPC — only the
on-disk mirror under chain/ (events.jsonl, king.json) and queue/. The
chain-backed BittensorChain class is heavier; the agent doesn't need it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from .agent_memory import RALPH_ROOT, read_memory

CHAIN_DIR = RALPH_ROOT / "chain"
QUEUE_DIR = RALPH_ROOT / "queue"

# Default noise-floor margin used by the validator. Matches the constant in
# validator/service.py and the canonical noise-floor study in /docs.
DEFAULT_NOISE_FLOOR = 0.013


# ----------------------------------------------------------------------------
# King + lineage
# ----------------------------------------------------------------------------

@dataclass
class KingSnapshot:
    miner_hotkey: str
    bundle_hash: str
    val_bpb: float
    benchmark_accuracy: float
    compute_cost_h100h: float
    crowned_at: float
    proof_dir: Optional[str]
    previous_king: Optional[dict]

    @classmethod
    def from_dict(cls, d: dict) -> "KingSnapshot":
        return cls(
            miner_hotkey=d["miner_hotkey"],
            bundle_hash=d.get("bundle_hash", ""),
            val_bpb=float(d.get("val_bpb", 0.0)),
            benchmark_accuracy=float(d.get("benchmark_accuracy", 0.0)),
            compute_cost_h100h=float(d.get("compute_cost_h100h", 0.0)),
            crowned_at=float(d.get("crowned_at", 0.0)),
            proof_dir=d.get("proof_dir"),
            previous_king=d.get("previous_king"),
        )


def get_king() -> Optional[KingSnapshot]:
    p = CHAIN_DIR / "king.json"
    if not p.exists():
        return None
    try:
        return KingSnapshot.from_dict(json.loads(p.read_text()))
    except (json.JSONDecodeError, KeyError):
        return None


def get_king_lineage(max_depth: int = 10) -> list[KingSnapshot]:
    """Walk king.previous_king recursively, newest-first."""
    out: list[KingSnapshot] = []
    cur = get_king()
    while cur is not None and len(out) < max_depth:
        out.append(cur)
        prev = cur.previous_king
        if not prev:
            break
        try:
            cur = KingSnapshot.from_dict(prev)
        except KeyError:
            break
    return out


# ----------------------------------------------------------------------------
# Chain events — submission_scored, meaningful_failure_archived, weights_set
# ----------------------------------------------------------------------------

def read_events(limit: int = 500) -> list[dict]:
    """Return the last `limit` events, oldest-first."""
    p = CHAIN_DIR / "events.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    out: list[dict] = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def recent_submission_events(limit: int = 50) -> list[dict]:
    """Filter chain events to just the submission_scored entries."""
    return [e for e in read_events(limit * 4) if e.get("type") == "submission_scored"][-limit:]


def recent_king_changes(limit: int = 10) -> list[dict]:
    """submission_scored events with classification=king_change."""
    return [
        e for e in read_events(limit * 20)
        if e.get("type") == "submission_scored" and e.get("classification") == "king_change"
    ][-limit:]


# ----------------------------------------------------------------------------
# Public negative-result corpus (meaningful_failure rationales)
# ----------------------------------------------------------------------------

@dataclass
class CorpusNegative:
    bundle_hash: str
    val_bpb: Optional[float]
    rationale_text: str
    archived_at: Optional[float]


def recent_meaningful_failures(limit: int = 20) -> list[CorpusNegative]:
    """Read rationales from queue/meaningful_failure/<bundle>/rationale.md.

    Bundles whose rationale is missing or > 200 KB are skipped (size cap
    matches the validator's defense-in-depth).
    """
    mf_dir = QUEUE_DIR / "meaningful_failure"
    if not mf_dir.exists():
        return []
    out: list[CorpusNegative] = []
    # Sort by directory mtime (newest first) so we get the most recent.
    sub = sorted(
        (d for d in mf_dir.iterdir() if d.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for d in sub[: limit * 2]:
        rationale_path = d / "rationale.md"
        if not rationale_path.exists():
            continue
        try:
            size = rationale_path.stat().st_size
        except OSError:
            continue
        if size <= 0 or size > 200_000:
            continue
        try:
            text = rationale_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        # val_bpb from sibling submission.json if available
        val_bpb: Optional[float] = None
        sub_json = d / "submission.json"
        if sub_json.exists():
            try:
                val_bpb = float(json.loads(sub_json.read_text()).get("val_bpb"))
            except (json.JSONDecodeError, ValueError, TypeError):
                val_bpb = None
        # archived_at: dir mtime
        try:
            archived_at: Optional[float] = d.stat().st_mtime
        except OSError:
            archived_at = None
        out.append(CorpusNegative(
            bundle_hash=d.name,
            val_bpb=val_bpb,
            rationale_text=text,
            archived_at=archived_at,
        ))
        if len(out) >= limit:
            break
    return out


# ----------------------------------------------------------------------------
# Per-agent avoid-list — axes already tried recently
# ----------------------------------------------------------------------------

def list_tried_axes_recent(agent_id: str, days: int = 14) -> list[dict]:
    """Returns a list of `{axis, parameter, classification, ts}` tuples from
    this agent's own memory.jsonl within the last `days`.

    The orchestrator uses this to instruct the LLM "don't pick from this list
    unless every fresh option is exhausted".
    """
    import time as _time
    cutoff = _time.time() - days * 86400
    out: list[dict] = []
    for entry in read_memory(agent_id):
        ts = entry.get("ts", 0)
        if ts < cutoff:
            continue
        out.append({
            "axis": entry.get("axis", ""),
            "parameter": entry.get("parameter", ""),
            "classification": entry.get("classification", ""),
            "ts": ts,
        })
    return out


# ----------------------------------------------------------------------------
# Noise floor
# ----------------------------------------------------------------------------

def get_noise_floor() -> float:
    """Currently a constant; later we can read it from a validator config
    file once that's persisted somewhere stable."""
    return DEFAULT_NOISE_FLOOR


# ----------------------------------------------------------------------------
# Prompt context formatter
# ----------------------------------------------------------------------------

def format_for_prompt(
    agent_id: str,
    n_lineage: int = 5,
    n_negatives: int = 10,
    n_private: int = 8,
    avoid_days: int = 14,
) -> str:
    """Assemble the compact context block that goes into the LLM prompt.

    Markdown formatted so the LLM can read it naturally. Does NOT include
    the agent's system prompt or the JSON schema — those are added by the
    orchestrator. This returns just the dynamic per-round context.
    """
    parts: list[str] = []

    king = get_king()
    if king is None:
        parts.append("## Current king\nNo king crowned yet — this would be the first submission.\n")
    else:
        parts.append(
            f"## Current king\n"
            f"- val_bpb: **{king.val_bpb:.4f}**\n"
            f"- benchmark_accuracy: {king.benchmark_accuracy:.3f}\n"
            f"- bundle_hash: {king.bundle_hash[:16]}…\n"
            f"- miner_hotkey: {king.miner_hotkey[:16]}…\n"
            f"- compute_cost_h100h: {king.compute_cost_h100h:.4f}\n"
        )

    lineage = get_king_lineage(max_depth=n_lineage)
    if len(lineage) > 1:
        parts.append("## Recent king lineage (newest first)")
        for k in lineage:
            parts.append(f"- val_bpb {k.val_bpb:.4f} · {k.miner_hotkey[:12]}… · bundle {k.bundle_hash[:12]}…")
        parts.append("")

    parts.append(
        f"## Noise floor margin\n{get_noise_floor():.4f} val_bpb — "
        "your patch must beat the king by more than this to crown.\n"
    )

    negs = recent_meaningful_failures(limit=n_negatives)
    if negs:
        parts.append("## Recent verified-negative rationales (public corpus)")
        for n in negs:
            bpb = f"val_bpb={n.val_bpb:.4f}" if n.val_bpb is not None else "val_bpb=?"
            # Trim rationale to first paragraph (or 500 chars) to keep prompt tight.
            preview = n.rationale_text.split("\n\n", 1)[0].strip()[:500]
            parts.append(f"- bundle {n.bundle_hash[:12]}… · {bpb}\n  > {preview}")
        parts.append("")

    avoid = list_tried_axes_recent(agent_id, days=avoid_days)
    if avoid:
        parts.append(f"## Axes YOU have tried in the last {avoid_days} days (avoid unless out of fresh angles)")
        for a in avoid:
            parts.append(f"- axis={a['axis']} · parameter={a['parameter']} · outcome={a['classification']}")
        parts.append("")

    private = read_memory(agent_id, last_n=n_private)
    if private:
        parts.append(f"## Your last {len(private)} attempts (private to you)")
        for e in private:
            bpb_s = f"val_bpb={e.get('val_bpb','?')}" if e.get("val_bpb") is not None else "val_bpb=?"
            parts.append(
                f"- round {e.get('round','?')} · axis={e.get('axis','?')} "
                f"· {e.get('classification','?')} · {bpb_s} · {e.get('hypothesis_slug','')}"
            )
        parts.append("")

    return "\n".join(parts).strip() + "\n"
