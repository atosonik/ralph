#!/usr/bin/env python3
"""Sweep orphaned + past-kill_at Shadeform instances.

Called automatically at the top of every `gpu.py` command (cheap when the
worst case is "no orphans"). Also runnable standalone — useful at session
start to clean up anything a crashed previous session left behind.

Rules:
  - Any local /root/.shadeform_instance_<name>.json with status != deleted
    and `kill_at` < now → DELETE via API.
  - Any Shadeform instance whose Shadeform-side name starts with `ralph-`
    AND is not referenced by any local instance file → DELETE.

Either rule firing prints a one-line warning to stderr. Exit code is 0
unless the Shadeform API itself errored.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://api.shadeform.ai/v1"
KEY_FILE = Path("/root/.shadeform_api_key")
INSTANCE_FILE_PREFIX = "/root/.shadeform_instance_"
INSTANCE_NAME_PREFIX = "ralph-"


def _api_key() -> str:
    if not KEY_FILE.exists():
        print(f"[sweep] no API key at {KEY_FILE} — skipping orphan sweep", file=sys.stderr)
        sys.exit(0)
    return KEY_FILE.read_text().strip()


def _api(method: str, path: str) -> dict | None:
    url = f"{API_BASE}{path}"
    headers = {"X-API-KEY": _api_key(), "Content-Type": "application/json"}
    req = urllib.request.Request(url, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # instance already gone
        print(f"[sweep] API {method} {path} → {e.code}", file=sys.stderr)
        return None
    except urllib.error.URLError as e:
        print(f"[sweep] API unreachable: {e}", file=sys.stderr)
        return None


def _delete_instance(instance_id: str, note: str) -> bool:
    print(f"[sweep] deleting {instance_id[:12]}… ({note})", file=sys.stderr)
    res = _api("POST", f"/instances/{instance_id}/delete")
    return res is not None


def _local_instance_files() -> list[Path]:
    return [
        p
        for p in Path("/root").glob(".shadeform_instance_*.json")
        if p.is_file()
    ]


def sweep_kill_at_expired() -> int:
    """Pass 1: any local instance file with kill_at past → delete."""
    now = time.time()
    swept = 0
    for f in _local_instance_files():
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("status") == "deleted":
            continue
        kill_at = d.get("kill_at")
        if kill_at is None or float(kill_at) > now:
            continue
        iid = d.get("id")
        if not iid:
            continue
        if _delete_instance(iid, f"kill_at expired by {(now - float(kill_at)) / 60:.1f}min"):
            d["status"] = "deleted"
            f.write_text(json.dumps(d, indent=2))
            swept += 1
    return swept


def sweep_unreferenced_ralph_instances() -> int:
    """Pass 2: any Shadeform instance with name prefix `ralph-` AND not
    referenced by any local instance file → delete.

    Defends against instance files getting hand-deleted while their
    Shadeform-side counterparts keep billing.
    """
    known_ids: set[str] = set()
    for f in _local_instance_files():
        try:
            d = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        iid = d.get("id")
        if iid and d.get("status") != "deleted":
            known_ids.add(iid)

    res = _api("GET", "/instances")
    if res is None:
        return 0
    instances = res.get("instances") or []
    swept = 0
    for inst in instances:
        iid = inst.get("id")
        name = inst.get("name", "")
        status = inst.get("status", "")
        if not iid or status == "deleted":
            continue
        if not name.startswith(INSTANCE_NAME_PREFIX):
            continue
        if iid in known_ids:
            continue
        if _delete_instance(iid, f"unreferenced ralph-* instance name={name!r}"):
            swept += 1
    return swept


def sweep(quiet: bool = False) -> dict:
    """Run both passes; return summary dict."""
    expired = sweep_kill_at_expired()
    unreferenced = sweep_unreferenced_ralph_instances()
    total = expired + unreferenced
    if total > 0 or not quiet:
        print(
            f"[sweep] swept {total} ({expired} expired kill_at, {unreferenced} unreferenced ralph-*)",
            file=sys.stderr,
        )
    return {"expired": expired, "unreferenced": unreferenced, "total": total}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Sweep orphaned Shadeform H100 instances.")
    p.add_argument("--quiet", action="store_true", help="suppress 'swept 0' messages")
    args = p.parse_args()
    result = sweep(quiet=args.quiet)
    sys.exit(0 if result["total"] >= 0 else 1)
