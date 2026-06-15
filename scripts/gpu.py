#!/usr/bin/env python3
"""
GPU lifecycle manager — Shadeform API.

Automates the rent → setup → work → backup → delete cycle for H100 instances.

Commands:
    python scripts/gpu.py rent          # Find cheapest H100, create instance, wait for ready
    python scripts/gpu.py status        # Show current instance status
    python scripts/gpu.py ssh [cmd]     # SSH into the instance (or run a command)
    python scripts/gpu.py backup        # SCP results from instance to local backup dir
    python scripts/gpu.py delete        # Delete the instance (stops billing)
    python scripts/gpu.py list-types    # List available H100 types + prices

The API key is read from /root/.shadeform_api_key (never stored in code or git).
Instance metadata is cached at /root/.shadeform_instance.json.

SSH uses the id_bitzic key at /root/.ssh/id_bitzic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional


def _sweep_orphans_safe() -> None:
    """Best-effort orphan sweep — never raises. Called at the top of every
    gpu.py command so a crashed previous session doesn't leak GPU billing.
    Costs an API call or two; cheap when there are no orphans."""
    try:
        from scripts.orphan_gpu_sweep import sweep
    except Exception:
        # Direct invocation (PYTHONPATH may not include scripts/)
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "orphan_gpu_sweep",
                str(__import__("pathlib").Path(__file__).resolve().parent / "orphan_gpu_sweep.py"),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            sweep = mod.sweep
        except Exception:
            return
    try:
        sweep(quiet=True)
    except Exception as e:
        print(f"[gpu] orphan sweep failed (non-fatal): {e}", flush=True)


API_BASE = "https://api.shadeform.ai/v1"
KEY_FILE = Path("/root/.shadeform_api_key")
SSH_KEY = Path("/root/.ssh/id_bitzic")
SSH_KEY_ID = os.environ.get("SHADEFORM_SSH_KEY_ID", "")
BACKUP_DIR = Path(__file__).resolve().parent.parent.parent / "backup_h100"


def _instance_file(name: str = "default") -> Path:
    """One state file per logical instance (e.g. ralph1, ralph2)."""
    return Path(f"/root/.shadeform_instance_{name}.json")


# Back-compat: legacy single-instance state file.
INSTANCE_FILE = _instance_file("default")


def _api_key() -> str:
    if not KEY_FILE.exists():
        print(f"ERROR: Put your Shadeform API key in {KEY_FILE}")
        sys.exit(1)
    return KEY_FILE.read_text().strip()


def _api(method: str, path: str, body: dict | None = None) -> dict:
    url = f"{API_BASE}{path}"
    headers = {"X-API-KEY": _api_key(), "Content-Type": "application/json"}
    data = json.dumps(body).encode() if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body_text = e.read().decode() if e.fp else ""
        print(f"API error {e.code}: {body_text}")
        sys.exit(1)


def _save_instance(data: dict, name: str = "default") -> None:
    _instance_file(name).write_text(json.dumps(data, indent=2))


def _load_instance(name: str = "default") -> Optional[dict]:
    if not _instance_file(name).exists():
        return None
    return json.loads(_instance_file(name).read_text())


# ---- Commands ----

def cmd_list_types(args):
    """List available H100 instances sorted by price."""
    resp = _api("GET", "/instances/types?gpu_type=H100&available=true&sort=price&num_gpus=1")
    types = resp.get("instance_types", [])
    if not types:
        print("No H100 instances available right now.")
        return
    print(f"{'Cloud':20s} {'Type':25s} {'$/hr':>7s} {'vRAM':>6s} {'RAM':>6s} {'vCPUs':>5s} {'Region'}")
    print("-" * 100)
    for t in types[:20]:
        cfg = t.get("configuration", {})
        regions = [a["region"] for a in t.get("availability", []) if a.get("available")]
        price = t.get("hourly_price", 0) / 100
        print(f"{t['cloud']:20s} {t.get('shade_instance_type','?'):25s} ${price:6.2f} "
              f"{cfg.get('vram_per_gpu_in_gb', '?'):>5}G {cfg.get('memory_in_gb', '?'):>5}G "
              f"{cfg.get('vcpus', '?'):>5} {', '.join(regions[:2])}")


def cmd_rent(args):
    """Rent the cheapest available H100."""
    _sweep_orphans_safe()
    name = getattr(args, "name", "default")
    existing = _load_instance(name)
    if existing and existing.get("status") not in ("deleted", "error", None):
        print(f"Instance '{name}' already exists: {existing.get('id', '?')} ({existing.get('status')})")
        print(f"Delete it first with: python scripts/gpu.py delete --name {name}")
        return

    # Default: Hyperstack H100. Override with --cloud / --region.
    preferred_cloud = args.cloud or "hyperstack"
    preferred_region = args.region or None

    print(f"Finding H100 on {preferred_cloud}...")
    resp = _api("GET", "/instances/types?gpu_type=H100&available=true&sort=price&num_gpus=1")
    types = resp.get("instance_types", [])

    # Filter for preferred cloud first, fall back to cheapest if unavailable.
    matches = [t for t in types if t["cloud"] == preferred_cloud]
    if not matches:
        print(f"  {preferred_cloud} not available, falling back to cheapest...")
        matches = types
    if not matches:
        print("No H100 instances available. Try again later.")
        return

    best = matches[0]
    regions = [a for a in best.get("availability", []) if a.get("available")]
    if preferred_region:
        regions = [a for a in regions if a["region"] == preferred_region] or regions
    if not regions:
        print("No available regions. Try again later.")
        return

    cloud = best["cloud"]
    region = regions[0]["region"]
    shade_type = best["shade_instance_type"]
    price = best.get("hourly_price", 0) / 100
    print(f"Selected: {cloud} / {shade_type} in {region} @ ${price:.2f}/hr")

    body = {
        "cloud": cloud,
        "region": region,
        "shade_instance_type": shade_type,
        "shade_cloud": True,
        "name": f"ralph-{name}-{int(time.time()) % 100000}",
    }
    if SSH_KEY_ID:
        body["ssh_key_id"] = SSH_KEY_ID

    print("Creating instance...")
    resp = _api("POST", "/instances/create", body)
    instance_id = resp.get("id")
    print(f"Instance created: {instance_id}")

    max_hours = float(getattr(args, "max_hours", 2.5))
    now_ts = time.time()
    instance_data = {
        "id": instance_id,
        "name": name,
        "cloud": cloud,
        "region": region,
        "type": shade_type,
        "price_per_hour": price,
        "created_at": now_ts,
        "kill_at": now_ts + max_hours * 3600,
        "max_hours": max_hours,
        "status": "pending",
    }
    _save_instance(instance_data, name)
    print(f"  kill_at: +{max_hours:.1f}h (safety: orphan sweep tears down after this)")

    print("Waiting for instance to be ready...")
    for i in range(60):
        time.sleep(15)
        info = _api("GET", f"/instances/{instance_id}/info")
        status = info.get("status", "unknown")
        ip = info.get("ip", "")
        print(f"  [{i*15}s] status={status} ip={ip}")
        instance_data["status"] = status
        instance_data["ip"] = ip
        instance_data["ssh_port"] = info.get("ssh_port", 22)
        instance_data["ssh_user"] = info.get("ssh_user", "root")
        _save_instance(instance_data, name)
        if status == "active" and ip:
            print("\nInstance ready!")
            print(f"  IP: {ip}")
            print(f"  SSH: ssh -i {SSH_KEY} {instance_data['ssh_user']}@{ip} -p {instance_data['ssh_port']}")
            print(f"  Cost: ${price:.2f}/hr")
            return
    print("Timeout waiting for instance. Check status with: python scripts/gpu.py status")


def cmd_status(args):
    """Show current instance status."""
    name = getattr(args, "name", "default")
    inst = _load_instance(name)
    if not inst:
        print(f"No active instance '{name}'. Rent one with: python scripts/gpu.py rent --name {name}")
        return
    if inst.get("id"):
        try:
            info = _api("GET", f"/instances/{inst['id']}/info")
            inst["status"] = info.get("status", inst.get("status"))
            inst["ip"] = info.get("ip", inst.get("ip"))
            _save_instance(inst, name)
        except Exception:
            pass
    hours = (time.time() - inst.get("created_at", time.time())) / 3600
    cost = hours * inst.get("price_per_hour", 0)
    print(f"ID:      {inst.get('id', '?')}")
    print(f"Status:  {inst.get('status', '?')}")
    print(f"IP:      {inst.get('ip', '?')}")
    print(f"Type:    {inst.get('type', '?')} on {inst.get('cloud', '?')}")
    print(f"Price:   ${inst.get('price_per_hour', 0):.2f}/hr")
    print(f"Uptime:  {hours:.1f}h (est. cost: ${cost:.2f})")
    ssh_user = inst.get("ssh_user", "root")
    ssh_ip = inst.get("ip", "?")
    ssh_port = inst.get("ssh_port", 22)
    print(f"SSH:     ssh -i {SSH_KEY} {ssh_user}@{ssh_ip} -p {ssh_port}")


def cmd_ssh(args):
    """SSH into the instance or run a command."""
    name = getattr(args, "name", "default")
    inst = _load_instance(name)
    if not inst or not inst.get("ip"):
        print(f"No active instance '{name}' with IP. Rent one first.")
        return
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no",
           "-i", str(SSH_KEY),
           "-p", str(inst.get("ssh_port", 22)),
           f"{inst.get('ssh_user', 'root')}@{inst['ip']}"]
    if args.command:
        cmd.extend(args.command)
    os.execvp("ssh", cmd)


def cmd_backup(args):
    """Backup results from the instance to local backup dir."""
    name = getattr(args, "name", "default")
    inst = _load_instance(name)
    if not inst or not inst.get("ip"):
        print(f"No active instance '{name}'.")
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ip = inst["ip"]
    port = inst.get("ssh_port", 22)
    user = inst.get("ssh_user", "root")
    scp_base = ["scp", "-o", "StrictHostKeyChecking=no",
                "-i", str(SSH_KEY), "-P", str(port)]

    files_to_backup = [
        ("runs/h100_calibration/calibration.json", "calibration.json"),
        ("runs/h100_noise_floor/noise_floor_summary.json", "noise_floor_summary.json"),
    ]
    # Find all final_state.json and training_log.jsonl
    result = subprocess.run(
        [
            "ssh", "-i", str(SSH_KEY), "-p", str(port), f"{user}@{ip}",
            "find /workspace/ralph/runs "
            "-name 'final_state.json' -o -name 'training_log.jsonl' "
            "-o -name 'checkpoint.pt' -o -name 'data_manifest.json' "
            "2>/dev/null | head -20",
        ],
        capture_output=True, text=True, timeout=15,
    )
    remote_files = [f.strip() for f in result.stdout.splitlines() if f.strip()]
    print(f"Found {len(remote_files)} files to backup")
    for rf in remote_files:
        local_name = rf.replace("/workspace/ralph/", "").replace("/", "_")
        local_path = BACKUP_DIR / local_name
        print(f"  {rf} → {local_path.name}")
        subprocess.run(scp_base + [f"{user}@{ip}:{rf}", str(local_path)], timeout=300)
    print(f"\nBackup saved to {BACKUP_DIR}")


def cmd_delete(args):
    """Delete the instance (stops billing)."""
    name = getattr(args, "name", "default")
    inst = _load_instance(name)
    if not inst or not inst.get("id"):
        print(f"No instance '{name}' to delete.")
        return
    instance_id = inst["id"]
    if not args.yes:
        confirm = input(f"Delete instance '{name}' ({instance_id})? This stops billing. [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
    print(f"Deleting instance {instance_id}...")
    _api("POST", f"/instances/{instance_id}/delete")
    inst["status"] = "deleted"
    inst["deleted_at"] = time.time()
    _save_instance(inst, name)
    hours = (inst["deleted_at"] - inst.get("created_at", inst["deleted_at"])) / 3600
    cost = hours * inst.get("price_per_hour", 0)
    print(f"Deleted. Total uptime: {hours:.1f}h, est. cost: ${cost:.2f}")


def main():
    p = argparse.ArgumentParser(description="GPU lifecycle manager (Shadeform)")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list-types", help="List available H100 instances")
    rent_p = sub.add_parser("rent", help="Rent H100 (default: Hyperstack)")
    rent_p.add_argument("--cloud", default=None, help="Cloud provider (default: hyperstack)")
    rent_p.add_argument("--region", default=None, help="Region (default: auto-pick cheapest)")
    rent_p.add_argument("--name", default="default", help="Logical instance name (state file suffix)")
    rent_p.add_argument(
        "--max-hours", type=float, default=2.5,
        help="Hard kill timer: instance auto-swept after this many hours from rent "
             "(default: 2.5). Defense against orchestrator crash leaking GPU billing.",
    )
    status_p = sub.add_parser("status", help="Show instance status")
    status_p.add_argument("--name", default="default")

    ssh_p = sub.add_parser("ssh", help="SSH into instance")
    ssh_p.add_argument("--name", default="default")
    ssh_p.add_argument("command", nargs="*", help="Command to run (or omit for interactive)")

    backup_p = sub.add_parser("backup", help="Backup results to local dir")
    backup_p.add_argument("--name", default="default")

    del_p = sub.add_parser("delete", help="Delete instance")
    del_p.add_argument("--name", default="default")
    del_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    args = p.parse_args()
    # Sweep is cheap when clean — defends against a crashed prior session
    # leaving an H100 billing in the background. Runs once per command.
    _sweep_orphans_safe()
    cmds = {
        "list-types": cmd_list_types,
        "rent": cmd_rent,
        "status": cmd_status,
        "ssh": cmd_ssh,
        "backup": cmd_backup,
        "delete": cmd_delete,
    }
    if args.cmd in cmds:
        cmds[args.cmd](args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
