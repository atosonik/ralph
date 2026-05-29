"""
HuggingFace Hub integration for proof bundle upload/download.

Miners upload proof bundles to HuggingFace after running the proof test.
Validators download bundles from HF URLs referenced in submission PRs.

The bundle structure on HF:
    karpaai/proof-bundles (dataset repo)
      submissions/<bundle_hash_prefix>/
        bundle_manifest.json
        checkpoint.pt
        training_log.jsonl
        calibration.json
        attestation.json (verified tier only)
        wandb_run_url.txt (if wandb was enabled)

Usage:
    # Miner uploads after proof test
    python -m miner.hub upload --proof-dir runs/proof_xxx --repo karpaai/proof-bundles

    # Validator downloads for scoring
    python -m miner.hub download --bundle-hash abc123 --repo karpaai/proof-bundles --out-dir /tmp/bundle

Requires: pip install 'karpa-subnet[hub]'
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def upload_bundle(
    proof_dir: Path,
    repo_id: str = "karpaai/proof-bundles",
    token: str | None = None,
) -> str:
    """Upload a proof bundle as a single HF PR. Returns the PR URL.

    Miners aren't org members on karpaai, so direct commits to main are
    forbidden. Instead we stage all the bundle files into one folder and
    push them via `create_pr=True` — community PR pattern. The validator
    side polls open PRs, scores, and the bot merges the winner.
    """
    from huggingface_hub import HfApi, CommitOperationAdd
    import tempfile
    import shutil

    api = HfApi(token=token)

    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception:
        pass

    proof_dir = Path(proof_dir)
    manifest = json.loads((proof_dir / "bundle_manifest.json").read_text())
    bundle_hash = manifest["bundle_hash"]
    prefix = f"submissions/{bundle_hash[:16]}"

    # Collect (local_path, name_in_bundle) pairs.
    files: list[tuple[Path, str]] = [
        (proof_dir / "bundle_manifest.json", "bundle_manifest.json"),
        (proof_dir / "calibration.json",     "calibration.json"),
    ]
    training_dir = proof_dir / "training"
    if training_dir.exists():
        for name in ["checkpoint.pt", "training_log.jsonl", "final_state.json",
                     "wandb_metrics.json", "wandb_run_url.txt"]:
            p = training_dir / name
            if p.exists():
                files.append((p, name))
    for name in ["attestation.json", "submission.json"]:
        p = proof_dir / name
        if p.exists():
            files.append((p, name))

    operations = [
        CommitOperationAdd(path_in_repo=f"{prefix}/{remote_name}", path_or_fileobj=str(local_path))
        for (local_path, remote_name) in files
        if local_path.exists()
    ]
    total_mb = sum(p.stat().st_size for (p, _) in files if p.exists()) / 1e6
    print(f"[hub] uploading {len(operations)} files ({total_mb:.1f} MB) as PR → {repo_id}/{prefix}")

    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Submit proof bundle {bundle_hash[:12]}",
        commit_description=f"Bundle hash: `{bundle_hash}`\nManifest sha256: `{manifest.get('manifest_sha256', '?')}`",
        create_pr=True,
    )

    pr_url = commit_info.pr_url or commit_info.commit_url
    print(f"[hub] PR: {pr_url}")
    return pr_url


def download_bundle(
    bundle_hash: str,
    repo_id: str = "karpaai/proof-bundles",
    out_dir: Path | None = None,
    token: str | None = None,
) -> Path:
    """Download a proof bundle from HuggingFace Hub. Returns local path."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    prefix = f"submissions/{bundle_hash[:16]}"
    if out_dir is None:
        out_dir = Path(f"/tmp/karpa_bundles/{bundle_hash[:16]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    training_dir = out_dir / "training"
    training_dir.mkdir(exist_ok=True)

    training_files = {"checkpoint.pt", "training_log.jsonl", "final_state.json"}

    try:
        tree = list_repo_tree(repo_id, path_in_repo=prefix, repo_type="dataset", token=token)
        files = [item.rfilename for item in tree if hasattr(item, "rfilename")]
    except Exception:
        files = [
            f"{prefix}/bundle_manifest.json",
            f"{prefix}/calibration.json",
            f"{prefix}/checkpoint.pt",
            f"{prefix}/training_log.jsonl",
            f"{prefix}/final_state.json",
            f"{prefix}/attestation.json",
            f"{prefix}/submission.json",
        ]

    for remote_path in files:
        filename = remote_path.split("/")[-1]
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                local_dir=str(out_dir / "_hf_cache"),
                token=token,
            )
            dest = (training_dir / filename) if filename in training_files else (out_dir / filename)
            import shutil
            shutil.copy2(local, dest)
            print(f"  {filename}: ok")
        except Exception as e:
            print(f"  {filename}: skipped ({e})")

    print(f"[hub] downloaded to {out_dir}")
    return out_dir


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command")

    up = sub.add_parser("upload")
    up.add_argument("--proof-dir", type=Path, required=True)
    up.add_argument("--repo", default="karpaai/proof-bundles")
    up.add_argument("--token", default=None)

    down = sub.add_parser("download")
    down.add_argument("--bundle-hash", required=True)
    down.add_argument("--repo", default="karpaai/proof-bundles")
    down.add_argument("--out-dir", type=Path, default=None)
    down.add_argument("--token", default=None)

    args = p.parse_args()
    if args.command == "upload":
        upload_bundle(args.proof_dir, args.repo, args.token)
    elif args.command == "download":
        download_bundle(args.bundle_hash, args.repo, args.out_dir, args.token)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
