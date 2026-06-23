"""
HuggingFace Hub integration for proof bundle upload/download.

Miners upload proof bundles to HuggingFace after running the proof test.
Validators download bundles from HF URLs referenced in submission PRs.

The bundle structure on HF:
    RalphLabsAI/proof-bundles (dataset repo)
      submissions/<bundle_hash_prefix>/
        bundle_manifest.json
        checkpoint.pt
        training_log.jsonl
        calibration.json
        attestation.json (verified tier only)
        wandb_run_url.txt (if wandb was enabled)

Usage:
    # Miner uploads after proof test
    python -m miner.hub upload --proof-dir runs/proof_xxx --repo RalphLabsAI/proof-bundles

    # Validator downloads for scoring
    python -m miner.hub download --bundle-hash abc123 --repo RalphLabsAI/proof-bundles --out-dir /tmp/bundle

Requires: pip install 'ralph-subnet[hub]'
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def upload_bundle(
    proof_dir: Path,
    repo_id: str = "RalphLabsAI/proof-bundles",
    token: str | None = None,
    rationale_text: str = "",
    patch_path: Path | None = None,
    encrypt: bool | None = None,
) -> str:
    """Upload a proof bundle as a single HF PR. Returns the PR URL.

    Miners aren't org members on RalphLabsAI, so direct commits to main are
    forbidden. Instead we stage all the bundle files into one folder and
    push them via `create_pr=True` — community PR pattern. The validator
    side polls open PRs, scores, and the bot merges the winner.

    When `rationale_text` is supplied, it becomes the PR description (so a
    human reading the HF PR sees the hypothesis before the file list) and
    also gets included as `rationale.md` in the bundle.

    The HF storage prefix uses the full 64-char `bundle_hash` (not a
    16-char slice) to eliminate the silent collision risk between honest
    miners who submit identical baseline bundles — both the storage path
    and the validator's poll-state key off this prefix.

    If `patch_path` is supplied (or a `patch.diff` already exists at the
    bundle root of `proof_dir`), it is uploaded as `patch.diff` so the
    validator's PR-match verifier can cross-check the GitHub PR diff
    against the bundle's diff (without it the verifier silently skips
    the check and treats the bundle as a baseline).
    """
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=token)

    try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=True)
    except Exception:
        pass

    proof_dir = Path(proof_dir)
    manifest = json.loads((proof_dir / "bundle_manifest.json").read_text())
    bundle_hash = manifest["bundle_hash"]
    prefix = f"submissions/{bundle_hash}"

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
    for name in ["attestation.json", "submission.json", "rationale.md", "patch.diff"]:
        p = proof_dir / name
        if p.exists():
            files.append((p, name))

    # Explicit patch_path overrides / supplements an in-tree patch.diff.
    if patch_path is not None:
        patch_path = Path(patch_path)
        if patch_path.exists():
            # Drop any auto-picked patch.diff in favor of the explicit one.
            files = [(lp, rn) for (lp, rn) in files if rn != "patch.diff"]
            files.append((patch_path, "patch.diff"))

    # Encrypt the bundle to the validator key unless explicitly disabled or no
    # key is configured (testnet / local). The decrypted blob reproduces this
    # exact directory layout, so the validator's verify path is unchanged.
    from proof import bundle_crypto

    do_enc = encrypt
    if do_enc is None:
        do_enc = os.environ.get("RALPH_DISABLE_BUNDLE_ENC") != "1"
    pubkey = None
    if do_enc:
        try:
            pubkey = bundle_crypto.load_validator_pubkey()
        except RuntimeError as e:
            print(f"[hub] bundle encryption off ({e}); uploading plaintext")
            do_enc = False

    if do_enc:
        import tempfile

        # Final arcnames mirror the on-disk layout the validator rebuilds:
        # training artifacts under training/, everything else at the root.
        tar_files = [
            (lp, f"training/{rn}" if Path(lp).parent.name == "training" else rn)
            for (lp, rn) in files
            if lp.exists()
        ]
        blob = bundle_crypto.encrypt(bundle_crypto.pack_bundle(tar_files), pubkey)
        tmp = Path(tempfile.mkdtemp())
        enc_path = tmp / bundle_crypto.ENC_FILENAME
        enc_path.write_bytes(blob)
        pub_manifest = {
            "bundle_hash": bundle_hash,
            "manifest_sha256": manifest.get("manifest_sha256"),
            "parent_hash": manifest.get("parent_hash"),
            "attestation_type": manifest.get("attestation_type"),
            "encrypted": True,
            "enc_scheme": bundle_crypto.ENC_SCHEME,
        }
        man_path = tmp / bundle_crypto.PUBLIC_MANIFEST
        man_path.write_text(json.dumps(pub_manifest, indent=2))
        operations = [
            CommitOperationAdd(path_in_repo=f"{prefix}/{bundle_crypto.ENC_FILENAME}",
                               path_or_fileobj=str(enc_path)),
            CommitOperationAdd(path_in_repo=f"{prefix}/{bundle_crypto.PUBLIC_MANIFEST}",
                               path_or_fileobj=str(man_path)),
        ]
        total_mb = enc_path.stat().st_size / 1e6
        print(f"[hub] uploading encrypted bundle ({total_mb:.1f} MB) as PR → {repo_id}/{prefix}")
    else:
        operations = [
            CommitOperationAdd(path_in_repo=f"{prefix}/{remote_name}", path_or_fileobj=str(local_path))
            for (local_path, remote_name) in files
            if local_path.exists()
        ]
        total_mb = sum(p.stat().st_size for (p, _) in files if p.exists()) / 1e6
        print(f"[hub] uploading {len(operations)} files ({total_mb:.1f} MB) as PR → {repo_id}/{prefix}")

    # PR description: rationale upfront, then bundle identification.
    # Cap rationale to ~60 KB so we don't trip the HF commit-description limit.
    _RATIONALE_MAX_BYTES = 60_000
    description_parts = []
    if rationale_text.strip():
        rationale_clean = rationale_text.rstrip()
        # Strip a trailing markdown `---` sign-off so we don't render two
        # consecutive horizontal rules when we append our own separator.
        while rationale_clean.endswith("---"):
            rationale_clean = rationale_clean[:-3].rstrip()
        rationale_bytes = rationale_clean.encode("utf-8")
        if len(rationale_bytes) > _RATIONALE_MAX_BYTES:
            rationale_clean = rationale_bytes[:_RATIONALE_MAX_BYTES].decode("utf-8", errors="ignore")
            rationale_clean += "\n\n_…rationale truncated; full text in bundle's rationale.md…_"
        description_parts.append(rationale_clean)
        description_parts.append("---")
    description_parts.append(
        f"**bundle_hash:** `{bundle_hash}`  \n"
        f"**manifest sha256:** `{manifest.get('manifest_sha256', '?')}`"
    )
    commit_description = "\n\n".join(description_parts)

    commit_info = api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=operations,
        commit_message=f"Submit proof bundle {bundle_hash[:12]}",
        commit_description=commit_description,
        create_pr=True,
    )

    pr_url = commit_info.pr_url or commit_info.commit_url
    print(f"[hub] PR: {pr_url}")
    return pr_url


def download_bundle(
    bundle_hash: str,
    repo_id: str = "RalphLabsAI/proof-bundles",
    out_dir: Path | None = None,
    token: str | None = None,
) -> Path:
    """Download a proof bundle from HuggingFace Hub. Returns local path."""
    from huggingface_hub import hf_hub_download, list_repo_tree

    from proof import bundle_crypto

    prefix = f"submissions/{bundle_hash}"
    if out_dir is None:
        out_dir = Path(f"/tmp/ralph_bundles/{bundle_hash[:16]}")
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

    enc_remote = f"{prefix}/{bundle_crypto.ENC_FILENAME}"
    if enc_remote in files:
        local = hf_hub_download(repo_id=repo_id, filename=enc_remote, repo_type="dataset",
                                local_dir=str(out_dir / "_hf_cache"), token=token)
        bundle_crypto.unpack_bundle(bundle_crypto.decrypt(Path(local).read_bytes()), out_dir)
        import shutil
        shutil.rmtree(out_dir / "_hf_cache", ignore_errors=True)
        print(f"[hub] downloaded + decrypted to {out_dir}")
        return out_dir

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
    up.add_argument("--repo", default="RalphLabsAI/proof-bundles")
    up.add_argument("--token", default=None)

    down = sub.add_parser("download")
    down.add_argument("--bundle-hash", required=True)
    down.add_argument("--repo", default="RalphLabsAI/proof-bundles")
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
