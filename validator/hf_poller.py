"""
HuggingFace Hub poller — fetches new submission bundles into the local queue.

The validator service calls poll_hub() each epoch to discover bundles that
miners have uploaded to the public dataset repo. New ones get downloaded
into `queue/pending/<bundle_hash>/` where the existing local-queue logic
picks them up.

State is tracked in queue/hf_state.json so we don't re-download already-
processed bundles after restarts. Each processed bundle is stamped with the
VALIDATOR_VERSION it was judged under; on a version bump, older-version entries
are reprocessed (re-downloaded + re-validated) so evaluation stays fair across
a logic upgrade.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

from validator.version import VALIDATOR_VERSION

DEFAULT_REPO = "RalphLabsAI/proof-bundles"


def _state_path(queue_dir: Path) -> Path:
    return queue_dir / "hf_state.json"


def _migrate_state(raw: dict) -> dict:
    """Normalise to {"validator_version": str, "processed": {bundle_id: version}}.

    Legacy state stored `processed` as a flat list of bundle_ids with no version;
    those are mapped to "legacy" so they re-process under the current version.
    """
    processed = raw.get("processed", {})
    if isinstance(processed, list):
        processed = {bid: "legacy" for bid in processed}
    elif not isinstance(processed, dict):
        processed = {}
    return {
        "validator_version": raw.get("validator_version", "legacy"),
        "processed": processed,
    }


def _load_state(queue_dir: Path) -> dict:
    p = _state_path(queue_dir)
    if not p.exists():
        return {"validator_version": VALIDATOR_VERSION, "processed": {}}
    try:
        return _migrate_state(json.loads(p.read_text()))
    except Exception:
        return {"validator_version": VALIDATOR_VERSION, "processed": {}}


def _save_state(queue_dir: Path, state: dict) -> None:
    _state_path(queue_dir).write_text(json.dumps(state, indent=2))


def list_remote_submissions(repo_id: str, token: Optional[str] = None) -> list[dict]:
    """Return the open HF PRs against the dataset, oldest-first.

    Each entry is a dict with bundle_id (= directory prefix under submissions/),
    pr_num, git_ref, and created_at (ISO-8601 PR creation time). Ordering is
    by created_at then pr_num so validation is first-come-first-served (fair),
    not by bundle-hash lexical order.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    try:
        discussions = api.get_repo_discussions(repo_id=repo_id, repo_type="dataset")
    except Exception as e:
        print(f"[hf_poller] get_repo_discussions failed: {e}")
        return []

    pending = []
    for d in discussions:
        if not d.is_pull_request:
            continue
        if d.status != "open":
            continue
        # Each PR has a git_reference like 'refs/pr/3'. Files under
        # submissions/<bundle_id>/ are what we want; find the bundle_id by
        # listing the PR's commit tree.
        ref = d.git_reference  # e.g. "refs/pr/3"
        try:
            files = api.list_repo_files(repo_id, repo_type="dataset", revision=ref)
        except Exception as e:
            print(f"[hf_poller] list PR #{d.num} files failed: {e}")
            continue
        created_at = d.created_at.isoformat() if getattr(d, "created_at", None) else None
        bundle_ids = {f.split("/")[1] for f in files if f.startswith("submissions/") and len(f.split("/")) >= 3}
        for bid in bundle_ids:
            pending.append(
                {"bundle_id": bid, "pr_num": d.num, "git_ref": ref, "created_at": created_at}
            )

    # First-come-first-validate: oldest PR first. created_at is ISO-8601 UTC so
    # lexical order is chronological; pr_num breaks ties / covers a missing time.
    pending.sort(key=lambda p: (p["created_at"] or "", p["pr_num"]))
    return pending


def download_one(
    bundle_id: str,
    repo_id: str,
    dest_dir: Path,
    token: Optional[str] = None,
    git_ref: str = "main",
    pr_num: int | None = None,
    created_at: str | None = None,
) -> bool:
    """Download all files for one bundle into dest_dir/<bundle_id>/.

    git_ref is the revision to read from — `main` for legacy direct-commit
    flows, `refs/pr/N` for PR-based submissions (the default since miners
    aren't org members on RalphLabsAI).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    from proof import bundle_crypto

    out = dest_dir / bundle_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    training_dir = out / "training"
    training_dir.mkdir()

    training_files = {
        "checkpoint.pt", "training_log.jsonl", "final_state.json",
        "wandb_metrics.json", "wandb_run_url.txt",
    }

    try:
        all_files = list_repo_files(repo_id, repo_type="dataset", token=token, revision=git_ref)
        prefix = f"submissions/{bundle_id}/"
        bundle_files = [f for f in all_files if f.startswith(prefix)]
    except Exception as e:
        print(f"[hf_poller] list files failed for {bundle_id} @ {git_ref}: {e}")
        return False

    if not bundle_files:
        print(f"[hf_poller] no files found for {bundle_id} @ {git_ref}")
        return False

    cache = out / "_hf_cache"
    enc_remote = f"{prefix}{bundle_crypto.ENC_FILENAME}"
    if enc_remote in bundle_files:
        # Encrypted submission: download the blob, decrypt with the validator
        # key, and unpack — reproduces the same dir a plaintext bundle would.
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=enc_remote, repo_type="dataset",
                local_dir=str(cache), token=token, revision=git_ref,
            )
            blob = bundle_crypto.decrypt(Path(local).read_bytes())
            bundle_crypto.unpack_bundle(blob, out)
            success = 1
        except Exception as e:
            print(f"[hf_poller] decrypt/unpack failed for {bundle_id}: {e}")
            success = 0
    else:
        # Legacy plaintext: download each file into the bundle dir.
        success = 0
        for remote_path in bundle_files:
            filename = remote_path.split("/")[-1]
            try:
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    repo_type="dataset",
                    local_dir=str(cache),
                    token=token,
                    revision=git_ref,
                )
                dest = (training_dir / filename) if filename in training_files else (out / filename)
                shutil.copy2(local, dest)
                success += 1
            except Exception as e:
                print(f"[hf_poller] download {filename} failed: {e}")

    if cache.exists():
        shutil.rmtree(cache)
    training_dir.mkdir(exist_ok=True)  # encrypted bundles without training/ still get the dir

    if success == 0:
        shutil.rmtree(out)
        return False

    # Annotate which PR this came from so the validator can merge later.
    if pr_num is not None:
        (out / ".hf_pr.json").write_text(json.dumps(
            {
                "pr_num": pr_num,
                "git_ref": git_ref,
                "repo_id": "RalphLabsAI/proof-bundles",
                "created_at": created_at,
            },
            indent=2,
        ))
    return True


def poll_hub(
    queue_dir: Path,
    repo_id: str = DEFAULT_REPO,
    token: Optional[str] = None,
    limit: int = 10,
) -> list[str]:
    """Fetch new submission bundles from HF into queue/pending/.

    Returns the list of newly-downloaded bundle IDs.
    """
    queue_dir = Path(queue_dir)
    pending = queue_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)

    state = _load_state(queue_dir)
    processed = state.get("processed", {})  # {bundle_id: validator_version}

    # A bundle counts as done only if it was judged by the CURRENT validator
    # version. Entries from an older version are reprocessed: not in `done` →
    # re-downloaded → re-judged → re-stamped below.
    done = {bid for bid, ver in processed.items() if ver == VALIDATOR_VERSION}

    remote_prs = list_remote_submissions(repo_id, token=token)  # oldest-first
    if not remote_prs:
        return []

    new = [p for p in remote_prs if p["bundle_id"] not in done]
    if not new:
        return []

    summary = [(p["bundle_id"][:8], f"PR#{p['pr_num']}") for p in new[:limit]]
    print(f"[hf_poller] found {len(new)} new PR-bundle(s) on HF Hub: {summary}")

    downloaded = []
    for sub in new[:limit]:
        bid = sub["bundle_id"]
        print(f"[hf_poller] downloading {bid} from PR #{sub['pr_num']} ({sub['git_ref']})...")
        if download_one(bid, repo_id, pending, token=token,
                        git_ref=sub["git_ref"], pr_num=sub["pr_num"],
                        created_at=sub.get("created_at")):
            downloaded.append(bid)
            processed[bid] = VALIDATOR_VERSION
        else:
            print(f"[hf_poller] skipped {bid} (download failed)")

    state["validator_version"] = VALIDATOR_VERSION
    state["processed"] = processed
    _save_state(queue_dir, state)
    return downloaded


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--queue-dir", type=Path, default=Path("queue"))
    p.add_argument("--repo", default=DEFAULT_REPO)
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--limit", type=int, default=10)
    args = p.parse_args()

    new = poll_hub(args.queue_dir, args.repo, args.token, args.limit)
    print(f"\nDownloaded {len(new)} bundle(s): {new}")


if __name__ == "__main__":
    main()
