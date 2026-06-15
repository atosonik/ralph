"""
HuggingFace Hub poller — fetches new submission bundles into the local queue.

The validator service calls poll_hub() each epoch to discover bundles that
miners have uploaded to the public dataset repo. New ones get downloaded
into `queue/pending/<bundle_hash>/` where the existing local-queue logic
picks them up.

State is tracked in queue/hf_state.json so we don't re-download already-
processed bundles after restarts.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Optional

DEFAULT_REPO = "RalphLabsAI/proof-bundles"


def _state_path(queue_dir: Path) -> Path:
    return queue_dir / "hf_state.json"


def _load_state(queue_dir: Path) -> dict:
    p = _state_path(queue_dir)
    if not p.exists():
        return {"processed": []}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {"processed": []}


def _save_state(queue_dir: Path, state: dict) -> None:
    _state_path(queue_dir).write_text(json.dumps(state, indent=2))


def list_remote_submissions(repo_id: str, token: Optional[str] = None) -> list[dict]:
    """Return the open HF PRs against the dataset, each as a dict with
    bundle_id (= directory prefix under submissions/), pr_num, git_ref."""
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
        bundle_ids = {f.split("/")[1] for f in files if f.startswith("submissions/") and len(f.split("/")) >= 3}
        for bid in bundle_ids:
            pending.append({"bundle_id": bid, "pr_num": d.num, "git_ref": ref})
    return pending


def download_one(
    bundle_id: str,
    repo_id: str,
    dest_dir: Path,
    token: Optional[str] = None,
    git_ref: str = "main",
    pr_num: int | None = None,
) -> bool:
    """Download all files for one bundle into dest_dir/<bundle_id>/.

    git_ref is the revision to read from — `main` for legacy direct-commit
    flows, `refs/pr/N` for PR-based submissions (the default since miners
    aren't org members on RalphLabsAI).
    """
    from huggingface_hub import hf_hub_download, list_repo_files

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

    success = 0
    for remote_path in bundle_files:
        filename = remote_path.split("/")[-1]
        try:
            local = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="dataset",
                local_dir=str(out / "_hf_cache"),
                token=token,
                revision=git_ref,
            )
            dest = (training_dir / filename) if filename in training_files else (out / filename)
            shutil.copy2(local, dest)
            success += 1
        except Exception as e:
            print(f"[hf_poller] download {filename} failed: {e}")

    cache = out / "_hf_cache"
    if cache.exists():
        shutil.rmtree(cache)

    if success == 0:
        shutil.rmtree(out)
        return False

    # Annotate which PR this came from so the validator can merge later.
    if pr_num is not None:
        (out / ".hf_pr.json").write_text(json.dumps(
            {"pr_num": pr_num, "git_ref": git_ref, "repo_id": "RalphLabsAI/proof-bundles"},
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
    processed = set(state.get("processed", []))

    remote_prs = list_remote_submissions(repo_id, token=token)
    if not remote_prs:
        return []

    new = [p for p in remote_prs if p["bundle_id"] not in processed]
    if not new:
        return []

    summary = [(p["bundle_id"][:8], f"PR#{p['pr_num']}") for p in new[:limit]]
    print(f"[hf_poller] found {len(new)} new PR-bundle(s) on HF Hub: {summary}")

    downloaded = []
    for sub in new[:limit]:
        bid = sub["bundle_id"]
        print(f"[hf_poller] downloading {bid} from PR #{sub['pr_num']} ({sub['git_ref']})...")
        if download_one(bid, repo_id, pending, token=token,
                        git_ref=sub["git_ref"], pr_num=sub["pr_num"]):
            downloaded.append(bid)
            processed.add(bid)
        else:
            print(f"[hf_poller] skipped {bid} (download failed)")

    state["processed"] = sorted(processed)
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
