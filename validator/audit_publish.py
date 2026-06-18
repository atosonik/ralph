"""Publish Ralph validator audit reports to a HuggingFace dataset repo.

validation-v2 Phase 2 (A). The on-chain `set_commitment` remains the trust
anchor; HF is just the off-chain store auditors pull the full signed report
from. This module uploads the per-epoch `<epoch_id>.json` envelope and upserts
`index.json` to a HF **dataset** repo (default `RalphLabsAI/audit-reports`).

Wiring contract (see validator/audit_report.write_report):
  * Local write always happens first and is authoritative.
  * HF publish is gated behind an explicit `hf_publish_enabled` flag (prod-only)
    and wrapped in try/except by the caller so a publish failure can NEVER break
    the validator's scoring / weight-setting / local report.

The layout on HF mirrors the local layout so an auditor's fetch code is the
same shape either way:
    <repo>/audit_reports/<epoch_id>.json   — full signed envelope
    <repo>/audit_reports/index.json        — append-only epoch index

The auditor resolves files via the public resolve URL:
    https://huggingface.co/datasets/<repo>/resolve/main/audit_reports/index.json
"""

from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_AUDIT_REPO = "RalphLabsAI/audit-reports"

# Where reports live inside the dataset repo. Mirrors write_report's local
# layout (<out_dir>/audit_reports/...) so the auditor's path logic is identical
# for local and remote.
_REPO_SUBDIR = "audit_reports"


def _index_path_in_repo() -> str:
    return f"{_REPO_SUBDIR}/index.json"


def _report_path_in_repo(epoch_id: str) -> str:
    return f"{_REPO_SUBDIR}/{epoch_id}.json"


def _fetch_remote_index(api, repo_id: str, token: Optional[str]) -> list[dict]:
    """Download the existing index.json from the dataset repo (if any).

    Returns [] if the repo has no index yet (first publish) or on any read
    failure — the caller upserts onto whatever we return, so a transient read
    miss degrades to "re-publish this epoch's entry" rather than data loss
    (the local index.json remains the complete authoritative copy).
    """
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    try:
        local = api.hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=_index_path_in_repo(),
            token=token,
        )
        with open(local, encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, list) else []
    except (EntryNotFoundError, RepositoryNotFoundError):
        return []
    except Exception:
        return []


def _index_entry(report_envelope: dict) -> dict:
    """Build the index.json entry for a report envelope — identical shape to
    the one write_report writes locally so local and remote indexes agree."""
    rj = report_envelope.get("report_json") or {}
    return {
        "epoch_id": rj.get("epoch_id"),
        "epoch_start_block": rj.get("epoch_start_block"),
        "epoch_end_block": rj.get("epoch_end_block"),
        "report_sha256": report_envelope.get("report_sha256"),
        "signer_hotkey": report_envelope.get("signer_hotkey"),
        "chain_commitment_block": report_envelope.get("chain_commitment_block"),
        "weights_set": report_envelope.get("weights_set"),
    }


def publish_report_hf(
    report_envelope: dict,
    repo_id: str = DEFAULT_AUDIT_REPO,
    token: Optional[str] = None,
) -> bool:
    """Upload `<epoch_id>.json` + upsert `index.json` to a HF dataset repo.

    Args:
      report_envelope: the signed envelope from build_envelope().
      repo_id: target HF dataset repo (default RalphLabsAI/audit-reports).
      token: HF write token; falls back to $HF_TOKEN when None.

    Returns True on success. Raises on failure — the validator caller wraps
    this in try/except so a publish error never affects scoring/weights/local
    write. (Returning a bool AND raising keeps both call styles honest: the
    caller treats any exception as "publish failed, continue".)

    Idempotent: re-publishing the same epoch_id overwrites the per-epoch file
    and replaces (not duplicates) its index entry.
    """
    from huggingface_hub import HfApi

    token = token or os.environ.get("HF_TOKEN")
    epoch_id = (report_envelope.get("report_json") or {}).get("epoch_id")
    if not epoch_id:
        raise ValueError("report_envelope has no report_json.epoch_id — cannot publish")

    api = HfApi(token=token)
    # Create-if-missing; exist_ok so steady-state publishes are a no-op here.
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, token=token)

    # 1. upload the per-epoch envelope.
    envelope_bytes = json.dumps(
        report_envelope, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    api.upload_file(
        path_or_fileobj=envelope_bytes,
        path_in_repo=_report_path_in_repo(epoch_id),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"audit report {epoch_id}",
    )

    # 2. upsert index.json: pull remote, replace-by-epoch_id, re-upload.
    index = _fetch_remote_index(api, repo_id, token)
    index = [e for e in index if e.get("epoch_id") != epoch_id]
    index.append(_index_entry(report_envelope))
    index_bytes = json.dumps(
        index, indent=2, sort_keys=True, ensure_ascii=False
    ).encode("utf-8")
    api.upload_file(
        path_or_fileobj=index_bytes,
        path_in_repo=_index_path_in_repo(),
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
        commit_message=f"index upsert {epoch_id}",
    )
    return True


__all__ = ["DEFAULT_AUDIT_REPO", "publish_report_hf"]
