"""Fetch audit reports from a validator's HF dataset repo.

Mirrors greencompute-audit/audit/fetch.py, but Ralph publishes to a HuggingFace
**dataset** repo instead of running a FastAPI server. We pull plain JSON over
the public `resolve` URL — no HF token needed for a public repo:

    https://huggingface.co/datasets/<repo>/resolve/main/audit_reports/index.json
    https://huggingface.co/datasets/<repo>/resolve/main/audit_reports/<epoch_id>.json

The layout (`audit_reports/...`) matches validator.audit_report.write_report and
validator.audit_publish, so local and remote stores are byte-shaped the same.
"""

from __future__ import annotations

import requests

DEFAULT_AUDIT_REPO = "RalphLabsAI/audit-reports"
_REPO_SUBDIR = "audit_reports"


class ReportClient:
    """HTTP client for a validator's published audit reports on HF."""

    def __init__(
        self,
        repo_id: str = DEFAULT_AUDIT_REPO,
        revision: str = "main",
        token: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self._timeout = timeout
        # requests follows the HF resolve 302 -> CDN redirect by default.
        self._session = requests.Session()
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    def _resolve_url(self, path_in_repo: str) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repo_id}"
            f"/resolve/{self.revision}/{path_in_repo}"
        )

    def list_reports(self) -> list[dict]:
        """Return the index.json list (one thin entry per epoch). Empty list if
        the index isn't available yet.

        Treats 404 (no index file), 401/403 (repo private or doesn't exist yet —
        HF returns 401 for an unauthenticated request to a missing/gated repo)
        as "no reports available to me" rather than a hard error. This keeps the
        auditor RUNNING (and, when weight-setting is enabled, falling back to the
        uid-0 burn) instead of crashing on EXIT_NETWORK while the validator
        hasn't published / made the repo public yet.
        """
        url = self._resolve_url(f"{_REPO_SUBDIR}/index.json")
        r = self._session.get(url, timeout=self._timeout)
        if r.status_code in (401, 403, 404):
            return []
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    def get_report(self, epoch_id: str) -> dict:
        """Return the full signed report envelope for one epoch_id."""
        url = self._resolve_url(f"{_REPO_SUBDIR}/{epoch_id}.json")
        r = self._session.get(url, timeout=self._timeout)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._session.close()


__all__ = ["DEFAULT_AUDIT_REPO", "ReportClient"]
