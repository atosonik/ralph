"""Open a karpaai/recipe PR from the miner's fork as part of submission.

The PR carries (a) the patch the miner is submitting, (b) the on-chain
metadata that ties it to the proof bundle on HuggingFace. Validators verify
the PR exists, is open, and that its diff byte-matches the bundle's
patch.diff before accepting the submission.

Required env vars on the miner box:
  KARPA_MINER_GH         — miner's GitHub username (e.g. "karpa1-gh")
  KARPA_MINER_GH_TOKEN   — PAT with `public_repo` scope; pushes to the
                           miner's fork, opens PR upstream.
  KARPA_RECIPE_FORK      — full URL of the miner's fork
                           (e.g. https://github.com/karpa1-gh/recipe.git)

Optional:
  KARPA_RECIPE_UPSTREAM  — defaults to "karpaai/recipe"
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

GH_API = "https://api.github.com"


def _gh_request(method: str, path: str, token: str, body: dict | None = None) -> dict:
    url = f"{GH_API}{path}" if path.startswith("/") else path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"github {method} {path} → {e.code}: {detail}") from None


def _run(cmd: list[str], cwd: Path) -> str:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"git command failed: {' '.join(cmd)}\nstdout: {r.stdout}\nstderr: {r.stderr}"
        )
    return r.stdout


def open_recipe_pr(
    patch_text: str,
    bundle_hash: str,
    miner_hotkey: str,
    miner_github: str,
    hf_bundle_url: str | None,
    signature_hex: str,
    fork_url: str,
    token: str,
    upstream: str = "karpaai/recipe",
) -> str:
    """Push a branch with the patch applied to the miner's fork, then open
    a PR upstream. Returns the PR URL.

    The branch name is deterministic: `submit/<bundle_hash[:12]>` so the
    same submission cannot accidentally collide with another, and so the
    validator can resolve the PR by bundle_hash alone if it has to.
    """
    if not token:
        raise RuntimeError("KARPA_MINER_GH_TOKEN is not set — cannot open PR")
    if not patch_text.strip():
        # Baseline submissions with an empty patch can't be a PR; skip.
        return ""

    short_hash = bundle_hash[:12]
    branch = f"submit/{short_hash}"
    title = f"[submit] {short_hash} — val by hotkey {miner_hotkey[:12]}…"
    body = "\n".join(
        line
        for line in (
            f"**bundle_hash:** `{bundle_hash}`",
            f"**miner_hotkey:** `{miner_hotkey}`",
            f"**miner_github:** @{miner_github}" if miner_github else None,
            f"**hf_bundle:** {hf_bundle_url}" if hf_bundle_url else None,
            f"**signature:** `{signature_hex[:32]}…`",
            "",
            "Submitted via `scripts/miner_run.py`. The validator will compare",
            "this PR's diff against the bundle's `patch.diff` byte-for-byte.",
        )
        if line is not None
    )

    workdir = Path(tempfile.mkdtemp(prefix="karpa_pr_"))
    try:
        # 1. Clone the fork
        _run(["git", "clone", "--depth=1", fork_url, str(workdir)], cwd=Path("/tmp"))

        # 2. Make sure we have upstream main + base off it (in case the fork is stale)
        upstream_url = f"https://github.com/{upstream}.git"
        _run(["git", "remote", "add", "upstream", upstream_url], cwd=workdir)
        _run(["git", "fetch", "--depth=1", "upstream", "main"], cwd=workdir)
        _run(["git", "checkout", "-B", branch, "upstream/main"], cwd=workdir)

        # 3. Apply the patch
        patch_path = workdir / ".karpa_submission.patch"
        patch_path.write_text(patch_text)
        _run(["git", "apply", "--whitespace=nowarn", str(patch_path)], cwd=workdir)
        patch_path.unlink()

        # 4. Commit
        _run(["git", "add", "-A"], cwd=workdir)
        _run(
            [
                "git",
                "-c", f"user.name={miner_github or 'karpa-miner'}",
                "-c", f"user.email={miner_github or 'miner'}@karpa.local",
                "commit",
                "-m", title,
                "-m", body,
            ],
            cwd=workdir,
        )

        # 5. Push to the miner's fork using token auth
        # Inject token into URL: https://<token>@github.com/<user>/recipe.git
        from urllib.parse import urlparse, urlunparse
        parsed = urlparse(fork_url)
        netloc = f"{token}@{parsed.netloc}" if parsed.username is None else parsed.netloc
        push_url = urlunparse(parsed._replace(netloc=netloc))
        _run(["git", "push", "--force-with-lease", push_url, branch], cwd=workdir)

        # 6. Open PR via REST API
        head_owner = parsed.path.lstrip("/").split("/")[0]
        pr = _gh_request(
            "POST",
            f"/repos/{upstream}/pulls",
            token,
            {
                "title": title,
                "body": body,
                "head": f"{head_owner}:{branch}",
                "base": "main",
                "maintainer_can_modify": True,
            },
        )
        return pr.get("html_url") or pr.get("url", "")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
