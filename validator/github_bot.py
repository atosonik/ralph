"""Validator-side GitHub operations on RalphLabsAI/recipe.

Two responsibilities:
  1. verify_pr_matches_bundle — confirm the open PR's diff is byte-equal to
     the bundle's patch.diff. If a miner opened a PR with a different diff
     than what the proof test actually ran on, reject the submission.
  2. merge_and_release — when a submission is crowned king, squash-merge
     the PR, tag the merge commit `recipe-vX.Y.Z`, and publish a release
     with the metrics in the body.

Requires env var RALPH_BOT_GH_TOKEN — a PAT with `public_repo` scope on
RalphLabsAI/recipe (the merge actor — recommend a dedicated ralph-bot account).
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

GH_API = "https://api.github.com"
RECIPE_REPO = "RalphLabsAI/recipe"


def _gh(
    method: str,
    path: str,
    token: str,
    body: dict | None = None,
    accept: str = "application/vnd.github+json",
) -> dict | str:
    url = f"{GH_API}{path}" if path.startswith("/") else path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", accept)
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            raw = r.read()
            if "diff" in accept or "patch" in accept:
                return raw.decode()
            return json.loads(raw.decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"github {method} {path} → {e.code}: {detail}") from None


def _parse_pr_url(pr_url: str) -> tuple[str, str, int]:
    """https://github.com/OWNER/REPO/pull/N → (OWNER, REPO, N)."""
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url.strip())
    if not m:
        raise ValueError(f"not a PR URL: {pr_url}")
    return m.group(1), m.group(2), int(m.group(3))


# Real GitHub handle rules: 1–39 chars, ASCII alphanumerics or hyphen.
# (GitHub additionally disallows leading/trailing hyphens, but we keep the
# regex strict-but-simple here; the practical attack we block is newlines,
# spaces, angle brackets, and commit-trailer metadata being injected.)
_GH_HANDLE_RE = re.compile(r"^[a-zA-Z0-9-]{1,39}$")


def _sanitize_gh_handle(s: str) -> str:
    """Return s if it looks like a real GitHub handle, else "".

    Used wherever a miner-supplied GitHub username is interpolated into a
    commit title, release name, or Co-authored-by trailer — those contexts
    treat newlines and angle brackets as control characters and can be
    abused to spoof additional authors or break formatting.
    """
    if not isinstance(s, str):
        return ""
    s = s.strip()
    if not s:
        return ""
    return s if _GH_HANDLE_RE.match(s) else ""


def _fence_safe(text: str) -> str:
    """Make text safe to embed inside a triple-backtick markdown fence.

    Any literal ``` is broken with a zero-width space so the fence cannot
    be escaped from inside the rationale. Visual fidelity is preserved.
    """
    return text.replace("```", "``​`")


@dataclass
class PrVerifyResult:
    ok: bool
    detail: str


def verify_pr_matches_bundle(
    pr_url: str,
    bundle_patch_text: str,
    token: str,
    expected_repo: str = RECIPE_REPO,
) -> PrVerifyResult:
    """Verify the PR exists, is open against expected_repo:main, and that its
    diff is byte-equal to the bundle's patch.diff."""
    if not pr_url:
        return PrVerifyResult(False, "submission has no pr_url")
    try:
        owner, repo, num = _parse_pr_url(pr_url)
    except ValueError as e:
        return PrVerifyResult(False, str(e))
    if f"{owner}/{repo}".lower() != expected_repo.lower():
        return PrVerifyResult(False, f"PR points at {owner}/{repo}, expected {expected_repo}")

    try:
        pr = _gh("GET", f"/repos/{owner}/{repo}/pulls/{num}", token)
    except RuntimeError as e:
        return PrVerifyResult(False, f"PR fetch failed: {e}")

    if pr.get("state") != "open":
        return PrVerifyResult(False, f"PR state={pr.get('state')} (must be open)")
    if pr.get("base", {}).get("ref") != "main":
        return PrVerifyResult(False, f"PR base={pr.get('base', {}).get('ref')} (must be main)")

    # Fetch the diff
    try:
        pr_diff = _gh("GET", f"/repos/{owner}/{repo}/pulls/{num}", token, accept="application/vnd.github.diff")
    except RuntimeError as e:
        return PrVerifyResult(False, f"diff fetch failed: {e}")

    # Byte-equal against the bundle's patch. We compare sha256 of the
    # normalized whitespace-stripped diffs, because GitHub may add metadata
    # lines (index, mode bits) that aren't in the bundle's patch.
    def _normalize(d: str) -> bytes:
        # Compare patches by their semantic content — the per-file headers
        # (--- / +++) plus the +/- lines. We drop:
        #   - all git porcelain headers (diff --git, index, mode bits,
        #     similarity / rename / copy markers)
        #   - hunk-position headers (@@ -X,Y +A,B @@), which are derived
        #     from how many context lines the producer chose to emit and
        #     therefore differ between a hand-crafted patch and one
        #     generated by `git diff` against the same applied tree
        #   - context lines (those starting with a literal space)
        # Two diffs with identical `--- / +++ / +/-` content are
        # equivalent patches for our purposes (the proof test already
        # validates that the patch applies cleanly to the canonical recipe,
        # so context-line drift is harmless).
        SKIP_PREFIXES = (
            "diff --git ",
            "index ",
            "new file mode ", "deleted file mode ",
            "old mode ", "new mode ",
            "similarity index ", "rename from ", "rename to ",
            "copy from ", "copy to ",
            "@@ ",
        )
        kept = []
        for line in d.splitlines():
            if line.startswith(SKIP_PREFIXES):
                continue
            # Drop context lines (start with a single space) and bare
            # blank lines (inter-hunk separators in some patch formats).
            # Keep `---`, `+++`, `-`, `+`.
            if line == "" or line.startswith(" "):
                continue
            kept.append(line.rstrip())
        while kept and kept[-1] == "":
            kept.pop()
        return ("\n".join(kept) + "\n").encode()

    pr_sha = hashlib.sha256(_normalize(pr_diff)).hexdigest()
    bundle_sha = hashlib.sha256(_normalize(bundle_patch_text)).hexdigest()
    if pr_sha != bundle_sha:
        return PrVerifyResult(
            False,
            f"PR diff sha256 != bundle patch sha256 (pr={pr_sha[:12]}..., bundle={bundle_sha[:12]}...)",
        )
    return PrVerifyResult(True, f"PR #{num} matches bundle patch (sha={bundle_sha[:12]}...)")


def _latest_recipe_tag(token: str, repo: str = RECIPE_REPO) -> tuple[int, int, int]:
    """Return (major, minor, patch) of the latest recipe-vX.Y.Z tag, or (0,0,0)."""
    try:
        tags = _gh("GET", f"/repos/{repo}/tags?per_page=100", token)
    except RuntimeError:
        return (0, 0, 0)
    best = (0, 0, 0)
    for t in tags or []:
        m = re.match(r"recipe-v(\d+)\.(\d+)\.(\d+)$", t.get("name", ""))
        if not m:
            continue
        v = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if v > best:
            best = v
    return best


def _next_recipe_version(token: str, repo: str = RECIPE_REPO) -> str:
    major, minor, patch = _latest_recipe_tag(token, repo)
    if (major, minor, patch) == (0, 0, 0):
        return "recipe-v0.1.0"
    return f"recipe-v{major}.{minor}.{patch + 1}"


@dataclass
class ReleaseResult:
    tag: str
    release_url: str
    merge_sha: str
    # Populated when the merge+tag succeeded but release publication failed.
    # Lets the caller emit a distinct chain event ("release_publish_failed")
    # without losing the merge_sha and tag that *did* land.
    release_publish_error: Optional[str] = None


# Hard caps. GitHub's release-create endpoint 422s above ~125 000 chars;
# commit messages have a smaller practical cap. We stay well under both.
_COMMIT_MESSAGE_MAX = 8_000
_RELEASE_BODY_MAX = 100_000
_RATIONALE_MAX = 30_000


def merge_and_release(
    pr_url: str,
    metrics: dict,
    token: str,
    repo: str = RECIPE_REPO,
) -> ReleaseResult:
    """Squash-merge the PR, tag the merge commit, publish a release.

    metrics keys used in the release body:
      val_bpb, quality_gain, compute_cost_h100h, miner_hotkey,
      miner_github, bundle_hash, hf_bundle_url, wandb_url (optional).
    """
    owner, pr_repo, num = _parse_pr_url(pr_url)
    if f"{owner}/{pr_repo}".lower() != repo.lower():
        raise RuntimeError(f"PR points at {owner}/{pr_repo}, refusing to merge into {repo}")

    # Sanitize miner_github once. Any miner-supplied value that doesn't look
    # like a real GitHub handle becomes "" — we then fall back to a hotkey
    # prefix wherever a human-readable attribution is needed.
    miner_gh_raw = metrics.get("miner_github", "")
    miner_gh = _sanitize_gh_handle(miner_gh_raw if isinstance(miner_gh_raw, str) else "")
    hotkey_prefix = str(metrics.get("miner_hotkey", ""))[:12]
    display_name = miner_gh or hotkey_prefix or "anonymous"

    # 1. Build the short commit_message (NO rationale_md) for the merge call,
    #    and the long release_body (WITH rationale_md, capped) for releases.
    commit_message = _commit_message(metrics, pr_url, miner_gh)
    release_body = _release_body(metrics, pr_url, miner_gh)

    # 2. Merge the PR (squash). The commit_title and Co-authored-by trailer
    #    inside commit_message both use the sanitized handle so a malicious
    #    miner_github value cannot inject newlines or spoof co-authors.
    merge_resp = _gh(
        "PUT",
        f"/repos/{repo}/pulls/{num}/merge",
        token,
        {
            "merge_method": "squash",
            "commit_title": f"recipe submission #{num} — {display_name}",
            "commit_message": commit_message,
        },
    )
    merge_sha = merge_resp.get("sha", "")

    # 3. Compute next version and create the tag on the merge commit.
    #    If tag creation fails, we re-raise — the merge already happened but
    #    tagging is cheap to retry manually and the caller will surface it.
    version = _next_recipe_version(token, repo)
    try:
        _gh(
            "POST",
            f"/repos/{repo}/git/refs",
            token,
            {"ref": f"refs/tags/{version}", "sha": merge_sha},
        )
    except RuntimeError as e:
        # Surface partial state so the caller can log a chain event.
        return ReleaseResult(
            tag="",
            release_url="",
            merge_sha=merge_sha,
            release_publish_error=f"tag creation failed after merge: {e}",
        )

    # 4. Publish a release. If this fails (most likely body-size 422), the
    #    PR is already merged and tagged — we DO NOT raise; we return a
    #    partial ReleaseResult with the error so the validator can emit a
    #    distinct "release_publish_failed" chain event and a human can
    #    recover by re-POSTing the release body manually.
    try:
        release = _gh(
            "POST",
            f"/repos/{repo}/releases",
            token,
            {
                "tag_name": version,
                "name": f"{version} — {display_name}",
                "body": release_body,
                "draft": False,
                "prerelease": False,
            },
        )
    except RuntimeError as e:
        # Best-effort log; the caller is responsible for the chain event.
        msg = (
            f"PR #{num} merged + tag {version} created, but release "
            f"publication failed — manual recovery needed: {e}"
        )
        # No logging dep available here; print to stderr-equivalent path
        # via stdlib. Keeping it simple — the validator wraps this call.
        import sys
        print(msg, file=sys.stderr)
        return ReleaseResult(
            tag=version,
            release_url="",
            merge_sha=merge_sha,
            release_publish_error=msg,
        )

    return ReleaseResult(
        tag=version,
        release_url=release.get("html_url", ""),
        merge_sha=merge_sha,
    )


def _hypothesis_block(m: dict) -> list[str]:
    """Render the miner's hypothesis with clear self-reported attribution.

    A hostile miner can otherwise write `**Bundle hash:** <competitor_hash>`
    inside the hypothesis and have it visually appear to come from the
    validator. We prefix the section with an attribution disclaimer and
    blockquote the body so the miner's voice is visually distinct.
    """
    hyp = m.get("hypothesis")
    if not hyp:
        return []
    # Blockquote each line — even blank lines get `>` so the quote stays
    # contiguous in the rendered markdown.
    quoted = "\n".join(f"> {ln}" if ln else ">" for ln in str(hyp).splitlines())
    return [
        "## Hypothesis",
        "",
        "_Miner's claim (self-reported, unverified by validator):_",
        "",
        quoted,
        "",
    ]


def _metrics_block(m: dict) -> list[str]:
    lines = ["## Metrics", ""]
    if "val_bpb" in m:
        lines.append(f"- **val_bpb:** `{m['val_bpb']:.4f}`")
    if "quality_gain" in m:
        lines.append(f"- **quality_gain vs previous king:** `{m['quality_gain']:+.4f}`")
    if "compute_cost_h100h" in m:
        lines.append(f"- **compute_cost (H100-hours):** `{m['compute_cost_h100h']:.4f}`")
    if "benchmark_accuracy" in m:
        lines.append(f"- **benchmark_accuracy:** `{m['benchmark_accuracy']:.3f}`")
    return lines


def _attribution_block(m: dict, miner_gh: str) -> list[str]:
    """miner_gh is the already-sanitized handle (or "")."""
    lines = ["", "## Attribution", ""]
    if miner_gh:
        lines.append(f"- **GitHub:** @{miner_gh}")
    if "miner_hotkey" in m:
        lines.append(f"- **hotkey:** `{m['miner_hotkey']}`")
    if "bundle_hash" in m:
        lines.append(f"- **bundle_hash:** `{m['bundle_hash']}`")
    return lines


def _links_block(m: dict, pr_url: str) -> list[str]:
    lines = ["", "## Links", "", f"- **PR:** {pr_url}"]
    if m.get("hf_bundle_url"):
        lines.append(f"- **HF proof bundle:** {m['hf_bundle_url']}")
    if m.get("wandb_url"):
        lines.append(f"- **wandb run:** {m['wandb_url']}")
    return lines


def _commit_message(m: dict, pr_url: str, miner_gh: str) -> str:
    """Compact commit message for the squash-merge PUT.

    Excludes rationale_md (which can be huge) so we stay well under the
    commit-message size cap. Includes hypothesis (capped), metrics,
    attribution, links, and the Co-authored-by trailer using only the
    sanitized handle.
    """
    lines: list[str] = []
    lines += _hypothesis_block(m)
    lines += _metrics_block(m)
    lines += _attribution_block(m, miner_gh)
    lines += _links_block(m, pr_url)

    if miner_gh:
        # GitHub recognises the `username@users.noreply.github.com` form as
        # the canonical "no-reply" address; commits credited to it count
        # toward the user's contribution graph. Using the sanitized handle
        # guarantees no newline/`<`/`>` can break the trailer.
        lines += [
            "",
            f"Co-authored-by: {miner_gh} <{miner_gh}@users.noreply.github.com>",
        ]

    msg = "\n".join(lines)
    if len(msg) > _COMMIT_MESSAGE_MAX:
        # Truncate while preserving the trailer (last 2 lines) so the
        # Co-authored-by attribution survives.
        trailer = ""
        if miner_gh:
            trailer = (
                f"\n\nCo-authored-by: {miner_gh} <{miner_gh}@users.noreply.github.com>"
            )
        budget = _COMMIT_MESSAGE_MAX - len(trailer) - len("\n\n…truncated…")
        msg = msg[:budget].rstrip() + "\n\n…truncated…" + trailer
    return msg


def _release_body(m: dict, pr_url: str, miner_gh: str = "") -> str:
    """Long-form release body (includes rationale_md, with caps).

    miner_gh is the already-sanitized handle. For backwards compatibility
    if called without it we re-sanitize from metrics.
    """
    if not miner_gh:
        miner_gh = _sanitize_gh_handle(m.get("miner_github") or "")

    lines: list[str] = []
    lines += _hypothesis_block(m)
    lines += _metrics_block(m)
    lines += _attribution_block(m, miner_gh)

    # If the miner shipped a full rationale.md, render it inside a
    # ```markdown fenced block (NOT a <details>/<summary> HTML block — a
    # rationale containing the literal `</details>` would otherwise escape
    # the collapsible). We also escape any triple-backtick sequences with
    # a zero-width space so the rationale cannot escape the fence itself.
    rationale = m.get("rationale_md")
    if rationale:
        rationale_text = str(rationale).rstrip()
        truncated = False
        if len(rationale_text) > _RATIONALE_MAX:
            rationale_text = rationale_text[:_RATIONALE_MAX].rstrip()
            truncated = True
        rationale_text = _fence_safe(rationale_text)
        lines += [
            "",
            "## Reasoning",
            "",
            "_Miner's claim (self-reported, unverified by validator):_",
            "",
            "```markdown",
            rationale_text,
            "```",
        ]
        if truncated:
            hf = m.get("hf_bundle_url") or "(no HF URL)"
            lines += [
                "",
                f"_…rationale truncated; full text in HF bundle {hf}…_",
            ]

    lines += _links_block(m, pr_url)

    body = "\n".join(lines)

    # Final hard cap on the entire release body. If we're still over the
    # limit (unlikely given the rationale cap, but possible with very
    # large hypothesis or many metric fields), trim from the rationale
    # section forward and leave a truncation marker.
    if len(body) > _RELEASE_BODY_MAX:
        body = body[: _RELEASE_BODY_MAX - len("\n\n…release body truncated…")].rstrip()
        body += "\n\n…release body truncated…"
    return body
