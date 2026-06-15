"""Validator-side merge of winning HF PRs on RalphLabsAI/proof-bundles.

Mirrors validator/github_bot.py but for HuggingFace dataset PRs. The miner
opens a community PR via `create_pr=True`; on king change the bot merges it
so the canonical bundle ends up on `main`.

Requires an HF token with merge rights on the dataset (org admin / write
collaborator). For testnet we can reuse the validator's own HF_TOKEN if the
operator's account is the org admin; for cleaner separation set
RALPH_BOT_HF_TOKEN to a token from a dedicated ralph-bot HF account that's
a Write member of the RalphLabsAI org.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HfMergeResult:
    pr_num: int
    merged: bool
    detail: str


def merge_pr(
    repo_id: str,
    pr_num: int,
    token: str,
    comment: str | None = None,
) -> HfMergeResult:
    """Merge the open PR via huggingface_hub. Returns a HfMergeResult.

    Failure is non-fatal — we surface the reason so the validator can log it
    and the operator can merge manually.
    """
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    try:
        if comment:
            api.comment_discussion(
                repo_id=repo_id, repo_type="dataset",
                discussion_num=pr_num, comment=comment,
            )
        api.merge_pull_request(
            repo_id=repo_id, repo_type="dataset",
            discussion_num=pr_num,
            comment=comment or "Auto-merged by Ralph validator (king crowned).",
        )
        return HfMergeResult(pr_num=pr_num, merged=True, detail="merged")
    except Exception as e:
        return HfMergeResult(pr_num=pr_num, merged=False, detail=f"merge failed: {e}")
