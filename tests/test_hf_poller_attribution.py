"""Tests for per-PR bundle attribution in the HF poller.

list_repo_files(revision="refs/pr/N") returns the cumulative tree (base main +
the PR's files), so the poller must isolate the bundle each PR actually adds —
primarily via the PR title (`Submit proof bundle <hash12>`), falling back to a
diff against main. These tests pin that behaviour with a fake HfApi.
"""

from datetime import datetime, timezone

import huggingface_hub

from validator.hf_poller import list_remote_submissions

A = "a" * 64  # already merged on main
B = "b" * 64  # added by PR 1
C = "c" * 64  # added by PR 2
D = "d" * 64  # added by PR 3
E = "e" * 64  # added by PR 4 (non-standard title → fallback)
X = "9" * 64  # in PR 3's cumulative tree but NOT on main (e.g. reverted mock)


def _files(*bundle_ids):
    out = ["README.md", ".gitattributes"]
    for bid in bundle_ids:
        out.append(f"submissions/{bid}/submission.json")
        out.append(f"submissions/{bid}/checkpoint.pt")
    return out


class _FakeDiscussion:
    def __init__(self, num, title, created, status="open", is_pr=True):
        self.num = num
        self.title = title
        self.status = status
        self.is_pull_request = is_pr
        self.git_reference = f"refs/pr/{num}"
        self.created_at = created


class _FakeApi:
    def __init__(self, discussions, trees):
        self._discussions = discussions
        self._trees = trees  # {None: main_files, "refs/pr/N": files}

    def get_repo_discussions(self, repo_id=None, repo_type=None):
        return iter(self._discussions)

    def list_repo_files(self, repo_id, repo_type=None, revision=None, token=None):
        return self._trees[revision]


def _install(monkeypatch, discussions, trees):
    api = _FakeApi(discussions, trees)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: api)


def test_attributes_only_each_prs_own_bundle(monkeypatch):
    def t(n):
        return datetime(2026, 6, n, tzinfo=timezone.utc)

    discussions = [
        _FakeDiscussion(2, f"Submit proof bundle {C[:12]}", t(2)),
        _FakeDiscussion(1, f"Submit proof bundle {B[:12]}", t(1)),  # out of order on purpose
        _FakeDiscussion(3, f"Submit proof bundle {D[:12]}", t(3)),
        _FakeDiscussion(4, "rebrand: tidy dataset", t(4)),          # no hash → fallback
        _FakeDiscussion(99, "an issue, not a PR", t(1), is_pr=False),
        _FakeDiscussion(98, f"Submit proof bundle {C[:12]}", t(1), status="closed"),
    ]
    trees = {
        None: _files(A),               # main
        "refs/pr/1": _files(A, B),     # cumulative: base + own
        "refs/pr/2": _files(A, B, C),
        "refs/pr/3": _files(A, X, D),  # X present here but gone from main
        "refs/pr/4": _files(A, E),     # title carries no hash
    }
    _install(monkeypatch, discussions, trees)

    subs = list_remote_submissions("RalphLabsAI/proof-bundles")

    # Oldest-first, one entry per open PR, each its own bundle.
    assert [(s["pr_num"], s["bundle_id"]) for s in subs] == [
        (1, B), (2, C), (3, D), (4, E),
    ]
    seen = {s["bundle_id"] for s in subs}
    assert A not in seen  # merged baseline never re-attributed
    assert X not in seen  # deleted-from-main bundle not mis-attributed to PR 3


def test_fallback_skips_pr_with_no_identifiable_bundle(monkeypatch):
    # Non-standard title AND nothing new vs main → nothing to attribute.
    discussions = [_FakeDiscussion(7, "housekeeping", datetime(2026, 6, 7, tzinfo=timezone.utc))]
    trees = {None: _files(A), "refs/pr/7": _files(A)}
    _install(monkeypatch, discussions, trees)
    assert list_remote_submissions("RalphLabsAI/proof-bundles") == []
