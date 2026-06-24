"""Tests for closing losing HF/GitHub PRs on terminal classification."""

import json

import validator.github_bot as github_bot
import validator.hf_bot as hf_bot
import validator.service as service


def _bundle(tmp_path, pr_num=42, pr_url="https://github.com/RalphLabsAI/recipe/pull/9"):
    d = tmp_path / "bundle"
    d.mkdir()
    (d / ".hf_pr.json").write_text(json.dumps(
        {"pr_num": pr_num, "repo_id": "RalphLabsAI/proof-bundles", "git_ref": f"refs/pr/{pr_num}"}
    ))
    (d / "submission.json").write_text(json.dumps({"pr_url": pr_url}))
    return d


def test_github_close_pr_noop_without_url():
    ok, detail = github_bot.close_pr("", token="t")
    assert ok is False and detail == "no pr_url"


def test_close_losing_prs_closes_both(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_CLOSE_LOSING_PRS", "1")
    monkeypatch.setenv("HF_TOKEN", "hf-tok")
    monkeypatch.setenv("RALPH_BOT_GH_TOKEN", "gh-tok")
    calls = {}

    def fake_hf_close(repo_id, pr_num, token, comment=None, repo_type="dataset"):
        calls["hf"] = (repo_id, pr_num, comment)
        return True, "closed"

    def fake_gh_close(pr_url, token, comment=None):
        calls["gh"] = (pr_url, comment)
        return True, "closed"

    monkeypatch.setattr(hf_bot, "close_pr", fake_hf_close)
    monkeypatch.setattr(github_bot, "close_pr", fake_gh_close)

    service._close_losing_prs(_bundle(tmp_path), "below threshold (gain -2.69)")

    assert calls["hf"][0] == "RalphLabsAI/proof-bundles"
    assert calls["hf"][1] == 42
    assert "below threshold" in calls["hf"][2]
    assert calls["gh"][0] == "https://github.com/RalphLabsAI/recipe/pull/9"
    assert "below threshold" in calls["gh"][1]


def test_close_losing_prs_disabled_flag(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_CLOSE_LOSING_PRS", "0")
    monkeypatch.setenv("HF_TOKEN", "hf-tok")
    called = {"n": 0}
    monkeypatch.setattr(hf_bot, "close_pr", lambda *a, **k: called.update(n=called["n"] + 1) or (True, "x"))
    service._close_losing_prs(_bundle(tmp_path), "reason")
    assert called["n"] == 0  # disabled → no close attempted


def test_close_losing_prs_noop_without_token(tmp_path, monkeypatch):
    monkeypatch.setenv("RALPH_CLOSE_LOSING_PRS", "1")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("RALPH_BOT_HF_TOKEN", raising=False)
    monkeypatch.delenv("RALPH_BOT_GH_TOKEN", raising=False)
    called = {"n": 0}
    monkeypatch.setattr(hf_bot, "close_pr", lambda *a, **k: called.update(n=called["n"] + 1) or (True, "x"))
    monkeypatch.setattr(github_bot, "close_pr", lambda *a, **k: called.update(n=called["n"] + 1) or (True, "x"))
    service._close_losing_prs(_bundle(tmp_path), "reason")
    assert called["n"] == 0  # no tokens → nothing attempted
