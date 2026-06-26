"""Shared pytest fixtures for the Ralph test suite."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _allow_synthetic_eval():
    """Default the CPU test suite into the synthetic hidden-eval fallback.

    No real held-out shard / benchmark mix ships in-repo, so the hidden-eval
    would otherwise fail closed (the production default — see
    eval.hidden_eval._synthetic_eval_allowed). The test suite runs on CPU
    against the reproducible synthetic stream, same spirit as the mock
    attestation relaxation. Tests that assert the fail-closed behavior opt out
    with `monkeypatch.delenv("RALPH_ALLOW_SYNTHETIC_EVAL", raising=False)`.
    """
    prev = os.environ.get("RALPH_ALLOW_SYNTHETIC_EVAL")
    os.environ["RALPH_ALLOW_SYNTHETIC_EVAL"] = "1"
    yield
    if prev is None:
        os.environ.pop("RALPH_ALLOW_SYNTHETIC_EVAL", None)
    else:
        os.environ["RALPH_ALLOW_SYNTHETIC_EVAL"] = prev
