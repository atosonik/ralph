"""Hidden-eval fail-closed guard.

A validator whose held-out shard / benchmark is missing must NOT silently fall
back to random tokens + placeholder questions — that scores every checkpoint
against noise (val_bpb ~3.9, benchmark_accuracy ~1.0) and crowns the
least-trained model. By default run_hidden_eval raises; the synthetic fallback
is reachable only behind RALPH_ALLOW_SYNTHETIC_EVAL=1 (CPU smoke / testnet).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eval.hidden_eval import run_hidden_eval


class _Tiny(torch.nn.Module):
    def __init__(self, vocab=50257, dim=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab, dim)
        self.out = torch.nn.Linear(dim, vocab)

    def forward(self, x):
        return self.out(self.embed(x)), None


def _model():
    torch.manual_seed(0)
    return _Tiny()


def _write_tokens(path, n=4096):
    np.arange(n, dtype=np.uint16).tofile(path)


def test_missing_tokens_fails_closed(tmp_path, monkeypatch):
    """No active_tokens.bin + flag unset → raise, naming the missing shard."""
    monkeypatch.delenv("RALPH_ALLOW_SYNTHETIC_EVAL", raising=False)
    with pytest.raises(FileNotFoundError, match="held-out eval stream not found"):
        run_hidden_eval(_model(), tmp_path, seq_len=32)


def test_missing_benchmark_fails_closed(tmp_path, monkeypatch):
    """Tokens present but no active_benchmark.json + flag unset → still raise."""
    monkeypatch.delenv("RALPH_ALLOW_SYNTHETIC_EVAL", raising=False)
    _write_tokens(tmp_path / "active_tokens.bin")
    with pytest.raises(FileNotFoundError, match="held-out benchmark not found"):
        run_hidden_eval(_model(), tmp_path, seq_len=32)


def test_synthetic_opt_in_runs(tmp_path, monkeypatch):
    """With RALPH_ALLOW_SYNTHETIC_EVAL=1, the fallback runs (no shard needed)."""
    monkeypatch.setenv("RALPH_ALLOW_SYNTHETIC_EVAL", "1")
    res = run_hidden_eval(_model(), tmp_path, seq_len=32)
    assert res.val_bpb > 0
    assert res.val_seq_len == 32
