"""Smoke tests for Ralph-base — runnable on CPU."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import RalphBase, RalphConfig

import ralph_bootstrap  # noqa: F401  — injects RALPH_RECIPE_DIR


def test_forward_pass_runs_on_cpu():
    cfg = RalphConfig(
        vocab_size=512,
        dim=64,
        n_layers=2,
        n_heads=2,
        head_dim=32,
        max_seq_len=32,
    )
    model = RalphBase(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=targets)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss is not None
    assert loss.dim() == 0
    print(f"  cpu smoke ok, loss = {loss.item():.4f}")


def test_default_param_count_around_50M():
    cfg = RalphConfig()
    model = RalphBase(cfg)
    n = model.num_parameters()
    n_no_embed = model.num_parameters(exclude_embeddings=True)
    print(f"  default config: {n / 1e6:.1f}M params total, {n_no_embed / 1e6:.1f}M excl embeddings")
    assert 40e6 < n < 80e6, f"expected ~50M, got {n}"


def test_backward_pass():
    cfg = RalphConfig(
        vocab_size=512, dim=64, n_layers=2, n_heads=2, head_dim=32, max_seq_len=32
    )
    model = RalphBase(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    targets = torch.randint(0, cfg.vocab_size, (2, 16))
    _, loss = model(idx, targets=targets)
    loss.backward()
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for _ in model.parameters())
    assert has_grad == total, f"only {has_grad}/{total} params got gradients"
    print(f"  backward ok, all {total} params got non-zero gradients")


def test_rope_invariant_to_sequence_length():
    # Same content at different positions should produce different outputs.
    cfg = RalphConfig(
        vocab_size=512, dim=64, n_layers=2, n_heads=2, head_dim=32, max_seq_len=32
    )
    model = RalphBase(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        out1, _ = model(idx)
        # Same tokens, longer context.
        idx_long = torch.cat([torch.zeros(1, 4, dtype=torch.long), idx], dim=-1)
        out2, _ = model(idx_long)
    # Outputs at the same content positions should differ — RoPE saw different offsets.
    same_position_diff = (out1[0, -1] - out2[0, -1]).abs().mean().item()
    assert same_position_diff > 0.01, "RoPE seems not to be applied — outputs identical at different positions"
    print(f"  rope position sensitivity ok, mean abs diff = {same_position_diff:.4f}")


if __name__ == "__main__":
    print("Running CPU smoke tests for RalphBase...")
    test_forward_pass_runs_on_cpu()
    test_default_param_count_around_50M()
    test_backward_pass()
    test_rope_invariant_to_sequence_length()
    print("All smoke tests passed.")
