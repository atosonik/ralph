"""Cross-repo karpa -> ralph rebrand compatibility (2026-06).

Runs in protocol CI (which clones the sibling recipe repo at
RALPH_RECIPE_DIR). Guards the contract that survived the class rename:

  1. `ralph_bootstrap` resolves the recipe sibling and `from model import
     RalphBase, RalphConfig` works (canonical names).
  2. The back-compat aliases `KarpaBase`/`KarpaConfig` still import and are
     the *same objects* — so any unmigrated importer keeps working.
  3. A checkpoint saved in the canonical `{"model": ..., "config":
     asdict(cfg)}` form round-trips and serializes NO class-name string,
     proving the rename is checkpoint-safe.
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import KarpaBase, KarpaConfig, RalphBase, RalphConfig  # noqa: E402

import ralph_bootstrap  # noqa: F401, E402  — injects RALPH_RECIPE_DIR


def test_aliases_are_canonical_classes():
    assert KarpaConfig is RalphConfig
    assert KarpaBase is RalphBase


def test_checkpoint_roundtrips_under_renamed_classes(tmp_path):
    cfg = RalphConfig(vocab_size=256, dim=32, n_layers=2, n_heads=2,
                      head_dim=16, max_seq_len=32, tie_embeddings=True)
    model = RalphBase(cfg)
    ckpt = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "config": asdict(cfg)}, ckpt)

    raw = ckpt.read_bytes()
    for needle in (b"KarpaBase", b"KarpaConfig", b"RalphBase", b"RalphConfig"):
        assert needle not in raw

    blob = torch.load(ckpt, weights_only=True, map_location="cpu")
    cfg_kwargs = {k: v for k, v in blob["config"].items()
                  if k in RalphConfig.__dataclass_fields__}
    # Rebuild via canonical class and via the back-compat alias.
    for klass in (RalphBase, KarpaBase):
        rebuilt = klass(RalphConfig(**cfg_kwargs))
        rebuilt.load_state_dict(blob["model"])
        assert rebuilt.num_parameters() == model.num_parameters()
