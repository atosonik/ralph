"""Single shared source-of-truth for container_measurement source files.

Imported by BOTH the miner-side proof runner and the validator. The two MUST
walk the same files in the same order or container_measurement diverges
across hosts and every honest verified-tier submission gets rejected at op2.

See deep_review_2026-05-31: critical #10/#11.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

# Files/extensions that contribute to the container_measurement.
# Keep tight: anything that affects training output must be here; nothing that
# varies per host (timestamps, lockfiles, secrets) should be.
_CONTRIBUTING_EXTS = {".py", ".yaml", ".yml", ".json", ".md"}
_RECIPE_DIRS = ("model", "recipe", "data", "configs")
_PROTOCOL_DIRS = ("eval", "calibration", "proof")
_PROTOCOL_FILES = ("restricted_files.yaml", "README.md")

# Repo-relative path prefixes that MUST NOT contribute to the measurement.
# eval/private/ holds the validator's SECRET held-out eval set (active_*.json):
# the validator has it, miners never do, so including it makes the validator's
# measurement impossible for any miner to reproduce — every honest verified-tier
# submission rejects at op2 with "container measurement mismatch". Excluding it
# lets both sides hash the identical shared tree. (eval/downstream/private_hard.py
# is kept — only the eval/private/ DIR is dropped.)
_EXCLUDED_REL_PREFIXES: tuple[tuple[str, ...], ...] = (("eval", "private"),)


def _is_excluded(rel: Path) -> bool:
    parts = rel.parts
    return any(parts[: len(pre)] == pre for pre in _EXCLUDED_REL_PREFIXES)


def list_proof_sources(
    ralph_root: Path,
    recipe_dir: Path | None = None,
) -> list[tuple[Path, Path]]:
    """Return a sorted list of (base, relative_path) tuples that contribute to
    the container_measurement.

    Sorted by repo + POSIX-relative path so the order is canonical regardless
    of filesystem layout.

    Args:
        ralph_root: the protocol repo root.
        recipe_dir: the recipe repo root. If None, expects the recipe dirs to
            live under ralph_root (legacy single-repo layout).
    """
    ralph_root = Path(ralph_root).resolve()
    pairs: list[tuple[Path, Path]] = []

    if recipe_dir is not None:
        recipe_dir = Path(recipe_dir).resolve()
        bases: Iterable[tuple[Path, tuple[str, ...]]] = (
            (recipe_dir, _RECIPE_DIRS),
            (ralph_root, _PROTOCOL_DIRS),
        )
    else:
        bases = ((ralph_root, _RECIPE_DIRS + _PROTOCOL_DIRS),)

    for base, dirs in bases:
        for d in dirs:
            root = base / d
            if not root.exists():
                continue
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                if "__pycache__" in p.parts:
                    continue
                if p.suffix not in _CONTRIBUTING_EXTS:
                    continue
                rel = p.relative_to(base)
                if _is_excluded(rel):
                    continue
                pairs.append((base, rel))

    for fname in _PROTOCOL_FILES:
        fp = ralph_root / fname
        if fp.exists():
            pairs.append((ralph_root, Path(fname)))

    # Canonical ordering: by POSIX relative path, ignoring which base it came from.
    pairs.sort(key=lambda x: x[1].as_posix())
    return pairs


def compute_container_measurement(
    ralph_root: Path,
    recipe_dir: Path | None = None,
) -> str:
    """Compute the container_measurement: a hash over the canonical source tree.

    Hashes repo-relative POSIX paths (not absolute paths). Two checkouts of the
    same content at different filesystem locations produce the same digest.
    """
    import hashlib

    h = hashlib.sha256()
    for base, rel in list_proof_sources(ralph_root, recipe_dir):
        h.update(rel.as_posix().encode("utf-8"))
        h.update(b"\x00")
        h.update((base / rel).read_bytes())
    return h.hexdigest()
