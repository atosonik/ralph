"""Bootstrap module — adds the karpaai/recipe sibling repo to sys.path so the
protocol code can `import model`, `from recipe.train ...`, etc., even though
the recipe lives in a separate repository.

Resolution order for the recipe directory:
  1. $KARPA_RECIPE_DIR if set
  2. ../recipe relative to this file (the sibling layout used in development)
  3. ./recipe relative to this file (fallback if someone vendored it back in)

Usage: at the top of any entry point — scripts, services, test runners —
add a single line:

    import karpa_bootstrap  # noqa: F401

The import has the side effect of inserting the recipe path into sys.path
and exposes RECIPE_DIR for tools that need the path itself.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parent.parent / "recipe"
_FALLBACK = Path(__file__).resolve().parent / "recipe"


def _resolve_recipe_dir() -> Path:
    env = os.environ.get("KARPA_RECIPE_DIR")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p
        raise RuntimeError(
            f"KARPA_RECIPE_DIR={env} does not exist. "
            "Clone karpaai/recipe and either point KARPA_RECIPE_DIR at it, "
            "or place it as a sibling of the karpa repo."
        )
    if _DEFAULT.exists():
        return _DEFAULT
    if _FALLBACK.exists():
        return _FALLBACK
    raise RuntimeError(
        f"Could not locate the recipe repo. Looked at "
        f"{_DEFAULT} and {_FALLBACK}. Either clone karpaai/recipe to "
        f"{_DEFAULT}, or set $KARPA_RECIPE_DIR to its path."
    )


RECIPE_DIR: Path = _resolve_recipe_dir()


def _inject() -> None:
    p = str(RECIPE_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


_inject()
