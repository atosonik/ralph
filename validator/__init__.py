"""
RESTRICTED — Validator client. Miners do not see this code at runtime.
"""

# Inject the sibling recipe repo onto sys.path before any submodule pulls in
# `from model import ...` / `from recipe.train import ...`.
import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
import ralph_bootstrap  # noqa: F401

from .scoring import ScoreReport, score_bundle
from .validator import (
    ValidatorReject,
    ValidatorResult,
    judge_submission,
)
from .version import VALIDATOR_VERSION

__all__ = [
    "ValidatorResult",
    "ValidatorReject",
    "judge_submission",
    "ScoreReport",
    "score_bundle",
    "VALIDATOR_VERSION",
]
