"""
Chain abstraction layer.

Provides a uniform interface for chain operations across backends:
  - LocalChain: JSON-file ledger (Phase 0 / testing)
  - BittensorChain: real Bittensor testnet/mainnet (Phase 0.5d+)

The rest of the codebase (miner, validator, router) calls the interface;
the backend is swapped by configuration, not by code changes.
"""

from .interface import ChainInterface
from .local import LocalChain
from .bittensor_chain import BittensorChain

__all__ = ["ChainInterface", "LocalChain", "BittensorChain"]
