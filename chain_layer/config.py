"""
Chain backend configuration.

Set KARPATHIAN_CHAIN=bittensor to use real Bittensor chain, or
KARPATHIAN_CHAIN=local (default) for JSON-file testing.
"""

from __future__ import annotations

import os
from pathlib import Path

from .interface import ChainInterface


def get_chain(karpathian_root: Path | None = None) -> ChainInterface:
    """Factory: returns the configured chain backend."""
    backend = os.environ.get("KARPATHIAN_CHAIN", "local")

    if backend == "bittensor":
        from .bittensor_chain import BittensorChain
        return BittensorChain(
            network=os.environ.get("BT_NETWORK", "test"),
            netuid=int(os.environ.get("BT_NETUID", "1")),
            wallet_name=os.environ.get("BT_WALLET", "default"),
            wallet_hotkey=os.environ.get("BT_HOTKEY", "default"),
            chain_dir=Path(karpathian_root / "chain") if karpathian_root else None,
        )
    else:
        from .local import LocalChain
        chain_dir = Path(karpathian_root / "chain") if karpathian_root else Path("chain")
        return LocalChain(chain_dir)
