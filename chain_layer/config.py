"""
Chain backend configuration.

Set RALPH_CHAIN=bittensor to use real Bittensor chain, or
RALPH_CHAIN=local (default) for JSON-file testing.

Reads from .env file if present (never committed to git).
"""

from __future__ import annotations

import os
from pathlib import Path

from .interface import ChainInterface


def _load_dotenv(ralph_root: Path | None = None) -> None:
    """Load .env file into os.environ if it exists. Does not override
    existing env vars (explicit exports take precedence).

    Warns loudly if the .env file has group/other read permissions — the
    file contains tokens and wallet passwords; mode 0644 is a security
    hazard on shared hosts."""
    candidates = []
    if ralph_root:
        candidates.append(Path(ralph_root) / ".env")
    candidates.append(Path.cwd() / ".env")
    for env_path in candidates:
        if env_path.exists():
            try:
                mode = env_path.stat().st_mode & 0o777
                if mode & 0o077:
                    import sys as _sys
                    print(
                        f"[chain] WARNING: {env_path} is mode {oct(mode)} — "
                        f"contains secrets but is readable by group/other. "
                        f"Run: chmod 600 {env_path}",
                        file=_sys.stderr,
                    )
            except OSError:
                pass
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
            return


def get_chain(ralph_root: Path | None = None) -> ChainInterface:
    """Factory: returns the configured chain backend."""
    _load_dotenv(ralph_root)
    backend = os.environ.get("RALPH_CHAIN", "local")

    if backend == "bittensor":
        from .bittensor_chain import BittensorChain
        return BittensorChain(
            network=os.environ.get("BT_NETWORK", "test"),
            netuid=int(os.environ.get("BT_NETUID", "1")),
            wallet_name=os.environ.get("BT_WALLET", "default"),
            wallet_hotkey=os.environ.get("BT_HOTKEY", "default"),
            chain_dir=Path(ralph_root / "chain") if ralph_root else None,
        )
    else:
        from .local import LocalChain
        chain_dir = Path(ralph_root / "chain") if ralph_root else Path("chain")
        return LocalChain(chain_dir)
