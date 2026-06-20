"""Optional auditor-side counter-weight (off by default).

When AUDITOR_SET_WEIGHTS_ENABLED is set, a registered auditor-validator sets its
OWN weights on netuid 40 from the independently-replayed scores, shadowing a
dishonest validator. Port of greencompute-audit/audit/weights.py, adapted to the
Ralph `bittensor` SDK (a Ralph dependency) instead of legacy substrate-interface.

KEY SECURITY PROPERTY — the auditor uses ITS OWN wallet, never the scored validator's:
  * You pass wallet IDENTIFIERS (coldkey/hotkey NAMES) via env vars, never the
    secret material. The bittensor wallet on disk is read at runtime.
  * Names-in-env: AUDITOR_WALLET_NAME / AUDITOR_WALLET_HOTKEY.
  * Disabled by default. Read-only verification (Gates 1-3) needs no keys.

Nothing is transmitted off-box; the validator being audited never sees the
auditor's wallet.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("ralph-auditor.weights")


# Counter-weight cadence: a validator must keep setting weights every epoch or
# its weights go stale and vTrust decays. The subnet enforces a minimum gap
# (weights_rate_limit ≈ 100 blocks on netuid 40); we default comfortably above it.
DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS = 300  # ≈ 1h at 12s/block


def is_enabled() -> bool:
    return os.environ.get("AUDITOR_SET_WEIGHTS_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def weight_set_interval_blocks() -> int:
    """Blocks between counter-weight sets (env AUDITOR_WEIGHT_INTERVAL_BLOCKS,
    default 300). Invalid/non-positive values fall back to the default."""
    raw = os.environ.get("AUDITOR_WEIGHT_INTERVAL_BLOCKS", "").strip()
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS


def is_weight_set_due(blocks_since: int | None, interval_blocks: int) -> bool:
    """True if the auditor should (re)set weights now.

    `blocks_since` is blocks elapsed since the auditor's hotkey last set weights
    (None = never set / unknown → due). Due once at least `interval_blocks` have
    elapsed. This is what makes counter-weighting a continuous epoch-cadence
    process rather than a one-shot.
    """
    if interval_blocks <= 0:
        raise ValueError(f"interval_blocks must be > 0; got {interval_blocks}")
    if blocks_since is None:
        return True
    return blocks_since >= interval_blocks


def auditor_hotkey_ss58() -> str | None:
    """The auditor's OWN hotkey ss58, read-only, for cadence queries
    (blocks-since-last-weight-set). None if no wallet is configured."""
    wallet = _load_wallet()
    if wallet is None:
        return None
    try:
        return wallet.hotkey.ss58_address
    except Exception:
        return None


def _load_wallet():
    """Load the auditor's OWN bittensor wallet from env-supplied NAMES.

    AUDITOR_WALLET_NAME is required; AUDITOR_WALLET_HOTKEY defaults to "default".
    Returns None when AUDITOR_WALLET_NAME is unset (caller treats as read-only).
    """
    name = os.environ.get("AUDITOR_WALLET_NAME", "").strip()
    hotkey = os.environ.get("AUDITOR_WALLET_HOTKEY", "default").strip()
    if not name:
        logger.warning(
            "AUDITOR_SET_WEIGHTS_ENABLED set but AUDITOR_WALLET_NAME unset — "
            "staying read-only (no counter-weight)."
        )
        return None
    import bittensor as bt

    return bt.Wallet(name=name, hotkey=hotkey)


def submit_weights(
    subtensor_url: str,
    netuid: int,
    weights_by_hotkey: dict[str, float],
) -> bool:
    """Set the auditor's own weights on `netuid` from the replayed scores.

    Maps each audited hotkey -> uid via the metagraph (dropping any not in it),
    normalizes, and submits a set_weights extrinsic signed by the auditor's OWN
    wallet. Returns True on success. Never raises into the audit loop.
    """
    wallet = _load_wallet()
    if wallet is None:
        return False
    if not weights_by_hotkey:
        logger.info("no replayed weights to publish; skipping counter-weight")
        return False

    try:
        import bittensor as bt

        subtensor = bt.Subtensor(network=subtensor_url)
        metagraph = subtensor.metagraph(netuid=netuid)
    except Exception:
        logger.exception("failed to connect subtensor / sync metagraph at %s", subtensor_url)
        return False

    try:
        hotkeys = list(metagraph.hotkeys)
        auditor_ss58 = wallet.hotkey.ss58_address
        if auditor_ss58 not in hotkeys:
            logger.warning(
                "auditor hotkey %s not registered on netuid=%d — cannot set "
                "weights (register first).",
                auditor_ss58,
                netuid,
            )
            return False

        import torch

        uids: list[int] = []
        vals: list[float] = []
        for hk, w in sorted(weights_by_hotkey.items()):
            if hk not in hotkeys:
                logger.info("hotkey %s not in metagraph; skipping", hk)
                continue
            uids.append(hotkeys.index(hk))
            vals.append(max(0.0, float(w)))
        if not uids:
            logger.warning("no audited hotkeys mapped to UIDs; nothing to publish")
            return False

        total = sum(vals) or 1.0
        vals = [v / total for v in vals]

        result = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=torch.tensor(uids, dtype=torch.int64),
            weights=torch.tensor(vals, dtype=torch.float32),
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        success = result.success if hasattr(result, "success") else bool(result)
        logger.info(
            "auditor counter-weight set_weights: success=%s (uids=%d, ss58=%s)",
            success,
            len(uids),
            auditor_ss58,
        )
        return bool(success)
    except Exception:
        logger.exception("auditor set_weights extrinsic failed")
        return False
    finally:
        try:
            subtensor.close()
        except Exception:
            pass


__all__ = [
    "DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS",
    "auditor_hotkey_ss58",
    "is_enabled",
    "is_weight_set_due",
    "submit_weights",
    "weight_set_interval_blocks",
]
