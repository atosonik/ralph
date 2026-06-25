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


# Counter-weight cadence. Preferred mode (SN51-style): set weights ~`lead`
# blocks BEFORE the subnet's tempo (Yuma consensus) boundary, so the auditor's
# weights are maximally fresh exactly when consensus consumes them — the
# strongest lever on vTrust=1.0. The flat block interval below is only the
# fallback used when the subnet tempo can't be read.
DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS = 300  # fallback when tempo is unavailable
DEFAULT_SET_LEAD_BLOCKS = 2  # set ~2 blocks before the tempo boundary (SN51)


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


def set_lead_blocks() -> int:
    """How many blocks before the tempo boundary to set weights (env
    AUDITOR_SET_LEAD_BLOCKS, default 2). Negative/invalid → default."""
    raw = os.environ.get("AUDITOR_SET_LEAD_BLOCKS", "").strip()
    if raw:
        try:
            v = int(raw)
            if v >= 0:
                return v
        except ValueError:
            pass
    return DEFAULT_SET_LEAD_BLOCKS


def blocks_until_next_epoch(block: int, netuid: int, tempo: int) -> int:
    """Blocks remaining until this subnet's next tempo (Yuma consensus) boundary,
    using the chain's own boundary formula offset by netuid (identical to SN51):

        blocks_left = tempo - (block + netuid + 1) % (tempo + 1)

    Result is in [0, tempo]; 0 == on the boundary this block.
    """
    if tempo <= 0:
        return 0
    return tempo - (block + netuid + 1) % (tempo + 1)


def is_weight_set_due_tempo(
    blocks_left: int | None,
    blocks_since: int | None,
    tempo: int | None,
    lead: int,
) -> bool:
    """SN51-style boundary timing for the auditor's counter-weight set.

    Due when within `lead` blocks of the next tempo boundary, so the weights are
    freshest when consensus evaluates them (best lever on vTrust=1.0). Plus two
    safeties so vTrust can never decay:
      - never-set yet (`blocks_since` None) → due immediately;
      - dead-man: more than 2 full tempos since the last set → due regardless of
        phase (covers a missed boundary / RPC gap).
    """
    if blocks_since is None:
        return True
    if tempo and blocks_since > tempo * 2:
        return True
    if blocks_left is not None and blocks_left <= lead:
        return True
    return False


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


def submit_burn_weights(
    subtensor_url: str,
    netuid: int,
    burn_uid: int | None = None,
) -> bool:
    """Set the auditor's own weights to 100% on the burn uid (default 0 = owner).

    Used when there is NO clean audit epoch to replay (e.g. the audit-reports
    repo is empty / 404) so the auditor-validator still sets weights every
    cadence — keeps its vTrust alive and burns to the owner uid instead of
    silently skipping. Signed by the auditor's OWN wallet. Never raises.
    """
    if burn_uid is None:
        burn_uid = int(os.environ.get("RALPH_BURN_UID", "0"))
    wallet = _load_wallet()
    if wallet is None:
        return False
    try:
        import bittensor as bt
        import torch

        subtensor = bt.Subtensor(network=subtensor_url)
        metagraph = subtensor.metagraph(netuid=netuid)
    except Exception:
        logger.exception("burn: failed to connect subtensor at %s", subtensor_url)
        return False
    try:
        auditor_ss58 = wallet.hotkey.ss58_address
        if auditor_ss58 not in list(metagraph.hotkeys):
            logger.warning(
                "burn: auditor hotkey %s not registered on netuid=%d — cannot set weights",
                auditor_ss58, netuid,
            )
            return False
        result = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=torch.tensor([burn_uid], dtype=torch.int64),
            weights=torch.tensor([1.0], dtype=torch.float32),
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )
        success = result.success if hasattr(result, "success") else bool(result)
        logger.info("auditor BURN set_weights -> uid %d: success=%s", burn_uid, success)
        return bool(success)
    except Exception:
        logger.exception("auditor burn set_weights failed")
        return False
    finally:
        try:
            subtensor.close()
        except Exception:
            pass


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
    "DEFAULT_SET_LEAD_BLOCKS",
    "DEFAULT_WEIGHT_SET_INTERVAL_BLOCKS",
    "auditor_hotkey_ss58",
    "blocks_until_next_epoch",
    "is_enabled",
    "is_weight_set_due",
    "is_weight_set_due_tempo",
    "set_lead_blocks",
    "submit_burn_weights",
    "submit_weights",
    "weight_set_interval_blocks",
]
