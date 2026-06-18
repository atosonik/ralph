"""Read the validator's on-chain audit commitment at a historical block.

Mirrors greencompute-audit/audit/chain.py. The validator commits the 64-hex
sha256 of the canonical report_json via `set_commitment` (see
chain_layer/bittensor_chain.py:commit_audit_root) — that commitment is the
trust anchor. Because each epoch's commitment OVERWRITES the previous one, the
auditor MUST query state at the report's historical `epoch_end_block`, which
requires an **archive** subtensor endpoint (lite nodes prune old state):

    wss://archive.chain.opentensor.ai:443/   (free, default)

We use the `bittensor` SDK (already a Ralph dependency) rather than the legacy
`substrateinterface` package the greencompute reference used. `get_commitment`
resolves a hotkey -> uid -> Commitments.CommitmentOf(netuid, hotkey) at the
given block_hash and decodes the Raw bytes back to the committed string.
"""

from __future__ import annotations

ARCHIVE_ENDPOINT_DEFAULT = "wss://archive.chain.opentensor.ai:443/"


class ChainClient:
    """Queries on-chain commitments from an archive subtensor.

    Pin `validator_hotkey` to the signer you trust (the report envelope carries
    `signer_hotkey`). Reading the commitment for that exact hotkey at the
    historical block is the whole point — a validator cannot retroactively edit
    a committed hash.
    """

    def __init__(
        self,
        subtensor_url: str = ARCHIVE_ENDPOINT_DEFAULT,
        netuid: int = 40,
        validator_hotkey: str | None = None,
    ) -> None:
        self.subtensor_url = subtensor_url
        self.netuid = netuid
        self.validator_hotkey = validator_hotkey
        self._subtensor = None

    def _connect(self):
        if self._subtensor is not None:
            return self._subtensor
        import bittensor as bt

        # network= accepts a raw ws(s):// chain endpoint; pass the archive URL.
        self._subtensor = bt.Subtensor(network=self.subtensor_url)
        return self._subtensor

    def get_commitment_hash(self, at_block: int, hotkey: str | None = None) -> str | None:
        """Return the 64-hex sha256 the validator committed for our netuid at
        `at_block`, or None if no commitment exists.

        `hotkey` (or the pinned `self.validator_hotkey`) is the signer to read
        the commitment for. A production auditor ALWAYS pins the exact hotkey it
        trusts — reading 'any' commitment would let a second validator's
        commitment masquerade as the signer's.
        """
        signer = hotkey or self.validator_hotkey
        if not signer:
            raise ValueError(
                "ChainClient.get_commitment_hash needs the signer hotkey "
                "(pass hotkey=... or set validator_hotkey) — never read 'any' "
                "commitment on the netuid."
            )
        subtensor = self._connect()
        try:
            raw = subtensor.get_commitment_metadata(
                netuid=self.netuid, hotkey_ss58=signer, block=at_block
            )
        except Exception:
            return None
        if not raw or isinstance(raw, str):
            # Empty string => no commitment at that block for this hotkey.
            # bittensor returns "" for "no commitment"; a real commitment is a
            # dict we decode below.
            if isinstance(raw, str) and raw:
                return _normalize_hex(raw)
            return None

        decoded = _decode_commitment(raw)
        return _normalize_hex(decoded) if decoded else None

    def close(self) -> None:
        sub = self._subtensor
        if sub is not None:
            try:
                sub.close()
            except Exception:
                pass


def _decode_commitment(raw) -> str | None:
    """Decode a Commitments.CommitmentOf value into the committed UTF-8 string.

    The bittensor SDK exposes `decode_metadata` for exactly this (Raw-field ->
    str). Fall back to manual Raw-field extraction if the helper moves.
    """
    try:
        from bittensor.core.chain_data.utils import decode_metadata

        return decode_metadata(raw)
    except Exception:
        pass
    # Manual fallback — mirror greencompute's Raw-field walk.
    info = raw.value if hasattr(raw, "value") else raw
    fields = info.get("fields") if isinstance(info, dict) else None
    if not fields:
        return None
    for field_list in fields:
        for entry in field_list:
            items = entry.items() if hasattr(entry, "items") else []
            for _tag, val in items:
                if isinstance(val, str):
                    if val.startswith("0x"):
                        try:
                            return bytes.fromhex(val[2:]).decode("utf-8")
                        except Exception:
                            return val[2:]
                    return val
    return None


def _normalize_hex(value: str) -> str:
    """Strip 0x and lowercase; if the value is a hex-encoded sha256 string,
    return the 64-hex digest. The committed data is the 64-char hex digest
    itself (see commit_audit_root), so most callers get it verbatim."""
    v = value.strip()
    if v.startswith("0x"):
        v = v[2:]
    v = v.lower()
    return v


__all__ = ["ARCHIVE_ENDPOINT_DEFAULT", "ChainClient"]
