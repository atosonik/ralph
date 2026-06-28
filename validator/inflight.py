"""One-in-flight-submission-per-hotkey guard (a dynamic cooldown).

NOT Ninja's one-shot rule (a hotkey is never "spent"). Instead: a hotkey may
have at most ONE submission being evaluated at a time. The moment that
submission is SCORED, the hotkey can submit again — zero penalty on honest fast
iterators, but the validator never runs more than one expensive op4 per hotkey
concurrently, and a miner can't overwrite/replace a submission mid-evaluation
and waste GPU.

This complements (does not replace) the on-chain `set_commitment` rate-limit and
op1 novelty-rejection. State is file-backed so it survives validator restarts.

Usage in the submission loop:
    guard = InFlightGuard(chain_dir / "inflight.json")
    ok, reason = guard.claim(hotkey, bundle_hash)
    if not ok:
        defer(bundle); continue          # try again a later epoch
    try:
        result = judge_submission(...)    # op1..op4
        ... score / crown ...
    finally:
        guard.release(hotkey, bundle_hash)
"""
from __future__ import annotations

import json
from pathlib import Path


class InFlightGuard:
    def __init__(self, state_path: Path | str):
        self.path = Path(state_path)
        self._state: dict[str, str] = self._load()

    def _load(self) -> dict[str, str]:
        if self.path.exists():
            try:
                d = json.loads(self.path.read_text())
                if isinstance(d, dict):
                    return {str(k): str(v) for k, v in d.items()}
            except Exception:  # noqa: BLE001 — corrupt state must not wedge the validator
                return {}
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._state, indent=2, sort_keys=True))

    def in_flight(self, hotkey: str) -> str | None:
        """The bundle_hash currently in flight for this hotkey, or None."""
        return self._state.get(hotkey)

    def claim(self, hotkey: str, bundle_hash: str) -> tuple[bool, str]:
        """Mark (hotkey, bundle) as in flight.

        OK if the hotkey has nothing in flight, or already has THIS bundle in
        flight (idempotent — safe across restarts / VALIDATOR_VERSION reprocess).
        Rejected if a DIFFERENT bundle is in flight — that one must be scored
        (released) first. Rejection is a DEFER, not a permanent reject: the
        caller should leave the bundle queued and retry a later epoch.
        """
        cur = self._state.get(hotkey)
        if cur is not None and cur != bundle_hash:
            return False, (
                f"hotkey has submission {cur[:16]}… in flight — one per hotkey at a "
                f"time; it must be scored before {bundle_hash[:16]}… is evaluated"
            )
        if cur != bundle_hash:
            self._state[hotkey] = bundle_hash
            self._save()
        return True, "claimed"

    def reconcile(self, valid_bundle_hashes: set[str]) -> None:
        """Drop in-flight claims whose bundle is no longer in the pending set.

        Crash-recovery: if the validator died after a bundle was scored+archived
        but before its claim was released, the stale claim would block that
        hotkey forever. Called at epoch start with the current pending bundle
        ids; any claim pointing at a bundle that's gone is cleared.
        """
        stale = [hk for hk, b in self._state.items() if b not in valid_bundle_hashes]
        if stale:
            for hk in stale:
                del self._state[hk]
            self._save()

    def release(self, hotkey: str, bundle_hash: str | None = None) -> None:
        """Clear the in-flight marker after the submission is scored.

        If bundle_hash is given, only release when it matches the in-flight one,
        so a stale release can't clear a newer claim.
        """
        cur = self._state.get(hotkey)
        if cur is None:
            return
        if bundle_hash is not None and cur != bundle_hash:
            return
        del self._state[hotkey]
        self._save()
