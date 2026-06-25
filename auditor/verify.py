"""Gate 1 — hash + signature verification.

THE FIDELITY ANCHOR: this module re-hashes the published report by importing
the validator's OWN `canonical_json` from validator.audit_report — it does NOT
reimplement the canonicalization. Byte-identical canonicalization is mandatory:
if the validator changed its encoding, the import makes the auditor compute the
same (different) bytes, so a drift is caught as an intended hash divergence
rather than silently passing. Do NOT copy the encoding here.

Checks, all hard failures:
  * self-consistency: report_sha256(report_json) == envelope.report_sha256
  * on-chain anchor:  == the hash committed on-chain at chain_commitment_block
                         (the block the validator's set_commitment landed at)
  * signature:        ed25519 valid for `signer_hotkey` over canonical bytes
"""

from __future__ import annotations

# IMPORT — never reimplement. canonical_json (and report_sha256, which wraps it)
# are the validator's; importing them is the whole guarantee.
from validator.audit_report import canonical_json, report_sha256


def verify_report(
    envelope: dict,
    *,
    expected_onchain_hash: str | None,
    signer_hotkey: str | None = None,
) -> None:
    """Raise AssertionError on any integrity failure (Gate-1 -> exit 1).

    Args:
      envelope: the signed report envelope (report_json, report_sha256,
        signature, signer_hotkey, ...).
      expected_onchain_hash: the 64-hex sha256 read from the chain at
        epoch_end_block. If None (chain unreachable / no commitment), the
        on-chain anchor check is skipped — caller decides whether that's a
        network failure vs an acceptable degraded run.
      signer_hotkey: ss58 to verify the signature against; defaults to the
        envelope's own signer_hotkey. The signature check is skipped only if no
        signature was attached (unsigned LocalChain parity reports).
    """
    report_json = envelope.get("report_json") or {}
    claimed_sha = (envelope.get("report_sha256") or "").lower()
    claimed_sig = envelope.get("signature") or ""
    claimed_signer = signer_hotkey or envelope.get("signer_hotkey") or ""

    # 1. self-consistency: the envelope's hash must match canonical(report_json).
    computed_sha = report_sha256(report_json)
    assert computed_sha == claimed_sha, (
        f"report self-hash mismatch: computed={computed_sha}, claimed={claimed_sha}"
    )

    # 2. on-chain anchor: the committed hash must equal the recomputed hash.
    if expected_onchain_hash:
        onchain = expected_onchain_hash.lower()
        assert computed_sha == onchain, (
            f"on-chain SHA256 mismatch: chain={onchain}, report={computed_sha} "
            "— validator committed a hash that does not match the published report"
        )

    # 3. signature: ed25519 over the canonical bytes for the signer hotkey.
    #    Skipped only when no signature is attached (unsigned local parity).
    if claimed_sig and claimed_signer:
        canonical = canonical_json(report_json)
        try:
            from bittensor_wallet import Keypair
        except ImportError:  # pragma: no cover
            from substrateinterface import Keypair  # type: ignore
        kp = Keypair(ss58_address=claimed_signer)
        ok = kp.verify(canonical, bytes.fromhex(claimed_sig))
        assert ok, f"ed25519 signature invalid for report (signer={claimed_signer})"


__all__ = ["verify_report"]
