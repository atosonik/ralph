"""Ninja-style (SN66) one-commitment-per-hotkey submission anchoring.

Ralph's legacy handshake commits an OPAQUE hash to the chain
(``sha256("karpa:handshake:{hotkey}:{patch_hash}:{nonce}")``): the validator
cannot read which bundle it points to — it must *reconstruct* the hash from a
submission it already has, and the embedded nonce makes the commitment race-prone
("submit immediately after committing, or set_commitment overwrites it").

SN66 Ninja solves this with a READABLE, content-addressed commitment in the
single on-chain slot each hotkey has:

    private-submission:<submission_id>:<sha256-of-agent.py>
    ("Only one accepted submission is eligible per miner hotkey registration.")

This module ports that pattern to Ralph. The commitment is::

    ralph-submission:<submission_id>:<bundle_sha256>

set on-chain via ``subtensor.set_commitment`` — itself a hotkey-signed extrinsic,
so the commitment is authenticated by the chain (no extra signature needed). Key
properties vs the legacy handshake:

  * **Readable / self-describing** — the validator reads the bundle hash straight
    from the chain and knows exactly which bundle is this hotkey's current one,
    without reconstructing anything or matching off-chain PR titles.
  * **One-per-hotkey is the feature** — overwriting the single slot is how a
    miner *updates* their submission; it is not a race. The commitment always
    names the current bundle.
  * **Content-addressed** — the validator verifies the bundle it holds hashes to
    the committed value, binding the on-chain pointer to exact bytes.

Scope: this is the op1 SUBMISSION-IDENTITY anchor. Attestation freshness (the
op2 nonce that flows into the TDX/NVIDIA quotes) is a separate concern and is
unchanged by this module.
"""
from __future__ import annotations

import re

PREFIX = "ralph-submission"
SIG_PREFIX = "ralph-submission-v1"

# A full, self-describing commitment string. submission_id is a short human/label
# token (<=128 of [A-Za-z0-9_.-]); the trailing field is a lowercase sha256.
COMMITMENT_RE = re.compile(rf"^{re.escape(PREFIX)}:[A-Za-z0-9_.-]{{1,128}}:[0-9a-f]{{64}}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def derive_submission_id(hotkey: str, content_sha256: str) -> str:
    """Stable, content-addressed id ``<hotkey[:16]>-<sha[:16]>`` (Ninja-style).

    Binding the id to BOTH the hotkey and the content means it changes whenever
    the bundle changes and cannot be made to masquerade as another hotkey's id.
    """
    safe = re.sub(r"[^A-Za-z0-9_.-]", "-", hotkey)[:16] or "hotkey"
    return f"{safe}-{content_sha256.lower()[:16]}"


def build_commitment(content_sha256: str, *, hotkey: str, submission_id: str | None = None) -> str:
    """Build the on-chain commitment string a miner sets via set_commitment."""
    sha = content_sha256.strip().lower()
    if not _SHA256_RE.match(sha):
        raise ValueError(f"content_sha256 must be 64 lowercase hex chars, got {content_sha256!r}")
    sid = submission_id or derive_submission_id(hotkey, sha)
    commitment = f"{PREFIX}:{sid}:{sha}"
    if not COMMITMENT_RE.match(commitment):
        raise ValueError(f"built commitment is malformed: {commitment!r}")
    return commitment


def parse_commitment(commitment: str) -> tuple[str, str]:
    """Return ``(submission_id, content_sha256)`` or raise ValueError."""
    c = (commitment or "").strip()
    if not COMMITMENT_RE.match(c):
        raise ValueError(f"not a valid {PREFIX} commitment: {commitment!r}")
    _, sid, sha = c.split(":", 2)
    return sid, sha


def signature_payload(hotkey: str, submission_id: str, content_sha256: str) -> bytes:
    """Optional hotkey-signature payload (parity with Ninja's tau-* payload).

    Not required when the commitment is set directly on-chain (set_commitment is
    already hotkey-signed); useful if a submission is relayed off-chain.
    """
    return f"{SIG_PREFIX}:{hotkey}:{submission_id}:{content_sha256.lower()}".encode()


def verify_commitment(commitment: str, *, hotkey: str, bundle_sha256: str) -> tuple[bool, str]:
    """Validator-side: does the on-chain commitment name THIS bundle for THIS hotkey?

    Verifies: well-formed; the committed content hash equals the bundle the
    validator holds; and the submission_id is the canonical derivation for
    (hotkey, content) — so a commitment cannot point at one hotkey's id while
    carrying another's content. Returns ``(ok, reason)``.
    """
    try:
        sid, sha = parse_commitment(commitment)
    except ValueError as e:
        return False, str(e)
    if sha != bundle_sha256.strip().lower():
        return False, (
            f"commitment content hash {sha[:16]}… != submitted bundle "
            f"{bundle_sha256.strip().lower()[:16]}…"
        )
    expected_sid = derive_submission_id(hotkey, sha)
    if sid != expected_sid:
        return False, f"submission_id {sid!r} is not canonical {expected_sid!r} for this hotkey/content"
    return True, "commitment verified (content-addressed, one-per-hotkey)"
