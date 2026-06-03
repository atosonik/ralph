"""
Miner-side tooling — outside the protocol's attestation surface.

The miner's autoresearch-style search agent runs separately on whatever
hardware the miner chooses; this directory contains only the bundling and
submission utilities the miner invokes after deciding on a candidate
patch. The search code itself is the miner's private IP.
"""

from .submit import (
    SubmissionBundle,
    assemble_submission,
    request_handshake_nonce,
    sign_submission,
)

__all__ = [
    "SubmissionBundle",
    "assemble_submission",
    "sign_submission",
    "request_handshake_nonce",
]
