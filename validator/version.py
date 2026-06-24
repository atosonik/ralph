"""Validator logic version.

Bump this whenever the verify/score logic changes in a way that could alter a
submission's outcome (op1–op4 checks, attestation verification, scoring/king
rule). The HF poller stamps this version onto every processed bundle in
`hf_state.json`; on a version bump, entries judged by an older version are
dropped from the processed set so they are re-downloaded and re-validated under
the new rules — keeping evaluation fair across a logic upgrade.
"""

VALIDATOR_VERSION = "v1"

__all__ = ["VALIDATOR_VERSION"]
