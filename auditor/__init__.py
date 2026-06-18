"""Ralph auditor — independent CPU-only verifier for subnet 40 (Ralph).

Anyone can run this against a validator's published audit reports to
prove, on-chain, whether a validator scored honestly — WITHOUT re-doing any GPU
work. Gates 1-3 (hash+sig, scoring replay, weight diff) are pure CPU.

The fidelity guarantee: this package IMPORTS the validator's canonical_json and
its weight/floor constants rather than copying them, so a unilateral validator
change makes the auditor diverge (an intended alarm), and the round-trip test in
tests/test_auditor_roundtrip.py proves the replay mirrors the scorer.
"""

__version__ = "0.1.0"
