"""
Phase 0 proof-test runner — Python entry-point standing in for the future
Karpa Docker. In Phase 0.5+ this becomes a signed Docker image whose
measurement is on-chain pinned, and the mock attestation is replaced by
real TDX + nvtrust quotes.

The entry point is `run_proof_test()`. It:
  1. Applies the miner's patch to a fresh checkout of the canonical recipe.
  2. Runs canonical training under fixed (config, seed, manifest).
  3. Runs the calibration benchmark in the same environment.
  4. Computes the bundle hash and produces a mock attestation chain.
  5. Emits a submission-ready directory with everything a validator needs.
"""

from .mock_attest import (
    MockAttestation,
    generate_mock_attestation,
    verify_mock_attestation,
)
from .real_attest import (
    RealAttestation,
    detect_capabilities,
    generate_attestation,
    verify_attestation,
)
from .runner import ProofTestBundle, run_proof_test

__all__ = [
    "run_proof_test",
    "ProofTestBundle",
    "generate_mock_attestation",
    "verify_mock_attestation",
    "MockAttestation",
    "RealAttestation",
    "generate_attestation",
    "verify_attestation",
    "detect_capabilities",
]
