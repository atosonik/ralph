"""Round-trip fidelity proof: validator audit report -> auditor verify/replay/diff.

This is the real proof the auditor faithfully mirrors the validator:

  1. Build a REAL signed report via validator.audit_report.build_report_json +
     build_envelope (fake scored submissions + a 90/10 weight snapshot).
  2. Drive the auditor's audit_epoch (Gate 1 hash+sig, Gate 2 replay, Gate 3
     diff) against it with in-memory fake chain/report clients -> assert CLEAN
     (exit 0).
  3. Mutate one weight in weight_snapshot -> Gate 3 catches it (exit 2).
  4. Mutate a byte of the report -> Gate 1 catches it (exit 1).
  5. Unit-assert the auditor's replay constants are the SAME OBJECTS / values as
     the validator's (imported, not copied) and that verify uses the validator's
     canonical_json.

A fixed generated_at keeps everything deterministic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bittensor_wallet import Keypair

import ralph_bootstrap  # noqa: F401
import validator.service as svc
from auditor import diff as auditor_diff
from auditor import replay as auditor_replay
from auditor import verify as auditor_verify
from auditor.main import (
    EXIT_CLEAN,
    EXIT_HASH_OR_SIG,
    EXIT_MATH_DIVERGE,
    audit_epoch,
)
from validator.audit_report import (
    build_envelope,
    build_report_json,
    canonical_json,
    report_sha256,
    sign_report,
)

FIXED_GENERATED_AT = "2026-06-18T00:00:00Z"


# --- fake scored submissions (shape mirrors score_and_decide() results) ---


def _scored_king() -> dict:
    return {
        "status": "accepted",
        "classification": "king_change",
        "weight_credit": 1.0,
        "miner_hotkey": "5Hkingalpha",
        "miner_github": "alice",
        "pr_url": "",
        "bundle_hash": "a" * 64,
        "val_bpb": 1.2000,
        "benchmark_accuracy": 0.42,
        "quality_gain": 0.05,
        "score": 0.9,
        "tier": "verified",
        "decisive": True,
        "accepted": True,
        "is_first": False,
        "parent_king_attestation_hash": "b" * 64,
        "attestation_hash": "c" * 64,
        "val_seq_len": 512,
        "sealed_stream_manifest_hash": "e" * 64,
        "tail_val_bpb": 1.30,
    }


def _scored_mf() -> dict:
    return {
        "status": "meaningful_failure",
        "classification": "meaningful_failure",
        "weight_credit": 0.1,
        "miner_hotkey": "5Hmfbeta",
        "miner_github": "bob",
        "pr_url": "",
        "bundle_hash": "d" * 64,
        "val_bpb": 1.2400,
        "benchmark_accuracy": 0.41,
        "quality_gain": -0.005,
        "score": 0.0,
        "tier": "verified",
        "decisive": False,
        "accepted": False,
        "is_first": False,
        "parent_king_attestation_hash": None,
        "attestation_hash": None,
        "val_seq_len": 512,
        "sealed_stream_manifest_hash": "f" * 64,
        "tail_val_bpb": 1.305,
    }


def _build_signed_envelope(weight_snapshot: dict[str, float], keypair: Keypair) -> dict:
    """Build a fully-signed report envelope the way the validator does."""
    report_json = build_report_json(
        epoch_id="40-1000",
        netuid=40,
        start_block=900,
        end_block=1000,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_king(), _scored_mf()],
        weight_snapshot=weight_snapshot,
        seed=0,
    )
    sig = sign_report(canonical_json(report_json), keypair)
    return build_envelope(
        report_json,
        signature=sig,
        signer_hotkey=keypair.ss58_address,
        chain_commitment_block=1000,
        weights_set=True,
    )


# --- in-memory fakes so audit_epoch runs with no network / no chain ---


class _FakeReportClient:
    """Returns one in-memory envelope, mimicking auditor.fetch.ReportClient."""

    def __init__(self, envelope: dict) -> None:
        self._envelope = envelope
        self._epoch_id = envelope["report_json"]["epoch_id"]

    def list_reports(self) -> list[dict]:
        rj = self._envelope["report_json"]
        return [{"epoch_id": rj["epoch_id"], "epoch_end_block": rj["epoch_end_block"]}]

    def get_report(self, epoch_id: str) -> dict:
        assert epoch_id == self._epoch_id
        return self._envelope


class _FakeChainClient:
    """Returns a pinned on-chain commitment hash, mimicking auditor.chain.ChainClient."""

    def __init__(self, onchain_hash: str | None) -> None:
        self._hash = onchain_hash
        self.subtensor_url = "wss://fake/"
        self.netuid = 40

    def get_commitment_hash(self, at_block: int, hotkey: str | None = None) -> str | None:
        return self._hash


# --------------------------------------------------------------------------
# 1. CLEAN round trip
# --------------------------------------------------------------------------


def test_roundtrip_clean_exit0():
    kp = Keypair.create_from_uri("//RoundtripSigner")
    snapshot = {"5Hkingalpha": 0.9, "5Hmfbeta": 0.1}  # the real 90/10 split
    env = _build_signed_envelope(snapshot, kp)

    # The on-chain commitment is the honest hash.
    chain = _FakeChainClient(env["report_sha256"])
    api = _FakeReportClient(env)

    code = audit_epoch("40-1000", chain, api)
    assert code == EXIT_CLEAN

    # And the replay reproduces the exact published weight vector.
    replayed = auditor_replay.replay_scoring(env["report_json"])
    assert replayed == snapshot
    assert auditor_diff.compare_weights(snapshot, replayed) == {}


# --------------------------------------------------------------------------
# 2. Mutated weight -> Gate 3 catches it (exit 2)
# --------------------------------------------------------------------------


def test_roundtrip_mutated_weight_exit2():
    kp = Keypair.create_from_uri("//RoundtripSigner")
    # The validator CLAIMS a tampered weight vector (king takes 0.95 instead of
    # the 0.9 the published submissions score to). The report is internally
    # self-consistent (hash + sig still valid) — only the math diverges.
    tampered = {"5Hkingalpha": 0.95, "5Hmfbeta": 0.05}
    env = _build_signed_envelope(tampered, kp)
    chain = _FakeChainClient(env["report_sha256"])  # commitment matches the (tampered) report
    api = _FakeReportClient(env)

    code = audit_epoch("40-1000", chain, api)
    assert code == EXIT_MATH_DIVERGE

    # Gate 1 still passes (the report is self-consistent); only Gate 3 fails.
    auditor_verify.verify_report(
        env, expected_onchain_hash=env["report_sha256"], signer_hotkey=kp.ss58_address
    )
    replayed = auditor_replay.replay_scoring(env["report_json"])
    discrepancies = auditor_diff.compare_weights(tampered, replayed)
    assert "5Hkingalpha" in discrepancies
    assert discrepancies["5Hkingalpha"]["replayed"] == pytest.approx(0.9)
    assert discrepancies["5Hkingalpha"]["claimed"] == pytest.approx(0.95)


# --------------------------------------------------------------------------
# 3. Mutated report byte -> Gate 1 catches it (exit 1)
# --------------------------------------------------------------------------


def test_roundtrip_mutated_report_byte_exit1():
    kp = Keypair.create_from_uri("//RoundtripSigner")
    snapshot = {"5Hkingalpha": 0.9, "5Hmfbeta": 0.1}
    env = _build_signed_envelope(snapshot, kp)

    # Tamper with the report_json AFTER signing/hashing: the published hash +
    # signature now describe a different report than the one served. The
    # on-chain commitment still holds the ORIGINAL (honest) hash.
    honest_hash = env["report_sha256"]
    env["report_json"]["submissions"][0]["eval_output"]["val_bpb"] = 0.0001

    chain = _FakeChainClient(honest_hash)
    api = _FakeReportClient(env)

    code = audit_epoch("40-1000", chain, api)
    assert code == EXIT_HASH_OR_SIG

    # Directly: the self-hash no longer matches the stale envelope hash.
    with pytest.raises(AssertionError):
        auditor_verify.verify_report(
            env, expected_onchain_hash=honest_hash, signer_hotkey=kp.ss58_address
        )
    assert report_sha256(env["report_json"]) != honest_hash


def test_roundtrip_onchain_hash_mismatch_exit1():
    """Even if the report is internally self-consistent, a commitment hash that
    doesn't match the report (validator edited the report after committing) is a
    Gate-1 failure."""
    kp = Keypair.create_from_uri("//RoundtripSigner")
    snapshot = {"5Hkingalpha": 0.9, "5Hmfbeta": 0.1}
    env = _build_signed_envelope(snapshot, kp)

    chain = _FakeChainClient("f" * 64)  # on-chain hash != report hash
    api = _FakeReportClient(env)

    code = audit_epoch("40-1000", chain, api)
    assert code == EXIT_HASH_OR_SIG


def test_roundtrip_bad_signature_exit1():
    """A signature from a different key than the named signer fails Gate 1."""
    kp = Keypair.create_from_uri("//RoundtripSigner")
    other = Keypair.create_from_uri("//Impostor")
    snapshot = {"5Hkingalpha": 0.9, "5Hmfbeta": 0.1}
    env = _build_signed_envelope(snapshot, kp)
    # Claim the impostor as signer while the signature is over kp -> verify fails.
    env["signer_hotkey"] = other.ss58_address

    chain = _FakeChainClient(env["report_sha256"])
    api = _FakeReportClient(env)
    code = audit_epoch("40-1000", chain, api)
    assert code == EXIT_HASH_OR_SIG


# --------------------------------------------------------------------------
# 4. Constants + canonical_json are IMPORTED from the validator, not copied
# --------------------------------------------------------------------------


def test_replay_constants_are_imported_identical():
    """The auditor's replay constants must be the SAME values as the validator's
    — imported, not hardcoded — so a unilateral validator change makes the
    auditor diverge (intended alarm)."""
    for name in (
        "KING_CHANGE_WEIGHT",
        "MEANINGFUL_FAILURE_WEIGHT",
        "PLAIN_FAILURE_WEIGHT",
        "NOISE_FLOOR_MARGIN_2X_MULTIPLIER",
        "KING_POOL_FRACTION",
        "MEANINGFUL_FAILURE_POOL_FRACTION",
    ):
        assert getattr(auditor_replay, name) == getattr(svc, name), name


def test_verify_uses_validator_canonical_json():
    """Gate 1 must re-hash with the validator's canonical_json object itself, so
    byte-identical canonicalization is guaranteed by construction."""
    import validator.audit_report as ar

    assert auditor_verify.canonical_json is ar.canonical_json
    # and report_sha256 -> the same bytes the validator hashes.
    obj = {"b": 2, "a": "ünïcode"}
    assert auditor_verify.report_sha256(obj) == ar.report_sha256(obj)


def test_classify_king_change_recomputed_not_trusted():
    """The king_change branch is recomputed from (decisive_vs_king or is_first),
    NOT taken from the published gate — so a validator that mislabels a decisive
    submission as 'plain_failure' is still scored as king_change by the auditor
    (and Gate 3 would then flag the weight)."""
    sub = {
        "miner_hotkey": "5Hk",
        "eval_output": {"gate": "plain_failure", "decisive_vs_king": True, "is_first": False},
    }
    classification, credit = auditor_replay.classify_from_report(sub)
    assert classification == "king_change"
    assert credit == svc.KING_CHANGE_WEIGHT

    # is_first alone also crowns (first-ever submission).
    sub2 = {
        "miner_hotkey": "5Hk",
        "eval_output": {"gate": "plain_failure", "decisive_vs_king": False, "is_first": True},
    }
    assert auditor_replay.classify_from_report(sub2)[0] == "king_change"
