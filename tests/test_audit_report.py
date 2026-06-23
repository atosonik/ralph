"""Tests for validator.audit_report — validation-v2 Phase 1.

Covers:
  * canonical_json determinism — same dict in any key order -> identical bytes.
  * report_sha256 stability across key orderings.
  * sign_report -> verify round-trip with a throwaway Keypair (incl. the
    auditor's ss58-only reconstruction path).
  * build_report_json shape from a couple of fake scored results + a snapshot.
  * write_report writes <out>/audit_reports/<epoch_id>.json + upserts index.json.

A fixed `generated_at` is used everywhere so the report is deterministic.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ralph_bootstrap  # noqa: F401
from validator.audit_report import (
    AUDIT_REPORT_SCHEMA_VERSION,
    build_envelope,
    build_report_json,
    canonical_json,
    report_sha256,
    sign_report,
    write_report,
)

FIXED_GENERATED_AT = "2026-06-18T00:00:00Z"


# --- a couple of fake scored results (shape mirrors score_and_decide()) ---

def _scored_king() -> dict:
    return {
        "status": "accepted",
        "classification": "king_change",
        "weight_credit": 1.0,
        "miner_hotkey": "5Hkingalpha",
        "miner_github": "alice",
        "pr_url": "https://github.com/x/y/pull/1",
        "bundle_hash": "a" * 64,
        "val_bpb": 1.2345,
        "benchmark_accuracy": 0.42,
        "quality_gain": 0.05,
        "score": 0.9,
        "tier": "verified",
        "decisive": True,
        "accepted": True,
        "is_first": False,
        "parent_king_attestation_hash": "b" * 64,
        "attestation_hash": "c" * 64,
        # validation-v2 Phase 1 reproducibility fields (now surfaced for real).
        "val_seq_len": 512,
        "sealed_stream_manifest_hash": "e" * 64,
        "tail_val_bpb": 1.3001,
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
        "tail_val_bpb": 1.3050,
    }


# --------------------------------------------------------------------------
# canonical_json / report_sha256 determinism
# --------------------------------------------------------------------------


def test_canonical_json_is_key_order_independent():
    a = {"z": 1, "a": {"y": 2, "x": 3}, "m": [3, 2, 1]}
    b = {"a": {"x": 3, "y": 2}, "m": [3, 2, 1], "z": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_exact_encoding():
    # Frozen separators (no spaces) + sorted keys + utf-8. The auditor depends
    # on this byte-for-byte.
    obj = {"b": 2, "a": "ünïcode"}
    out = canonical_json(obj)
    assert out == b'{"a":"\xc3\xbcn\xc3\xafcode","b":2}'


def test_report_sha256_stable_across_key_order():
    a = {"z": 1, "a": 2, "nested": {"q": 9, "p": 8}}
    b = {"a": 2, "nested": {"p": 8, "q": 9}, "z": 1}
    assert report_sha256(a) == report_sha256(b)
    # 64-hex
    h = report_sha256(a)
    assert len(h) == 64 and all(c in "0123456789abcdef" for c in h)


def test_report_sha256_changes_on_content_change():
    assert report_sha256({"a": 1}) != report_sha256({"a": 2})


# --------------------------------------------------------------------------
# sign -> verify round-trip
# --------------------------------------------------------------------------


def test_sign_report_verify_roundtrip():
    from bittensor_wallet import Keypair

    kp = Keypair.create_from_uri("//AuditTestSigner")
    report = build_report_json(
        epoch_id="40-100",
        netuid=40,
        start_block=90,
        end_block=100,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_king()],
        weight_snapshot={"5Hkingalpha": 1.0},
    )
    cbytes = canonical_json(report)
    sig_hex = sign_report(cbytes, kp)

    # signer verifies directly
    assert kp.verify(cbytes, bytes.fromhex(sig_hex))
    # tampered bytes fail
    assert not kp.verify(cbytes + b"x", bytes.fromhex(sig_hex))

    # auditor path: reconstruct a verifier Keypair from ONLY the ss58 hotkey.
    verifier = Keypair(ss58_address=kp.ss58_address)
    assert verifier.verify(cbytes, bytes.fromhex(sig_hex))


# --------------------------------------------------------------------------
# build_report_json shape
# --------------------------------------------------------------------------


def test_build_report_json_shape():
    snapshot = {"5Hkingalpha": 0.9, "5Hmfbeta": 0.1}
    report = build_report_json(
        epoch_id="40-200",
        netuid=40,
        start_block=190,
        end_block=200,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_king(), _scored_mf()],
        weight_snapshot=snapshot,
        seed=7,
    )

    # top-level
    assert report["schema_version"] == AUDIT_REPORT_SCHEMA_VERSION
    assert report["epoch_id"] == "40-200"
    assert report["netuid"] == 40
    assert report["epoch_start_block"] == 190
    assert report["epoch_end_block"] == 200
    assert report["generated_at"] == FIXED_GENERATED_AT

    # submissions
    assert len(report["submissions"]) == 2
    s0 = report["submissions"][0]
    assert s0["miner_hotkey"] == "5Hkingalpha"
    assert s0["submission_hash"] == "a" * 64
    assert s0["bundle_sha256"] == "a" * 64
    assert s0["parent_king_attestation_hash"] == "b" * 64
    assert s0["attestation_hash"] == "c" * 64

    ei = s0["eval_input"]
    assert ei["seed"] == 7
    assert ei["bundle_sha256"] == "a" * 64
    # ladder rung dims carried for reproduction
    labels = [r["scale_label"] for r in ei["ladder_rungs"]]
    assert labels == ["S1", "S2", "S3"]
    assert ei["ladder_rungs"][2] == {"scale_label": "S3", "dim": 768, "n_layers": 12}
    # validation-v2 Phase 1: these are now populated for REAL (were GAP/None).
    assert ei["val_seq_len"] == 512
    assert ei["sealed_stream_manifest_hash"] == "e" * 64

    eo = s0["eval_output"]
    assert eo["val_bpb"] == 1.2345
    assert eo["decisive_vs_king"] is True
    assert eo["gate"] == "king_change"
    assert eo["tail_val_bpb"] == 1.3001  # now populated for real

    # scorecards: final_score from scorer, weight from snapshot
    assert len(report["scorecards"]) == 2
    sc0 = report["scorecards"][0]
    assert sc0["miner_hotkey"] == "5Hkingalpha"
    assert sc0["final_score"] == 0.9
    assert sc0["weight"] == 0.9

    # weight_snapshot embedded verbatim
    assert report["weight_snapshot"]["netuid"] == 40
    assert report["weight_snapshot"]["weights"] == snapshot
    assert report["weight_snapshot"]["created_at"] == FIXED_GENERATED_AT


def test_reproducibility_fields_populated_in_report():
    """validation-v2 Phase 1: the three Gate-4 reproducibility fields
    (val_seq_len, sealed_stream_manifest_hash, tail_val_bpb) flow from a scored
    result into eval_input / eval_output with REAL values (not None)."""
    report = build_report_json(
        epoch_id="40-210",
        netuid=40,
        start_block=200,
        end_block=210,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_king()],
        weight_snapshot={"5Hkingalpha": 1.0},
        seed=3,
    )
    s0 = report["submissions"][0]
    ei = s0["eval_input"]
    eo = s0["eval_output"]

    # Keys still present (auditors branch on them) AND now non-None.
    assert "val_seq_len" in ei and ei["val_seq_len"] == 512
    assert ei["val_seq_len"] is not None
    assert "sealed_stream_manifest_hash" in ei
    assert ei["sealed_stream_manifest_hash"] == "e" * 64
    assert ei["sealed_stream_manifest_hash"] is not None
    assert "tail_val_bpb" in eo and eo["tail_val_bpb"] == 1.3001
    assert eo["tail_val_bpb"] is not None


def test_build_envelope_records_weights_set():
    """The envelope carries weights_set (decision-vs-landed) OUTSIDE the signed
    report_json, so the hash is identical regardless of whether the extrinsic
    landed."""
    report = build_report_json(
        epoch_id="40-220",
        netuid=40,
        start_block=210,
        end_block=220,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_king()],
        weight_snapshot={"5Hkingalpha": 1.0},
    )
    env_landed = build_envelope(
        report, signature="", signer_hotkey="",
        chain_commitment_block=1, weights_set=True,
    )
    env_ratelimited = build_envelope(
        report, signature="", signer_hotkey="",
        chain_commitment_block=1, weights_set=False,
    )
    assert env_landed["weights_set"] is True
    assert env_ratelimited["weights_set"] is False
    # weights_set lives in the envelope only — the signed/hashed report_json is
    # identical either way.
    assert env_landed["report_sha256"] == env_ratelimited["report_sha256"]
    # default (omitted) is False
    env_default = build_envelope(
        report, signature="", signer_hotkey="", chain_commitment_block=None,
    )
    assert env_default["weights_set"] is False


def test_build_report_json_is_canonicalizable_deterministic():
    # Same inputs -> identical canonical bytes (no hidden datetime.now()).
    kwargs = dict(
        epoch_id="40-300",
        netuid=40,
        start_block=290,
        end_block=300,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_king()],
        weight_snapshot={"5Hkingalpha": 1.0},
    )
    r1 = build_report_json(**kwargs)
    r2 = build_report_json(**kwargs)
    assert canonical_json(r1) == canonical_json(r2)
    assert report_sha256(r1) == report_sha256(r2)


# --------------------------------------------------------------------------
# _generate_audit_report ordering — report is written + records weights_set
# regardless of whether set_weights landed (Task 1 ordering fix).
# --------------------------------------------------------------------------


class _StubChain:
    """Minimal chain for _generate_audit_report: no wallet (unsigned, local
    parity), a fixed end block, and a commit_audit_root that records the sha."""

    def __init__(self, chain_dir: Path):
        self.chain_dir = chain_dir
        self.committed_sha: str | None = None

    def get_current_block(self) -> int:
        return 999

    def commit_audit_root(self, sha: str) -> int:
        self.committed_sha = sha
        return 999


@pytest.mark.parametrize("weights_set", [True, False])
def test_generate_audit_report_records_weights_set(tmp_path, weights_set):
    """The audit report is built + anchored + written and stamps weights_set —
    in particular weights_set=False (a rate-limited epoch where set_weights
    returned early) STILL produces a full report. This is the Task-1 ordering
    contract: the report documents the DECISION, independent of the extrinsic."""
    from validator.service import _generate_audit_report

    chain = _StubChain(tmp_path)
    _generate_audit_report(
        chain,
        scored_results=[_scored_king()],
        weight_snapshot={"5Hkingalpha": 1.0},
        epoch_start_block=990,
        netuid=40,
        eval_seed=0,
        weights_set=weights_set,
    )

    # On-chain anchor happened.
    assert chain.committed_sha is not None
    # Report written to chain_dir/audit_reports/40-999.json.
    report_path = tmp_path / "audit_reports" / "40-999.json"
    assert report_path.exists()
    env = json.loads(report_path.read_text())
    assert env["weights_set"] is weights_set
    assert env["report_sha256"] == chain.committed_sha
    # Reproducibility fields flowed through to the persisted report.
    ei = env["report_json"]["submissions"][0]["eval_input"]
    assert ei["val_seq_len"] == 512
    assert ei["sealed_stream_manifest_hash"] == "e" * 64
    eo = env["report_json"]["submissions"][0]["eval_output"]
    assert eo["tail_val_bpb"] == 1.3001
    # index records weights_set too.
    index = json.loads((tmp_path / "audit_reports" / "index.json").read_text())
    assert index[0]["weights_set"] is weights_set


# --------------------------------------------------------------------------
# write_report + index upsert
# --------------------------------------------------------------------------


def test_write_report_and_index(tmp_path):
    from bittensor_wallet import Keypair

    kp = Keypair.create_from_uri("//WriteTest")
    report = build_report_json(
        epoch_id="40-400",
        netuid=40,
        start_block=390,
        end_block=400,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_king()],
        weight_snapshot={"5Hkingalpha": 1.0},
    )
    sig = sign_report(canonical_json(report), kp)
    env = build_envelope(
        report,
        signature=sig,
        signer_hotkey=kp.ss58_address,
        chain_commitment_block=12345,
        weights_set=True,
    )
    assert env["report_sha256"] == report_sha256(report)

    path = write_report(env, tmp_path)
    assert path == tmp_path / "audit_reports" / "40-400.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["report_sha256"] == env["report_sha256"]
    assert loaded["chain_commitment_block"] == 12345
    assert loaded["weights_set"] is True

    index = json.loads((tmp_path / "audit_reports" / "index.json").read_text())
    assert len(index) == 1
    assert index[0]["epoch_id"] == "40-400"
    assert index[0]["report_sha256"] == env["report_sha256"]
    assert index[0]["signer_hotkey"] == kp.ss58_address
    assert index[0]["chain_commitment_block"] == 12345
    assert index[0]["weights_set"] is True

    # second epoch appends; re-writing same epoch_id replaces (idempotent).
    report2 = build_report_json(
        epoch_id="40-500",
        netuid=40,
        start_block=490,
        end_block=500,
        generated_at=FIXED_GENERATED_AT,
        scored=[_scored_mf()],
        weight_snapshot={"5Hmfbeta": 1.0},
    )
    env2 = build_envelope(
        report2, signature="", signer_hotkey="", chain_commitment_block=500,
    )
    write_report(env2, tmp_path)
    write_report(env, tmp_path)  # re-write 40-400 -> must NOT duplicate
    index = json.loads((tmp_path / "audit_reports" / "index.json").read_text())
    epoch_ids = sorted(e["epoch_id"] for e in index)
    assert epoch_ids == ["40-400", "40-500"]


# --------------------------------------------------------------------------
# Phase 2 (A): HF publish — gated, idempotent, never breaks the local write.
# --------------------------------------------------------------------------


class _FakeHfApi:
    """Records create_repo / upload_file calls; never touches the network.

    Mimics the slice of huggingface_hub.HfApi that audit_publish uses, plus
    hf_hub_download so _fetch_remote_index can read back an uploaded index.
    """

    def __init__(self, token=None):
        self.token = token
        self.created = []
        self.uploads = {}  # path_in_repo -> bytes

    def create_repo(self, repo_id, repo_type=None, exist_ok=False, private=False, token=None):
        self.created.append((repo_id, repo_type, exist_ok, private))

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type,
                    token=None, commit_message=None):
        data = path_or_fileobj
        if hasattr(data, "read"):
            data = data.read()
        self.uploads[path_in_repo] = data

    def hf_hub_download(self, *, repo_id, repo_type, filename, token=None):
        # Serve back whatever upload_file last stored for this filename.
        import tempfile
        if filename not in self.uploads:
            from huggingface_hub.errors import EntryNotFoundError
            raise EntryNotFoundError(f"{filename} not found")
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "wb") as f:
            data = self.uploads[filename]
            f.write(data if isinstance(data, bytes) else data.encode("utf-8"))
        return path


def _signed_env(epoch_id="40-700"):
    from bittensor_wallet import Keypair

    kp = Keypair.create_from_uri("//PublishTest")
    report = build_report_json(
        epoch_id=epoch_id, netuid=40, start_block=690, end_block=700,
        generated_at=FIXED_GENERATED_AT, scored=[_scored_king()],
        weight_snapshot={"5Hkingalpha": 1.0},
    )
    sig = sign_report(canonical_json(report), kp)
    return build_envelope(
        report, signature=sig, signer_hotkey=kp.ss58_address,
        chain_commitment_block=700, weights_set=True,
    )


def test_publish_report_hf_uploads_envelope_and_index(monkeypatch):
    import validator.audit_publish as ap

    fake = _FakeHfApi()
    monkeypatch.setattr(ap, "HfApi", lambda token=None: fake, raising=False)
    # HfApi is imported inside the function — patch the module symbol the
    # function resolves via `from huggingface_hub import HfApi`.
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake)

    env = _signed_env("40-700")
    ok = ap.publish_report_hf(env, repo_id="RalphLabsAI/audit-reports", token="x")
    assert ok is True

    # dataset repo created exist_ok; envelope + index uploaded under audit_reports/.
    assert fake.created and fake.created[0][1] == "dataset" and fake.created[0][2] is True
    assert "audit_reports/40-700.json" in fake.uploads
    assert "audit_reports/index.json" in fake.uploads

    idx = json.loads(fake.uploads["audit_reports/index.json"])
    assert isinstance(idx, list) and idx[0]["epoch_id"] == "40-700"
    assert idx[0]["report_sha256"] == env["report_sha256"]


def test_publish_report_hf_index_upsert_is_idempotent(monkeypatch):
    import huggingface_hub

    import validator.audit_publish as ap

    fake = _FakeHfApi()
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda token=None: fake)

    ap.publish_report_hf(_signed_env("40-700"), token="x")
    ap.publish_report_hf(_signed_env("40-710"), token="x")
    ap.publish_report_hf(_signed_env("40-700"), token="x")  # re-publish -> no dup

    idx = json.loads(fake.uploads["audit_reports/index.json"])
    epoch_ids = sorted(e["epoch_id"] for e in idx)
    assert epoch_ids == ["40-700", "40-710"]


def test_write_report_hf_publish_gated_off_by_default(tmp_path, monkeypatch):
    """hf_publish_enabled defaults False -> publish_report_hf is never called,
    local write still happens."""
    import validator.audit_publish as ap

    called = {"n": 0}

    def _boom(*a, **k):
        called["n"] += 1
        raise AssertionError("should not be called when gated off")

    monkeypatch.setattr(ap, "publish_report_hf", _boom)
    env = _signed_env("40-720")
    path = write_report(env, tmp_path)  # default: hf_publish_enabled=False
    assert path.exists()
    assert called["n"] == 0


def test_write_report_hf_publish_failure_never_breaks_local(tmp_path, monkeypatch):
    """A publish exception is swallowed; the local report is still written."""
    import validator.audit_publish as ap

    monkeypatch.setattr(
        ap, "publish_report_hf",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("HF down")),
    )
    env = _signed_env("40-730")
    path = write_report(env, tmp_path, hf_publish_enabled=True, hf_token="x")
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["report_sha256"] == env["report_sha256"]
