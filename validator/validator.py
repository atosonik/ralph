"""
Validator client. Four cheap operations per submission (§5.4), plus the
audit decision.

  Operation 1: diff scan + bundle integrity
  Operation 2: attestation chain verification
  Operation 3: training-log plausibility
  Operation 4: hidden-eval inference

Phase 0 attestation is mock (HMAC-signed JSON). Phase 0.5+ swaps in real
TDX + nvtrust quote verification — only proof.mock_attest.verify_mock_attestation
changes.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import run_hidden_eval, HiddenEvalResult
from model import KarpathianBase, KarpathianConfig
from proof.mock_attest import (
    MockAttestation,
    compute_container_measurement,
    verify_mock_attestation,
)
from miner.submit import verify_signature, lookup_handshake


@dataclass
class ValidatorReject:
    reason: str
    detail: str = ""


@dataclass
class ValidatorResult:
    miner_hotkey: str
    bundle_hash: str
    handshake_nonce: str
    hidden_eval: HiddenEvalResult | None = None
    training_summary: dict | None = None
    calibration: dict | None = None
    operations: dict = field(default_factory=dict)
    rejected: ValidatorReject | None = None

    def to_dict(self) -> dict:
        return {
            "miner_hotkey": self.miner_hotkey,
            "bundle_hash": self.bundle_hash,
            "handshake_nonce": self.handshake_nonce,
            "hidden_eval": asdict(self.hidden_eval) if self.hidden_eval else None,
            "training_summary": self.training_summary,
            "calibration": self.calibration,
            "operations": self.operations,
            "rejected": asdict(self.rejected) if self.rejected else None,
        }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _list_proof_sources(base: Path) -> list[Path]:
    """Same set as proof/runner.py's _list_proof_sources — must match for
    the container_measurement to be reproducible across miner+validator."""
    dirs = ["model", "recipe", "data", "eval", "calibration", "proof"]
    files = ["restricted_files.yaml", "README.md"]
    out: list[Path] = []
    for d in dirs:
        path = base / d
        if not path.exists():
            continue
        for p in sorted(path.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".json", ".md"}:
                out.append(p)
    for f in files:
        path = base / f
        if path.exists():
            out.append(path)
    return out


def op1_diff_and_integrity(
    karpathian_root: Path,
    submission_payload: dict,
    proof_dir: Path,
) -> tuple[bool, str]:
    """Verify the submission signature + bundle-hash integrity."""
    # Verify miner signature.
    ok = verify_signature(
        miner_hotkey=submission_payload["miner_hotkey"],
        bundle_hash=submission_payload["bundle_hash"],
        handshake_nonce=submission_payload["handshake_nonce"],
        signature_hex=submission_payload["signature_hex"],
        public_key_hex=submission_payload["public_key_hex"],
    )
    if not ok:
        return False, "submission signature invalid"

    # Verify handshake nonce was actually committed on-chain.
    chain_entry = lookup_handshake(karpathian_root, submission_payload["handshake_nonce"])
    if chain_entry is None:
        return False, "handshake nonce not found on chain"
    if chain_entry["miner_hotkey"] != submission_payload["miner_hotkey"]:
        return False, "handshake nonce was committed by a different miner"

    # Verify bundle manifest hashes match what's on disk.
    manifest = json.loads((proof_dir / "bundle_manifest.json").read_text())
    pairs = [
        ("checkpoint", proof_dir / "training" / "checkpoint.pt", manifest["checkpoint_sha256"]),
        ("training_log", proof_dir / "training" / "training_log.jsonl", manifest["training_log_sha256"]),
        ("calibration", proof_dir / "calibration.json", manifest["calibration_sha256"]),
    ]
    # Attestation is optional (unverified tier has no attestation.json).
    if manifest.get("attestation_sha256"):
        pairs.append(("attestation", proof_dir / "attestation.json", manifest["attestation_sha256"]))
    for name, path, expected in pairs:
        if not path.exists():
            return False, f"missing artifact {name} at {path}"
        actual = _file_sha256(path)
        if actual != expected:
            return False, f"{name} hash mismatch (expected {expected[:8]}, got {actual[:8]})"
    return True, "ok"


def op2_attestation_verify(
    karpathian_root: Path,
    submission_payload: dict,
    proof_dir: Path,
) -> tuple[bool, str, str]:
    """Verify attestation. Returns (ok, detail, tier).

    Tier-aware (whitepaper v1.1 §5.4):
      - If attestation.json is present and valid → tier = "verified"
      - If attestation.json is absent → tier = "unverified" (NOT rejected)
      - If attestation.json is present but INVALID → rejected (moral hazard:
        a failed verified claim is not silently downgraded to unverified)
    """
    att_path = proof_dir / "attestation.json"
    if not att_path.exists():
        return True, "no attestation — scoring as unverified (α=0.5)", "unverified"

    att_text = att_path.read_text()
    att = MockAttestation.from_json(att_text)
    expected_measurement = compute_container_measurement(_list_proof_sources(karpathian_root))
    ok, errors = verify_mock_attestation(
        att,
        expected_container_measurement=expected_measurement,
        expected_handshake_nonce=submission_payload["handshake_nonce"],
        expected_bundle_hash=submission_payload["bundle_hash"],
    )
    if not ok:
        return False, "verified-tier claim failed: " + "; ".join(errors), "rejected"
    return True, "attestation verified — scoring as verified (α=1.0)", "verified"


def op3_log_plausibility(proof_dir: Path) -> tuple[bool, str]:
    """Cheap sanity checks on the training log."""
    log_path = proof_dir / "training" / "training_log.jsonl"
    lines = [json.loads(l) for l in log_path.read_text().splitlines() if l.strip()]
    if not lines:
        return False, "empty training log"
    losses = [e["loss"] for e in lines]
    # No NaN / Inf
    for v in losses:
        if v != v or math.isinf(v):
            return False, "NaN/Inf in training loss"
    # Loss should not have suddenly exploded.
    if losses[-1] > 50.0:
        return False, f"final loss suspiciously high ({losses[-1]:.2f})"
    # No suspicious resume (training time should monotonically increase).
    elapsed = [e["elapsed_s"] for e in lines]
    for prev, cur in zip(elapsed, elapsed[1:]):
        if cur < prev - 0.5:  # tolerate small clock jitter
            return False, f"non-monotonic elapsed_s at step (prev={prev:.2f}, cur={cur:.2f})"
    return True, f"loss[0]={losses[0]:.3f} -> loss[-1]={losses[-1]:.3f}"


def op4_hidden_eval(
    karpathian_root: Path,
    proof_dir: Path,
) -> tuple[bool, str, HiddenEvalResult | None]:
    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    saved = ckpt["config"]
    cfg = KarpathianConfig(
        vocab_size=saved["vocab_size"],
        dim=saved["dim"],
        n_layers=saved["n_layers"],
        n_heads=saved["n_heads"],
        head_dim=saved["head_dim"],
        ffn_mult=saved["ffn_mult"],
        max_seq_len=saved["max_seq_len"],
    )
    model = KarpathianBase(cfg)
    model.load_state_dict(ckpt["model"])
    if torch.cuda.is_available():
        model = model.cuda()
    result = run_hidden_eval(model, karpathian_root / "eval" / "private", seq_len=cfg.max_seq_len // 2)
    return True, f"val_bpb={result.val_bpb:.4f} bench={result.benchmark_accuracy:.3f}", result


def judge_submission(
    karpathian_root: Path,
    proof_dir: Path,
) -> ValidatorResult:
    """Run the four ops in order. Any failure shorts out and returns a rejection."""
    sub_path = proof_dir / "submission.json"
    if not sub_path.exists():
        return ValidatorResult(
            miner_hotkey="?",
            bundle_hash="?",
            handshake_nonce="?",
            rejected=ValidatorReject("missing_submission_json", str(sub_path)),
        )
    submission = json.loads(sub_path.read_text())
    result = ValidatorResult(
        miner_hotkey=submission["miner_hotkey"],
        bundle_hash=submission["bundle_hash"],
        handshake_nonce=submission["handshake_nonce"],
    )

    ok, detail = op1_diff_and_integrity(karpathian_root, submission, proof_dir)
    result.operations["op1_diff_integrity"] = {"ok": ok, "detail": detail}
    if not ok:
        result.rejected = ValidatorReject("op1_diff_integrity", detail)
        return result

    ok, detail, tier = op2_attestation_verify(karpathian_root, submission, proof_dir)
    result.operations["op2_attestation"] = {"ok": ok, "detail": detail, "tier": tier}
    if not ok:
        result.rejected = ValidatorReject("op2_attestation", detail)
        return result

    ok, detail = op3_log_plausibility(proof_dir)
    result.operations["op3_log_plausibility"] = {"ok": ok, "detail": detail}
    if not ok:
        result.rejected = ValidatorReject("op3_log_plausibility", detail)
        return result

    ok, detail, hidden_eval = op4_hidden_eval(karpathian_root, proof_dir)
    result.operations["op4_hidden_eval"] = {"ok": ok, "detail": detail}
    result.hidden_eval = hidden_eval

    # Attach training + calibration summaries for downstream scoring.
    final_state_path = proof_dir / "training" / "final_state.json"
    if final_state_path.exists():
        result.training_summary = json.loads(final_state_path.read_text())
    cal_path = proof_dir / "calibration.json"
    if cal_path.exists():
        result.calibration = json.loads(cal_path.read_text())

    return result


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--karpathian-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--proof-dir", type=Path, required=True)
    args = p.parse_args()

    res = judge_submission(args.karpathian_root, args.proof_dir)
    print(json.dumps(res.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
