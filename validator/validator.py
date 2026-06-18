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

from model import RalphBase, RalphConfig

from eval import HiddenEvalResult, run_hidden_eval
from miner.submit import lookup_handshake, verify_signature
from proof.mock_attest import (
    MockAttestation,
    verify_mock_attestation,
)
from proof.real_attest import (
    RealAttestation,
)
from proof.real_attest import (
    verify_attestation as verify_real_attestation,
)
from proof.runner import _load_restricted_paths, scan_diff_for_restricted
from proof.sources import compute_container_measurement

# Hard-coded sanity bounds for the miner-submitted model config. The validator
# loads checkpoint['config'] from an attacker-controlled file; without bounds
# a malicious config could OOM the host or allocate gigabytes of memory before
# the actual RCE prevention (weights_only=True) gets a chance. These bounds
# are 10x the largest legitimate config we ship today (h100_scale.json).
MAX_VOCAB_SIZE = 200_000
MAX_DIM = 8192
MAX_N_LAYERS = 64
MAX_N_HEADS = 64
MAX_HEAD_DIM = 256
MAX_MAX_SEQ_LEN = 8192
MAX_CHECKPOINT_BYTES = 5 * 1024 * 1024 * 1024  # 5 GiB


@dataclass
class ValidatorReject:
    reason: str
    detail: str = ""


@dataclass
class ValidatorResult:
    miner_hotkey: str
    bundle_hash: str
    handshake_nonce: str
    miner_github: str = ""  # self-declared attribution; informational only
    pr_url: str = ""  # PR against RalphLabsAI/recipe (verified later in service.py)
    hidden_eval: HiddenEvalResult | None = None
    training_summary: dict | None = None
    calibration: dict | None = None
    operations: dict = field(default_factory=dict)
    rejected: ValidatorReject | None = None

    def to_dict(self) -> dict:
        return {
            "miner_hotkey": self.miner_hotkey,
            "miner_github": self.miner_github,
            "pr_url": self.pr_url,
            "bundle_hash": self.bundle_hash,
            "handshake_nonce": self.handshake_nonce,
            # Use to_legacy_dict() so the chain payload stays byte-equivalent
            # to pre-v0.11 form when downstream is None (B1-D12 contract).
            "hidden_eval": self.hidden_eval.to_legacy_dict() if self.hidden_eval else None,
            "training_summary": self.training_summary,
            "calibration": self.calibration,
            "operations": self.operations,
            "rejected": asdict(self.rejected) if self.rejected else None,
        }


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_load_checkpoint_config(ckpt_path: Path) -> dict:
    """Load checkpoint['config'] safely.

    We need the config (vocab_size, dim, n_layers, ...) to instantiate the
    model BEFORE loading state_dict. The legacy on-disk format stores config
    inside the checkpoint pickle, which means loading the config exposes us
    to the same pickle-RCE we're trying to avoid.

    Strategy:
      1. Prefer a sibling `checkpoint_config.json` (new format, JSON-only).
      2. Fall back to weights_only=True + extra unpickler patches.
      3. Reject if both fail.

    Bounds-check every field before returning — the values are then passed to
    RalphConfig which allocates tensors proportional to them.
    """
    sidecar = ckpt_path.parent / "checkpoint_config.json"
    if sidecar.exists():
        cfg = json.loads(sidecar.read_text())
    else:
        # Legacy path: extract config from the pickle. We use weights_only=True
        # which since PyTorch 2.4 raises on any arbitrary class reducer, but
        # still permits primitives. The "config" key is plain dict-of-prims so
        # this works for honest miners; malicious pickles raise.
        try:
            ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"checkpoint config could not be loaded safely: {e}. "
                f"Miner must ship checkpoint_config.json alongside checkpoint.pt."
            )
        cfg = ckpt.get("config", {})
        if not isinstance(cfg, dict):
            raise RuntimeError("checkpoint['config'] is not a dict")
    bounds = [
        ("vocab_size", MAX_VOCAB_SIZE),
        ("dim", MAX_DIM),
        ("n_layers", MAX_N_LAYERS),
        ("n_heads", MAX_N_HEADS),
        ("head_dim", MAX_HEAD_DIM),
        ("max_seq_len", MAX_MAX_SEQ_LEN),
    ]
    for key, ceiling in bounds:
        v = cfg.get(key)
        if not isinstance(v, int) or v <= 0 or v > ceiling:
            raise RuntimeError(f"checkpoint config {key}={v!r} out of bounds (max {ceiling})")
    ffn_mult = cfg.get("ffn_mult", 8 / 3)
    if not isinstance(ffn_mult, (int, float)) or not (0.1 <= float(ffn_mult) <= 32.0):
        raise RuntimeError(f"checkpoint config ffn_mult={ffn_mult!r} out of bounds")
    return cfg


def _safe_load_checkpoint_weights(ckpt_path: Path, expected_keys: set[str] | None = None) -> dict:
    """Load checkpoint['model'] state_dict via weights_only=True.

    The size cap is checked BEFORE torch.load to bound memory and prevent
    20GB checkpoint DoS. weights_only=True (PyTorch 2.4+) rejects any
    non-tensor / non-primitive payload so a malicious pickle can't fire a
    reducer.
    """
    size = ckpt_path.stat().st_size
    if size > MAX_CHECKPOINT_BYTES:
        raise RuntimeError(f"checkpoint too large: {size} bytes (max {MAX_CHECKPOINT_BYTES})")
    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    state_dict = ckpt.get("model", ckpt if isinstance(ckpt, dict) and "model" not in ckpt else None)
    if state_dict is None:
        raise RuntimeError("checkpoint has no 'model' state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"checkpoint['model'] is not a dict: {type(state_dict).__name__}")
    return state_dict


def op1_diff_and_integrity(
    ralph_root: Path,
    submission_payload: dict,
    proof_dir: Path,
) -> tuple[bool, str]:
    """Verify miner signature + bundle integrity + restricted-file scan.

    Whitepaper §5.4 Operation 1. Hardened per deep_review_2026-05-31:
      #8  patch.diff is now integrity-checked vs manifest['patch_sha256'],
          and bundle_hash is recomputed from disk and required to match
          submission_payload['bundle_hash'] + manifest['bundle_hash'].
      #9  if on-chain handshake is enforced, patch_hash from the chain
          entry must equal manifest['patch_sha256'].
      #12 the restricted-file scanner is now invoked on the validator
          side (was previously only miner-side).
    """
    # Verify miner signature — hypothesis is part of the signed payload
    # post-fix so miners can't replace a "we're investigating X" rationale
    # with "we proved X" after the fact.
    submission_hypothesis = submission_payload.get("hypothesis", "")
    ok = verify_signature(
        miner_hotkey=submission_payload["miner_hotkey"],
        bundle_hash=submission_payload["bundle_hash"],
        handshake_nonce=submission_payload["handshake_nonce"],
        signature_hex=submission_payload["signature_hex"],
        public_key_hex=submission_payload["public_key_hex"],
        hypothesis=submission_hypothesis,
    )
    if not ok:
        return False, "submission signature invalid"

    # Verify bundle manifest hashes match what's on disk.
    manifest_path = proof_dir / "bundle_manifest.json"
    if not manifest_path.exists():
        return False, "missing bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    pairs = [
        ("checkpoint", proof_dir / "training" / "checkpoint.pt", manifest.get("checkpoint_sha256")),
        ("training_log", proof_dir / "training" / "training_log.jsonl", manifest.get("training_log_sha256")),
        ("calibration", proof_dir / "calibration.json", manifest.get("calibration_sha256")),
    ]
    # Attestation is now required (single attested-execution tier).
    if manifest.get("attestation_sha256"):
        pairs.append(("attestation", proof_dir / "attestation.json", manifest["attestation_sha256"]))
    # patch.diff: integrity-check whenever manifest declares a hash for it.
    patch_path = proof_dir / "patch.diff"
    patch_sha = manifest.get("patch_sha256")
    if patch_sha:
        pairs.append(("patch", patch_path, patch_sha))
    for name, path, expected in pairs:
        if expected is None:
            return False, f"manifest missing {name} hash"
        if not path.exists():
            return False, f"missing artifact {name} at {path}"
        actual = _file_sha256(path)
        if actual != expected:
            return False, f"{name} hash mismatch (expected {expected[:8]}, got {actual[:8]})"

    # Recompute bundle_hash from disk and require it match BOTH the
    # submission's signed-over hash AND the manifest's declared bundle hash.
    # The recipe for bundle_hash is the same as in proof.runner.run_proof_test.
    bundle_components: list[bytes] = []
    if patch_path.exists():
        bundle_components.append(_file_sha256(patch_path).encode())
    else:
        # Baseline bundles can ship without patch.diff (empty patch).
        bundle_components.append(b"")
    bundle_components.append(_file_sha256(proof_dir / "training" / "checkpoint.pt").encode())
    bundle_components.append(_file_sha256(proof_dir / "training" / "training_log.jsonl").encode())
    bundle_components.append(_file_sha256(proof_dir / "calibration.json").encode())
    recomputed = hashlib.sha256(b"".join(bundle_components)).hexdigest()
    if submission_payload.get("bundle_hash") != recomputed:
        return False, (
            f"bundle_hash mismatch: signed={submission_payload.get('bundle_hash','?')[:12]}, "
            f"recomputed={recomputed[:12]}"
        )
    if manifest.get("bundle_hash") != recomputed:
        return False, (
            f"manifest bundle_hash mismatch: manifest={manifest.get('bundle_hash','?')[:12]}, "
            f"recomputed={recomputed[:12]}"
        )

    # Verify handshake nonce was committed on-chain. Until on-chain commits
    # work reliably for the test, the lookup may fail for legitimate cross-host
    # miners — set RALPH_SKIP_HANDSHAKE=1 to skip in that mode.
    import os as _os
    if not _os.environ.get("RALPH_SKIP_HANDSHAKE"):
        chain_entry = lookup_handshake(ralph_root, submission_payload["handshake_nonce"])
        if chain_entry is None:
            return False, "handshake nonce not found on chain"
        if chain_entry.get("miner_hotkey") != submission_payload["miner_hotkey"]:
            return False, "handshake nonce was committed by a different miner"
        # #9: cross-check that the patch the miner committed on-chain matches
        # the patch in the bundle. Without this, a miner can commit "look
        # I'm running patch X" and ship a bundle for entirely different patch Y.
        chain_patch_hash = chain_entry.get("patch_hash")
        if chain_patch_hash and patch_sha and chain_patch_hash != patch_sha:
            return False, (
                f"on-chain patch_hash mismatch: chain={chain_patch_hash[:12]}, "
                f"bundle={patch_sha[:12]}"
            )

    # #12: restricted-file scanner now runs on the validator side. The
    # miner-side scan in proof.runner was bypassable by a miner who skipped
    # invoking the runner.
    if patch_path.exists():
        patch_text = patch_path.read_text(encoding="utf-8", errors="replace")
        restricted_yaml = ralph_root / "restricted_files.yaml"
        if restricted_yaml.exists():
            patterns = _load_restricted_paths(restricted_yaml)
            violations = scan_diff_for_restricted(patch_text, patterns)
            if violations:
                return False, f"patch touches restricted paths: {violations}"

    return True, "ok"


def op2_attestation_verify(
    ralph_root: Path,
    submission_payload: dict,
    proof_dir: Path,
) -> tuple[bool, str, str]:
    """Verify attestation. Returns (ok, detail, tier).

    Whitepaper v1.2 §5.4 — single attested-execution tier. There is no
    "unverified" path; a submission without a valid attestation chain is
    REJECTED outright.

    On mainnet (RALPH_ALLOW_MOCK_ATTESTATION unset or != "1") only real_*
    attestation types are accepted. Mock attestations exist in the repo as
    open-source code so anyone can forge them — they are explicitly rejected
    on mainnet. Testnet operators can set RALPH_ALLOW_MOCK_ATTESTATION=1 to
    accept mocks (with a loud warning).

    See deep_review_2026-05-31 critical #3/#4/#5.
    """
    import os as _os

    att_path = proof_dir / "attestation.json"
    if not att_path.exists():
        # Whitepaper v1.2: no attestation = rejection. The legacy
        # "unverified tier α=0.5" path is retired.
        return False, "missing attestation.json — single attested-execution tier required (v1.2 §5.4)", "rejected"

    att_text = att_path.read_text()
    att_data = json.loads(att_text)
    from ralph_bootstrap import RECIPE_DIR
    expected_measurement = compute_container_measurement(ralph_root, recipe_dir=RECIPE_DIR)

    allow_mock = _os.environ.get("RALPH_ALLOW_MOCK_ATTESTATION") == "1"

    # Auto-detect attestation format: real (has attestation_type field) vs legacy mock
    if "attestation_type" in att_data:
        att = RealAttestation.from_json(att_text)
        att_type_label = att.attestation_type
        is_real = att_type_label.startswith("real_")
        if not is_real and not allow_mock:
            return False, (
                f"attestation_type={att_type_label!r} is not real_*; mock "
                "attestations rejected on mainnet (set RALPH_ALLOW_MOCK_ATTESTATION=1 "
                "for testnet)"
            ), "rejected"
        if not is_real and allow_mock:
            import sys as _sys
            print(
                "[attest] WARNING: RALPH_ALLOW_MOCK_ATTESTATION=1 — accepting "
                f"mock attestation_type={att_type_label!r}. MUST NOT BE SET ON MAINNET.",
                file=_sys.stderr,
            )
        ok, errors = verify_real_attestation(
            att,
            expected_container_measurement=expected_measurement,
            expected_handshake_nonce=submission_payload["handshake_nonce"],
            expected_bundle_hash=submission_payload["bundle_hash"],
        )
    else:
        # Legacy mock format (no attestation_type field).
        if not allow_mock:
            return False, (
                "legacy mock attestation rejected on mainnet "
                "(set RALPH_ALLOW_MOCK_ATTESTATION=1 for testnet)"
            ), "rejected"
        import sys as _sys
        print(
            "[attest] WARNING: RALPH_ALLOW_MOCK_ATTESTATION=1 — accepting "
            "legacy mock attestation. MUST NOT BE SET ON MAINNET.",
            file=_sys.stderr,
        )
        att = MockAttestation.from_json(att_text)
        ok, errors = verify_mock_attestation(
            att,
            expected_container_measurement=expected_measurement,
            expected_handshake_nonce=submission_payload["handshake_nonce"],
            expected_bundle_hash=submission_payload["bundle_hash"],
        )
        att_type_label = "mock"
        is_real = False

    if not ok:
        return False, f"attestation verification failed ({att_type_label}): " + "; ".join(errors), "rejected"

    # v1.2: single attested-execution tier. No α discount.
    detail = f"attestation verified ({att_type_label})"
    return True, detail, "verified"


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


def _is_state_dict_shape_mismatch(err: Exception) -> bool:
    """Detect when a load_state_dict failure is due to architecture divergence
    between the validator's canonical RalphBase and the miner's trained model.

    These are the recoverable cases — the miner's patch added/removed/renamed
    parameters relative to canonical. We can retry under a patched-workdir
    subprocess. Other RuntimeErrors (corrupt tensors, etc.) re-raise.
    """
    msg = str(err)
    return (
        "Unexpected key" in msg
        or "Missing key" in msg
        or "size mismatch" in msg
    )


def _patched_hidden_eval(
    ralph_root: Path,
    proof_dir: Path,
    ckpt_path: Path,
) -> tuple[bool, str, HiddenEvalResult | None]:
    """Fallback path when canonical RalphBase can't load the miner's checkpoint.

    Creates a temp workdir, applies the miner's patch.diff against a copy of
    the canonical recipe, and runs eval_in_workdir.py as a subprocess so the
    patched model code is loaded fresh (no module-reload hazards in our own
    interpreter). Returns the same triple shape as op4_hidden_eval so callers
    don't need to branch on the path.
    """
    import shutil
    import subprocess
    import tempfile

    from proof.runner import apply_patch

    patch_path = proof_dir / "patch.diff"
    if not patch_path.exists():
        return False, "state_dict mismatch and no patch.diff to retry with", None

    try:
        from ralph_bootstrap import RECIPE_DIR
    except Exception as e:
        return False, f"patched-eval setup: bootstrap import failed: {e}", None

    with tempfile.TemporaryDirectory(prefix="ralph_patched_eval_") as tmp:
        workdir = Path(tmp) / "workdir"
        workdir.mkdir(parents=True)
        # Mirror the proof.runner layout: recipe sources from RECIPE_DIR, eval
        # / calibration from the ralph protocol root.
        for sub in ("model", "recipe", "data", "configs"):
            src = RECIPE_DIR / sub
            if src.exists():
                shutil.copytree(src, workdir / sub, dirs_exist_ok=True)
        for sub in ("eval", "calibration"):
            src = ralph_root / sub
            if src.exists():
                shutil.copytree(src, workdir / sub, dirs_exist_ok=True)

        try:
            apply_patch(workdir, patch_path)
        except Exception as e:
            return False, f"patched-eval: patch apply failed: {str(e)[:200]}", None

        helper = Path(__file__).resolve().parent / "eval_in_workdir.py"
        if not helper.exists():
            return False, f"patched-eval: helper script missing at {helper}", None

        try:
            res = subprocess.run(
                [sys.executable, str(helper), str(workdir), str(ckpt_path), str(ralph_root)],
                capture_output=True,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            return False, "patched-eval subprocess timed out (>240s)", None
        if res.returncode != 0:
            tail = (res.stderr or "")[-300:]
            return False, f"patched-eval subprocess exit={res.returncode}: {tail}", None

        marker = "RALPH_EVAL_RESULT "
        line = next(
            (ln for ln in (res.stdout or "").splitlines() if ln.startswith(marker)),
            None,
        )
        if line is None:
            return False, "patched-eval: no RALPH_EVAL_RESULT line in stdout", None

        fields: dict[str, str] = {}
        for tok in line[len(marker):].split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = v
        required = ("val_bpb", "benchmark_acc", "tokens_evaluated", "benchmark_examples", "eval_set_hash")
        if not all(k in fields for k in required):
            return False, f"patched-eval: malformed result line: {line!r}", None

        # Optional audit-reproducibility fields (validation-v2 Phase 1). Older
        # helper versions don't emit them; "none" maps to None. Parse
        # defensively so a malformed optional field never fails the eval.
        def _opt_float(key: str) -> float | None:
            v = fields.get(key)
            if v is None or v == "none":
                return None
            try:
                return float(v)
            except ValueError:
                return None

        def _opt_int(key: str) -> int | None:
            v = fields.get(key)
            if v is None or v == "none":
                return None
            try:
                return int(v)
            except ValueError:
                return None

        def _opt_str(key: str) -> str | None:
            v = fields.get(key)
            return None if (v is None or v == "none") else v

        try:
            result = HiddenEvalResult(
                val_bpb=float(fields["val_bpb"]),
                benchmark_accuracy=float(fields["benchmark_acc"]),
                tokens_evaluated=int(fields["tokens_evaluated"]),
                benchmark_examples=int(fields["benchmark_examples"]),
                eval_set_hash=fields["eval_set_hash"],
                val_seq_len=_opt_int("val_seq_len"),
                sealed_stream_manifest_hash=_opt_str("sealed_stream_manifest_hash"),
                tail_val_bpb=_opt_float("tail_val_bpb"),
            )
        except ValueError as e:
            return False, f"patched-eval: result line parse error: {e}", None
        return (
            True,
            f"val_bpb={result.val_bpb:.4f} bench={result.benchmark_accuracy:.3f} (patched-eval)",
            result,
        )


def op4_hidden_eval(
    ralph_root: Path,
    proof_dir: Path,
) -> tuple[bool, str, HiddenEvalResult | None]:
    ckpt_path = proof_dir / "training" / "checkpoint.pt"
    if not ckpt_path.exists():
        return False, f"missing checkpoint at {ckpt_path}", None
    # Load config + weights using the SAFE path — no pickle reducers, bounds-checked.
    saved = _safe_load_checkpoint_config(ckpt_path)
    state_dict = _safe_load_checkpoint_weights(ckpt_path)
    cfg = RalphConfig(
        vocab_size=saved["vocab_size"],
        dim=saved["dim"],
        n_layers=saved["n_layers"],
        n_heads=saved["n_heads"],
        head_dim=saved["head_dim"],
        ffn_mult=saved.get("ffn_mult", 8 / 3),
        max_seq_len=saved["max_seq_len"],
    )
    try:
        model = RalphBase(cfg)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        # Architecture divergence between canonical RalphBase and the miner's
        # trained model (typically a structural patch that adds parameters).
        # Retry under the patched workdir so the actually-trained model code
        # is what scores the checkpoint.
        if _is_state_dict_shape_mismatch(e):
            return _patched_hidden_eval(ralph_root, proof_dir, ckpt_path)
        raise
    if torch.cuda.is_available():
        model = model.cuda()
    result = run_hidden_eval(model, ralph_root / "eval" / "private", seq_len=cfg.max_seq_len // 2)
    return True, f"val_bpb={result.val_bpb:.4f} bench={result.benchmark_accuracy:.3f}", result


def judge_submission(
    ralph_root: Path,
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
        miner_github=submission.get("miner_github", ""),
        pr_url=submission.get("pr_url", ""),
        bundle_hash=submission["bundle_hash"],
        handshake_nonce=submission["handshake_nonce"],
    )

    ok, detail = op1_diff_and_integrity(ralph_root, submission, proof_dir)
    result.operations["op1_diff_integrity"] = {"ok": ok, "detail": detail}
    if not ok:
        result.rejected = ValidatorReject("op1_diff_integrity", detail)
        return result

    ok, detail, tier = op2_attestation_verify(ralph_root, submission, proof_dir)
    result.operations["op2_attestation"] = {"ok": ok, "detail": detail, "tier": tier}
    if not ok:
        result.rejected = ValidatorReject("op2_attestation", detail)
        return result

    ok, detail = op3_log_plausibility(proof_dir)
    result.operations["op3_log_plausibility"] = {"ok": ok, "detail": detail}
    if not ok:
        result.rejected = ValidatorReject("op3_log_plausibility", detail)
        return result

    ok, detail, hidden_eval = op4_hidden_eval(ralph_root, proof_dir)
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
    p.add_argument("--ralph-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--proof-dir", type=Path, required=True)
    args = p.parse_args()

    res = judge_submission(args.ralph_root, args.proof_dir)
    print(json.dumps(res.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
