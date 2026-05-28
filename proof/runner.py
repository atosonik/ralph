"""
Proof-test runner.

Given a miner's submission directory containing:
  - patch.diff           (unified diff against canonical recipe)
  - proof_request.json   (handshake_nonce, declared seed, declared config)

The runner:
  1. Applies the patch to a working copy of the canonical recipe.
  2. Invokes the restricted-files diff scanner — refuses to proceed if the
     patch touches eval/, calibration/, validator/, or proof/.
  3. Runs canonical training (recipe/train.py) under the declared config + seed.
  4. Runs the calibration benchmark in the same environment.
  5. Computes a bundle hash of (patch || checkpoint || training_log || calibration).
  6. Generates the mock attestation chain incorporating the bundle hash.
  7. Writes everything to the output bundle directory.

In Phase 0.5+ steps 1-2 happen INSIDE the signed Docker image whose
measurement is on-chain pinned. The patch is applied inside the container;
the container refuses to start if the post-patch tree differs from the
canonical recipe on any restricted path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration import run_calibration
from proof.mock_attest import compute_container_measurement
from proof.real_attest import generate_attestation, detect_capabilities


def _list_proof_sources(karpa_root: Path) -> list[Path]:
    """Source files contributing to the container measurement.

    Covers both repos: protocol (eval, calibration, proof) lives in
    karpa_root; recipe (model, recipe, data) lives in RECIPE_DIR.
    """
    from karpa_bootstrap import RECIPE_DIR
    sources: list[tuple[Path, list[str]]] = [
        (RECIPE_DIR, ["model", "recipe", "data", "configs"]),
        (karpa_root, ["eval", "calibration", "proof"]),
    ]
    out: list[Path] = []
    for base, dirs in sources:
        for d in dirs:
            path = base / d
            if not path.exists():
                continue
            for p in sorted(path.rglob("*")):
                if p.is_file() and "__pycache__" not in p.parts and p.suffix in {".py", ".yaml", ".json", ".md"}:
                    out.append(p)
    for f in ["restricted_files.yaml", "README.md"]:
        path = karpa_root / f
        if path.exists():
            out.append(path)
    return out


def _load_restricted_paths(restricted_yaml: Path) -> list[str]:
    """Minimal YAML parsing — we only need the `restricted_paths` list."""
    text = restricted_yaml.read_text()
    patterns: list[str] = []
    in_list = False
    for line in text.splitlines():
        line_stripped = line.strip()
        if line_stripped.startswith("restricted_paths:"):
            in_list = True
            continue
        if in_list:
            if line_stripped.startswith("- "):
                value = line_stripped[2:].strip().strip('"').strip("'")
                patterns.append(value)
            elif line_stripped and not line.startswith((" ", "\t", "#")):
                in_list = False
    return patterns


def _path_matches(pattern: str, path: str) -> bool:
    """Glob-style match supporting prefix/**."""
    if pattern.endswith("/**"):
        prefix = pattern[:-3]
        return path == prefix.rstrip("/") or path.startswith(prefix)
    return path == pattern


def scan_diff_for_restricted(patch_text: str, restricted_patterns: list[str]) -> list[str]:
    """Return restricted paths the patch touches. Empty list = clean."""
    violations: list[str] = []
    for line in patch_text.splitlines():
        # unified diff: lines starting with `+++ ` or `--- ` carry file paths
        if line.startswith(("+++ ", "--- ")):
            rest = line[4:].strip()
            if rest in ("/dev/null",):
                continue
            # strip leading "a/" or "b/" if present
            for prefix in ("a/", "b/"):
                if rest.startswith(prefix):
                    rest = rest[len(prefix):]
            for pat in restricted_patterns:
                if _path_matches(pat, rest) and rest not in violations:
                    violations.append(rest)
    return violations


def apply_patch(workdir: Path, patch_path: Path) -> None:
    """Apply a unified diff using `git apply` (no git repo needed with --3way? No,
    we use plain patch). We use `patch -p1` since it's simpler."""
    if patch_path.stat().st_size == 0:
        # Empty patch == unchanged canonical baseline. Common for noise-floor
        # calibration runs. Nothing to do.
        return
    result = subprocess.run(
        ["patch", "-p1", "-i", str(patch_path.resolve()), "--no-backup-if-mismatch"],
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"patch failed:\n{result.stdout}\n{result.stderr}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    return sha256_hex(path.read_bytes())


@dataclass
class ProofTestBundle:
    """The artifacts a proof-test run produces."""
    workdir: Path
    patch_path: Path
    checkpoint_path: Path
    training_log_path: Path
    final_state_path: Path
    calibration_path: Path
    attestation_path: Path
    bundle_manifest_path: Path
    bundle_hash: str


def run_proof_test(
    karpa_root: Path,
    submission_dir: Path,
    out_dir: Path,
    total_steps_override: int | None = None,
    tier: str = "verified",
) -> ProofTestBundle:
    submission_dir = Path(submission_dir)
    out_dir = Path(out_dir)
    karpa_root = Path(karpa_root)

    proof_request = json.loads((submission_dir / "proof_request.json").read_text())
    handshake_nonce = proof_request["handshake_nonce"]
    declared_seed = int(proof_request.get("seed", 1337))
    config_overrides = proof_request.get("config_overrides", {})
    config_path = proof_request.get("config_path")
    patch_path = submission_dir / "patch.diff"

    # 1. Compute container measurement BEFORE patching (the canonical recipe's
    #    measurement). The container_measurement in Phase 0.5+ is the Docker
    #    image digest, which is independent of any miner patch.
    container_measurement = compute_container_measurement(_list_proof_sources(karpa_root))

    # 2. Scan the patch for restricted-file violations.
    restricted_yaml = karpa_root / "restricted_files.yaml"
    restricted_patterns = _load_restricted_paths(restricted_yaml)
    patch_text = patch_path.read_text() if patch_path.exists() else ""
    violations = scan_diff_for_restricted(patch_text, restricted_patterns)
    if violations:
        raise RuntimeError(f"patch touches restricted paths: {violations}")

    # 3. Create a working copy of the canonical recipe and apply the patch.
    #    The recipe (model/, recipe/, data/, configs/) lives in the sibling
    #    karpaai/recipe repo; eval/, calibration/ stay in the protocol repo.
    out_dir.mkdir(parents=True, exist_ok=True)
    workdir = out_dir / "workdir"
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    from karpa_bootstrap import RECIPE_DIR
    for sub in ("model", "recipe", "data", "configs"):
        src = RECIPE_DIR / sub
        if src.exists():
            shutil.copytree(src, workdir / sub, dirs_exist_ok=True)
    for sub in ("eval", "calibration"):
        src = karpa_root / sub
        if src.exists():
            shutil.copytree(src, workdir / sub, dirs_exist_ok=True)
    # Copy required top-level files.
    for f in ("restricted_files.yaml",):
        src = karpa_root / f
        if src.exists():
            shutil.copy2(src, workdir / f)
    if patch_text:
        # Save the patch into workdir for apply.
        local_patch = workdir / "submission.patch"
        local_patch.write_text(patch_text)
        apply_patch(workdir, local_patch)

    # 4. Run canonical training. We invoke recipe.train inside workdir.
    train_out = out_dir / "training"
    train_out.mkdir(parents=True, exist_ok=True)
    train_cmd = [
        sys.executable,
        "-m", "recipe.train",
        "--out-dir", str(train_out.resolve()),
        "--seed", str(declared_seed),
    ]
    if config_path:
        # config_path is recipe-relative (e.g. "configs/h100_proxy.json"); use the
        # workdir copy so any patch to the config is honoured.
        train_cmd += ["--config", str((workdir / config_path).resolve())]
    if total_steps_override is not None:
        train_cmd += ["--total-steps", str(total_steps_override)]
    # Manifest lives in the recipe repo (data.prepare writes it there). The
    # canonical training reads from that manifest, not the workdir copy, so the
    # shard byte-content is bound to the recipe checkout's manifest hash.
    train_cmd += ["--manifest", str((RECIPE_DIR / "data" / "data_manifest.json").resolve())]
    print(f"[proof] running training: {' '.join(train_cmd)}")
    train_result = subprocess.run(
        train_cmd,
        cwd=workdir,
        env={**__import__("os").environ, "PYTHONPATH": str(workdir.resolve())},
        capture_output=True,
        text=True,
    )
    if train_result.returncode != 0:
        raise RuntimeError(f"training failed:\nstdout:\n{train_result.stdout}\nstderr:\n{train_result.stderr}")
    print(train_result.stdout)

    # 5. Run calibration benchmark in the same environment.
    cal_result = run_calibration(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    cal_path = out_dir / "calibration.json"
    cal_path.write_text(json.dumps(asdict(cal_result), indent=2))

    # 6. Compute bundle hash + epoch records from training log timeline.
    checkpoint_path = train_out / "checkpoint.pt"
    training_log_path = train_out / "training_log.jsonl"
    final_state_path = train_out / "final_state.json"

    bundle_hash = sha256_hex(
        (
            file_hash(patch_path).encode() if patch_path.exists() else b""
        )
        + file_hash(checkpoint_path).encode()
        + file_hash(training_log_path).encode()
        + file_hash(cal_path).encode()
    )

    # 7. Build attestation chain (verified tier) or skip (unverified tier).
    att_path = None
    if tier == "verified":
        caps = detect_capabilities()
        epoch_records = _build_epoch_records(training_log_path, handshake_nonce)
        attestation = generate_attestation(
            container_measurement=container_measurement,
            handshake_nonce=handshake_nonce,
            epoch_records=epoch_records,
            bundle_hash=bundle_hash,
        )
        att_path = out_dir / "attestation.json"
        att_path.write_text(attestation.to_json())
        print(f"[proof] tier=verified, attestation_type={attestation.attestation_type} "
              f"(tdx={caps['tdx']}, nvcc={caps['nvcc']})")
    else:
        print(f"[proof] tier=unverified, no attestation chain generated")

    # 8. Bundle manifest: a single JSON listing every artifact + its hash.
    manifest = {
        "patch_sha256": file_hash(patch_path) if patch_path.exists() else None,
        "checkpoint_sha256": file_hash(checkpoint_path),
        "training_log_sha256": file_hash(training_log_path),
        "calibration_sha256": file_hash(cal_path),
        "final_state_sha256": file_hash(final_state_path),
        "attestation_sha256": file_hash(att_path) if att_path else None,
        "bundle_hash": bundle_hash,
        "container_measurement": container_measurement,
        "handshake_nonce": handshake_nonce,
        "declared_seed": declared_seed,
        "tier": tier,
    }
    manifest_path = out_dir / "bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    return ProofTestBundle(
        workdir=workdir,
        patch_path=patch_path,
        checkpoint_path=checkpoint_path,
        training_log_path=training_log_path,
        final_state_path=final_state_path,
        calibration_path=cal_path,
        attestation_path=att_path if att_path else Path("/dev/null"),
        bundle_manifest_path=manifest_path,
        bundle_hash=bundle_hash,
    )


def _build_epoch_records(
    training_log_path: Path,
    handshake_nonce: str,
) -> list[tuple[int, float, str]]:
    """Convert the training log into a sequence of (epoch_idx, timestamp,
    rolling_hash) records, simulating per-epoch self-attestations."""
    records: list[tuple[int, float, str]] = []
    rolling = handshake_nonce
    BUCKET = 10
    bucket_idx = 0
    bucket_lines: list[str] = []
    last_ts: float = time.time()
    for raw in training_log_path.read_text().splitlines():
        bucket_lines.append(raw)
        if len(bucket_lines) >= BUCKET:
            entry = json.loads(raw)
            ts = float(entry.get("elapsed_s", 0.0))
            rolling = hashlib.sha256(
                (rolling + "\n".join(bucket_lines)).encode()
            ).hexdigest()
            records.append((bucket_idx, last_ts + ts, rolling))
            bucket_idx += 1
            bucket_lines.clear()
    if bucket_lines:
        rolling = hashlib.sha256(
            (rolling + "\n".join(bucket_lines)).encode()
        ).hexdigest()
        records.append((bucket_idx, last_ts, rolling))
    return records


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--karpa-root", type=Path, default=Path(__file__).resolve().parent.parent)
    p.add_argument("--submission", type=Path, required=True, help="dir with patch.diff + proof_request.json")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--total-steps", type=int, default=None)
    p.add_argument("--tier", choices=["verified", "unverified"], default="verified",
                   help="verified = full attestation chain; unverified = no attestation, α=0.5 scoring discount")
    args = p.parse_args()

    bundle = run_proof_test(
        karpa_root=args.karpa_root,
        submission_dir=args.submission,
        out_dir=args.out_dir,
        total_steps_override=args.total_steps,
        tier=args.tier,
    )
    print(f"\n[proof] DONE bundle_hash={bundle.bundle_hash[:16]}...")
    print(f"        attestation: {bundle.attestation_path}")
    print(f"        manifest:    {bundle.bundle_manifest_path}")


if __name__ == "__main__":
    main()
