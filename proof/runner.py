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
from proof.real_attest import detect_capabilities, generate_attestation
from proof.sources import compute_container_measurement

# Allowlist of env vars the patched training subprocess is permitted to see.
# Miners control recipe/train.py via patch.diff, so anything inherited by the
# training subprocess can be exfiltrated through the training_log (which gets
# published) or a network call. Keep this tight; reject everything else.
# See deep_review_2026-05-31 critical #2.
_TRAINING_ENV_ALLOWLIST = (
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "LC_CTYPE",
    "TERM", "TMPDIR", "TEMP", "TMP",
    # GPU / CUDA
    "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "CUDA_DEVICE_ORDER",
    "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH",
    # Torch perf knobs (read-only — can't exfil)
    "TORCH_USE_CUDA_DSA", "TORCH_CUDA_ARCH_LIST", "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    # HF cache locations (NOT tokens — tokens scrubbed below)
    "HF_HOME", "HF_DATASETS_CACHE", "TRANSFORMERS_CACHE",
    # wandb run-mode toggle, NOT api key (api key handled below)
    "WANDB_MODE", "WANDB_DIR",
)
# Env vars that must be SCRUBBED even if a downstream library would otherwise
# read them. These are the secrets the miner subprocess must never see.
_TRAINING_ENV_BLOCKLIST = (
    "BT_WALLET_PASSWORD", "BT_WALLET", "BT_HOTKEY",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_TOKEN",
    "KARPA_BOT_GH_TOKEN", "KARPA_BOT_HF_TOKEN",
    "SHADEFORM_API_KEY", "WANDB_API_KEY",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "AWS_SECRET_ACCESS_KEY", "AWS_ACCESS_KEY_ID", "AWS_SESSION_TOKEN",
    "GCP_SERVICE_ACCOUNT_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "GH_TOKEN", "GITHUB_TOKEN",
)


def _sanitized_env(extra: dict | None = None) -> dict:
    """Build an env for the training subprocess: allowlist-only + extras."""
    import os as _os
    out = {}
    for k in _TRAINING_ENV_ALLOWLIST:
        v = _os.environ.get(k)
        if v is not None:
            out[k] = v
    # Defense-in-depth: pop blocklist explicitly even if allowlist matched.
    for k in _TRAINING_ENV_BLOCKLIST:
        out.pop(k, None)
    if extra:
        for k, v in extra.items():
            if k in _TRAINING_ENV_BLOCKLIST:
                raise ValueError(f"refusing to inject blocklisted env var into training subprocess: {k}")
            out[k] = v
    return out


# Token-redaction set for subprocess stderr/stdout in error paths. Populated
# lazily — secrets read once at first redaction call so we don't keep them in
# module scope longer than needed.
_REDACT_CACHE: tuple[str, ...] | None = None


def _redacted(text: str) -> str:
    """Scrub known secret values from text before logging / re-raising."""
    global _REDACT_CACHE
    if _REDACT_CACHE is None:
        import os as _os
        vals = []
        for k in _TRAINING_ENV_BLOCKLIST:
            v = _os.environ.get(k, "")
            if len(v) >= 8:  # avoid redacting common short strings
                vals.append(v)
        _REDACT_CACHE = tuple(vals)
    out = text
    for v in _REDACT_CACHE:
        out = out.replace(v, "<REDACTED>")
    return out


def _list_proof_sources(karpa_root: Path) -> list[Path]:
    """DEPRECATED — kept for any in-tree caller. New code uses
    proof.sources.list_proof_sources which produces (base, rel) tuples
    that hash to the same digest from any filesystem layout."""
    from karpa_bootstrap import RECIPE_DIR
    from proof.sources import list_proof_sources
    return [base / rel for base, rel in list_proof_sources(karpa_root, RECIPE_DIR)]


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
    import os.path
    # Normalize the path: collapse ./, .., duplicate slashes. POSIX style.
    path = os.path.normpath(path).replace("\\", "/")
    if path.startswith("./"):
        path = path[2:]
    if pattern.endswith("/**"):
        prefix = pattern[:-3].rstrip("/")
        return path == prefix or path.startswith(prefix + "/")
    return path == pattern


def _extract_diff_paths(patch_text: str) -> list[str]:
    """Extract every file path a unified diff touches.

    Handles:
      - `--- a/path` / `+++ b/path` headers (the common case)
      - Tab-suffixed paths: `--- a/path\t2024-01-01...`
      - Git extended headers: `diff --git a/foo b/bar`, `rename from`,
        `rename to`, `copy from`, `copy to`
      - `Index: path` and `Only in:` markers (rare but possible)
      - /dev/null sentinels (file create/delete) — ignored

    Deduplicated, POSIX-normalized.
    """
    paths: list[str] = []

    def _add(raw: str) -> None:
        raw = raw.strip()
        if not raw or raw == "/dev/null":
            return
        # Strip tab-suffix timestamps from GNU diff output
        if "\t" in raw:
            raw = raw.split("\t", 1)[0].strip()
        # Strip leading quotes from `diff --git "a/file with spaces" "b/..."`
        if raw.startswith('"') and raw.endswith('"'):
            raw = raw[1:-1]
        # Strip leading a/ or b/ prefix (standard git)
        for prefix in ("a/", "b/", "c/"):
            if raw.startswith(prefix):
                raw = raw[len(prefix):]
                break
        # Normalize relative components, collapse backslashes
        import os.path
        norm = os.path.normpath(raw).replace("\\", "/")
        if norm.startswith("./"):
            norm = norm[2:]
        if norm and norm not in paths:
            paths.append(norm)

    for line in patch_text.splitlines():
        if line.startswith(("+++ ", "--- ")):
            _add(line[4:])
        elif line.startswith("diff --git "):
            # `diff --git a/foo b/bar` — extract both file names
            rest = line[len("diff --git "):]
            # Handle quoted paths with spaces
            if rest.startswith('"'):
                # Find the closing quote
                end_first = rest.find('" "', 1)
                if end_first > 0:
                    _add(rest[1:end_first])
                    _add(rest[end_first + 3:].rstrip('"'))
                    continue
            # Split on first space group
            parts = rest.split()
            if len(parts) >= 2:
                _add(parts[0])
                _add(parts[1])
        elif line.startswith(("rename from ", "rename to ", "copy from ", "copy to ")):
            _add(line.split(" ", 2)[2])
        elif line.startswith("Index: "):
            _add(line[7:])
    return paths


def scan_diff_for_restricted(patch_text: str, restricted_patterns: list[str]) -> list[str]:
    """Return restricted paths the patch touches. Empty list = clean.

    Hardened against scanner bypasses (deep_review_2026-05-31 high #1):
      - Tab-suffixed paths (`--- a/file\t2024-...`)
      - Rename/copy headers (`rename from`, `rename to`, `copy from`,
        `copy to`)
      - `diff --git` lines (capture both source + destination)
      - Path traversal via `..` (normalized away)
      - Backslash path separators (normalized to forward slash)
    """
    violations: list[str] = []
    for path in _extract_diff_paths(patch_text):
        for pat in restricted_patterns:
            if _path_matches(pat, path) and path not in violations:
                violations.append(path)
                break
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
    from karpa_bootstrap import RECIPE_DIR
    container_measurement = compute_container_measurement(karpa_root, recipe_dir=RECIPE_DIR)

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
        # SAFE env: allowlist + scrubbed blocklist. The training subprocess is
        # miner-controlled via patch.diff — it must not see HF/GH tokens, the
        # wallet password, or any cloud creds. See deep_review_2026-05-31 #2.
        env=_sanitized_env(extra={"PYTHONPATH": str(workdir.resolve())}),
        capture_output=True,
        text=True,
    )
    if train_result.returncode != 0:
        # Redact known secrets from stdout/stderr before re-raising or
        # logging — even with the env allowlist above, defense-in-depth.
        raise RuntimeError(
            "training failed:\nstdout:\n" + _redacted(train_result.stdout)
            + "\nstderr:\n" + _redacted(train_result.stderr)
        )
    print(_redacted(train_result.stdout))

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
        print("[proof] tier=unverified, no attestation chain generated")

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
