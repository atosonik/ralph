"""Hardened Docker sandbox for executing UNTRUSTED miner code on the validator.

The validator must run miner-authored Python (the patched `model.py` for op4
hidden-eval, and the full patched recipe for the re-train audit). Today that runs
in a bare same-user subprocess with the validator's filesystem, network, and —
until the env-sanitize stopgap — its secrets. This module contains that
execution inside a hardened container so a malicious submission cannot reach the
seal privkey, the wallet, the hidden eval set, or the host.

Design reference: Ralph-Validator-Sandbox-Plan.md.

Non-negotiable controls (enforced in `build_docker_argv`, asserted in tests):
  --network none · --read-only · non-root · --cap-drop ALL ·
  --security-opt no-new-privileges · --pids-limit · --memory(==swap) · --cpus ·
  --ipc=private · a SINGLE pinned GPU device (never `--gpus all`) · NO secrets in
  env or mounts · image pinned by digest · a host-side wall-clock watchdog that
  `docker kill`s on overrun.

FAIL-CLOSED: `preflight()` raises `SandboxUnavailable` if the runtime or its
hardening config cannot be verified. Callers MUST treat that as a rejected
submission and MUST NOT fall back to a bare subprocess.

This first cut provides the runner, the preflight gate, and the result parser.
Call-site wiring (op4 + audit), the in-container entrypoint, and the sandbox
image are landed alongside it.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

from proof.runner import _TRAINING_ENV_BLOCKLIST, _redacted

# Version floors that close the 2024-2026 escape classes (Leaky Vessels, the
# Nov-2025 runc masked-path trio, NVIDIAScape). See the design doc §5.
MIN_DOCKER_VERSION = (25, 0, 2)
MIN_RUNC_VERSION = (1, 2, 8)
MIN_TOOLKIT_VERSION = (1, 17, 8)

# Flags that must NEVER appear in a sandbox invocation.
FORBIDDEN_FLAGS = ("--privileged", "--pid=host", "--net=host", "--network=host", "--userns=host")

_RESULT_RE = re.compile(r"^RALPH_EVAL_RESULT ")


class SandboxUnavailable(RuntimeError):
    """The sandbox runtime or its required hardening config is not verifiably
    present. Callers MUST reject the submission — never fall back to bare exec."""


@dataclass(frozen=True)
class Mount:
    """A read-only-by-default bind mount into the sandbox."""
    host: Path
    container: str
    ro: bool = True

    def as_flag(self) -> str:
        return f"{Path(self.host).resolve()}:{self.container}:{'ro' if self.ro else 'rw'}"


@dataclass(frozen=True)
class SandboxConfig:
    """Tunables for one sandboxed run. `image` MUST be pinned by digest."""
    image: str  # e.g. "ralph-eval-sandbox@sha256:..."
    gpu_device: int | None = 0  # a SINGLE device index, or None for CPU-only
    memory: str = "16g"
    cpus: str = "8"
    pids_limit: int = 512
    uid_gid: str = "65534:65534"  # nobody:nogroup
    scratch_size: str = "8g"
    seccomp_profile: Path | None = None  # path to a custom seccomp json
    apparmor_profile: str | None = None  # loaded profile name, e.g. "ralph-sandbox"
    network: str = "none"  # never anything else for untrusted code


@dataclass
class SandboxResult:
    returncode: int
    stdout: str
    stderr: str  # already redacted
    timed_out: bool
    eval_line: str | None = None  # the parsed RALPH_EVAL_RESULT line, if present


# ---------------------------------------------------------------------------
# Argv construction (PURE + unit-tested — this is the security boundary)
# ---------------------------------------------------------------------------

def build_container_env(env_extra: dict[str, str] | None) -> dict[str, str]:
    """Minimal, explicit container env. Never forwards host env; rejects any
    blocklisted key. The image supplies PATH/torch; we add only safe knobs."""
    env: dict[str, str] = {
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": "/scratch",
        "TMPDIR": "/scratch/tmp",
    }
    for k, v in (env_extra or {}).items():
        if k in _TRAINING_ENV_BLOCKLIST:
            raise ValueError(f"refusing to inject blocklisted env var into sandbox: {k}")
        env[k] = v
    return env


def build_docker_argv(
    cfg: SandboxConfig,
    *,
    container_argv: list[str],
    mounts: list[Mount],
    out_dir: Path,
    name: str,
    env: dict[str, str],
) -> list[str]:
    """Build the hardened `docker run` argv. Raises on any unsafe input so a
    mistake is a crash, not a silent weakening of the boundary."""
    if "@sha256:" not in cfg.image:
        raise ValueError(f"sandbox image must be pinned by digest, got {cfg.image!r}")
    if cfg.gpu_device is not None and cfg.gpu_device < 0:
        raise ValueError("gpu_device must be a non-negative index or None")
    for k in env:
        if k in _TRAINING_ENV_BLOCKLIST:
            raise ValueError(f"blocklisted env var in sandbox env: {k}")

    argv: list[str] = [
        "docker", "run", "--rm",
        "--name", name,
        f"--network={cfg.network}",
        "--read-only",
        f"--user={cfg.uid_gid}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--ipc=private",
        f"--pids-limit={cfg.pids_limit}",
        f"--memory={cfg.memory}",
        f"--memory-swap={cfg.memory}",  # == memory ⇒ swap disabled
        f"--cpus={cfg.cpus}",
        "--tmpfs", f"/scratch:rw,nosuid,nodev,noexec,size={cfg.scratch_size},mode=1777",
    ]
    if cfg.seccomp_profile is not None:
        argv += ["--security-opt", f"seccomp={Path(cfg.seccomp_profile).resolve()}"]
    if cfg.apparmor_profile is not None:
        argv += ["--security-opt", f"apparmor={cfg.apparmor_profile}"]
    if cfg.gpu_device is not None:
        # A SINGLE pinned device. Never "all" — that exposes every GPU + the
        # full driver ioctl surface for every device.
        #
        # Documented residual: the container is ephemeral (--rm), so its CUDA
        # context + VRAM are freed on exit, but freed VRAM is not guaranteed
        # ZEROED before the next container reuses it. The eval set is mounted
        # read-only into EVERY submission's container anyway, so reading a prior
        # run's residual VRAM gives a miner nothing they don't already get — the
        # only leak is a prior miner's model weights to a later one (low). The
        # clean mitigation is driver/MIG-level scrub, not an in-band hack.
        argv += ["--gpus", f"device={cfg.gpu_device}"]

    for m in mounts:
        argv += ["-v", m.as_flag()]
    # The one writable output mount (validated, single result line).
    argv += ["-v", f"{Path(out_dir).resolve()}:/out:rw"]

    for k, v in env.items():
        argv += ["-e", f"{k}={v}"]

    argv.append(cfg.image)
    argv += container_argv

    _assert_argv_safe(argv)
    return argv


def _assert_argv_safe(argv: list[str]) -> None:
    """Defense-in-depth: reject an argv that smuggled in a dangerous flag or a
    secret-bearing mount. A bug here should fail the build, not the box."""
    joined = " ".join(argv)
    for bad in FORBIDDEN_FLAGS:
        if bad in argv or bad in joined:
            raise ValueError(f"forbidden flag in sandbox argv: {bad}")
    if "--gpus" in argv:
        i = argv.index("--gpus")
        if i + 1 < len(argv) and argv[i + 1] in ("all", "--gpus=all") or "--gpus=all" in joined:
            raise ValueError("refusing `--gpus all`: pin a single device")
    # No secret paths or the docker socket may be mounted.
    secret_markers = (
        "docker.sock", ".bittensor", "ralph_validator_enc_key",
        "/root/.ssh", "id_rsa", "wallet",
    )
    for tok in argv:
        if tok == "-v" or tok.startswith("-v"):
            continue
        if ":" in tok and ("ro" in tok.split(":")[-1] or "rw" in tok.split(":")[-1]):
            low = tok.lower()
            for marker in secret_markers:
                if marker in low:
                    raise ValueError(f"refusing to mount a secret path into sandbox: {tok}")


# ---------------------------------------------------------------------------
# Preflight (FAIL-CLOSED runtime + hardening verification)
# ---------------------------------------------------------------------------

def _run(cmd: list[str], timeout: int = 15) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _parse_version(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text)
    return tuple(int(x) for x in m.groups()) if m else None


def _check_docker(reasons: list[str]) -> None:
    if shutil.which("docker") is None:
        reasons.append("docker not on PATH")
        return
    try:
        info = _run(["docker", "info", "--format", "{{json .}}"])
    except Exception as e:  # noqa: BLE001
        reasons.append(f"`docker info` failed: {e}")
        return
    if info.returncode != 0:
        reasons.append("docker daemon unreachable")
        return
    ver = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    v = _parse_version(ver.stdout)
    if v is None or v < MIN_DOCKER_VERSION:
        reasons.append(f"docker engine {v} < required {MIN_DOCKER_VERSION}")
    # userns-remap must be ACTIVE (the highest-value 2025/26 control).
    sec = _run(["docker", "info", "--format", "{{.SecurityOptions}}"])
    if "name=userns" not in sec.stdout:
        reasons.append("docker userns-remap is NOT active (daemon needs userns-remap)")


def _check_runc(reasons: list[str]) -> None:
    if shutil.which("runc") is None:
        return  # rootless/containerd-shim setups may not expose runc on PATH
    v = _parse_version(_run(["runc", "--version"]).stdout)
    if v is None or v < MIN_RUNC_VERSION:
        reasons.append(f"runc {v} < required {MIN_RUNC_VERSION} (Leaky Vessels / masked-path)")


def _check_nvidia_toolkit(reasons: list[str], require_gpu: bool) -> None:
    if not require_gpu:
        return
    if shutil.which("nvidia-ctk") is None:
        reasons.append("nvidia-container-toolkit (nvidia-ctk) not found")
        return
    v = _parse_version(_run(["nvidia-ctk", "--version"]).stdout)
    if v is None or v < MIN_TOOLKIT_VERSION:
        reasons.append(f"nvidia-container-toolkit {v} < required {MIN_TOOLKIT_VERSION} (NVIDIAScape)")
    # The CVE-2025-23266 / CVE-2024-0132 mitigation is a CONFIG flag, not a
    # version: the cuda-compat-lib hook must be disabled.
    cfg_path = Path("/etc/nvidia-container-runtime/config.toml")
    if cfg_path.exists():
        text = cfg_path.read_text(errors="ignore")
        if "disable-cuda-compat-lib-hook" not in text or "disable-cuda-compat-lib-hook = true" not in text:
            reasons.append("nvidia toolkit `disable-cuda-compat-lib-hook=true` not set")
    else:
        reasons.append("nvidia-container-runtime config.toml not found (cannot verify hook-disable)")


def _check_image(reasons: list[str], image: str) -> None:
    if "@sha256:" not in image:
        reasons.append(f"sandbox image not pinned by digest: {image}")
        return
    res = _run(["docker", "image", "inspect", image])
    if res.returncode != 0:
        reasons.append(f"sandbox image not present locally: {image}")


def preflight(cfg: SandboxConfig, *, require_gpu: bool = True) -> None:
    """Verify the sandbox runtime + hardening config. Raise SandboxUnavailable
    listing EVERY failed check. This is the fail-closed gate; never bypass it in
    production (RALPH_TEST_MODE is honored only for CI stubbing by callers)."""
    reasons: list[str] = []
    _check_docker(reasons)
    _check_runc(reasons)
    _check_nvidia_toolkit(reasons, require_gpu)
    _check_image(reasons, cfg.image)
    if reasons:
        raise SandboxUnavailable("sandbox preflight failed: " + "; ".join(reasons))


# ---------------------------------------------------------------------------
# Run (preflight → docker run with a host-side watchdog)
# ---------------------------------------------------------------------------

def run_in_sandbox(
    cfg: SandboxConfig,
    *,
    container_argv: list[str],
    mounts: list[Mount],
    out_dir: Path,
    timeout_s: int,
    env_extra: dict[str, str] | None = None,
    skip_preflight: bool = False,
) -> SandboxResult:
    """Run `container_argv` in a hardened container. Fail-closed: preflight must
    pass (unless `skip_preflight`, which callers set ONLY under RALPH_TEST_MODE).
    A wall-clock overrun triggers `docker kill`, tearing down the PID namespace.
    """
    if not skip_preflight:
        preflight(cfg, require_gpu=cfg.gpu_device is not None)

    name = f"ralph-sbx-{uuid.uuid4().hex[:12]}"
    env = build_container_env(env_extra)
    argv = build_docker_argv(
        cfg, container_argv=container_argv, mounts=mounts,
        out_dir=out_dir, name=name, env=env,
    )

    timed_out = False
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
        rc, out, err = proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        # Tear down the container; --rm cleans up after the kill.
        try:
            _run(["docker", "kill", name], timeout=20)
        except Exception:  # noqa: BLE001
            pass
        rc = 124
        out = (e.stdout.decode() if isinstance(e.stdout, bytes) else (e.stdout or "")) if e.stdout else ""
        err = f"sandbox timed out after {timeout_s}s and was killed"

    eval_line = next((ln for ln in out.splitlines() if _RESULT_RE.match(ln)), None)
    return SandboxResult(
        returncode=rc,
        stdout=out,
        stderr=_redacted(err),
        timed_out=timed_out,
        eval_line=eval_line,
    )


# ---------------------------------------------------------------------------
# Output handling (cap the covert channel; trust only typed, quantized fields)
# ---------------------------------------------------------------------------

# Ranking precision: val_bpb decisive margins are ~0.01-0.05 bpb, so 3 decimals
# is far below ranking resolution while capping the per-field covert-channel
# capacity. NOTE: this only caps leakage; the real fix is host-side reduction so
# the score is validator-produced, not miner-printed (design doc §7 #2).
_QUANTIZE_DECIMALS = 3
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def parse_eval_line(line: str) -> dict[str, object]:
    """Strictly parse + type-validate + quantize the single result line. Drops
    every field the validator does not rank on. Raises ValueError on anything
    malformed (never echo raw miner stdout downstream)."""
    if not _RESULT_RE.match(line):
        raise ValueError("not a RALPH_EVAL_RESULT line")
    fields: dict[str, str] = {}
    for tok in line[len("RALPH_EVAL_RESULT "):].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            fields[k] = v

    def _qfloat(key: str) -> float:
        return round(float(fields[key]), _QUANTIZE_DECIMALS)

    out: dict[str, object] = {
        "val_bpb": _qfloat("val_bpb"),
        "benchmark_acc": _qfloat("benchmark_acc"),
        "tokens_evaluated": int(fields["tokens_evaluated"]),
        "benchmark_examples": int(fields["benchmark_examples"]),
    }
    h = fields.get("eval_set_hash", "")
    if not _HEX64.match(h):
        # Not fatal here (host should recompute over the full stream — §7 #2),
        # but never propagate an unvalidated string.
        h = ""
    out["eval_set_hash"] = h
    return out
