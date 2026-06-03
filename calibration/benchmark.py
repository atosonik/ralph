"""
Deterministic calibration benchmark.

Runs a fixed workload — matmul + attention + collective — whose wall-clock
on a given accelerator is a runtime fingerprint of the hardware family.
Validator checks the fingerprint against the claimed hardware in the
submission bundle.

Reference timings to be populated from real-hardware runs on H100-SXM 80GB
during Phase 0.5 commissioning. Phase 0 stores the local timings and lets
the validator's plausibility check flag obvious mismatches (e.g. claimed
H100 but benchmark runs at A100 speed).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

# Fixed workload parameters — these MUST NOT change between runs that wish to
# be cross-comparable. Changing them is a recipe-wide event coordinated by the
# subnet-owner team alongside container-measurement updates.
CALIB_VERSION = "phase0-v1"
MATMUL_SIZE = 2048
MATMUL_REPEATS = 20
ATTN_BATCH = 4
ATTN_HEADS = 16
ATTN_SEQ = 1024
ATTN_HEAD_DIM = 64
ATTN_REPEATS = 20
DTYPE = torch.float32  # Phase 0 default; H100 path will use bf16 later.


@dataclass
class CalibrationResult:
    version: str
    device: str
    dtype: str
    matmul_ms: float
    attention_ms: float
    collective_ms: float
    total_ms: float
    gpu_name: str | None
    cuda_available: bool


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _matmul_bench(device: torch.device) -> float:
    g = torch.Generator(device=device).manual_seed(1)
    a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, generator=g, device=device, dtype=DTYPE)
    b = torch.randn(MATMUL_SIZE, MATMUL_SIZE, generator=g, device=device, dtype=DTYPE)
    # Warmup
    for _ in range(2):
        c = a @ b
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(MATMUL_REPEATS):
        c = a @ b
    _sync(device)
    return (time.perf_counter() - t0) * 1000 / MATMUL_REPEATS


def _attention_bench(device: torch.device) -> float:
    g = torch.Generator(device=device).manual_seed(2)
    q = torch.randn(ATTN_BATCH, ATTN_HEADS, ATTN_SEQ, ATTN_HEAD_DIM, generator=g, device=device, dtype=DTYPE)
    k = torch.randn(ATTN_BATCH, ATTN_HEADS, ATTN_SEQ, ATTN_HEAD_DIM, generator=g, device=device, dtype=DTYPE)
    v = torch.randn(ATTN_BATCH, ATTN_HEADS, ATTN_SEQ, ATTN_HEAD_DIM, generator=g, device=device, dtype=DTYPE)
    # Warmup
    for _ in range(2):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(ATTN_REPEATS):
        _ = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    _sync(device)
    return (time.perf_counter() - t0) * 1000 / ATTN_REPEATS


def _collective_bench(device: torch.device) -> float:
    """
    On a single GPU we stub this as a sum-reduce over a moderate tensor.
    On multi-GPU runs (Phase 1 FSDP) this becomes a real all_reduce.
    """
    g = torch.Generator(device=device).manual_seed(3)
    x = torch.randn(1 << 22, generator=g, device=device, dtype=DTYPE)  # ~16MB
    # Warmup
    for _ in range(2):
        _ = x.sum()
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = x.sum()
    _sync(device)
    return (time.perf_counter() - t0) * 1000 / 20


def run_calibration(device: torch.device | None = None) -> CalibrationResult:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else None
    cuda_available = torch.cuda.is_available()

    matmul_ms = _matmul_bench(device)
    attention_ms = _attention_bench(device)
    collective_ms = _collective_bench(device)

    return CalibrationResult(
        version=CALIB_VERSION,
        device=str(device),
        dtype=str(DTYPE).replace("torch.", ""),
        matmul_ms=matmul_ms,
        attention_ms=attention_ms,
        collective_ms=collective_ms,
        total_ms=matmul_ms + attention_ms + collective_ms,
        gpu_name=gpu_name,
        cuda_available=cuda_available,
    )


if __name__ == "__main__":
    import json
    result = run_calibration()
    print(json.dumps(asdict(result), indent=2))
