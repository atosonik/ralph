"""
RESTRICTED. Miners may not patch any file in this directory. The
restricted-file diff scanner enforces this in validator/.
"""

from .benchmark import compute_benchmark_score
from .hidden_eval import HiddenEvalResult, run_hidden_eval
from .sealed_streams import (
    MANIFEST_VERSION,
    SealedStreamBatch,
    SealedStreamManifest,
    SealedStreamSpec,
    bytes_per_token_for,
    compute_streams_root_hash,
    load_stream,
    read_manifest,
    select_active_streams,
    write_manifest,
)
from .val_bpb import compute_val_bpb

__all__ = [
    "HiddenEvalResult",
    "MANIFEST_VERSION",
    "SealedStreamBatch",
    "SealedStreamManifest",
    "SealedStreamSpec",
    "bytes_per_token_for",
    "compute_benchmark_score",
    "compute_streams_root_hash",
    "compute_val_bpb",
    "load_stream",
    "read_manifest",
    "run_hidden_eval",
    "select_active_streams",
    "write_manifest",
]
