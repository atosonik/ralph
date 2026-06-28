#!/usr/bin/env python3
"""In-container entrypoint for the validator execution sandbox (op4 hidden-eval).

Runs INSIDE the hardened container (network none, non-root, read-only rootfs, no
secrets). It loads the miner's (possibly patched) model, runs the forward pass
over the held-out stream, and emits the **per-position NLLs** — NOT a reduced
score. The host (`validator/sandbox.py` → `eval.host_reduce`) computes val_bpb
from that array, owning the formula, token count, bytes_per_token, tail mask, and
eval-set hash. The container never prints the crowning number.

The eval/scoring code is the CANONICAL (image-baked / installed) package; only the
MODEL is imported from the patched workdir. Trusted helpers are imported BEFORE
the workdir is placed on sys.path so miner code cannot shadow them.

Container layout (mounts, all ro except /out):
  /work/workdir   patched recipe tree (model/, ... ; already patch-applied)
  /in/checkpoint.pt   the miner's checkpoint
  /eval-private/active_tokens.bin   the held-out stream (host-mounted ro)
  /out            the single writable dir — receives nlls.npy + manifest.json

Outputs:
  /out/nlls.npy        float32 per-position NLLs (window-row-major order)
  /out/manifest.json   {status, seq_len, tokens_emitted, model_config}

Exit codes: 0 ok · 1 setup/import/load failure · 2 eval crash · 3 bad args.

TODO(benchmark): emit per-example benchmark correctness for host reduction too,
so wiring op4 through the sandbox does not drop benchmark_accuracy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def prepare_workdir(canon_dir: Path, patch_path: Path, dest_workdir: Path) -> Path:
    """copytree the canonical recipe tree into a scratch workdir and apply the
    miner patch — done INSIDE the container so `patch -p1` path-traversal and
    symlink dereference are contained by the read-only/non-root namespace, not
    run on the host. Returns the workdir.
    """
    import shutil

    from proof.runner import apply_patch

    canon_dir = Path(canon_dir)
    dest_workdir = Path(dest_workdir)
    dest_workdir.mkdir(parents=True, exist_ok=True)
    for sub in ("model", "recipe", "data", "configs", "eval", "calibration"):
        src = canon_dir / sub
        if src.exists():
            # symlinks=False (default) copies content, but we never mount secrets
            # into /canon, so there is nothing sensitive to dereference here.
            shutil.copytree(src, dest_workdir / sub, dirs_exist_ok=True)
    if Path(patch_path).exists():
        apply_patch(dest_workdir, Path(patch_path))
    return dest_workdir


def run_sandbox_eval(
    workdir: Path,
    ckpt_path: Path,
    eval_dir: Path,
    out_dir: Path,
    *,
    batch_size: int = 8,
):
    """Produce per-position NLLs (for host val_bpb reduction) + benchmark accuracy.

    Importable + unit-testable in-process (no container) so the produce→reduce
    equivalence can be proven on CPU.
    """
    import numpy as np

    # Trusted helpers FIRST (canonical/installed), before the workdir goes on the
    # path — miner code in workdir must not be able to shadow the reducer.
    from eval.benchmark import compute_benchmark_score
    from eval.val_bpb import load_eval_tokens, per_position_nlls

    sys.path.insert(0, str(Path(workdir).resolve()))
    import torch
    from model import RalphBase, RalphConfig

    # Checkpoint config: sidecar JSON if present, else the embedded "config".
    sidecar = Path(ckpt_path).parent / "checkpoint_config.json"
    if sidecar.exists():
        saved = json.loads(sidecar.read_text())
    else:
        saved = torch.load(ckpt_path, weights_only=True, map_location="cpu").get("config", {})

    fields = RalphConfig.__dataclass_fields__
    cfg = RalphConfig(**{k: v for k, v in saved.items() if k in fields})

    ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
    state_dict = ckpt.get("model", ckpt)
    model = RalphBase(cfg)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()

    seq_len = cfg.max_seq_len // 2
    eval_dir = Path(eval_dir)
    tokens = np.asarray(load_eval_tokens(eval_dir / "active_tokens.bin"))
    nlls = per_position_nlls(model, tokens, seq_len, batch_size)

    # Benchmark accuracy — cheap (the ~1.5k held-out examples, not the token
    # stream). Contained but miner-computed; the crown-critical val_bpb is the
    # one the HOST reduces from nlls. No benchmark file -> 0.0.
    benchmark_accuracy = 0.0
    benchmark_examples = 0
    bpath = eval_dir / "active_benchmark.json"
    if bpath.exists():
        examples = json.loads(bpath.read_text())
        bench = compute_benchmark_score(model, examples)
        benchmark_accuracy = float(bench["benchmark_accuracy"])
        benchmark_examples = int(bench["n_examples"])

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "nlls.npy", nlls)
    manifest = {
        "status": "ok",
        "seq_len": int(seq_len),
        "tokens_emitted": int(nlls.shape[0]),
        "benchmark_accuracy": benchmark_accuracy,
        "benchmark_examples": benchmark_examples,
        "model_config": {
            "vocab_size": cfg.vocab_size,
            "dim": cfg.dim,
            "n_layers": cfg.n_layers,
            "max_seq_len": cfg.max_seq_len,
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    return nlls


def run_prepare_and_eval(
    canon_dir: Path,
    patch_path: Path,
    ckpt_path: Path,
    eval_dir: Path,
    out_dir: Path,
    *,
    batch_size: int = 8,
):
    """Full container flow: prepare the patched workdir IN here (traversal
    contained), then eval. The host-side runner calls this via __main__."""
    import tempfile

    workdir = prepare_workdir(canon_dir, patch_path, Path(tempfile.mkdtemp(prefix="ralph_sbx_")) / "workdir")
    return run_sandbox_eval(workdir, ckpt_path, eval_dir, out_dir, batch_size=batch_size)


def main(argv: list[str]) -> int:
    if len(argv) != 6:
        print(f"usage: {argv[0]} <canon_dir> <patch.diff> <ckpt_path> <eval_dir> <out_dir>", file=sys.stderr)
        return 3
    canon_dir, patch_path, ckpt_path, eval_dir, out_dir = (Path(a) for a in argv[1:6])
    if not canon_dir.is_dir() or not ckpt_path.is_file() or not eval_dir.is_dir():
        print("ERROR: canon_dir/ckpt/eval_dir must exist", file=sys.stderr)
        return 3
    try:
        run_prepare_and_eval(canon_dir, patch_path, ckpt_path, eval_dir, out_dir)
    except (ImportError, KeyError, RuntimeError) as e:
        print(f"ERROR: setup/load failed: {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: eval crashed: {e}", file=sys.stderr)
        return 2
    print("RALPH_SANDBOX_EVAL ok")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
