#!/usr/bin/env python3
"""Subprocess helper for op4_hidden_eval's patched-model fallback.

When a miner submits a structural patch that ADDS new parameters to RalphBase
(e.g. QK-Norm adding q_norm/k_norm weights), the validator's canonical
RalphBase rejects the checkpoint at load_state_dict time. This script is
spawned in that case: it imports the PATCHED model from a workdir where the
patch has already been applied, instantiates the model, loads the state_dict
strict, runs hidden_eval, and prints the result for the parent to parse.

Args (positional):
  1. workdir   — directory containing the patched recipe tree (model/, eval/, ...)
  2. ckpt_path — path to the miner's checkpoint.pt
  3. ralph_root — repo root (for eval/ data location if not in workdir)

Output (stdout, last line):
  RALPH_EVAL_RESULT val_bpb=<float> benchmark_acc=<float>

Exit codes:
  0  — eval ran successfully (result line printed)
  1  — patched model could not be imported or state_dict load failed
  2  — eval crashed
  3  — invalid args
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 4:
        print(f"ERROR: usage: {sys.argv[0]} <workdir> <ckpt_path> <ralph_root>", file=sys.stderr)
        return 3
    workdir = Path(sys.argv[1]).resolve()
    ckpt_path = Path(sys.argv[2]).resolve()
    ralph_root = Path(sys.argv[3]).resolve()

    if not workdir.is_dir():
        print(f"ERROR: workdir {workdir} not a directory", file=sys.stderr)
        return 3
    if not ckpt_path.is_file():
        print(f"ERROR: ckpt {ckpt_path} not a file", file=sys.stderr)
        return 3

    # Put the workdir FIRST on sys.path so its patched `model/` package wins
    # over any model package that lives under ralph_root or its installed
    # parent. The ralph_root entry is appended so eval/ + dependencies still
    # resolve when the workdir doesn't include them.
    sys.path.insert(0, str(workdir))
    sys.path.insert(1, str(ralph_root))

    try:
        import torch
        from model import RalphBase, RalphConfig

        from eval import run_hidden_eval
    except Exception as e:
        print(f"ERROR: import failed: {e}", file=sys.stderr)
        return 1

    sidecar = ckpt_path.parent / "checkpoint_config.json"
    if sidecar.exists():
        saved = json.loads(sidecar.read_text())
    else:
        try:
            full = torch.load(ckpt_path, weights_only=True, map_location="cpu")
            saved = full.get("config", {})
        except Exception as e:
            print(f"ERROR: cannot read checkpoint config: {e}", file=sys.stderr)
            return 1

    try:
        cfg_kwargs = dict(
            vocab_size=saved["vocab_size"],
            dim=saved["dim"],
            n_layers=saved["n_layers"],
            n_heads=saved["n_heads"],
            head_dim=saved["head_dim"],
            ffn_mult=saved.get("ffn_mult", 8 / 3),
            max_seq_len=saved["max_seq_len"],
        )
        for k, v in saved.items():
            if k in cfg_kwargs or k in ("rms_norm_eps", "rope_theta", "tie_embeddings"):
                cfg_kwargs.setdefault(k, v)
        cfg = RalphConfig(**{k: v for k, v in cfg_kwargs.items() if k in RalphConfig.__dataclass_fields__})
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        state_dict = ckpt.get("model", ckpt)
    except Exception as e:
        print(f"ERROR: cfg/checkpoint parse failed: {e}", file=sys.stderr)
        return 1

    try:
        model = RalphBase(cfg)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"ERROR: patched model state_dict load failed: {e}", file=sys.stderr)
        return 1

    if torch.cuda.is_available():
        model = model.cuda()

    try:
        eval_root = workdir if (workdir / "eval" / "private").is_dir() else ralph_root
        result = run_hidden_eval(model, eval_root / "eval" / "private", seq_len=cfg.max_seq_len // 2)
    except Exception as e:
        print(f"ERROR: hidden_eval crashed: {e}", file=sys.stderr)
        return 2

    print(
        f"RALPH_EVAL_RESULT val_bpb={result.val_bpb:.6f} "
        f"benchmark_acc={result.benchmark_accuracy:.6f} "
        f"tokens_evaluated={result.tokens_evaluated} "
        f"benchmark_examples={result.benchmark_examples} "
        f"eval_set_hash={result.eval_set_hash}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
