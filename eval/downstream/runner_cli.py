"""Production CLI entrypoint for the downstream-eval subprocess (B1).

Invoked as `python -m eval.downstream.runner_cli ...` from the
validator's `run_eval_in_subprocess` wrapper (eval/downstream/
runner_subprocess.py). Reads its `EvalConfig` from a JSON file
specified via --config, loads a miner-submitted checkpoint with
torch.load(weights_only=True), builds a `KarpaBase` model, and
delegates to `run_downstream_eval`. Writes the resulting
`DownstreamReport` as JSON to --output.

Closes:

  * **B1-D5** — subprocess isolation reality. `torch.load(
    weights_only=True)` blocks pickle-deserialization RCE during
    checkpoint load; it does NOT block arbitrary Python code
    executed inside the loaded model's `forward()` method. Once
    state_dict tensors are back in memory and `model(input_ids)` is
    called, any code the miner has wired into KarpaBase's forward
    path runs in this subprocess. The only containment B1 provides
    is OS-level process isolation (separate PID, separate Python
    interpreter), which the `run_eval_in_subprocess` wrapper
    establishes. Seccomp / landlock are deferred to a named
    follow-up phase before mainnet activation.

  * **B1-D13** — structural-patch CLI args. --patch and
    --karpa-root are accepted at the CLI surface so the contract is
    stable. The actual `apply_patch` integration (~150 LOC + 2 days)
    is deferred to a follow-up PR; for now, passing --patch raises
    a clean NotImplementedError pointing at the follow-up ticket.
    Submissions that don't use a structural patch (the common case)
    are unaffected.

What this module ships:

  * `main(argv)` — the argparse + orchestration entrypoint.
  * `_build_parser()` — the argparse.ArgumentParser; exposed for
    test inspection.
  * `_load_checkpoint(path)` — torch.load + dict-shape validation
    helper. Returns (config_dict, state_dict). Closes the B1-D5
    weights_only=True requirement.
  * `_import_karpa_model(karpa_root)` — sys.path bootstrap +
    `from model import KarpaBase, KarpaConfig` import helper.

What this module does NOT ship (separate follow-up PR):

  * The `apply_patch` integration that turns --patch into a
    modified workdir. The args are accepted; full handling lands
    when the apply_patch helper is ready.

  * The DCLM bundle download + SHA-pin commit. `core22.load_task_examples`
    is implemented and reads from a local bundle dir; the operational
    `wget` + sha256sum step that pins the bundle constant is a
    separate one-time concern (B1-D2 protocol). Similarly,
    `private_hard.load_task_examples` is implemented and reads from
    `{bundle_dir}/private_hard/{task}.jsonl` — the operator's HF
    download + re-keying step populates that subdir.

Reference scope: docs/build_scope/02_scope_B1.md "runner_cli.py".
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

from .grader import read_hardness_index_jsonl
from .runner import (
    EvalConfig,
    run_downstream_eval,
)
from .runner_subprocess import serialize_report


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Exposed for arg-surface tests."""
    p = argparse.ArgumentParser(
        prog="eval.downstream.runner_cli",
        description=(
            "Subprocess entrypoint for the downstream-eval harness. "
            "Loads a miner checkpoint with weights_only=True, runs the "
            "configured tasks, and writes a DownstreamReport JSON."
        ),
    )
    p.add_argument("--checkpoint", required=True, type=Path,
                   help="Path to the miner-submitted checkpoint .pt file.")
    p.add_argument("--config", required=True, type=Path,
                   help="Path to the EvalConfig JSON.")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write the DownstreamReport JSON.")
    p.add_argument("--bundle-sha", required=True,
                   help="Pinned SHA256 of the DCLM eval bundle (B1-D2).")
    p.add_argument("--bundle-dir", required=True, type=Path,
                   help="Path to the local unpacked DCLM eval bundle.")
    p.add_argument("--vocab-size", required=True, type=int,
                   help="Expected tokenizer vocab size (must be 50257 per "
                        "B1-D6).")
    p.add_argument("--hardness-index", default=None, type=Path,
                   help="Optional path to the hardness-index JSONL. "
                        "Required when --config tasks include any "
                        "private_hard task.")
    p.add_argument("--patch", default=None, type=Path,
                   help="Structural-patch path (B1-D13). Args accepted; "
                        "full apply_patch integration deferred.")
    p.add_argument("--karpa-root", default=None, type=Path,
                   help="Karpa repo root for structural-patch application "
                        "(B1-D13) and `from model import ...` resolution.")
    return p


def _load_checkpoint(path: Path) -> tuple[dict, dict]:
    """Load a checkpoint with `torch.load(weights_only=True)`.

    Returns:
      `(config_dict, state_dict)`. `config_dict` is the KarpaConfig
      kwargs the checkpoint was saved with; `state_dict` is the
      tensor state for `model.load_state_dict`.

    Accepted checkpoint shapes:
      * `{"model": state_dict, "config": dict}` — canonical v0.10 form
      * `{"state_dict": state_dict, "config": dict}` — legacy variant
      * `{"model": state_dict}` + sibling `checkpoint_config.json` —
        the sidecar pattern used by validator/eval_in_workdir.py

    Raises:
      ValueError on any unrecognized shape, with a message naming the
      specific failure mode (not a dict / missing config / etc.).
    """
    full = torch.load(path, weights_only=True, map_location="cpu")
    if not isinstance(full, dict):
        raise ValueError(
            f"checkpoint at {path} must be a dict; got {type(full).__name__}"
        )
    if "model" in full:
        state_dict = full["model"]
    elif "state_dict" in full:
        state_dict = full["state_dict"]
    else:
        raise ValueError(
            f"checkpoint at {path} has no 'model' or 'state_dict' key; "
            f"top-level keys: {sorted(full.keys())}"
        )

    config = full.get("config")
    if config is None:
        sidecar = path.parent / "checkpoint_config.json"
        if sidecar.exists():
            config = json.loads(sidecar.read_text())
    if not isinstance(config, dict):
        raise ValueError(
            f"checkpoint at {path} missing 'config' dict and no "
            "checkpoint_config.json sidecar in the same directory"
        )
    return config, state_dict


def _import_karpa_model(karpa_root: Path | None):
    """Inject the recipe path into sys.path and import KarpaBase + Config.

    If `karpa_root` is provided, prepend it to sys.path so a patched
    `model/` package wins over any installed one. Otherwise import
    `karpa_bootstrap` from the karpa repo (already on sys.path via
    the subprocess wrapper's PYTHONPATH) which injects the sibling
    `../recipe` per its resolution order.

    Returns: (KarpaBase, KarpaConfig) classes.
    """
    if karpa_root is not None:
        sys.path.insert(0, str(karpa_root.resolve()))
    else:
        import karpa_bootstrap  # noqa: F401  (side-effect import)
    from model import KarpaBase, KarpaConfig  # type: ignore
    return KarpaBase, KarpaConfig


def _build_task_loaders(
    tasks: tuple[str, ...],
    bundle_dir: Path,
):
    """Build a {task_name: () -> raw_rows} dict for run_downstream_eval.

    Dispatches by task name:
      * task in PRIVATE_HARD_TASK_SPECS → private_hard.load_task_examples
        reads from `{bundle_dir}/private_hard/{task_name}.jsonl`.
      * otherwise (assumes task in TASK_SPECS) → core22.load_task_examples
        reads from `{bundle_dir}/{task_name}.jsonl`.

    The `private_hard` subdir convention keeps CORE-22 + private-hard
    JSONLs from colliding in the same flat directory; the operator's
    bundle-prep step is responsible for placing files under the right
    subtree.
    """
    from .core22 import load_task_examples as core22_load
    from .private_hard import (
        PRIVATE_HARD_TASK_SPECS,
    )
    from .private_hard import (
        load_task_examples as private_hard_load,
    )
    private_hard_dir = Path(bundle_dir) / "private_hard"
    loaders: dict[str, object] = {}
    for task in tasks:
        if task in PRIVATE_HARD_TASK_SPECS:
            loaders[task] = (lambda t=task: private_hard_load(private_hard_dir, t))
        else:
            loaders[task] = (lambda t=task: core22_load(bundle_dir, t))
    return loaders


def _build_tokenize_fn():
    """Return the GPT-2 BPE tokenize callable used by the harness."""
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return lambda text: enc.encode(text)


def _apply_patch_to_workdir(patch_path: Path, karpa_root: Path) -> Path:
    """Copy `karpa_root` to a tmp workdir + apply `patch_path` via `patch -p1`.

    Returns the path of the patched workdir. The caller adds it to
    `sys.path` so `import model` resolves to the PATCHED model package.

    Empty patches are tolerated (no-op canonical baseline). Patch failures
    raise RuntimeError with the patch tool's stdout+stderr; partial
    application is not allowed (`--no-backup-if-mismatch`).

    Closes B1-D13 in full.
    """
    karpa_root = karpa_root.resolve()
    patch_path = patch_path.resolve()
    if not karpa_root.exists():
        raise FileNotFoundError(f"karpa_root not found: {karpa_root}")
    if not patch_path.exists():
        raise FileNotFoundError(f"patch not found: {patch_path}")
    workdir = Path(tempfile.mkdtemp(prefix="karpa_patched_"))
    # Copy the karpa root into the workdir. We use a child dir so
    # `workdir` itself is a clean container and can be cleaned up
    # whole-tree on completion.
    target = workdir / "karpa_root"
    shutil.copytree(karpa_root, target, symlinks=True)
    if patch_path.stat().st_size == 0:
        # Empty patch == canonical baseline. Nothing to do.
        return target
    result = subprocess.run(
        ["patch", "-p1", "-i", str(patch_path), "--no-backup-if-mismatch"],
        cwd=target,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"patch failed (rc={result.returncode}):\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return target


def main(argv: list[str] | None = None) -> int:
    """Subprocess entrypoint. Returns the process exit code.

    Flow:
      1. Parse args.
      2. Reject --patch (B1-D13 args accepted, application deferred).
      3. Read EvalConfig JSON.
      4. Import KarpaBase / KarpaConfig (sys.path bootstrap).
      5. Load checkpoint with weights_only=True.
      6. Construct model + load state_dict.
      7. Verify vocab matches.
      8. Build forward_logits + tokenize + task_loaders.
      9. Load hardness_index if provided.
      10. Run downstream eval.
      11. Serialize report + write to --output.
    """
    args = _build_parser().parse_args(argv)

    config = EvalConfig.from_dict(json.loads(args.config.read_text()))

    # Structural-patch handling (closes B1-D13): if --patch is supplied, apply
    # it to a tmp copy of --karpa-root and import the model package from the
    # patched workdir. Without --patch, _import_karpa_model uses the karpa
    # root as-is (sys.path insert + `from model import ...`).
    if args.patch is not None:
        if args.karpa_root is None:
            raise ValueError(
                "--patch requires --karpa-root so the patched recipe tree "
                "has a base to patch against"
            )
        patched_workdir = _apply_patch_to_workdir(args.patch, args.karpa_root)
        # Re-route the model import to the patched workdir.
        sys.path.insert(0, str(patched_workdir))

    KarpaBase, KarpaConfig = _import_karpa_model(args.karpa_root)
    ckpt_config, state_dict = _load_checkpoint(args.checkpoint)

    cfg_kwargs = {
        k: v for k, v in ckpt_config.items()
        if k in KarpaConfig.__dataclass_fields__
    }
    model_cfg = KarpaConfig(**cfg_kwargs)
    model = KarpaBase(model_cfg)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    if model_cfg.vocab_size != args.vocab_size:
        raise ValueError(
            f"checkpoint vocab_size={model_cfg.vocab_size} does not match "
            f"--vocab-size={args.vocab_size}; the validator and miner "
            "are out of sync on tokenizer choice"
        )

    hardness_index = None
    if args.hardness_index is not None:
        hardness_index = read_hardness_index_jsonl(args.hardness_index)

    tokenize = _build_tokenize_fn()
    task_loaders = _build_task_loaders(config.tasks, args.bundle_dir)

    def forward_logits(input_ids: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()
        with torch.no_grad():
            out = model(input_ids)
        # KarpaBase returns (logits, optional_loss) or logits; scorer's
        # _extract_logits handles either form.
        return out

    report = run_downstream_eval(
        forward_logits,
        config=config,
        task_loaders=task_loaders,
        tokenize=tokenize,
        bundle_sha256=args.bundle_sha,
        vocab_size=model_cfg.vocab_size,
        hardness_index=hardness_index,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serialize_report(report)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
