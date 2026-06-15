"""Cache + re-key the 4 private-hard HF datasets to canonical Ralph JSONL.

Pulls each of the 4 datasets in `eval/downstream/private_hard.HF_DATASET_IDS`
from HuggingFace Hub via the `datasets` library, iterates the rows, and
re-keys each into the canonical Ralph schema that
`private_hard.load_task_examples` consumes:

  * MC tasks (arc_challenge_hard, tiny_arc, tiny_mmlu):
      {"id": str, "query": str, "choices": [str, ...], "gold": int}
  * Schema tasks (winogrande_hard):
      {"id": str, "contexts": [str, ...], "continuations": [str, ...],
       "gold": int}

Each per-task JSONL is written to
`eval/private/downstream_pool/private_hard/{task_name}.jsonl`. A
manifest file `manifest.json` carries the HF revision SHA for each
dataset so the operator can re-pull deterministically later.

USAGE:
    pip install datasets huggingface_hub
    python scripts/cache_hf_assets.py \\
        [--output-dir eval/private/downstream_pool/private_hard] \\
        [--task arc_challenge_hard ... --task tiny_mmlu]  # subset, default = all 4
        [--hf-token <token>]   # optional, for gated datasets
        [--revision <sha>]     # pin a specific revision per task (advanced)

OUTPUTS:
  {output_dir}/
    arc_challenge_hard.jsonl    canonical MC schema, ordered by upstream id
    winogrande_hard.jsonl       canonical schema (variants) schema
    tiny_arc.jsonl              canonical MC
    tiny_mmlu.jsonl             canonical MC
    manifest.json               {task -> {hf_id, hf_config, hf_revision, row_count}}

Notes on per-task field mapping (the operator should re-verify against the
upstream dataset card if HF changes the schema):

  * arc_challenge_hard (allenai/ai2_arc / ARC-Challenge):
      `question`     -> query
      `choices.text` -> choices  (the `choices` column is a dict
                                  {"text": [...], "label": [...]})
      `answerKey`    -> gold (after mapping the label letter to its index
                              in `choices.label`)
  * winogrande_hard (allenai/winogrande / winogrande_xl):
      `sentence`     -> two variants (contexts/continuations split on the
                        underscore placeholder per the wino convention)
      `option1` / `option2` -> insert into each context to produce 2
                                continuations
      `answer`        -> 0 if option1, 1 if option2
  * tiny_arc + tiny_mmlu (tinyBenchmarks/tinyArc, tinyBenchmarks/tinyMMLU):
      tinyBenchmarks ships canonical-friendly schemas already. We accept
      MMLU-style 4-choice items keyed by `question` / `choices` / `answer`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.downstream.private_hard import (  # noqa: E402
    HF_DATASET_IDS,
    PRIVATE_HARD_TASK_SPECS,
    PRIVATE_HARD_TASKS,
)


def _load_hf(
    hf_id: str,
    hf_config: str | None,
    *,
    hf_token: str | None,
    revision: str | None,
    split: str = "validation",
):
    """Load one HF dataset split. Network-required; raises ImportError if
    the `datasets` package isn't installed (operator: `pip install datasets`).
    """
    from datasets import load_dataset
    kwargs: dict = {"split": split}
    if hf_config is not None:
        kwargs["name"] = hf_config
    if hf_token is not None:
        kwargs["token"] = hf_token
    if revision is not None:
        kwargs["revision"] = revision
    return load_dataset(hf_id, **kwargs)


# ----------------------------------------------------------------------------
# Per-task converters
# ----------------------------------------------------------------------------


def convert_arc_row(row: dict) -> dict | None:
    """allenai/ai2_arc / ARC-Challenge → canonical MC row.

    Returns None on a malformed row (caller skips). ARC's choices field
    is `{"text": [...], "label": [...]}`; the answerKey is the LABEL of
    the correct choice (e.g. "A", "B", "C", "D" or "1", "2", ...).
    """
    try:
        item_id = str(row["id"])
        query = str(row["question"])
        ch_field = row["choices"]
        if isinstance(ch_field, dict):
            choices = list(ch_field.get("text", []))
            labels = list(ch_field.get("label", []))
        elif isinstance(ch_field, list):
            # Some HF revisions ship choices as a list of dicts.
            choices = [str(c.get("text", c)) for c in ch_field]
            labels = [str(c.get("label", "")) for c in ch_field]
        else:
            return None
        if not choices or len(choices) != len(labels):
            return None
        answer_key = str(row["answerKey"]).strip()
        try:
            gold = labels.index(answer_key)
        except ValueError:
            return None
        return {"id": item_id, "query": query,
                "choices": [str(c) for c in choices], "gold": gold}
    except (KeyError, TypeError, ValueError):
        return None


def convert_winogrande_row(row: dict) -> dict | None:
    """allenai/winogrande / winogrande_xl → canonical schema row.

    Winogrande sentences contain a single `_` placeholder. We construct
    TWO variants by substituting option1 and option2 in, and split the
    resulting sentence at the placeholder location into context +
    continuation (the continuation is everything from the option onward
    to the end of the sentence).
    """
    try:
        sentence = str(row["sentence"])
        if "_" not in sentence:
            return None
        prefix, _, suffix = sentence.partition("_")
        option1 = str(row["option1"])
        option2 = str(row["option2"])
        answer = str(row["answer"]).strip()
        if answer not in ("1", "2"):
            return None
        gold = 0 if answer == "1" else 1
        # Two variants, each with prefix as context and "{option}{suffix}"
        # as continuation. This matches the schema_score convention.
        return {
            "id": str(row.get("qID", row.get("id", ""))),
            "contexts": [prefix, prefix],
            "continuations": [option1 + suffix, option2 + suffix],
            "gold": gold,
        }
    except (KeyError, TypeError, ValueError):
        return None


def convert_tinybench_mc_row(row: dict) -> dict | None:
    """tinyBenchmarks/tinyArc and tinyMMLU → canonical MC row.

    Both tinyBenchmarks variants ship with `question` / `choices` /
    `answer` fields; `choices` is a list of strings and `answer` is the
    integer index of the correct choice (0-based). `id` is the upstream
    item id or `input_id` field.
    """
    try:
        item_id = str(
            row.get("id") or row.get("input_id") or row.get("item_id") or ""
        )
        query = str(row["question"])
        choices_field = row["choices"]
        if isinstance(choices_field, dict):
            choices = list(choices_field.get("text", []))
        else:
            choices = list(choices_field)
        choices = [str(c) for c in choices]
        if not choices:
            return None
        gold = int(row.get("answer", row.get("gold", -1)))
        if not 0 <= gold < len(choices):
            return None
        return {"id": item_id, "query": query, "choices": choices, "gold": gold}
    except (KeyError, TypeError, ValueError):
        return None


# Per-task converter dispatch table. The operator can swap converters here
# if HF rotates a schema. Note: tinyBenchmarks/tinyAI2_arc ships ARC's
# native schema (`choices` is `{text, label}`, `answerKey` is a label
# letter), so tiny_arc routes to convert_arc_row. tinyMMLU uses the
# flatter `{question, choices: [str], answer: int}` schema and routes to
# convert_tinybench_mc_row.
TASK_CONVERTERS = {
    "arc_challenge_hard": convert_arc_row,
    "winogrande_hard": convert_winogrande_row,
    "tiny_arc": convert_arc_row,
    "tiny_mmlu": convert_tinybench_mc_row,
}

# Per-task default split. ARC/Winogrande/tinyArc ship a `validation` split;
# tinyMMLU only ships `test` and `dev`, so default to `test` (the held-out
# items; `dev` is the IRT++ calibration set). `--split` on the CLI still
# overrides every task globally if the operator asks for it.
TASK_DEFAULT_SPLITS = {
    "arc_challenge_hard": "validation",
    "winogrande_hard": "validation",
    "tiny_arc": "validation",
    "tiny_mmlu": "test",
}


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------


def cache_one_task(
    task_name: str,
    *,
    output_dir: Path,
    hf_token: str | None = None,
    revision: str | None = None,
    split: str = "validation",
) -> dict:
    """Pull + re-key one private-hard task. Returns a manifest entry.

    Skips rows that fail per-row conversion (logged in the manifest as
    `n_rows_skipped`). The output JSONL is written atomically via a
    `.tmp` rename so a crashed pull never leaves a half-written file.
    """
    if task_name not in HF_DATASET_IDS:
        raise ValueError(
            f"unknown private-hard task {task_name!r}; expected one of "
            f"{sorted(PRIVATE_HARD_TASKS)}"
        )
    hf_id, hf_config = HF_DATASET_IDS[task_name]
    converter = TASK_CONVERTERS[task_name]
    ds = _load_hf(hf_id, hf_config, hf_token=hf_token,
                  revision=revision, split=split)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{task_name}.jsonl"
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    n_in = 0
    n_out = 0
    n_skipped = 0
    with tmp_path.open("w") as f:
        for row in ds:
            n_in += 1
            converted = converter(row)
            if converted is None:
                n_skipped += 1
                continue
            f.write(json.dumps(converted) + "\n")
            n_out += 1
    tmp_path.replace(out_path)

    # Best-effort revision pin: some HF versions expose `_info.version`
    # or expose nothing; fall back to a placeholder if unavailable.
    pinned_revision = revision or _best_effort_revision(ds)

    return {
        "task": task_name,
        "hf_id": hf_id,
        "hf_config": hf_config,
        "hf_revision_pinned": pinned_revision,
        "mode": PRIVATE_HARD_TASK_SPECS[task_name].mode,
        "n_rows_in": n_in,
        "n_rows_written": n_out,
        "n_rows_skipped": n_skipped,
        "output_path": str(out_path),
    }


def _best_effort_revision(ds) -> str:
    """Try a few attributes to recover the HF revision SHA. Returns
    "unknown" if nothing usable surfaces."""
    info = getattr(ds, "info", None)
    if info is not None:
        for attr in ("version", "revision", "dataset_version"):
            v = getattr(info, attr, None)
            if v:
                return str(v)
    return "unknown"


def cache_all(
    *,
    output_dir: Path,
    tasks: tuple[str, ...] | None = None,
    hf_token: str | None = None,
    revisions: dict[str, str] | None = None,
    split: str | None = None,
) -> dict:
    """Drive cache_one_task across the requested tasks. Writes the
    manifest. Returns the manifest dict.

    If `split` is `None`, each task uses `TASK_DEFAULT_SPLITS[task]`.
    Passing a non-None `split` forces the same split for every task
    (operator override; useful when probing schemas).
    """
    tasks = tasks or PRIVATE_HARD_TASKS
    revisions = revisions or {}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for task in tasks:
        task_split = split if split is not None else TASK_DEFAULT_SPLITS[task]
        entries.append(cache_one_task(
            task,
            output_dir=output_dir,
            hf_token=hf_token,
            revision=revisions.get(task),
            split=task_split,
        ))

    manifest = {
        "_meta": "ralph-private-hard-cache-manifest",
        "version": "v1",
        "tasks": entries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    return manifest


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scripts.cache_hf_assets")
    p.add_argument("--output-dir", type=Path,
                   default=Path("eval/private/downstream_pool/private_hard"))
    p.add_argument("--task", action="append", dest="tasks", default=None,
                   help="repeat for each task; default = all 4 private-hard tasks")
    p.add_argument("--hf-token", default=None)
    p.add_argument("--revision", action="append", default=None,
                   help="pin a specific HF revision in the form task=<sha>; "
                        "repeat for multiple tasks")
    p.add_argument("--split", default=None,
                   help="HF dataset split to pull. If omitted, each task "
                        "uses its TASK_DEFAULT_SPLITS entry "
                        "(validation for ARC/winogrande/tinyArc; "
                        "test for tinyMMLU, which lacks a validation split).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    revisions: dict[str, str] = {}
    for entry in (args.revision or []):
        if "=" not in entry:
            print(f"ERROR: --revision must be task=<sha>, got {entry!r}",
                  file=sys.stderr)
            return 2
        task, sha = entry.split("=", 1)
        revisions[task.strip()] = sha.strip()
    try:
        manifest = cache_all(
            output_dir=args.output_dir,
            tasks=tuple(args.tasks) if args.tasks else None,
            hf_token=args.hf_token,
            revisions=revisions,
            split=args.split,
        )
    except Exception as e:
        print(f"cache_hf_assets FAILED: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 1
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
