#!/usr/bin/env python3
"""
End-to-end miner script — runs on a remote H100 to participate in Karpa.

Flow:
  1. Hash the patch file (or empty patch for baseline)
  2. Request handshake nonce — commits (hotkey, patch_hash, nonce) on-chain
  3. Run the canonical proof test (training in the official Docker image, or
     direct Python with --no-docker for testing)
  4. Assemble + sign the submission bundle
  5. Upload the bundle to HuggingFace Hub for validators to pick up

After this finishes, validators worldwide will find the bundle by polling
the HF dataset repo and score it on their side.

Usage on a remote H100:
    # Setup .env with BT_NETWORK=test BT_NETUID=16 BT_WALLET=... BT_HOTKEY=... HF_TOKEN=...
    python scripts/miner_run.py --patch patches/raise_lr.diff --label round1
    python scripts/miner_run.py --baseline --label baseline   # empty patch
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import karpa_bootstrap  # noqa: F401  — injects KARPA_RECIPE_DIR onto sys.path
from chain_layer.config import get_chain
from miner.hub import upload_bundle
from miner.submit import sign_submission
from proof.runner import run_proof_test

KARPA_ROOT = Path(__file__).resolve().parent.parent


def _get_hotkey_ss58(wallet_name: str, hotkey_name: str) -> str:
    import bittensor as bt
    w = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
    return w.hotkey.ss58_address


def _summarise_rationale(rationale_text: str, max_chars: int = 280) -> str:
    """Pull a short summary out of a rationale markdown file.
    Preference: the **Summary:** bold line if present; otherwise the first
    non-heading paragraph; otherwise the first non-empty line.
    Trimmed/ellipsed to max_chars.
    """
    if not rationale_text.strip():
        return ""
    import re as _re
    # Single-line capture anchored at start of a line — stops at end of line so
    # a missing trailing \n\n doesn't slurp the rest of the file.
    m = _re.search(r"^\s*\*\*Summary:\*\*\s*(.+)$", rationale_text, _re.MULTILINE)
    if m:
        snippet = m.group(1).strip()
    else:
        paragraphs = [p.strip() for p in rationale_text.split("\n\n")]
        # For each paragraph, strip any leading heading lines (starting with "#")
        # so a paragraph that bundles "# Heading\nBody" still yields its body.
        normalised = []
        for p in paragraphs:
            lines = p.split("\n")
            while lines and lines[0].lstrip().startswith("#"):
                lines.pop(0)
            cleaned = "\n".join(lines).strip()
            normalised.append(cleaned)
        body = [p for p in normalised if p]
        if body:
            snippet = body[0]
        elif paragraphs:
            # Heading-only fallback: strip leading "#" and whitespace so the
            # literal marker doesn't leak into the summary.
            snippet = paragraphs[0].lstrip("#").strip()
        else:
            snippet = ""
    snippet = " ".join(snippet.split())  # collapse whitespace + newlines
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 1].rstrip() + "…"
    return snippet


def run_miner(
    patch_path: Path | None,
    label: str,
    config_path: str,
    tier: str,
    hf_repo: str,
    hf_token: str | None,
    seed: int,
    skip_upload: bool,
    rationale_path: Path | None = None,
) -> dict:
    import os

    wallet_name = os.environ.get("BT_WALLET", "default")
    hotkey_name = os.environ.get("BT_HOTKEY", "default")
    miner_gh = os.environ.get("KARPA_MINER_GH", "")
    miner_hotkey = _get_hotkey_ss58(wallet_name, hotkey_name)

    print(f"\n{'='*60}")
    print(f"  KARPA MINER — {label}")
    print(f"{'='*60}")
    print(f"  wallet: {wallet_name}/{hotkey_name}")
    print(f"  hotkey: {miner_hotkey}")
    if miner_gh:
        print(f"  gh:     {miner_gh}")
    print(f"  config: {config_path}")
    print(f"  tier:   {tier}")
    print(f"  hf:     {hf_repo}")

    chain = get_chain(KARPA_ROOT)
    if not chain.is_hotkey_registered(miner_hotkey):
        raise RuntimeError(
            f"hotkey {miner_hotkey} is NOT registered on netuid "
            f"{getattr(chain, 'netuid', '?')}. Register first via btcli."
        )

    # ---- 1. Prepare submission directory -----------------------------------
    sub_dir = KARPA_ROOT / f"runs/miner/{label}_sub"
    proof_dir = KARPA_ROOT / f"runs/miner/{label}_proof"
    for d in [sub_dir, proof_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    target_patch = sub_dir / "patch.diff"
    if patch_path is None:
        target_patch.write_text("")
        patch_text = ""
    else:
        patch_text = patch_path.read_text()
        target_patch.write_text(patch_text)
    patch_hash = hashlib.sha256(patch_text.encode()).hexdigest()
    print(f"  patch_hash: {patch_hash[:24]}...  ({len(patch_text)} bytes)")

    # Read the rationale upfront so we can fail fast if the file's missing
    # AND so we can embed it into the proof bundle before run_proof_test()
    # snapshots the bundle_manifest.
    rationale_text = ""
    rationale_summary = ""
    if rationale_path is not None:
        # Hardened loader: resolve strict (raises if missing), reject symlinks
        # and non-regular files, cap size before reading, decode strict-utf8.
        try:
            resolved = rationale_path.resolve(strict=True)
        except FileNotFoundError:
            raise FileNotFoundError(f"--rationale path does not exist: {rationale_path}")
        if rationale_path.is_symlink() or resolved.is_symlink():
            raise ValueError(f"--rationale path must not be a symlink: {rationale_path}")
        if not resolved.is_file():
            raise ValueError(f"--rationale path must be a regular file: {rationale_path}")
        st = resolved.stat()
        if stat.S_IFMT(st.st_mode) != stat.S_IFREG:
            raise ValueError(f"--rationale path must be a regular file: {rationale_path}")
        _RATIONALE_MAX_BYTES = 65536
        if st.st_size > _RATIONALE_MAX_BYTES:
            raise ValueError(
                f"--rationale file too large: {st.st_size} bytes "
                f"(max {_RATIONALE_MAX_BYTES}): {rationale_path}"
            )
        if resolved.suffix.lower() not in {".md", ".txt"}:
            print(f"  WARNING: --rationale extension is '{resolved.suffix}' "
                  f"(expected .md or .txt); reading anyway.")
        try:
            rationale_text = resolved.read_text(encoding="utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise ValueError(
                f"--rationale file is not valid UTF-8 ({e}): {rationale_path}"
            ) from e
        rationale_summary = _summarise_rationale(rationale_text)
        print(f"  rationale: {rationale_summary[:80]}{'…' if len(rationale_summary) > 80 else ''}")

    # ---- 2. Handshake — commit on-chain ------------------------------------
    print("\n[1/6] handshake — committing (hotkey, patch_hash, nonce) on-chain...")
    nonce = chain.request_handshake_nonce(miner_hotkey, patch_hash)
    print(f"      nonce: {nonce[:32]}...")

    (sub_dir / "proof_request.json").write_text(json.dumps({
        "handshake_nonce": nonce,
        "seed": seed,
        "config_path": config_path,
        "miner_hotkey": miner_hotkey,
    }, indent=2))

    # ---- 3. Run the proof test ---------------------------------------------
    print("\n[2/6] proof test — running canonical training...")
    t0 = time.time()
    bundle = run_proof_test(
        karpa_root=KARPA_ROOT,
        submission_dir=sub_dir,
        out_dir=proof_dir,
        tier=tier,
    )
    elapsed = time.time() - t0
    print(f"      bundle_hash: {bundle.bundle_hash[:24]}...")
    print(f"      elapsed:     {elapsed:.1f}s")

    # Write rationale.md into proof_dir so HF upload picks it up. This is the
    # only on-disk copy that actually ships — the bundle manifest already
    # hashes its inputs; rationale.md is an additional human-facing artifact
    # that travels with the bundle. Always write when a rationale was provided
    # (even if empty) so behaviour matches the submission.json hypothesis field.
    if rationale_path is not None:
        (proof_dir / "rationale.md").write_text(rationale_text)

    # ---- 4. Sign submission ------------------------------------------------
    print("\n[3/6] signing submission...")
    # Sign over bundle_hash + nonce + hotkey + hypothesis-hash. Hypothesis
    # is folded in so the miner can't swap the rationale post-merge.
    sig = sign_submission(KARPA_ROOT, miner_hotkey, bundle.bundle_hash, nonce, hypothesis=rationale_summary)
    print(f"      signed by {sig['public_key_hex'][:24]}...")

    # ---- 5. Open PR against karpaai/recipe (before HF upload so it ends up
    #         in the HF PR's submission.json) -------------------------------
    pr_url = ""
    fork_url = os.environ.get("KARPA_RECIPE_FORK", "")
    gh_token = os.environ.get("KARPA_MINER_GH_TOKEN", "")
    upstream = os.environ.get("KARPA_RECIPE_UPSTREAM", "karpaai/recipe")
    if not patch_text.strip():
        print("\n[4/6] skipping recipe PR (baseline submission, empty patch)")
    elif skip_upload:
        print("\n[4/6] skipping recipe PR (--skip-upload also implies no PR)")
    elif not fork_url or not gh_token:
        print("\n[4/6] WARNING: KARPA_RECIPE_FORK or KARPA_MINER_GH_TOKEN missing — not opening recipe PR")
    else:
        print(f"\n[4/6] opening recipe PR against {upstream}...")
        from miner.github_pr import open_recipe_pr
        try:
            pr_url = open_recipe_pr(
                patch_text=patch_text,
                bundle_hash=bundle.bundle_hash,
                miner_hotkey=miner_hotkey,
                miner_github=miner_gh,
                hf_bundle_url="",  # not known yet; HF PR is opened next
                signature_hex=sig["signature_hex"],
                fork_url=fork_url,
                token=gh_token,
                upstream=upstream,
                rationale_text=rationale_text,
            )
            print(f"      recipe PR: {pr_url}")
        except Exception as e:
            print(f"      WARNING: recipe PR open failed ({e}). Submission still uploaded to HF.")

    submission = {
        "miner_hotkey": miner_hotkey,
        "miner_github": miner_gh,
        "handshake_nonce": nonce,
        "patch_path": str(target_patch),
        "proof_dir": str(proof_dir),
        "bundle_hash": bundle.bundle_hash,
        "signature_hex": sig["signature_hex"],
        "public_key_hex": sig["public_key_hex"],
        "submitted_at": time.time(),
        "label": label,
        "pr_url": pr_url,
        "hf_bundle_url": "",  # filled by validator/log only; the PR itself IS the bundle on HF
        "hypothesis": rationale_summary,  # short machine-readable, full markdown in rationale.md
    }
    (proof_dir / "submission.json").write_text(json.dumps(submission, indent=2, sort_keys=True))

    # ---- 6. Upload bundle as a single HF PR (includes submission.json) -----
    if skip_upload:
        print("\n[5/6] skipping HF upload (--skip-upload)")
        url = None
    else:
        print(f"\n[5/6] uploading bundle to HF Hub {hf_repo} as PR...")
        # patch.diff lives in sub_dir (the submission staging area). Pass it
        # explicitly so the bundle on HF carries it — the validator's
        # PR-match verifier reads bundle_dir/patch.diff to byte-equal the
        # GitHub PR's diff; without it the check silently no-ops.
        url = upload_bundle(
            proof_dir,
            repo_id=hf_repo,
            token=hf_token,
            rationale_text=rationale_text,
            patch_path=target_patch if patch_text.strip() else None,
        )

    # ---- 7. Done -----------------------------------------------------------
    print("\n[6/6] DONE")
    print(f"  bundle_hash: {bundle.bundle_hash}")
    print(f"  proof_dir:   {proof_dir}")
    if url:
        print(f"  hf url:      {url}")
    if pr_url:
        print(f"  pr:          {pr_url}")
    print("\nValidators will now find this on HF Hub and score it.")
    print(f"Track status: tail -f {KARPA_ROOT}/chain*/events.jsonl  (on validator host)")

    return {
        "miner_hotkey": miner_hotkey,
        "bundle_hash": bundle.bundle_hash,
        "patch_hash": patch_hash,
        "nonce": nonce,
        "pr_url": pr_url,
        "proof_dir": str(proof_dir),
        "hf_url": url,
        "elapsed_s": elapsed,
    }


def main() -> None:
    import os

    p = argparse.ArgumentParser(description="Karpa end-to-end miner")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--patch", type=Path, help="Path to patch file to submit")
    g.add_argument("--baseline", action="store_true", help="Submit empty patch (baseline)")
    p.add_argument("--label", required=True, help="Human label for this run (used in paths)")
    p.add_argument("--config", default="configs/proxy_cpu_smoke.json",
                   help="Recipe config (default: proxy_cpu_smoke.json — use proxy_h100.json on H100)")
    p.add_argument("--tier", default="unverified", choices=["verified", "unverified"],
                   help="Attestation tier (verified requires CC; default: unverified)")
    p.add_argument("--hf-repo", default=os.environ.get("KARPA_HF_REPO", "karpaai/proof-bundles"),
                   help="HF dataset repo to upload to")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF API token (defaults to $HF_TOKEN)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rationale", type=Path, default=None,
                   help="Markdown file with the hypothesis / reasoning behind this patch. "
                        "Travels with the bundle, becomes the GH PR body and the HF PR description, "
                        "and seeds submission.json's hypothesis field.")
    p.add_argument("--skip-upload", action="store_true",
                   help="Run locally but skip HF upload (for testing)")
    args = p.parse_args()

    result = run_miner(
        patch_path=args.patch,
        label=args.label,
        config_path=args.config,
        tier=args.tier,
        hf_repo=args.hf_repo,
        hf_token=args.hf_token,
        seed=args.seed,
        skip_upload=args.skip_upload,
        rationale_path=args.rationale,
    )
    print(f"\n{json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()
