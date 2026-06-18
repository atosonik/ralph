"""Owner-validator audit report — validation-v2 Phase 1.

The owner validator (H200) is the single node that does the heavy hidden-eval
scoring, sets weights, and — added here — anchors a per-epoch **audit
commitment** on-chain so anyone can re-derive its decisions from published data.

This module builds that per-epoch report:

  1. `build_report_json(...)` assembles a deterministic dict from the epoch's
     scored submissions + the weight snapshot the validator just set.
  2. `canonical_json(...)` / `report_sha256(...)` produce the byte-exact
     canonicalization + hash. THE AUDITOR DEPENDS ON THIS BEING
     BYTE-IDENTICAL — `json.dumps(obj, sort_keys=True,
     separators=(",",":"), ensure_ascii=False).encode("utf-8")`. Do not
     change the separators / sort / encoding without a coordinated auditor
     bump.
  3. `sign_report(...)` ed25519-signs the canonical bytes with the validator
     hotkey `Keypair`.
  4. `write_report(...)` writes `<out_dir>/audit_reports/<epoch_id>.json` and
     upserts an `index.json` so an auditor can enumerate epochs.

The owner cannot lie two ways: (a) commit a hash != raw report -> caught by
re-hash; (b) commit a report whose weights != scorer(raw data) -> caught by
replay. See docs/rearch_2026_06/childkey_owner_auditor_architecture.md.

Design note — field availability: the three Gate-4 reproducibility fields
(`val_seq_len`, `sealed_stream_manifest_hash`, `tail_val_bpb`) are now populated
for REAL — the hidden-eval surfaces the context length it used, a content hash
of the sealed eval set it scored against, and a long-context tail probe (BPB
over the tail positions `[val_seq_len//2 :]`, recorded only — the scorer does
not consume it yet). A few idealized-schema fields the current scorer still does
not emit (`downstream_acc`, `fraud_penalty`, standalone `recipe_diff_sha256`)
remain `None` (clearly stubbed) so the schema shape is stable and an auditor can
rely on the keys existing; they get filled in as the pipeline grows. Everything
the current scorer DOES emit (bundle/submission sha256, miner hotkey,
parent_king_attestation_hash, val_bpb, tail_val_bpb, val_seq_len,
sealed_stream_manifest_hash, decision/gate, decisive_vs_king, seed, ladder rung
dims, final score/weight) is populated for real.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Schema / format constants — frozen. A unilateral change to any of these is
# an intended divergence from the auditor and must be coordinated.
# ---------------------------------------------------------------------------

AUDIT_REPORT_SCHEMA_VERSION = "v1"

# The canonical ladder rung dimensions the validator pins (mirrors
# validator/ladder.py _STANDARD_RUNGS_DEFAULT). Carried in eval_input so an
# auditor can reproduce the eval at any tier without guessing the geometry.
STANDARD_LADDER_RUNGS: tuple[dict[str, int], ...] = (
    {"scale_label_index": 0, "dim": 256, "n_layers": 4},   # S1 ~1.5M
    {"scale_label_index": 1, "dim": 512, "n_layers": 12},  # S2 ~18M
    {"scale_label_index": 2, "dim": 768, "n_layers": 12},  # S3 ~124M
)
_STANDARD_LADDER_RUNGS_LABELED: tuple[dict[str, Any], ...] = (
    {"scale_label": "S1", "dim": 256, "n_layers": 4},
    {"scale_label": "S2", "dim": 512, "n_layers": 12},
    {"scale_label": "S3", "dim": 768, "n_layers": 12},
)


# ---------------------------------------------------------------------------
# Canonicalization + hashing — MUST stay byte-identical validator <-> auditor.
# ---------------------------------------------------------------------------


def canonical_json(obj: Any) -> bytes:
    """Deterministic, byte-exact JSON encoding.

    The auditor re-hashes the published report and compares to the on-chain
    commitment; if this encoding drifts even by a separator the hash check
    fails. Keep this EXACTLY:

        json.dumps(obj, sort_keys=True, separators=(",",":"),
                   ensure_ascii=False).encode("utf-8")
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def report_sha256(obj: Any) -> str:
    """Return the 64-hex sha256 of `canonical_json(obj)`."""
    return hashlib.sha256(canonical_json(obj)).hexdigest()


def sign_report(canonical_bytes: bytes, keypair: Any) -> str:
    """ed25519-sign `canonical_bytes` with a bittensor `Keypair`.

    Returns the signature as lowercase hex. `keypair.sign(bytes)` returns raw
    signature bytes (64 for sr25519/ed25519); an auditor reconstructs the
    `Keypair` from the signer's ss58 hotkey and calls `.verify(bytes, sig)`.
    """
    sig = keypair.sign(canonical_bytes)
    # bittensor Keypair.sign returns bytes; be defensive about a hex-str impl.
    if isinstance(sig, bytes):
        return sig.hex()
    if isinstance(sig, str):
        return sig[2:] if sig.startswith("0x") else sig
    return bytes(sig).hex()


# ---------------------------------------------------------------------------
# Report construction
# ---------------------------------------------------------------------------


def _eval_input_for(scored: dict, seed: int) -> dict:
    """Build the `eval_input` block — everything an auditor needs to reproduce
    the eval. `val_seq_len` and `sealed_stream_manifest_hash` are now surfaced
    by the hidden-eval (Gate-4 reproducibility); the bundle_hash remains the
    submission anchor."""
    return {
        "bundle_sha256": scored.get("bundle_hash"),
        # Content hash of the sealed eval set the validator scored against —
        # lets an auditor confirm it re-runs over the identical held-out data.
        "sealed_stream_manifest_hash": scored.get("sealed_stream_manifest_hash"),
        "seed": seed,
        # The context length the hidden-eval used (RalphConfig.max_seq_len//2).
        "val_seq_len": scored.get("val_seq_len"),
        "ladder_rungs": list(_STANDARD_LADDER_RUNGS_LABELED),
    }


def _eval_output_for(scored: dict) -> dict:
    """Build the `eval_output` block from the scorer's result dict."""
    return {
        "val_bpb": scored.get("val_bpb"),
        # Long-context tail probe: BPB over the tail positions [val_seq_len//2:].
        # Recorded only — the scorer does not consume it yet.
        "tail_val_bpb": scored.get("tail_val_bpb"),
        "benchmark_accuracy": scored.get("benchmark_accuracy"),
        "quality_gain": scored.get("quality_gain"),
        "score": scored.get("score"),
        "tier": scored.get("tier"),
        # gate / decision: the scorer's classification + the decisive flag.
        "gate": scored.get("classification"),
        "status": scored.get("status"),
        "decisive_vs_king": bool(scored.get("decisive", False)),
        "accepted": bool(scored.get("accepted", False)),
        "is_first": bool(scored.get("is_first", False)),
    }


def _submission_entry(scored: dict, seed: int, observed_at: Any) -> dict:
    """One `submissions[]` entry from a single score_and_decide result dict."""
    return {
        "submission_hash": scored.get("bundle_hash"),
        "bundle_sha256": scored.get("bundle_hash"),
        "miner_hotkey": scored.get("miner_hotkey"),
        "miner_github": scored.get("miner_github", ""),
        # recipe diff identity. The scorer dict doesn't carry a standalone
        # recipe_diff_sha256 yet (the bundle_hash folds the patch sha in), so
        # we surface whatever was passed alongside, else None.
        "recipe_diff_sha256": scored.get("recipe_diff_sha256"),  # GAP if not threaded
        "parent_king_attestation_hash": scored.get("parent_king_attestation_hash"),
        "eval_input": _eval_input_for(scored, seed),
        "eval_output": _eval_output_for(scored),
        # the submission's own attestation hash (king_attestation_hash once crowned).
        "attestation_hash": scored.get("attestation_hash"),
        "observed_at": observed_at,
    }


def build_report_json(
    epoch_id: str,
    netuid: int,
    start_block: int,
    end_block: int,
    generated_at: Any,
    scored: list[dict],
    weight_snapshot: dict[str, float],
    *,
    seed: int = 0,
    created_at: Any = None,
) -> dict:
    """Assemble the per-epoch `report_json` from the epoch's scored submissions.

    Args:
      epoch_id:   stable id, e.g. "40-<end_block>".
      netuid:     subnet uid (40 on mainnet).
      start_block/end_block: epoch block range.
      generated_at: timestamp string. Passed IN (not computed) so the report
        is deterministic / testable. Do NOT call datetime.now() here.
      scored:     list of score_and_decide() result dicts (the per-submission
        view). Rejected/non-scored entries are tolerated — they contribute a
        thin entry (no eval_output fields) rather than being dropped, so the
        history stays append-only and complete.
      weight_snapshot: {hotkey: weight} the validator set this epoch (round_scores).
      seed:       the eval seed (carried into each eval_input for reproduction).
      created_at: optional timestamp for the embedded weight_snapshot /
        scorecards; defaults to `generated_at` when None (keeps it injectable).

    Returns a plain dict — hash/sign/serialize it via the other helpers.
    """
    if created_at is None:
        created_at = generated_at

    submissions = [_submission_entry(s, seed, generated_at) for s in scored]

    # scorecards: final_score + weight per hotkey. final_score comes from the
    # scorer dict; weight comes from the snapshot the validator actually set
    # (round_scores after the 90/10 pool split).
    scorecards = []
    for s in scored:
        hk = s.get("miner_hotkey")
        scorecards.append({
            "miner_hotkey": hk,
            "final_score": s.get("score"),
            "quality_gain": s.get("quality_gain"),
            "fraud_penalty": s.get("fraud_penalty"),  # GAP: not yet emitted
            "weight": weight_snapshot.get(hk),
            "classification": s.get("classification"),
            "computed_at": created_at,
        })

    return {
        "schema_version": AUDIT_REPORT_SCHEMA_VERSION,
        "epoch_id": epoch_id,
        "netuid": netuid,
        "epoch_start_block": start_block,
        "epoch_end_block": end_block,
        "generated_at": generated_at,
        "submissions": submissions,
        "scorecards": scorecards,
        "weight_snapshot": {
            "netuid": netuid,
            "weights": dict(weight_snapshot),
            "created_at": created_at,
        },
    }


def build_envelope(
    report_json: dict,
    *,
    signature: str,
    signer_hotkey: str,
    chain_commitment_block: Optional[int],
    weights_set: bool = False,
) -> dict:
    """Wrap `report_json` with its hash, signature, signer, and commit block.

    `weights_set` is an ENVELOPE-level field (deliberately NOT inside the signed
    `report_json`): the report records what the validator DECIDED this epoch,
    while `weights_set` records whether the weight extrinsic actually landed
    on-chain (set_weights can rate-limit and return early). Keeping it out of
    `report_json` means the signed/hashed decision record is identical whether
    or not the extrinsic landed, and an auditor reads decision-vs-landed from
    the two separate places.
    """
    return {
        "report_json": report_json,
        "report_sha256": report_sha256(report_json),
        "signature": signature,
        "signer_hotkey": signer_hotkey,
        "chain_commitment_block": chain_commitment_block,
        "weights_set": bool(weights_set),
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_report(report_envelope: dict, out_dir: Path) -> Path:
    """Write the signed report envelope + upsert the epoch index.

    Layout:
      <out_dir>/audit_reports/<epoch_id>.json   — the full signed envelope
      <out_dir>/audit_reports/index.json        — append-only epoch index

    Returns the path to the written per-epoch report.
    """
    out_dir = Path(out_dir)
    reports_dir = out_dir / "audit_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    report_json = report_envelope["report_json"]
    epoch_id = report_json["epoch_id"]

    report_path = reports_dir / f"{epoch_id}.json"
    report_path.write_text(
        json.dumps(report_envelope, indent=2, sort_keys=True, ensure_ascii=False)
    )

    # Upsert index.json. Append (or replace-by-epoch_id) the index entry.
    index_path = reports_dir / "index.json"
    index: list[dict] = []
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text())
            if isinstance(loaded, list):
                index = loaded
        except (json.JSONDecodeError, OSError):
            index = []

    entry = {
        "epoch_id": epoch_id,
        "epoch_start_block": report_json.get("epoch_start_block"),
        "epoch_end_block": report_json.get("epoch_end_block"),
        "report_sha256": report_envelope.get("report_sha256"),
        "signer_hotkey": report_envelope.get("signer_hotkey"),
        "chain_commitment_block": report_envelope.get("chain_commitment_block"),
        "weights_set": report_envelope.get("weights_set"),
    }
    # Replace any existing entry for the same epoch_id (idempotent re-run),
    # else append.
    index = [e for e in index if e.get("epoch_id") != epoch_id]
    index.append(entry)
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False)
    )

    # TODO Phase 2: publish to HF — upload `report_path` + `index.json` to
    # RalphLabsAI/audit-reports so auditors pull from the Hub. The on-chain
    # commitment remains the trust anchor; HF is just the off-chain store.
    # e.g. huggingface_hub.upload_file(path_or_fileobj=report_path,
    #          path_in_repo=f"audit_reports/{epoch_id}.json",
    #          repo_id="RalphLabsAI/audit-reports", repo_type="dataset")

    return report_path


__all__ = [
    "AUDIT_REPORT_SCHEMA_VERSION",
    "STANDARD_LADDER_RUNGS",
    "build_envelope",
    "build_report_json",
    "canonical_json",
    "report_sha256",
    "sign_report",
    "write_report",
]
