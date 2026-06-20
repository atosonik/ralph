"""Ralph auditor CLI — independent CPU-only verifier for subnet 40.

Per epoch it runs three gates against a validator's published audit report:

  Gate 1 (verify.py): recompute sha256(canonical_json(report_json)) and assert
      it equals the envelope hash AND the on-chain commitment at epoch_end_block,
      and that the ed25519 signature is valid. Fail -> exit 1.
  Gate 2 (replay.py): recompute the epoch weight vector from the published raw
      data, importing the validator's weight/floor constants.
  Gate 3 (diff.py):   diff replayed vs claimed weights (tol 1e-4). Fail -> exit 2.

Exit codes (worst across the pass): 0 clean / 1 hash-or-sig / 2 math-diverge /
3 network. A clean epoch advances the local `.audit_state` watermark so `--loop`
only re-audits new epochs.

    python -m auditor --once
    python -m auditor --loop
    python -m auditor --epoch 40-1234567
    python -m auditor --help

Env:
    AUDIT_REPO            HF dataset repo (default RalphLabsAI/audit-reports)
    SUBTENSOR_URL         archive endpoint (default wss://archive.chain.opentensor.ai:443/)
    NETUID                subnet (default 40)
    VALIDATOR_HOTKEY      signer ss58 to read the on-chain commitment for
                          (default: the report's own signer_hotkey)
    HF_TOKEN              only for a private report repo (public needs none)
    AUDIT_INTERVAL_SECONDS  --loop sleep (default 300)
    AUDITOR_SET_WEIGHTS_ENABLED  opt-in counter-weight (default off)
    AUDITOR_WEIGHT_INTERVAL_BLOCKS  re-set weights every N blocks (default 300)

Counter-weight (when enabled) runs on a BLOCK cadence, not per-report: each pass
reads how many blocks since the auditor's hotkey last set weights and re-sets
from the latest clean epoch's replayed scores once ~300 blocks have elapsed, so
the auditor-validator keeps its weights (and vTrust) fresh every epoch.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Make the sibling `validator` package importable so verify/replay can IMPORT
# the validator's canonical_json + weight constants (the fidelity guarantee).
# auditor/ lives at the repo root next to validator/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from auditor.chain import ARCHIVE_ENDPOINT_DEFAULT, ChainClient  # noqa: E402
from auditor.diff import compare_weights  # noqa: E402
from auditor.fetch import DEFAULT_AUDIT_REPO, ReportClient  # noqa: E402
from auditor.replay import replay_scoring  # noqa: E402
from auditor.verify import verify_report  # noqa: E402

logger = logging.getLogger("ralph-auditor")

EXIT_CLEAN = 0
EXIT_HASH_OR_SIG = 1
EXIT_MATH_DIVERGE = 2
EXIT_NETWORK = 3

STATE_FILE = Path(".audit_state")
PUBLISHED_FILE = Path(".audit_published")
LAST_CLEAN_EPOCH_FILE = Path(".audit_last_clean_epoch")  # epoch_id of the most recent clean epoch


def _read_int_file(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _write_int_file(path: Path, value: int) -> None:
    path.write_text(str(value))


def _read_str_file(path: Path) -> str | None:
    if not path.exists():
        return None
    val = path.read_text().strip()
    return val or None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def audit_epoch(epoch_id: str, chain: ChainClient, api: ReportClient) -> int:
    """Run Gates 1-3 for a single epoch. Returns its exit code."""
    logger.info("auditing epoch %s", epoch_id)

    try:
        envelope = api.get_report(epoch_id)
    except Exception as exc:
        logger.error("epoch %s: failed to fetch report: %s", epoch_id, exc)
        return EXIT_NETWORK

    report_json = envelope.get("report_json") or {}
    end_block = report_json.get("epoch_end_block")
    signer = envelope.get("signer_hotkey") or os.environ.get("VALIDATOR_HOTKEY") or None

    # Read the on-chain commitment at the historical block. A network failure
    # here is EXIT_NETWORK, not a verification failure — we don't penalize the
    # validator for the auditor's RPC being down.
    on_chain_hash = None
    if end_block is not None and signer:
        try:
            on_chain_hash = chain.get_commitment_hash(int(end_block), hotkey=signer)
        except Exception as exc:
            logger.error("epoch %s: chain query failed: %s", epoch_id, exc)
            return EXIT_NETWORK
        if on_chain_hash is None:
            logger.warning(
                "epoch %s: no on-chain commitment found at block %s for signer %s "
                "— verifying self-consistency + signature only",
                epoch_id, end_block, signer,
            )

    # Gate 1: hash (self + on-chain) + signature.
    try:
        verify_report(envelope, expected_onchain_hash=on_chain_hash, signer_hotkey=signer)
    except AssertionError as exc:
        logger.error("epoch %s: Gate-1 FAIL — %s", epoch_id, exc)
        return EXIT_HASH_OR_SIG

    # Gate 2: replay the weight derivation from published raw data.
    replayed = replay_scoring(report_json)

    # Gate 3: diff replayed vs claimed.
    claimed = (report_json.get("weight_snapshot") or {}).get("weights") or {}
    discrepancies = compare_weights(claimed, replayed)
    if discrepancies:
        for hk, d in discrepancies.items():
            logger.error(
                "epoch %s: Gate-3 weight mismatch %s — claimed=%.6f replayed=%.6f Δ=%.6f",
                epoch_id, hk, d["claimed"], d["replayed"], d["delta"],
            )
        return EXIT_MATH_DIVERGE

    logger.info(
        "epoch %s: CLEAN — hash==on-chain, signature valid, weights replay-match (%d miners)",
        epoch_id, len(claimed),
    )
    return EXIT_CLEAN


def audit_new_epochs(chain: ChainClient, api: ReportClient) -> int:
    """Audit every epoch newer than the local watermark. Returns the worst code.

    On a clean epoch the `.audit_state` watermark advances and the epoch_id is
    recorded as the latest-clean epoch (consumed by `maybe_counter_weight`).
    Weight-setting is decoupled from this pass — it runs on a block cadence (see
    `maybe_counter_weight`) so the auditor keeps setting weights every epoch even
    when no new report dropped this cycle.
    """
    last_audited = _read_int_file(STATE_FILE)
    try:
        reports = api.list_reports()
    except Exception as exc:
        logger.error("failed to list reports: %s", exc)
        return EXIT_NETWORK

    sorted_reports = sorted(reports, key=lambda r: r.get("epoch_end_block") or 0)
    worst = EXIT_CLEAN

    for r in sorted_reports:
        end_block = r.get("epoch_end_block")
        if last_audited is not None and end_block is not None and end_block <= last_audited:
            continue
        code = audit_epoch(r["epoch_id"], chain, api)
        worst = max(worst, code)
        if code == EXIT_CLEAN and end_block is not None:
            _write_int_file(STATE_FILE, end_block)
            LAST_CLEAN_EPOCH_FILE.write_text(str(r["epoch_id"]))

    return worst


def maybe_counter_weight(chain: ChainClient, api: ReportClient) -> None:
    """Set the auditor's OWN weights on a block cadence (off unless enabled).

    Continuous epoch-cadence process, NOT one-shot: each call reads how many
    blocks since the auditor last set weights and, if at least
    `weight_set_interval_blocks` have elapsed (≈300 blocks ≈ 1h), re-sets weights
    from the latest CLEAN epoch's independently-replayed scores — shadowing the
    honest validator and keeping the auditor's own vTrust from decaying. Re-uses
    the last clean scores when no new report appeared this cycle. Never raises
    into the loop.
    """
    from auditor.weights import (
        auditor_hotkey_ss58,
        is_weight_set_due,
        submit_weights,
        weight_set_interval_blocks,
    )
    from auditor.weights import (
        is_enabled as cw_enabled,
    )

    if not cw_enabled():
        return
    hotkey = auditor_hotkey_ss58()
    if hotkey is None:
        return  # no wallet configured → stay read-only

    interval = weight_set_interval_blocks()
    try:
        blocks_since = chain.blocks_since_weight_set(hotkey)
        current = chain.get_current_block()
    except Exception:
        logger.exception("counter-weight: chain query failed; skipping this tick")
        return

    if not is_weight_set_due(blocks_since, interval):
        logger.info(
            "counter-weight: %s/%s blocks since last set (block %s) — not due",
            blocks_since, interval, current,
        )
        return

    epoch_id = _read_str_file(LAST_CLEAN_EPOCH_FILE)
    if not epoch_id:
        logger.info("counter-weight: due but no clean epoch audited yet — skipping")
        return

    logger.info(
        "counter-weight: due (%s≥%s blocks since last set, block %s) — setting from %s",
        blocks_since, interval, current, epoch_id,
    )
    try:
        envelope = api.get_report(epoch_id)
        replayed = replay_scoring(envelope.get("report_json") or {})
        ok = submit_weights(
            subtensor_url=chain.subtensor_url,
            netuid=chain.netuid,
            weights_by_hotkey=replayed,
        )
        if ok:
            _write_int_file(PUBLISHED_FILE, current)
    except Exception:
        logger.exception("counter-weight step failed for %s", epoch_id)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="auditor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Ralph auditor — independent CPU-only verifier for subnet 40.\n"
            "Gate 1 (hash + on-chain commitment + signature), Gate 2 (weight "
            "replay from published data), Gate 3 (weight diff, tol 1e-4) against\n"
            "a validator's published audit reports.\n"
            "Exit codes: 0 clean, 1 hash-or-sig fail, 2 math diverge, 3 network.\n"
            "Needs an ARCHIVE subtensor (SUBTENSOR_URL) — each epoch's "
            "commitment overwrites the last."
        ),
    )
    p.add_argument("--once", action="store_true", help="Run one audit pass over new epochs and exit.")
    p.add_argument("--loop", dest="loop_", action="store_true", help="Run continuously every AUDIT_INTERVAL_SECONDS.")
    p.add_argument("--epoch", default=None, help="Audit only this epoch_id and exit.")
    p.add_argument("--repo", default=None, help=f"HF dataset repo (default {DEFAULT_AUDIT_REPO}).")
    p.add_argument("--netuid", type=int, default=None, help="Subnet netuid (default 40).")
    p.add_argument(
        "--subtensor-url", dest="subtensor_url", default=None,
        help=f"Archive endpoint (default {ARCHIVE_ENDPOINT_DEFAULT}).",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging.")
    args = p.parse_args(argv)

    _setup_logging(args.verbose)

    repo = args.repo or os.environ.get("AUDIT_REPO", DEFAULT_AUDIT_REPO)
    netuid = args.netuid if args.netuid is not None else int(os.environ.get("NETUID", "40"))
    subtensor_url = args.subtensor_url or os.environ.get("SUBTENSOR_URL", ARCHIVE_ENDPOINT_DEFAULT)
    interval = int(os.environ.get("AUDIT_INTERVAL_SECONDS", "300"))
    validator_hotkey = os.environ.get("VALIDATOR_HOTKEY") or None
    token = os.environ.get("HF_TOKEN") or None

    chain = ChainClient(subtensor_url=subtensor_url, netuid=netuid, validator_hotkey=validator_hotkey)
    api = ReportClient(repo_id=repo, token=token)

    if args.epoch:
        sys.exit(audit_epoch(args.epoch, chain, api))

    if args.loop_:
        while True:
            try:
                audit_new_epochs(chain, api)
                maybe_counter_weight(chain, api)  # block-cadence weight-set each tick
            except Exception:
                logger.exception("audit loop iteration failed")
            time.sleep(interval)

    # default (and --once) -> single pass.
    code = audit_new_epochs(chain, api)
    maybe_counter_weight(chain, api)
    sys.exit(code)


if __name__ == "__main__":
    main()
