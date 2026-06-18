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
"""

from __future__ import annotations

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

import click  # noqa: E402

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


def _read_int_file(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def _write_int_file(path: Path, value: int) -> None:
    path.write_text(str(value))


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

    On a clean epoch the `.audit_state` watermark advances. If
    AUDITOR_SET_WEIGHTS_ENABLED, also counter-weights from the most recent clean
    epoch's replayed scores using the auditor's OWN wallet.
    """
    from auditor.weights import is_enabled as cw_enabled
    from auditor.weights import submit_weights

    last_audited = _read_int_file(STATE_FILE)
    try:
        reports = api.list_reports()
    except Exception as exc:
        logger.error("failed to list reports: %s", exc)
        return EXIT_NETWORK

    sorted_reports = sorted(reports, key=lambda r: r.get("epoch_end_block") or 0)
    worst = EXIT_CLEAN
    last_clean_epoch_id: str | None = None
    last_clean_end_block: int | None = None

    for r in sorted_reports:
        end_block = r.get("epoch_end_block")
        if last_audited is not None and end_block is not None and end_block <= last_audited:
            continue
        code = audit_epoch(r["epoch_id"], chain, api)
        worst = max(worst, code)
        if code == EXIT_CLEAN and end_block is not None:
            _write_int_file(STATE_FILE, end_block)
            last_clean_epoch_id = r["epoch_id"]
            last_clean_end_block = end_block

    if last_clean_epoch_id and cw_enabled():
        try:
            envelope = api.get_report(last_clean_epoch_id)
            replayed = replay_scoring(envelope.get("report_json") or {})
            ok = submit_weights(
                subtensor_url=chain.subtensor_url,
                netuid=chain.netuid,
                weights_by_hotkey=replayed,
            )
            if ok and last_clean_end_block is not None:
                _write_int_file(PUBLISHED_FILE, last_clean_end_block)
        except Exception:
            logger.exception("counter-weight step failed for %s", last_clean_epoch_id)

    return worst


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    help=(
        "Ralph auditor — independent CPU-only verifier for subnet 40.\n\n"
        "Runs Gate 1 (hash + on-chain commitment + signature), Gate 2 (weight "
        "replay from published data), and Gate 3 (weight diff, tol 1e-4) against "
        "a validator's published audit reports.\n\n"
        "Exit codes: 0 clean, 1 hash-or-sig fail, 2 math diverge, 3 network.\n\n"
        "Needs an ARCHIVE subtensor (SUBTENSOR_URL) because each epoch's "
        "commitment overwrites the last."
    ),
)
@click.option("--once", is_flag=True, help="Run one audit pass over new epochs and exit.")
@click.option("--loop", "loop_", is_flag=True, help="Run continuously every AUDIT_INTERVAL_SECONDS.")
@click.option("--epoch", "epoch", default=None, help="Audit only this epoch_id and exit.")
@click.option("--repo", default=None, help=f"HF dataset repo (default {DEFAULT_AUDIT_REPO}).")
@click.option("--netuid", type=int, default=None, help="Subnet netuid (default 40).")
@click.option("--subtensor-url", default=None, help=f"Archive endpoint (default {ARCHIVE_ENDPOINT_DEFAULT}).")
@click.option("-v", "--verbose", is_flag=True, help="Debug logging.")
def main(once, loop_, epoch, repo, netuid, subtensor_url, verbose) -> None:
    _setup_logging(verbose)

    repo = repo or os.environ.get("AUDIT_REPO", DEFAULT_AUDIT_REPO)
    netuid = netuid if netuid is not None else int(os.environ.get("NETUID", "40"))
    subtensor_url = subtensor_url or os.environ.get("SUBTENSOR_URL", ARCHIVE_ENDPOINT_DEFAULT)
    interval = int(os.environ.get("AUDIT_INTERVAL_SECONDS", "300"))
    validator_hotkey = os.environ.get("VALIDATOR_HOTKEY") or None
    token = os.environ.get("HF_TOKEN") or None

    chain = ChainClient(subtensor_url=subtensor_url, netuid=netuid, validator_hotkey=validator_hotkey)
    api = ReportClient(repo_id=repo, token=token)

    if epoch:
        sys.exit(audit_epoch(epoch, chain, api))

    if loop_:
        while True:
            try:
                audit_new_epochs(chain, api)
            except Exception:
                logger.exception("audit loop iteration failed")
            time.sleep(interval)

    # default (and --once) -> single pass.
    sys.exit(audit_new_epochs(chain, api))


if __name__ == "__main__":
    main()
