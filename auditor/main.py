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
    AUDIT_INTERVAL_SECONDS  --loop: how often the Gates-1-3 audit pass runs (default 300)
    AUDITOR_SET_WEIGHTS_ENABLED  opt-in counter-weight (default off)
    AUDITOR_WEIGHT_POLL_SECONDS  --loop poll cadence when counter-weighting (default 12 ≈ 1 block)
    AUDITOR_SET_LEAD_BLOCKS  set weights this many blocks before the tempo boundary (default 2)
    AUDITOR_WEIGHT_INTERVAL_BLOCKS  flat-cadence fallback when tempo can't be read (default 300)

Counter-weight (when enabled) is timed to the subnet TEMPO BOUNDARY, SN51-style:
the --loop polls at block cadence and sets weights ~AUDITOR_SET_LEAD_BLOCKS before
the next Yuma-consensus boundary, so the auditor-validator's weights are freshest
exactly when consensus evaluates them (best lever on vTrust). It falls back to a
flat block interval if the tempo can't be read, and a dead-man set (> 2 tempos
since the last set) keeps vTrust from decaying if a boundary is ever missed.
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
    """Configure the 'ralph-auditor' logger so it survives bittensor's logging
    hijack.

    Importing bittensor (auditor.chain) runs bt.logging, which raises the ROOT
    logger to WARNING ("Enabling default logging (Warning level)") and silences
    any INFO that propagates to root — so a plain logging.basicConfig() makes the
    auditor go SILENT after startup (validators saw exactly this). We instead give
    OUR logger its own handler + level and set propagate=False, so the auditor's
    output is independent of whatever bittensor does to the root logger.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False  # don't route through root (bittensor pins it to WARNING)


def audit_epoch(epoch_id: str, chain: ChainClient, api: ReportClient) -> int:
    """Run Gates 1-3 for a single epoch. Returns its exit code."""
    logger.info("auditing epoch %s", epoch_id)

    try:
        envelope = api.get_report(epoch_id)
    except Exception as exc:
        logger.error("epoch %s: failed to fetch report: %s", epoch_id, exc)
        return EXIT_NETWORK

    report_json = envelope.get("report_json") or {}
    # The validator's set_commitment extrinsic lands a few blocks AFTER
    # epoch_end_block (snapshot → hash → sign → commit), and commitments
    # OVERWRITE, so the historical value is only readable at the block the commit
    # actually landed (chain_commitment_block). Querying at epoch_end_block reads
    # the PREVIOUS epoch's commitment → Gate 1 always fails → the auditor never
    # advances its clean-epoch watermark. Fall back to epoch_end_block only for
    # older reports that predate chain_commitment_block.
    end_block = envelope.get("chain_commitment_block")
    if end_block is None:
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

    if not reports:
        logger.info(
            "no audit reports published yet (repo empty / not public) — "
            "nothing to verify this pass"
        )
        return EXIT_CLEAN

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
    """Set the auditor's OWN weights, timed to the tempo boundary (off unless
    enabled).

    Continuous process, NOT one-shot: each tick checks how close we are to the
    subnet's next tempo (Yuma consensus) boundary and, when within `lead` blocks
    of it (SN51-style — maximally fresh when consensus evaluates, the strongest
    lever on vTrust), re-sets weights from the latest CLEAN epoch's independently-
    replayed scores — shadowing the honest validator. Falls back to a flat block
    interval if the tempo can't be read, and to a BURN-to-uid-0 set when no clean
    epoch is available. A rate-limit floor prevents double-sets near the boundary.
    Never raises into the loop.
    """
    from auditor.weights import (
        auditor_hotkey_ss58,
        blocks_until_next_epoch,
        is_weight_set_due,
        is_weight_set_due_tempo,
        set_lead_blocks,
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

    try:
        blocks_since = chain.blocks_since_weight_set(hotkey)
        current = chain.get_current_block()
        tempo = chain.tempo()
    except Exception:
        logger.exception("counter-weight: chain query failed; skipping this tick")
        return

    if not tempo or tempo <= 0:
        # Tempo unreadable / degenerate → fall back to the flat block interval
        # (which has its own never-set / interval-based cadence).
        interval = weight_set_interval_blocks()
        if not is_weight_set_due(blocks_since, interval):
            logger.debug(
                "counter-weight: %s/%s blocks since last set (block %s) — not due (flat fallback)",
                blocks_since, interval, current,
            )
            return
    else:
        lead = set_lead_blocks()
        # Compute blocks-to-boundary from the block we already read (no second
        # RPC; keeps blocks_left consistent with `current`).
        blocks_left = blocks_until_next_epoch(current, chain.netuid, tempo)
        if not is_weight_set_due_tempo(blocks_left, blocks_since, tempo, lead):
            logger.debug(
                "counter-weight: block %s — %s blocks to tempo boundary (lead %s, %s since last set) — not due",
                current, blocks_left, lead, blocks_since,
            )
            return
        # Double-set guard. The fire window is a few blocks wide (blocks_left in
        # {lead..0}), so without a guard we'd re-submit on every tick of it. The
        # chain's blocks_since lags (we set wait_for_finalization=False), so use a
        # LOCAL last-fired marker (.audit_published). Size it to the rate-limit
        # window but cap it BELOW tempo so it can never suppress the once-per-
        # tempo boundary set; exempt the dead-man / never-set rescue path.
        deadman = blocks_since is None or blocks_since > tempo * 2
        rl = chain.weights_rate_limit() or 0
        floor = max(rl if 0 < rl < tempo else 0, lead + 1)
        last_pub = _read_int_file(PUBLISHED_FILE)
        if not deadman and last_pub is not None and (current - last_pub) < floor:
            logger.debug(
                "counter-weight: boundary (blocks_left %s) but last set at block %s "
                "(< %s blocks ago) — double-set guard, skipping", blocks_left, last_pub, floor,
            )
            return
        logger.info(
            "counter-weight: block %s — %s blocks to tempo boundary (≤ lead %s) → setting weights",
            current, blocks_left, lead,
        )

    epoch_id = _read_str_file(LAST_CLEAN_EPOCH_FILE)
    if not epoch_id:
        # No clean epoch to replay (e.g. the audit-reports repo is empty / 404).
        # BURN FALLBACK: still set weights to uid 0 so the auditor-validator
        # keeps its vTrust alive + burns to the owner, instead of skipping.
        from auditor.weights import submit_burn_weights

        logger.info(
            "counter-weight: due but no clean epoch (empty/404 audit repo) — "
            "BURN fallback to uid 0"
        )
        if submit_burn_weights(subtensor_url=chain.subtensor_url, netuid=chain.netuid):
            _write_int_file(PUBLISHED_FILE, current)
        return

    logger.info(
        "counter-weight: due (block %s, %s blocks since last set) — setting from %s",
        current, blocks_since, epoch_id,
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
        from auditor.weights import is_enabled as _cw_enabled
        # Counter-weight timing (SN51-style) must catch the ~2-block window just
        # before the tempo boundary, so poll at block cadence when weight-setting
        # is enabled; the heavier Gates-1-3 audit pass still runs only every
        # AUDIT_INTERVAL_SECONDS. With weights off, fall back to the audit cadence.
        poll = int(os.environ.get("AUDITOR_WEIGHT_POLL_SECONDS", "12")) if _cw_enabled() else interval
        logger.info(
            "auditor loop started — repo=%s netuid=%s audit-interval=%ss poll=%ss (subtensor=%s)",
            repo, netuid, interval, poll, subtensor_url,
        )
        last_audit = 0.0
        while True:
            try:
                now = time.time()
                if now - last_audit >= interval:
                    code = audit_new_epochs(chain, api)
                    last_audit = now
                    logger.info("audit pass complete (exit=%s)", code)
                maybe_counter_weight(chain, api)  # cheap per-tick tempo-boundary check
            except Exception:
                logger.exception("audit loop iteration failed")
            time.sleep(poll)

    # default (and --once) -> single pass.
    code = audit_new_epochs(chain, api)
    maybe_counter_weight(chain, api)
    sys.exit(code)


if __name__ == "__main__":
    main()
