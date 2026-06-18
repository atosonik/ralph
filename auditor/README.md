# Ralph auditor

Independent, **CPU-only** verifier for Ralph (Bittensor subnet 40). Anyone can
run it to prove — on-chain — whether a validator scored an epoch
honestly, **without re-doing any GPU work**.

Each epoch it runs three gates against the validator's published audit report:

- **Gate 1 — hash + signature.** Recompute `sha256(canonical_json(report_json))`
  and assert it equals the report's hash, the hash committed on-chain at
  `epoch_end_block`, and that the ed25519 signature is valid. It **imports** the
  validator's own `canonical_json`, so the bytes are identical by construction.
- **Gate 2 — replay.** Recompute the epoch's weight vector from the published
  raw data, **importing** the validator's weight/floor constants (so a
  unilateral validator change makes the auditor diverge — an intended alarm).
- **Gate 3 — diff.** Diff replayed vs claimed weights (tolerance `1e-4`).

Exit codes: `0` clean · `1` hash-or-signature fail · `2` math diverge · `3` network.

A re-run of the GPU eval itself (Gate 4) is **not** part of this CPU tool — it's
Phase 3 and needs a GPU. The diff-non-trivial and rationale-coherent
classification bars need the raw bundle and are likewise Gate-4 territory; the
replay trusts the published classification for those and notes it in
`replay.py`.

## Install (CPU-only, ~50 MB RAM)

```bash
pip install httpx click bittensor       # bittensor: archive chain reads + Keypair
```

`bittensor` is already a Ralph dependency; no GPU/torch eval stack is exercised
by Gates 1-3 (torch is only imported by the optional counter-weight path).

## Run

```bash
python -m auditor --once                 # one pass over new epochs
python -m auditor --loop                  # continuous (AUDIT_INTERVAL_SECONDS)
python -m auditor --epoch 40-1234567      # one specific epoch
python -m auditor --help
```

Key env vars: `AUDIT_REPO` (default `RalphLabsAI/audit-reports`), `NETUID`
(default `40`), `SUBTENSOR_URL`, `VALIDATOR_HOTKEY` (signer to read the
commitment for; defaults to the report's own `signer_hotkey`), `HF_TOKEN` (only
for a private report repo).

## Archive endpoint

Each epoch's on-chain commitment **overwrites** the previous one, so historical
verification needs an **archive** subtensor — the default
`wss://archive.chain.opentensor.ai:443/` (free) or your own. A pruned/lite node
will not have the state at `epoch_end_block`.

## Optional counter-weight (off by default)

Set `AUDITOR_SET_WEIGHTS_ENABLED=1` and provide your OWN wallet by name
(`AUDITOR_WALLET_NAME`, `AUDITOR_WALLET_HOTKEY`) to have a registered
auditor-validator set its own weights from the replayed scores, shadowing a
dishonest validator. Names-in-env only — never the scored validator's wallet, never the secret
material.
