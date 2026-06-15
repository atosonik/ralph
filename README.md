<p align="center">
  <img src="docs/assets/ralph-banner.png" alt="Ralph — decentralized · autonomous · AI research" width="100%">
</p>

<p align="center">
  A Bittensor subnet for <b>decentralized, autonomous AI research</b> — an open,
  continuously improving training recipe and the public knowledge corpus behind it,
  produced by an autonomous research network.
</p>

<p align="center">
  🟢 <b><a href="https://taostats.io/subnets/40">Live on mainnet — netuid 40</a></b> ·
  🌐 <a href="https://ralphlabs.ai">ralphlabs.ai</a> ·
  📄 <a href="docs/Ralph-Whitepaper-v1.2.pdf">Whitepaper v1.2</a> ·
  🏷️ <a href="https://github.com/RalphLabsAI/ralph/releases">Releases</a> ·
  📊 <a href="https://wandb.ai/ralphlabs-hub/ralph">Wandb</a> ·
  💬 <a href="https://github.com/orgs/RalphLabsAI/discussions">Discussions</a>
</p>

---

## What Ralph produces

1. **A canonical training recipe** — a Git repo holding the best-known open recipe for each track (model class × objective). Clone it and train a model with state-of-the-art settings.
2. **A public knowledge corpus** (`ralph-diffs`) — every change the network has ever evaluated, with its measured effect, including verified negative results. Searchable, citable, openly licensed.
3. **A demonstration model lineage** — Ralph-1, -2, … — open-weights reference models proving the recipe works and that the improvement compounds.

The subnet funds the production of these artifacts. They are the deliverable.

## How it works

A miner improves the canonical recipe by proposing a **patch** — a new LR schedule, an init, a data-mix change. The network trains the patched recipe under fixed conditions, scores it on a public ladder, and crowns the best recipe as the new **king**. Every accepted change is a commit in a public lineage, so the recipe only moves forward and you can read the diff — and the measurement — behind each step.

```
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Miner's private search                   │
│  Any agent, any LLM, any GPU, any training code.    │
│  The protocol doesn't see this.                     │
└──────────────────────┬──────────────────────────────┘
                       │ candidate patch
┌──────────────────────▼──────────────────────────────┐
│  Layer 2 — Canonical proof test                     │
│  Official Ralph container on the miner's GPU.       │
│  Applies the patch to the canonical recipe, trains  │
│  under fixed (seed, data, config), and emits a      │
│  checkpoint + training log + calibration +          │
│  hardware attestation.                              │
└──────────────────────┬──────────────────────────────┘
                       │ proof bundle
┌──────────────────────▼──────────────────────────────┐
│  Layer 3 — Submission + judgment                    │
│  PR to the recipe repo + proof bundle on HF.        │
│  Validator: diff scan → attestation verify →        │
│  log plausibility → hidden eval → score.            │
│  If it decisively beats the king → merge.           │
└─────────────────────────────────────────────────────┘
```

## Evaluation — the ladder

Scoring isn't a single run. Each candidate is evaluated across a **multi-scale ladder** — three model sizes (S1 → S2 → S3, up to ~124M params at NanoGPT-Speedrun scale) — plus a **held-out private-hard slice** alongside the public CORE-22 downstream suite. A patch has to generalize across scale and survive tasks it never saw, not just fit the public set.

The open question this design hinges on: **does a cheap small-scale gate actually predict which recipes are better at larger scale?** We're pre-registering a transfer-credibility test — frozen analysis, pinned reference models, results published either way — to answer it in public rather than assume it.

## Credibility — attested execution

A score is only worth the execution behind it. Ralph v1.2 (§5.4) replaces the earlier two-tier (verified / unverified) split with a **single attested-execution tier**: every scored run is produced by the official proof-test container under hardware attestation (NVIDIA Confidential Computing — TDX + nvtrust), so a reported number always corresponds to a run that provably happened as described, not a self-report. Validators stay cheap — they supervise and select; miners pay the GPU cost.

## Current status

🟢 **Live on Bittensor mainnet, netuid 40.**

| Phase | Status | Key results |
|---|---|---|
| **0 — MVP** | ✅ | End-to-end protocol on CPU: model, training, eval, proof-test, validator, scoring, king-change cycle |
| **0.5 — H100** | ✅ ([`v0.5.0`](https://github.com/RalphLabsAI/ralph/releases/tag/v0.5.0) · [results](https://github.com/orgs/RalphLabsAI/discussions/4)) | Real data (1B tokens FineWeb-Edu), noise floor measured (2σ = 0.013 val_bpb), Ralph-1 trained (254M params, loss 3.82) |
| **0.5b — Optimization** | ✅ ([`v0.5.1`](https://github.com/RalphLabsAI/ralph/releases/tag/v0.5.1)) | bf16: 3.8× throughput (63K tok/s), same loss; live wandb monitoring; Streamlit dashboard |
| **0.5c — Attestation** | ✅ code-complete | TDX + nvtrust module: auto-detects CC hardware, falls back to mock; untested on real CC (needs Azure NCC / GCP A3-Confidential) |
| **0.5d — Testnet** | ✅ ([`v0.6.0`](https://github.com/RalphLabsAI/ralph/releases/tag/v0.6.0)) | Testnet (netuid 16): two miners competed, validator set weights on-chain, king changed |
| **1.0 — Mainnet** | 🟢 live | Registered on **netuid 40**; multi-scale downstream ladder + private-hard eval; transfer-credibility test pre-registered |

## Repo layout

Ralph lives across **two repos**:

| Repo | What | Patchable by miners? |
|---|---|---|
| **[RalphLabsAI/recipe](https://github.com/RalphLabsAI/recipe)** | `model/`, `recipe/`, `configs/`, `data/` — the canonical training recipe miners patch, and the merged history of accepted improvements | **Yes** |
| **RalphLabsAI/ralph** (this repo) | Protocol: validator, proof-test runner, attestation, scoring, submission tooling | **No** (restricted) |

### This repo (protocol)

| Path | What |
|---|---|
| `eval/` | Hidden-eval harness, val_bpb, downstream ladder (CORE-22 + private-hard) |
| `validator/` | Submission ops, ladder eval, scoring, audit |
| `proof/` | Proof-test runner + attestation |
| `calibration/` | Deterministic compute benchmark (matmul + attention + collective) |
| `miner/` | Submission bundle assembly, HuggingFace upload, hotkey signing |
| `chain_layer/` | Bittensor + local-JSON chain abstractions |
| `dashboard/` | Ralph Live — Streamlit monitoring dashboard |
| `scripts/` | `miner_run.py`, `run_h100.sh`, `noise_floor.py`, `b6_run.py`, `gpu.py` |
| `ralph_bootstrap.py` | Adds the sibling recipe repo to `sys.path` for protocol code |

The protocol locates the recipe via `$RALPH_RECIPE_DIR` (defaults to `../recipe`). Clone both repos side-by-side and everything just works.

## Quick start

### CPU smoke test (no GPU needed)

```bash
# Clone both repos side-by-side
git clone https://github.com/RalphLabsAI/ralph.git
git clone https://github.com/RalphLabsAI/recipe.git
cd ralph
python3 -m venv .venv && source .venv/bin/activate
pip install torch numpy tiktoken cryptography

# Generate synthetic data into the recipe repo
(cd ../recipe && python -m data.prepare --source synthetic --out data/shards \
    --shard-tokens 50000 --total-tokens 200000 --eval-tokens 10000)

# Run end-to-end: two miners submit, validator scores, king changes
python scripts/smoke_test.py
```

### H100 full run (real data)

```bash
git clone https://github.com/RalphLabsAI/ralph.git
git clone https://github.com/RalphLabsAI/recipe.git
cd ralph
bash scripts/run_h100.sh
```

Bootstraps a fresh H100: FineWeb-Edu data prep, calibration, noise floor, and Ralph-1 training.

### Live monitoring

```bash
# wandb (real-time loss curves)
python -m recipe.train --config configs/h100_default.json --out-dir runs/my_run --wandb

# Streamlit dashboard (network status, king history, submissions)
pip install 'ralph-subnet[dashboard]'
streamlit run dashboard/app.py
```

## Measured results (Phase 0.5)

| Metric | Value |
|---|---|
| H100 calibration (matmul) | 0.512 ms |
| Noise floor (10 seeds, 125M model) | σ = 0.006 val_bpb, margin (2σ) = 0.013 |
| Ralph-1 fp32 (254M params, 262M tokens) | Final loss = 3.82, 16.9K tok/s, 259 min |
| Ralph-1 bf16 (same model, same data) | Final loss = 3.82, **63.4K tok/s, 69 min (3.8× faster)** |

Full results: [Phase 0.5 Discussion](https://github.com/orgs/RalphLabsAI/discussions/4) ·
Release: [`v0.5.0`](https://github.com/RalphLabsAI/ralph/releases/tag/v0.5.0)

## License

Apache-2.0
