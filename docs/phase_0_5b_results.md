# Phase 0.5b — bf16 Mixed Precision: 3.8× Speedup, Same Quality

**Date:** May 27, 2026
**Hardware:** NVIDIA H100 PCIe 80GB (Shadeform / ShadeCloud)
**Code:** [`RalphLabsAI/ralph`](https://github.com/RalphLabsAI/ralph/tree/v0.5.1) @ [`v0.5.1`](https://github.com/RalphLabsAI/ralph/releases/tag/v0.5.1)

---

## What changed

One code change: `use_bf16=True` in `recipe/train.py`. PyTorch `torch.amp.autocast` with bfloat16 on the H100's tensor cores. Everything else identical — same model, same data, same seed, same config.

## fp32 vs bf16 — head-to-head

| Metric | fp32 (Phase 0.5) | bf16 (Phase 0.5b) | Change |
|---|---|---|---|
| **Final loss** | 3.8173 | 3.8163 | -0.001 (identical) |
| **Throughput** | 16,882 tok/s | **63,361 tok/s** | **3.8×** |
| **Wall-clock** | 258.8 min | **69.0 min** | **3.8× faster** |
| Tokens trained | 262,144,000 | 262,144,000 | same |
| Parameters | 253,872,128 | 253,872,128 | same |
| Seed | 1337 | 1337 | same |
| Data | FineWeb-Edu 1B tokens | FineWeb-Edu 1B tokens | same |
| Config | `h100_default.json` | `h100_default.json` | same |

**Key takeaway:** bf16 is free performance. No quality degradation at all — loss within 0.001 of fp32. The H100's tensor cores are designed for bf16; running fp32 was leaving 73% of the hardware on the table.

## What this means for miners

A proof-test that took 4.3 hours in fp32 now takes 69 minutes. At ~$3/hr for an H100, that's:
- fp32: ~$13 per Ralph-1 proof test
- bf16: ~$3.50 per Ralph-1 proof test

Cheaper proof tests → lower barrier to submit → more patches competing → faster recipe improvement.

## New infrastructure in this release

| Feature | What it does |
|---|---|
| **wandb live monitoring** | `--wandb` flag streams loss/lr/grad_norm/throughput to [wandb.ai/ralphlabs-hub/ralph](https://wandb.ai/ralphlabs-hub/ralph). Public — anyone can view. |
| **wandb metrics in proof bundles** | Training traces auto-exported as `wandb_metrics.json` alongside checkpoints. Miners submit their complete training story, not just the final number. |
| **Ralph Live dashboard** | Streamlit dashboard with auto-refreshing loss curves, king status, noise floor stats. `streamlit run dashboard/app.py` |
| **HuggingFace Hub integration** | `miner/hub.py` — upload/download proof bundles to HuggingFace. Checkpoints + logs + wandb traces + attestation, all content-addressed. |

## Live training run

Watch the bf16 Ralph-1 training on wandb (public):

🔗 **wandb:** https://wandb.ai/ralphlabs-hub/ralph

## How to reproduce

```bash
git clone https://github.com/RalphLabsAI/ralph.git
cd ralph && git checkout v0.5.1
bash scripts/run_h100.sh  # bf16 is now the default
```

Or run just the training step with wandb:

```bash
python -m recipe.train \
    --config configs/h100_default.json \
    --out-dir runs/ralph1_bf16 \
    --seed 1337 \
    --wandb
```

## What's next

- **Phase 0.5c** ✅ (just shipped): Real TDX + nvtrust attestation module — code-complete, auto-detects CC hardware, falls back to mock. Ready for when CC instances are available.
- **Phase 0.5d**: Bittensor testnet integration — replace JSON-file chain with real on-chain commitments.
- **Phase 1.0**: Subnet launch.

---

🔗 **Repo:** [github.com/RalphLabsAI/ralph](https://github.com/RalphLabsAI/ralph)
🏷️ **This milestone:** [`v0.5.1`](https://github.com/RalphLabsAI/ralph/releases/tag/v0.5.1)
💬 **Discussions:** [github.com/orgs/RalphLabsAI/discussions](https://github.com/orgs/RalphLabsAI/discussions)
📊 **wandb:** [wandb.ai/ralphlabs-hub/ralph](https://wandb.ai/ralphlabs-hub/ralph)
