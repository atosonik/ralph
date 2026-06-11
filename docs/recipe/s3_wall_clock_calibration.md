# S₃ rung wall-clock calibration — measurement record

**Pre-work #4 of the Cross-Scale Downstream Pareto build.** Locks the empirical wall-clock for an 800-step S₃ baseline run on a single H100, so B5's calibration cost and B6's per-submission cost estimates use real numbers instead of the master-plan placeholder.

**Status:** measured 2026-06-10. The master-plan ~70-min target was ~30× off. The real number is 2.3 min.

---

## Setup

| Field | Value |
|---|---|
| Hardware | NVIDIA H100 PCIe, 80 GB VRAM, driver 570.195.03 |
| Provider | Hyperstack via Shadeform |
| Container | shade_os ubuntu22.04 cuda12.8 (default Shadeform image) |
| Python | 3.10.x |
| PyTorch | 2.6.0 + cu124 (default cu128 wheel was driver-incompatible; downgraded) |
| Precision | bf16 |
| Determinism | `torch.use_deterministic_algorithms(True)` (with warning that CuBLAS + Flash-Attn still drop nondeterminism) |
| Config | [`recipe/configs/h100_s3_d768_l12.json`](../../../recipe/configs/h100_s3_d768_l12.json) |
| Data | Existing repo shards (5 × 50k tokens GPT-2 BPE; cycled mod-N during the 100-step run) |

### Recipe config (the S₃ spec being calibrated)

```json
{
  "vocab_size": 50257,
  "dim": 768, "n_layers": 12, "n_heads": 12, "head_dim": 64,
  "ffn_mult": 2.6875,
  "max_seq_len": 1024, "seq_len": 1024,
  "batch_size": 16, "micro_batch_size": 16,
  "total_steps": 800, "warmup_steps": 80,
  "max_lr": 3e-4, "min_lr": 3e-5,
  "weight_decay": 0.1, "grad_clip": 1.0
}
```

Resulting model: **125,320,704 total params · 86,723,328 non-embedding params** (sits in the modded-nanogpt 124M coordination band).

---

## Measurement

Ran 100 steps (12.5% of the 800-step target) on the H100. Captured `tokens_per_sec` and `elapsed_s` from each row of `training_log.jsonl`.

### Steady-state window (steps 20-99, excluding warmup)

| Statistic | Value |
|---|---:|
| **Median tokens/sec** | **94,461** |
| Mean tokens/sec | 93,819 |
| StDev tokens/sec | 1,874 (1.99% coefficient of variation) |
| Per-step time | **0.173 sec** |
| Final 100-step wall-clock | 17.7 sec |

### Extrapolation to the full 800-step S₃ baseline

| Field | Value |
|---|---:|
| Tokens per step (batch × seq) | 16,384 |
| Total tokens at 800 steps | 13.1 M |
| **Extrapolated wall-clock at 800 steps** | **138.5 sec ≈ 2.3 min** |
| vs master-plan target | −67.7 min (the target was ~30× too long) |

### Sanity check

H100 PCIe theoretical bf16 peak: ~756 TFLOPS. Effective MFU at the measured throughput:

* FLOPs/token (6·N_nonembed): 6 × 86.7e6 = 5.2e8
* Sustained FLOPs/sec: 94,461 × 5.2e8 = 4.9e13 = **49 TFLOPS effective**
* **Realised MFU: ~6.5%**

That's low for an H100, but believable for batch=16 (small batch dramatically reduces MFU on H100). modded-nanogpt's records use batch=32–64 + 8×H100; at batch=64 a single H100 would hit ~25-30% MFU.

---

## Implications

### 1. The master plan's wall-clock estimate was wrong

The master plan ([docs/build_scope/00_MASTER_PLAN.md](../build_scope/00_MASTER_PLAN.md) §3 S₃ row) stated "**~70 min on a single H100 in bf16**" for the 800-step S₃ rung. Empirical measurement says **~2.3 min**. The 30× gap is large enough that downstream cost estimates need to be re-anchored.

The intended training amount (13.1 M tokens at 800 steps) IS consistent with the recommendation's pinned S₃ tokens-seen target ([docs/king_criterion_review/00_RECOMMENDATION.md](../king_criterion_review/00_RECOMMENDATION.md) §4.2 ladder table). So the spec stands; only the time estimate was off.

### 2. B5 calibration is much cheaper than estimated

The master plan ([docs/build_scope/00_MASTER_PLAN.md](../build_scope/00_MASTER_PLAN.md) §4 budget table) estimated B5 calibration at **~430 H100-hours ≈ $350-600**. With the measured 2.3-min S₃ wall-clock:

* B5b 10-stream calibration: 900 runs × (S₁ ~30s + S₂ ~5min + S₃ ~2.3min) ≈ **~115 H100-hours**
* At Shadeform spot $0.80/hr: **~$92** (was $350-600)
* At Hyperstack reserved $1.90/hr: **~$220**

**v0.10/v0.11 budget headroom**: an order of magnitude more than the master plan assumed.

### 3. Per-submission cost for miners drops sharply

Master plan §3 stated **~$2-5 per submission** (assuming the ladder dominated by S₃ at ~70 min). At the measured S₃ wall-clock:

* S₁ ~30s + S₂ ~5 min + S₃ ~2.3 min = **~8 min per submission**
* At $0.80/hr Shadeform spot: **~$0.11/submission**
* At $1.90/hr Hyperstack: **~$0.25/submission**

This is a meaningful improvement for miner economics — closer to the **$0.05/submission** the prior small-scale Pareto rule promised, while still hitting the 124M coordination point. The Cross-Scale Downstream Pareto pivot's barrier to entry just dropped by ~10× from what we publicly committed to.

### 4. Public-post correction needed

The public Twitter/GH-Discussion posts ([twitter_posts/karpaai_account/11_cross_scale_downstream_pareto_pivot.md](../../twitter_posts/karpaai_account/11_cross_scale_downstream_pareto_pivot.md), [github_discussions/karpaai_repo/02_cross_scale_downstream_pareto_pivot.md](../../github_discussions/karpaai_repo/02_cross_scale_downstream_pareto_pivot.md)) reference "~70-minute run at d=768 / 124M nonembed params on FineWeb-Edu" as the S₃ rung. **That number is now empirically false** at the current spec. Two ways to handle:

1. **Adopt the measured number** and post a brief correction via the `/karpa-announce` skill (event-type=followup). Cleanest.
2. **Raise total_steps to hit the original 70-min target.** Recommended steps for 70 min: **24,214** (vs 800). At 24k steps the model would see 397M tokens — far closer to Chinchilla-optimal for 124M (~2.5B tokens, 16% of optimal). Materially different recipe in research substance.

Recommendation: **(1)**. The 800-step spec was pinned by the recommendation document with explicit token-budget reasoning; we should not move it just to recover an inaccurate time estimate. Issue the followup correction.

### 5. MFU is the obvious optimisation lever

At 6.5% MFU we're leaving 4-5× throughput on the table. Future S₃ tuning could:
* Raise batch_size from 16 → 64 (single-H100 VRAM permitting, which it does at this model size) → expect ~25-30% MFU
* Add gradient accumulation if VRAM is tight
* This would either halve per-submission cost OR allow ~4× more steps in the same wall-clock budget

This is a B5 follow-up, not a Pre-work #4 deliverable.

---

## Sources

* Run log: `/tmp/karpa_s3_measurement/run3.log` (local-only; not committed)
* Recipe config: [`recipe/configs/h100_s3_d768_l12.json`](../../../recipe/configs/h100_s3_d768_l12.json)
* Master plan: [`docs/build_scope/00_MASTER_PLAN.md`](../build_scope/00_MASTER_PLAN.md)
* Recommendation (S₃ spec source): [`docs/king_criterion_review/00_RECOMMENDATION.md`](../king_criterion_review/00_RECOMMENDATION.md)
* OLMo-2-1B reference recipe (for context on the S₃ → 1B transfer test): [`experiments/2026-06-transfer-credibility/notes/olmo_2_1b_anchor.md`](../../experiments/2026-06-transfer-credibility/notes/olmo_2_1b_anchor.md)

## Real-money spend

| Item | Value |
|---|---:|
| Instance | NVIDIA H100 PCIe via Hyperstack ($1.90/hr) |
| Wall-clock alive | ~40 minutes (rent + bootstrap + 2 train runs + delete) |
| **Spend** | **~$1.30** (well under the master plan's $5 budget) |
| Cumulative session H100 spend | this is the first H100 spend of this session-arc |
