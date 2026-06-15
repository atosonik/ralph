# Hardness-subset license decision — record

**Decision:** Pre-swap OpenBookQA + SciQ → **ARC-Challenge bottom-quintile + winogrande + tinyARC + tinyMMLU**.
**Reason:** OpenBookQA and SciQ are CC-BY-NC; Ralph's gate output produces TAO emissions
(commercial activity), so the NC clause is incompatible with the protocol's use. The four
chosen alternatives are all commercial-permitting.
**Status:** Decided 2026-06-10 ahead of any B1 engineering work.
**Pre-registration discipline:** This decision is recorded BEFORE the build phase that
depends on it (B1: CORE-22 + private hardness-graded subset eval harness — see
[docs/build_scope/02_scope_B1.md](../build_scope/02_scope_B1.md)). The fact that we are
recording it pre-build is the audit trail.

## The question

The Cross-Scale Downstream Pareto rule (publicly committed in [twitter_posts/RalphLabsAI_account/11_cross_scale_downstream_pareto_pivot.md](../../twitter_posts/RalphLabsAI_account/11_cross_scale_downstream_pareto_pivot.md) and [github_discussions/RalphLabsAI_repo/02_cross_scale_downstream_pareto_pivot.md](../../github_discussions/RalphLabsAI_repo/02_cross_scale_downstream_pareto_pivot.md)) scores each accepted submission at the S₃ rung (~124M params) on:

1. **CORE-22** — the public 22-dataset eval bundle Karpathy uses in [nanochat #420](https://github.com/karpathy/nanochat/discussions/420).
2. **A private hardness-graded subset** — kept off the public eval surface to defeat overfitting against a fully-public eval set.

The hardness-graded subset's original candidate list was: HellaSwag-hard / ARC-easy / **OpenBookQA** / **TinyMMLU**, plus **SciQ** as a backup. This document resolves a license risk on the two bolded candidates.

## Facts (verified at decision time)

| Dataset | License (as of 2026-06-10) | Source | Commercial use under license? |
|---|---|---|---|
| **OpenBookQA** | CC-BY-NC-SA-4.0 | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) | **NO** — Non-Commercial clause |
| **SciQ** | CC-BY-NC-3.0 | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | **NO** — Non-Commercial clause |
| ARC-Challenge | CC-BY-SA-4.0 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | YES (share-alike) |
| ARC-Easy | CC-BY-SA-4.0 | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | YES (share-alike) |
| winogrande | CC-BY-4.0 | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | YES |
| tinyMMLU | MIT | [tinyBenchmarks/tinyMMLU](https://huggingface.co/datasets/tinyBenchmarks/tinyMMLU) | YES |
| HellaSwag | MIT | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | YES |

(License strings sampled from each dataset's HF Hub `dataset_info.license` field on 2026-06-10. A re-verification at B1 start is required before code lands.)

## The argument for pre-swapping

Ralph's gate emits TAO weights, which are economic outputs of a commercial network. Embedding a CC-BY-NC dataset's bytes inside that gate is a use that the NC clause does not permit. The risk is small in absolute terms (low likelihood of enforcement), but the cost of being wrong later — having to redesign the hardness subset mid-build, recalibrate the per-cell noise floors in B5, and explain the change publicly — is high.

The four alternative datasets satisfy three constraints simultaneously:

1. **Commercial-permitting licenses** (CC-BY, CC-BY-SA, or MIT).
2. **Hardness-graded selection feasible** — ARC-Challenge bottom-quintile is straightforward to extract; tinyMMLU is already pre-curated; winogrande and tinyARC have well-known difficulty proxies (length, distractor similarity).
3. **Public-rank-correlated outcomes** at the ~124M scale — all four have published 100M-1B scale rank evidence in the OLMo / DCLM / lm-evaluation-harness literature, so we are not flying blind on the metric.

## What this decides

For the **private hardness-graded subset** at the S₃ rung, the canonical pool becomes:

1. **ARC-Challenge bottom-quintile** — sourced from `allenai/ai2_arc`, ARC-Challenge split, ordered by composite-difficulty score (TBD: define in B1; candidate proxy is multi-distractor entropy from a 100M-class probe model), retain the bottom 20% by accuracy.
2. **winogrande** — sourced from `allenai/winogrande`, the `winogrande_xl` configuration, full set.
3. **tinyARC** — sourced from `tinyBenchmarks/tinyARC` (or constructed from `ai2_arc` if `tinyBenchmarks` doesn't ship it; verify in B1).
4. **tinyMMLU** — sourced from `tinyBenchmarks/tinyMMLU`, all 14 task subsets.

## What this DOES NOT decide

This document only covers the private hardness-graded subset. CORE-22 (the public side of the metric) has its own license profile across 22 different sources; B1 owns enumerating and clearing each of those individually. If any of the 22 turns out NC-encumbered, B1 will need a parallel decision doc for that subset.

## Re-verification trigger

If at the start of B1 implementation, any of the four alternative datasets above has changed its HF Hub license, this decision document is invalidated and B1 must produce a new one. Verification command:

```bash
huggingface-cli scan-licenses allenai/ai2_arc allenai/winogrande tinyBenchmarks/tinyARC tinyBenchmarks/tinyMMLU Rowan/hellaswag
```

(Or the equivalent `datasets.load_dataset(..., trust_remote_code=False).info.license` check in code.)
