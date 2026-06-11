# B1 deferred decisions

15 fix-before-execution items from the B1 critique that must be resolved
BEFORE the remaining B1 modules (scorer / core22 / private_hard / grader /
calibration / runner) land. Recorded here so future sessions don't re-discover
them.

Each item has an owning phase, a one-line decision recommendation, and a
status field. Update status as you close items.

## Decisions blocking scorer.py / core22.py

### B1-D1 — TinyMMLU IRT++ provenance
**Owner:** B1 (private_hard.py author)
**Decision:** Confirm `tinyBenchmarks` Python package license. If it ships
under a non-commercial / unknown license, swap TinyMMLU for a 4th
commercial-clean task (ARC-Challenge mid-quintile is the natural fallback).
**Recommendation:** Verify license before any `pip install tinyBenchmarks`.
**Status:** OPEN

### B1-D2 — DCLM bundle SHA pin
**Owner:** B1 (core22.py author)
**Decision:** Download the current DCLM CORE eval bundle, compute its SHA256,
and pin it in `eval/downstream/core22.py` as a module-level constant.
**Recommendation:** Pin first; revisit only if the upstream bundle gets
materially better (the rotation cost is high, do it once).
**Status:** OPEN

### B1-D3 — 22-vs-23 task answer for `bigbench_language_identification`
**Owner:** B1 (core22.py author)
**Decision:** DCLM's CORE-22 vs CORE-23: include or exclude
`bigbench_language_identification`. Karpathy's nanochat uses the 22-task
variant; some other downstream evals use the 23-task variant.
**Recommendation:** Match Karpathy's 22-task selection. Document the choice
in core22.py docstring + the public Cross-Scale Downstream Pareto post.
**Status:** OPEN

### B1-D4 — DCLM CC-BY-SA share-alike review
**Owner:** Legal-not-engineering
**Decision:** Does shipping CC-BY-SA-4.0 sub-datasets (ARC, BoolQ, COPA)
inside the attested validator container constitute redistribution that
triggers share-alike obligations?
**Recommendation:** Get a one-line written posture before B5 calibration
runs commit the bundle to validator nodes.
**Status:** OPEN

## Decisions blocking runner.py / scorer.py

### B1-D5 — Subprocess isolation reality check
**Owner:** B1 (runner.py author)
**Decision:** `weights_only=True` blocks pickle RCE but does NOT block code
execution from inside `model.forward()`. Either (a) accept this and document
that miners' patched model code runs with subprocess-level isolation only
(seccomp/landlock is a separate phase), OR (b) pull seccomp/landlock into B1's
scope.
**Recommendation:** Accept (a) for B1. Subprocess isolation is necessary but
not sufficient against arbitrary `forward()` code. Land seccomp/landlock as a
named follow-up phase before mainnet activation.
**Status:** OPEN

### B1-D6 — Tokenizer equivalence enforcement
**Owner:** B1 (runner.py author)
**Decision:** Reject submissions whose checkpoint config has
`vocab_size != 50257` (GPT-2 BPE) at the runner level with a clean error.
**Recommendation:** Land. Document the vocab-divergence rejection in the
op4_ladder_eval failure-mode table.
**Status:** OPEN

### B1-D7 — Determinism specification
**Owner:** B1 (runner.py author)
**Decision:** Pin determinism test to specific dtype + device +
`torch.use_deterministic_algorithms` mode. Bit-identical assertions otherwise
flake on driver / kernel variants.
**Recommendation:** Use `torch.use_deterministic_algorithms(True)` +
`CUBLAS_WORKSPACE_CONFIG=:4096:8` in the runner subprocess + accept that
some CUDA kernels will be slower. Acceptable cost for reproducibility.
**Status:** OPEN

## Decisions blocking calibration.py / aggregate.py / chain wiring

### B1-D8 — Calibration data sourcing
**Owner:** B1 (calibration.py author)
**Decision:** Where do N=10 baseline checkpoints come from?
Options: (a) B1 trains them (≈ 10 × 70min H100 = ~$15), (b) reuse Track A2
baselines (adds hard dependency on A2), (c) ship `noise_floors_v0.json` as
null-filled placeholders for B1 ship and complete in B5.
**Recommendation:** (a). The cost is trivial and option (c) leaves the
aggregator without floors which is brittle.
**Status:** OPEN

### B1-D9 — `validator/scoring.py::score_ladder` reference removed from aggregate.py
**Owner:** B1 (aggregate.py author) — **CLOSED in this commit**
**Decision:** The original scope referenced a `score_ladder` symbol in
`validator/scoring.py` that doesn't exist. aggregate.py is the predicate-half
on its own; B2 will wire it into validator/scoring.py separately.
**Status:** **CLOSED 2026-06-10** — aggregate.py ships as a standalone kernel
in eval/downstream/. No fictional symbol references.

### B1-D10 — Restricted-files scanner path correction
**Owner:** B1 (op1 author)
**Decision:** Add `eval/downstream/**` + `eval/private/downstream_pool/**` +
`eval/private/hardness/**` + `eval/private/calibration/**` to
`karpa/restricted_files.yaml` (NOT the imaginary
`karpa/validator/restricted_files.yaml`).
**Recommendation:** Land in the same PR as core22.py.
**Status:** OPEN

### B1-D11 — Container measurement bump
**Owner:** B1 (proof/sources.py author)
**Decision:** Adding eval/downstream/ to the attested container changes the
measurement. Either (a) include in B1 (lands the new measurement on chain),
or (b) ship B1 in non-attested mode for testnet and bump the measurement at
mainnet activation.
**Recommendation:** (b). Faster B1 ship; container bump is a B7 deliverable.
**Status:** OPEN

### B1-D12 — `HiddenEvalResult` schema-versioning test
**Owner:** B1 (eval/hidden_eval.py author)
**Decision:** Add a test that confirms an old serialized `HiddenEvalResult`
(without `downstream`) deserializes cleanly post-change AND that a new one
with `downstream=None` is byte-equivalent to a legacy serialization.
**Recommendation:** Land. Prevents source-compat surprises during the
v0.10 → v0.11 transition.
**Status:** OPEN

## Decisions blocking the runner CLI contract

### B1-D13 — runner CLI structural-patch handling
**Owner:** B1 (runner.py author)
**Decision:** Does the structural-patch fallback path need `--patch` and
`--karpa-root` CLI args? If yes, add them and budget for `apply_patch`
integration (~150 LOC + 2 days). If no, document that structural patches
WILL fail the new downstream runner and the legacy path handles them.
**Recommendation:** Add the args. The whole point of B1 → B7 is that
structural patches work cleanly under the new rule.
**Status:** OPEN

### B1-D14 — LOC re-budget
**Owner:** B1 (whole-phase author)
**Decision:** Critic was right that mid-LOC estimates of 22-32 days are low.
Realistic budget: 35-50 days for code + 2200-3000 LOC of tests.
**Recommendation:** Update the master plan effort estimate.
**Status:** **CLOSED 2026-06-10** — master plan already shows 35-50 day
range; this aligns with the critic's number.

### B1-D15 — GPT-2-small ±1% baseline source
**Owner:** B1 (calibration.py author)
**Decision:** Cite a specific public number for the GPT-2-small CORE-22
acceptance criterion or replace with a different reference model whose
CORE-22 score is published.
**Recommendation:** Cite the OLMo-2 paper's CORE-22 baselines as the
reference (they publish per-task numbers at multiple scales including
~125M).
**Status:** OPEN

## What B1 ships this commit (closed)

- `eval/downstream/__init__.py` — re-exports + RESTRICTED marker
- `eval/downstream/types.py` — DownstreamReport / CellResult /
  NoiseFloorTable / ParetoVerdict / TaskSpec dataclasses + cell-key conventions
  + HARNESS_VERSION pin
- `eval/downstream/aggregate.py` — the Cross-Scale Downstream Pareto kernel
- `tests/test_downstream_types.py` — schema-stability tests (19 cases)
- `tests/test_downstream_aggregate.py` — Pareto kernel tests (21 cases)

All 201 tests passing. Ruff clean. Zero external dependencies added.

## What B1 still owes (open, in dependency order)

1. **B1-D1, B1-D2, B1-D3, B1-D4** — license / sourcing decisions blocking
   scorer + core22 + private_hard
2. `eval/downstream/scorer.py` — score_mc / score_schema / score_lm kernels
3. `eval/downstream/core22.py` — DCLM bundle adapter
4. `eval/downstream/private_hard.py` — 4-task hardness subset adapter (uses
   the swap recorded in docs/license/hardness_subset_decision.md)
5. `eval/downstream/grader.py` — offline grader
6. `eval/downstream/calibration.py` — N=10 baseline runs → noise_floors_v1.json
7. `eval/downstream/runner.py` — subprocess-isolated entrypoint (consumes
   B1-D5, B1-D6, B1-D7, B1-D13 decisions)
8. **B1-D8, B1-D11** — calibration sourcing + container measurement strategy
9. `eval/hidden_eval.py` — additive change for `include_downstream=True`
   (consumes B1-D12)
10. `proof/runner.py::scan_diff_for_restricted` glob extension (B1-D10)
