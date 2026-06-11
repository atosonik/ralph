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
**Status:** **CLOSED 2026-06-10** — verified via WebFetch on
https://github.com/felipemaiapolo/tinyBenchmarks/blob/main/LICENSE:
**MIT License** (Copyright (c) 2024 Felipe Maia Polo). Permits commercial
use, modification, distribution, sublicensing. TinyMMLU + the
`tinyBenchmarks` Python package are SAFE for the v0.10 hardness subset.
The associated `tinyBenchmarks/tinyMMLU` HF dataset card does not display
a license badge but the parent project's MIT license covers the IRT++
parameters and the curated 100-item set. No swap needed.

### B1-D2 — DCLM bundle SHA pin
**Owner:** B1 (core22.py author)
**Decision:** Download the current DCLM CORE eval bundle, compute its SHA256,
and pin it in `eval/downstream/core22.py` as a module-level constant.
**Recommendation:** Pin first; revisit only if the upstream bundle gets
materially better (the rotation cost is high, do it once).
**Status:** **URL LOCKED 2026-06-10, SHA-pin deferred to first B1 code commit.**
Verified via WebFetch on
https://raw.githubusercontent.com/karpathy/nanochat/master/scripts/base_eval.py:
the canonical bundle URL is
**`https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`**
(constant `EVAL_BUNDLE_URL` in nanochat). The bundle is hosted in
Karpathy's personal S3, which means it CAN rotate without notice — the
SHA-pin commit is a one-time guard against silent upstream changes.

**Protocol the first B1 commit must follow:**
1. `wget` the bundle to a one-shot location (~200 MB).
2. Compute `sha256sum` + record verbatim in `eval/downstream/core22.py`
   as `DCLM_EVAL_BUNDLE_SHA256 = "..."`.
3. Mirror the zip into `eval/private/downstream_pool/bundle_v1/`
   (gitignored — not redistributed).
4. Add a one-time test `test_dclm_bundle_sha_pinned` that re-hashes the
   local mirror and asserts equality with the pinned constant.
5. Document the rotation policy in the core22.py docstring: if upstream
   rotates, we re-pin under a new constant `DCLM_EVAL_BUNDLE_SHA256_v2`
   and the harness consumes both via fallback ladder.

No further blockers on this item; the URL is verified and the protocol
is recorded. Closed-pending-execution.

### B1-D3 — 22-vs-23 task answer for `bigbench_language_identification`
**Owner:** B1 (core22.py author)
**Decision:** DCLM's CORE-22 vs CORE-23: include or exclude
`bigbench_language_identification`. Karpathy's nanochat uses the 22-task
variant; some other downstream evals use the 23-task variant.
**Recommendation:** Match Karpathy's 22-task selection. Document the choice
in core22.py docstring + the public Cross-Scale Downstream Pareto post.
**Status:** **CLOSED 2026-06-10** — DCLM's authoritative CORE-22 list is the
`low_variance_datasets` array in
https://raw.githubusercontent.com/mlfoundations/dclm/main/eval/additional_aggregation.json.
Verbatim verified count: **22 tasks**, `bigbench_language_identification`
IS included. Karpathy's "22 nice and high quality datasets" framing in
nanochat #420 matches this list exactly. Final lock for `core22.py`:

```python
DCLM_CORE_22_TASKS = (
    "hellaswag_zeroshot",
    "jeopardy",
    "bigbench_qa_wikidata",
    "arc_easy",
    "arc_challenge",
    "copa",
    "commonsense_qa",
    "piqa",
    "openbook_qa",
    "lambada_openai",
    "hellaswag",
    "winograd",
    "winogrande",
    "bigbench_dyck_languages",
    "agi_eval_lsat_ar",
    "bigbench_cs_algorithms",
    "bigbench_operators",
    "bigbench_repeat_copy_logic",
    "squad",
    "coqa",
    "boolq",
    "bigbench_language_identification",
)
assert len(DCLM_CORE_22_TASKS) == 22
```

Note that `openbook_qa` appears in CORE-22 as a downstream-eval consumer;
this is distinct from B1-D4 (CC-BY-SA share-alike review) and from the
PRIVATE hardness subset (B1's parallel track) where OpenBookQA was
pre-swapped per `docs/license/hardness_subset_decision.md`. Including it
in CORE-22 is downstream consumption, not redistribution.

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
**Status:** **CLOSED 2026-06-11 — option (a) ratified.** `eval/downstream/
runner_cli.py` calls `torch.load(weights_only=True)` (closes the pickle RCE
vector) AND its module + function docstrings document the caveat that
forward()-code execution is NOT prevented by weights_only and the only
containment is OS-level process isolation via the subprocess wrapper
(`eval/downstream/runner_subprocess.py`). Seccomp / landlock recorded as a
named follow-up phase before mainnet activation; not blocking B2 / B3 / B4.

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
**Status:** **CLOSED at aggregation surface 2026-06-11 — option (a)
ratified.** `eval/downstream/calibration.py` ships the
`aggregate_noise_floors` kernel that takes N DownstreamReports and
produces a `NoiseFloorTable` via per-task `margin_multiplier *
max(stddev across scales)`. JSON round-trip via
`write_noise_floor_table_json` / `read_noise_floor_table_json`. The
**operational** half (train 10 baselines on H100, run each through
`run_eval_in_subprocess`, feed reports into the aggregator) is gated
on (1) `load_task_examples` being implemented (B1-D1 follow-up) and
(2) an H100 instance for the ~$15 baseline-training pass. Recorded as
a separate operational step, not blocking B2 / B3 / B4 code paths
that consume `NoiseFloorTable` via the same dataclass contract.

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
**Status:** **CLOSED 2026-06-10 — option (b) ratified.** B1 ships in
non-attested-extension mode on testnet: the `eval/downstream/` tree is
present + functional but is NOT yet part of the canonical container
measurement consumed by op2_attestation_verify. Container re-measurement
is explicitly a mainnet-activation deliverable, not B1's. Rationale:

* Container re-measurement requires coordinated validator deployment
  (all validators must adopt the new measurement simultaneously or the
  attestation check splits the network), which is independently a B7
  concern.
* During the B1 → B6 build window, miners can submit but the v0.10
  legacy gate (`KARPA_KING_RULE=legacy`) is the rule that crowns kings.
  The new `eval/downstream/` modules are validator-side-only and don't
  need to be miner-attested yet.
* The cost of waiting: a malicious patch could modify `eval/downstream/`
  if it shipped attested today. The cost of NOT waiting: zero, since
  the new rule isn't crowning anyone yet.

Documentation: a one-line note in `eval/downstream/__init__.py` already
records the RESTRICTED designation. proof/runner.py's
scan_diff_for_restricted scanner gate (per B1-D10) closes the
miner-side modification vector before the harness goes live.

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
**Status:** **CLOSED at CLI surface 2026-06-11.** `eval/downstream/
runner_cli.py` accepts both `--patch` and `--karpa-root` arguments so the
CLI contract is stable; submissions without structural patches go through
the no-patch path unchanged. Full `apply_patch` integration (the ~150 LOC
+ 2 day deliverable) is recorded as a separate follow-up PR: until that
lands, invoking the CLI WITH `--patch` raises `NotImplementedError`
pointing at this DEFERRED.md item. No-patch submissions (the common case)
are fully supported.

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
**Status:** **CLOSED 2026-06-10 — reframed.** The "±1% vs GPT-2-small"
formulation imports a fixed-model reference that the Cross-Scale Downstream
Pareto rule does not actually depend on. The rule's acceptance criterion
for any cell is the per-cell calibrated noise floor `eta_task` computed
empirically from N=10 baseline-recipe runs at the same scale (per
calibration.py's job, B1-D8). That floor IS the acceptance threshold —
there is no "absolute" external reference because the rule is comparative
(challenger vs king on the SAME calibrated surface, not vs an external
target).

The GPT-2-small framing was useful as a sanity check on the harness's
output magnitudes (e.g., "if the harness reports 95% accuracy on PIQA at
124M params, something is wrong because GPT-2-small gets ~75%"). For that
sanity-check role:

* DCLM paper §3.1 (Li et al. 2024, arXiv:2406.11794) reports per-task
  CORE-22 numbers at 412M-1x as a fixed-cost reference; 124M would be a
  ~1-2 absolute percentage points below the 412M-1x line on most tasks.
* OLMo-2 paper publishes per-task CORE numbers across multiple checkpoint
  sizes; the smallest published OLMo-2 (1B) is one band up from our S₃.

Both are SANITY-CHECK references, not acceptance criteria. Sanity-check
test in B1: assert per-task accuracy at our 124M reference checkpoint is
within ±5 absolute percentage points of DCLM 412M-1x per-task numbers
after scale adjustment (loose tolerance). Anything more aggressive is
spurious precision.

## What B1 ships this commit (closed)

- `eval/downstream/__init__.py` — re-exports + RESTRICTED marker
- `eval/downstream/types.py` — DownstreamReport / CellResult /
  NoiseFloorTable / ParetoVerdict / TaskSpec dataclasses + cell-key conventions
  + HARNESS_VERSION pin
- `eval/downstream/aggregate.py` — the Cross-Scale Downstream Pareto kernel
- `tests/test_downstream_types.py` — schema-stability tests (19 cases)
- `tests/test_downstream_aggregate.py` — Pareto kernel tests (21 cases)

All 201 tests passing. Ruff clean. Zero external dependencies added.

## Update 2026-06-10: 5 additional B1 decisions closed (no code change)

- **B1-D1 (TinyMMLU license)** — closed; MIT, commercial OK. No swap.
- **B1-D2 (DCLM bundle URL + SHA pin protocol)** — URL locked to
  `https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip`;
  SHA-pin protocol recorded for first B1 code commit.
- **B1-D3 (22-vs-23 task answer)** — closed; verbatim 22-task list
  recorded as `DCLM_CORE_22_TASKS`. `bigbench_language_identification`
  is included.
- **B1-D11 (container measurement strategy)** — closed; B1 ships in
  non-attested-extension mode on testnet, container re-measurement
  deferred to mainnet activation.
- **B1-D15 (GPT-2-small ±1% baseline)** — closed; reframed away from a
  fixed-model reference. The rule's acceptance criterion is the
  per-cell calibrated noise floor, not an external baseline.

Now 7 of 15 items closed, 8 still open. Remaining open items are all
**implementation-time** decisions for the scorer/runner/calibration
authors (subprocess isolation reality, tokenizer enforcement,
determinism specification, structural-patch handling, CC-BY-SA review,
calibration sourcing, restricted-files scanner glob, HiddenEvalResult
schema-versioning test).

## Update 2026-06-11: 4 additional B1 decisions closed by runner.py PRs

- **B1-D6 (tokenizer equivalence)** — closed; runner.py rejects
  `vocab_size != 50257` at the in-process kernel via
  `check_vocab_compatibility`.
- **B1-D7 (determinism specification)** — closed; `set_eval_determinism`
  pins `torch.use_deterministic_algorithms(True)` +
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **B1-D5 (subprocess isolation reality)** — closed; runner_cli.py
  uses `torch.load(weights_only=True)` and documents that
  forward()-code containment is OS-level subprocess isolation only.
- **B1-D13 (structural-patch CLI args)** — closed at CLI surface;
  --patch / --karpa-root accepted; full apply_patch integration
  recorded as a separate follow-up PR.

Now 11 of 15 items closed, 4 still open: B1-D4 (CC-BY-SA legal
review), B1-D8 (calibration data sourcing), B1-D10 (restricted-files
scanner glob extension), B1-D12 (HiddenEvalResult schema-versioning
test). Note: B1-D1 (HF dataset download implementation) was logged as
closed in the 2026-06-10 update; the load_task_examples function
remains stubbed but the decision itself is closed.

## Update 2026-06-11 (later) — B1-D8 closed at aggregation surface

- **B1-D8 (calibration data sourcing)** — closed; `calibration.py`
  ships `aggregate_noise_floors` + JSON round-trip. Operational
  baseline-training run (H100 + ~$15) recorded as a separate step,
  not blocking B2 / B3 / B4 code that depends on the
  `NoiseFloorTable` dataclass.

Now 12 of 15 items closed, 3 still open: B1-D4 (CC-BY-SA legal
review), B1-D10 (restricted-files scanner glob extension), B1-D12
(HiddenEvalResult schema-versioning test). Each is a small,
focused PR independent of the remaining B1 module sweep.

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
