# Whitepaper v1.1 → v1.2 manual update guide

Two kinds of edits below: a mechanical Karpathian→Karpa sweep, and a content rewrite for §5.4 / §5.5 that collapses the two-tier credibility model into the single attested-execution tier.

Source file: `Karpathian-Whitepaper-v1.1.docx`. Target: `Karpa-Whitepaper-v1.2.docx`.

---

## Part A — Brand rename (find-and-replace)

Use Word's Find & Replace dialog with **Match case** on. Run these in the order shown so compound terms aren't already collapsed by the bare-word pass.

| # | Find | Replace with | Approx. count |
|---|---|---|---|
| 1 | `Karpathian-1` | `Karpa-1` | 2 |
| 2 | `Karpathian-2` | `Karpa-2` | 1 |
| 3 | `Karpathian Live` | `Karpa Live` | 1 (table 14) |
| 4 | `Karpathian Docker` | `Karpa Docker` | ~5 |
| 5 | `KARPATHIAN` | `KARPA` | 2 (title + page header) |
| 6 | `Karpathian` | `Karpa` | ~110 (body + tables + footnotes) |

### Header / footer / title

- **Cover-page title**: `KARPATHIAN` → `KARPA` (caught by row 5).
- **Page header on every page**: `KARPATHIANDecentralized Autonomous AI Research · Whitepaper v1.1` → `KARPADecentralized Autonomous AI Research · Whitepaper v1.2`. The missing space between `KARPA` and `Decentralized` is an existing layout quirk; preserve it.
- **Cover-page version line**: bump `Whitepaper v1.1` → `Whitepaper v1.2`.

### What NOT to change

- `https://github.com/karpathy/autoresearch` — Andrej Karpathy's repo, the inspiration. Appears at the §3.1 intro and footnote [5]. Not us.
- All external `*.ai` URLs in the references section: `epoch.ai`, `primeintellect.ai`, `o-mega.ai`, `subnetalpha.ai`, etc.
- References [1]–[21] generally — leave the citations intact.

### File rename

`Karpathian-Whitepaper-v1.1.docx` → `Karpa-Whitepaper-v1.2.docx`. Keep the v1.1 file on disk for lineage; publish v1.2.

---

## Part B — Single attested-execution tier rewrite (§5.4 / §5.5 / §9 / glossary)

All snippets below already use "Karpa", so they're paste-ready regardless of whether you do Part A first.

### B.1 — §5.4: Operation 4 wrap-up paragraph (around L106)

In the paragraph starting *"Why this is enough."*, replace the parenthetical clause (2):

> **(2) for verified submissions, the hardware that ran the training was genuine and was running the official miner container under the declared workload — for unverified submissions, the calibration benchmark constrains hardware claims and the 0.5× credibility factor absorbs the residual uncertainty;**

with:

> (2) the hardware that ran the training was a genuine Confidential-Computing GPU running the official Karpa container under the declared workload, with all attestations bound to the validator-issued nonce;

---

### B.2 — §5.4: replace the entire "Hardware attestation: a two-tier credibility model" subsection (L109–L113 + table 8)

**New subhead:** `Hardware attestation: the single attested-execution tier`

**Body (replaces the four body paragraphs):**

> Karpa's scoring (Section 5.5) prices a contribution as quality improvement per normalized compute dollar produced under canonical training conditions. Two failure modes have to be foreclosed for that score to mean anything: the miner fabricating the result (Section 5.7's audit), and the miner running real training on cheap hardware while claiming expensive hardware — the denominator attack. Karpa addresses the denominator attack architecturally rather than statistically. There is one proof-test tier, and every submission to it carries a full hardware attestation chain. Submissions without a valid chain are rejected, not discounted.

> **The canonical proof-test environment.** Every miner runs the official Karpa Docker image inside a Confidential VM on a Confidential-Computing-capable GPU — H100, H200, or B200 — hosted under Intel TDX or AMD SEV-SNP. The Docker's measurement is pinned on-chain. The CVM's signed attestation chain binds, in one nonce-coupled flow, the GPU report, the TDX (or SEV-SNP) quote, the container measurement, the data-manifest hash, and a rolling hash of the training log. Any deviation — non-official container, attestation missing, nonce mismatch, manifest mismatch — fails verification, and the submission is rejected with no recourse to a lower tier.

> **Why one tier.** The earlier two-tier design (verified at face value; unverified at a 0.5× credibility discount) priced the lie-about-hardware attack out of profitability statistically. The container-measurement layer adopted here forecloses it deterministically: a miner cannot produce a valid submission off the official image on a non-CC GPU, period. The 0.5× factor and its calibration are therefore retired. The trade-off is consumer-GPU exclusion at launch: a miner without access to a CC-enabled cloud cannot participate in the launch track. This is acknowledged honestly as Phase 4 work (Section 8). Once a credible cheaper trust primitive is available — fractional CC slots on commodity clouds, a hybrid attestation model, or a different per-track equilibrium — the network can lower the floor without giving up the architectural guarantee. Until then, the floor is the floor.

> **Miner search remains unconstrained.** Only the proof step is subject to this requirement. A miner's private autonomous-research loop (Section 5.3) runs on whatever hardware they choose; the proof is what the network actually scores.

**Replace table 8** with this one-row spec:

| Requirement | Treatment in scoring |
|---|---|
| Official Karpa Docker (on-chain pinned measurement) running inside a Confidential VM on a Confidential-Computing-capable GPU (H100 / H200 / B200) under Intel TDX or AMD SEV-SNP; full attestation chain valid per §5.4. | Compute claim accepted at the calibration-normalized H100 reference cost; no credibility factor. |

---

### B.3 — §5.4: replace the entire "Failure semantics" paragraph (around L132)

> **Failure semantics.** A submission whose attestation chain fails verification is rejected outright. There is no fallback tier to absorb the failure. A miner uncertain whether their attestation environment will produce a clean chain should not submit; mining is permissionless and free, so a deferred submission costs the miner nothing. To avoid penalising honest miners for transient infrastructure issues, the validator surfaces the specific failure reason on rejection, and a miner whose chain fails for a documented infrastructure cause — Intel PCS collateral expiry, NVIDIA root certificate rotation, a coordinated network upgrade to a new container measurement — may re-submit within a grace window without burning a fresh on-chain handshake.

---

### B.4 — §5.5: replace the "A hardware-independent compute unit, scored under a credibility factor." paragraph (around L140)

**New subhead:** `A hardware-independent compute unit.`

> "Per compute dollar" only means something if the denominator is comparable across machines. Karpa normalizes compute to a reference accelerator — the NVIDIA H100-SXM 80GB — with the in-bundle calibration benchmark calibrated against published H100 reference timings; submissions are reported in normalized H100-hours rather than raw GPU-hours, so a discovery is rewarded for the efficiency of the idea, not the speed or expense of the hardware that happened to run it. Trustworthiness of the denominator is no longer a scoring parameter: it is a precondition. The proof-test attestation chain (Section 5.4) verifies that the run actually used the declared hardware, and submissions without a valid chain are rejected, not discounted. The earlier two-tier credibility factor (verified α = 1.0, unverified α = 0.5) is removed from the scoring expression.

---

### B.5 — §9 Risks table: add (or expand) two rows

**Container-correctness row** — strengthen if it exists, add if it doesn't:

| Risk | Mitigation |
|---|---|
| **Official Docker as trust anchor.** With the single-tier design, the official Karpa Docker is the trust anchor for the entire network: a flaw that lets user-supplied weights into the signing path would invalidate every submission produced during the vulnerable window. | Reproducible builds and published measurements so any operator can verify the image; coordinated multi-week notice for image rotations; documented security-out-of-cycle release path with abbreviated notice when the flaw is in the old image (Section 5.4 upgrade cadence); medium-term, governance path for transferring the signing key from the team to a multi-party body. |

**Consumer-GPU exclusion row** — add:

| Risk | Mitigation |
|---|---|
| **Consumer-GPU exclusion at launch.** The single-tier requirement excludes miners without access to CC-enabled cloud GPUs from the launch track. This narrows the launch miner population and depends on CC cloud pricing for participation cost. | CC cloud capacity is general-availability on Azure NCC, GCP Confidential VMs, and selected neoclouds, and prices have tracked standard H100 rentals within ~10–20%. Phase 4 (Section 8) commits to a hybrid trust path that lowers this floor as a credible cheaper primitive becomes available. |

---

### B.6 — Glossary (table 21): one targeted edit

In the **NVIDIA Confidential Computing** row, replace the trailing clause:

> ~~Enables verified-tier scoring in Karpa.~~

with:

> The hardware primitive Karpa relies on for the proof-test attestation chain (Section 5.4); every submission requires it.

No new glossary entry needed — "verified tier" and "unverified tier" no longer exist in the document after these edits.

---

## Checklist

- [ ] Part A.1–A.6 find-and-replace passes run (case-sensitive, in order)
- [ ] Page header brand + version updated
- [ ] Cover-page version `v1.1` → `v1.2`
- [ ] §5.4 Op-4 wrap-up parenthetical replaced (B.1)
- [ ] §5.4 "Hardware attestation: a two-tier credibility model" subsection + table 8 replaced (B.2)
- [ ] §5.4 "Failure semantics" paragraph replaced (B.3)
- [ ] §5.5 "A hardware-independent compute unit" paragraph + subhead replaced (B.4)
- [ ] §9 risks table: container-correctness row + consumer-GPU-exclusion row (B.5)
- [ ] Glossary: NVIDIA CC entry trailing clause swapped (B.6)
- [ ] File saved as `Karpa-Whitepaper-v1.2.docx`
