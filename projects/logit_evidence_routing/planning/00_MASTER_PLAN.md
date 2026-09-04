# 00 — Master Execution Plan

## Objective

Produce an evidence-bounded ICLR study of how fine-grained visual information is
represented and used across a frozen VLM. Do not assume a bottleneck, a
representation–utilization gap, or a monotonic loss of information in advance.

## Testable hypotheses

- **H1 — Visual-to-language bottleneck:** attribute information is accessible in
  vision features but decreases at the projector or in the LLM.
- **H2 — Representation–utilization gap:** information remains probe-accessible
  in LLM states, while normal answer generation performs substantially worse.
- **H3 — Discriminative and semantic localization differ:** a map can select
  species-useful regions without making their human-readable attributes directly
  decodable.
- **H4 — Evidence transforms or moves:** local patch information becomes
  redistributed rather than simply weakening with depth.

These are competing possibilities. Report a null or mixed trajectory if that is
what the measurements support.

## Fixed scope

- Dataset: official CUB-200-2011 split, using boxes, parts, and official
  attributes where applicable.
- Initial model: `llava-hf/llava-1.5-7b-hf` at a pinned revision.
- Frozen: vision encoder, projector, LLM, LM head.
- Trainable diagnostics: linear probes first; no high-capacity MLP until linear
  accessibility is understood.
- Same images, labels, preprocessing, and probe protocol at every stage.
- Species accuracy is secondary to attribute recoverability and localization.

## Gates and deliverables

### Gate 1 — Phase 01 sanity (now)

Audit the 95% Vision-CLS selection result for official-split membership,
train/validation disjointness, cache-to-manifest identity, exact duplicate cached
features, three probe seeds, box localization, part proximity, and qualitative
examples. Deliver `phase1_sanity_report.md` and stop if it fails.

### Gate 2 — Representation cache

For the same images, cache selected early/middle/late/final vision layers,
projector output, and early/middle/late/final LLM visual-token states with spatial
metadata. Deliver a documented, resumable schema. Do not repeatedly run the VLM
for each probe.

### Gate 3 — Accessibility and localization

Run the same linear attribute probe at every compatible stage. Separately compare
Vision-CLS attention, LLM visual attention, dense semantic similarity, and Logit
Lens where mathematically valid. Deliver stage-wise CSVs and figures.

### Gate 4 — Identify the transition

Write `INTERMEDIATE_FINDINGS.md` with the strongest observation, numbers,
alternative explanations, best-supported hypothesis, and one minimal causal
test. Do not silently convert a diagnostic into the proposed method.

### Gate 5 — Causal test

At the identified stage, remove or replace top evidence tokens and compare with
random matched controls. Only then discuss whether recoverable evidence is used.

## Immediate calendar

| Dates | Work | Required output |
|---|---|---|
| Sep 4–5 | Correct and audit Phase 01 | sanity report + corrected concept rows |
| Sep 5–8 | Stage-aligned cache | schema + resumable extractor |
| Sep 8–12 | Attribute probes | recoverability-by-stage CSV/figure |
| Sep 12–15 | Spatial/semantic comparison | quantitative localization tables |
| Sep 15–17 | Transition analysis | `INTERMEDIATE_FINDINGS.md` |
| Sep 17–18 | Evidence-bounded abstract | abstract submission |
| Sep 19–22 | Minimal causal study + draft | main causal table + complete draft |
| Sep 22–25 | Reproduction and freeze | anonymous final submission |

This is a tight schedule. A careful single-model CUB study is preferable to an
under-validated multi-model or multi-dataset sweep.

## Required run record

Every run saves commit hash, exact model revision, preprocessing, source split,
image IDs, layer definitions, representation shape/dtype, probe seed, metrics,
runtime, and peak memory. The untouched official test split is used only after
the analysis choices are frozen.

## Claim discipline

- Attention is not a causal explanation.
- Probe accuracy is accessibility, not proof of model use.
- Bounding-box overlap is object localization, not attribute localization.
- Logit confidence is not semantic identity.
- A difference between selectors does not isolate a representational stage.
- The Phase 01 validation split has already informed method choice and cannot be
  presented as final held-out evidence.
