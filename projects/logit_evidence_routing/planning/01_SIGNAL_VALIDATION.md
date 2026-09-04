# 01 — Phase 01 Sanity Gate

## Question

Is the 95% Vision-CLS Top-32 result a reproducible routing phenomenon rather
than split leakage, cache misalignment, or an evaluation artifact?

## Required audit

Run `scripts/write_phase1_sanity_report.py` after the matched Phase 01B probes.
It must verify:

- every pilot image belongs to the official CUB training partition;
- the development train and validation image IDs and paths are disjoint;
- cache ID, path, label, class name, and split match the manifest;
- no exact duplicate global cached features appear across images;
- Vision-CLS probe results exist for at least three initialization seeds;
- qualitative selected-patch figures exist;
- broad bird-box localization and visible-part proximity are quantified.

The report must state that the three runs use one fixed pilot split. Do not call
them independent dataset replications.

## Phase 01 measurement contract

- Selector: final-layer CLS-to-patch attention from LLaVA's frozen CLIP tower.
- Selected representation: late LLM visual-token state at `layer_offset=-2`.
- Aggregation: mean of exactly K selected original hidden states.
- Readout: same linear 20-way species probe for every selector.
- Development data: 8 train + 4 validation images per class for 20 classes,
  sampled only from the official training partition.
- Probe seeds: 0, 1, 2.
- Primary budgets: K=16 and K=32.

## Interpretation

A successful audit establishes that Vision-CLS attention is a strong routing
signal for late LLM representations on this development pilot. It does not yet
establish:

- attribute accessibility in raw vision features;
- a projector or language bottleneck;
- fine-grained part localization from the broad bounding box alone;
- causal use by normal VLM answer generation;
- generalization to the untouched official CUB test split.

## Exit rule

- **Pass:** integrity checks pass and the result remains strong across the three
  probe seeds. Freeze Phase 01 analysis choices and build the stage cache.
- **Stop:** any image/split/cache identity check fails, or the result does not
  reproduce. Diagnose before proceeding.
- **Pass with anomaly:** preserve anomalies such as weak top-1 pointing despite
  strong Top-K box coverage; carry them into the tracing hypotheses.
