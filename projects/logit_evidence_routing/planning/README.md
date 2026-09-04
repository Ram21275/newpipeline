# ICLR 2027 Execution Pack — Fine-Grained Evidence Tracing

This directory is the canonical research plan. The repository directory keeps
its historical `logit_evidence_routing` name so existing Kaggle paths and caches
continue to work; Logit Lens is no longer the proposed contribution.

## Research question

> Where does fine-grained visual evidence exist inside a VLM, how does it change
> through the vision encoder, multimodal projector, and language model, and does
> the final VLM use evidence that remains recoverable internally?

The study traces the same CUB images and attributes through:

`image → vision layers → projector → LLM layers → generated answer`

It separates four properties that must not be conflated:

1. spatial or discriminative importance;
2. linear accessibility of an attribute;
3. direct semantic readability;
4. causal use by the final prediction.

Vision attention, LLM attention, dense image–text similarity, linear probes,
Logit Lens, and later interventions are measurement tools for different
properties. The goal is a high-level empirical finding, not a contest in which
one tool must win.

## Current evidence boundary

The Phase 01B pilot found Vision-CLS Top-32 selection at 95.0% species-probe
accuracy versus 54.4% for Random-32 and 66.3% for mean pooling all 576 patches.
It also placed 80.8% of selected patch centers inside the bird box versus 47.1%
for random selection.

This result is promising but narrower than it first appears:

- Vision-CLS attention selected patches, while the linear classifier consumed
  late LLM patch states. It is evidence for a strong vision-side routing signal,
  not direct proof that a vision-layer representation itself classifies at 95%.
- The three runs vary probe initialization on one fixed development split. They
  do not measure variability across data splits.
- CUB boxes cover whole birds, not the fine-grained part or attribute responsible
  for a species decision.
- `logit_concept` and `attention_logit_fusion` from commit `e89d9e6` are invalid:
  tokenization included standalone whitespace token 29871. Repair and rerun
  those rows before interpreting them.

## Reference that motivated the original direction

Arsh Naqvi, [*Using Logit Space of VLMs for Attention to Detail*](https://www.arsh-naqvi.xyz/blog/logit-space-vlm-attention-to-detail),
describes a private trauma pipeline using attention for candidate localization
and a frozen LM head for patch-level semantic filtering. It motivates inspecting
the vision/language interface. It is not a public-benchmark result and does not
establish that generic logit confidence is a superior localizer on CUB.

## Execution order

1. `00_MASTER_PLAN.md` — scope, hypotheses, gates, and timeline.
2. `01_SIGNAL_VALIDATION.md` — audit the Phase 01 result.
3. `01B_LOCALIZATION_SELECTOR_VALIDATION.md` — selector result and correction.
4. `02_DATA_AND_CACHE_PIPELINE.md` — stage-aligned representation cache.
5. `03_BASELINES.md` — matched probes and answer baselines.
6. `04_METHOD_LOGIT_EVIDENCE_ROUTING.md` — tracing framework; legacy filename.
7. `05_MAIN_EXPERIMENTS.md` — layer-wise, spatial, semantic, and use measurements.
8. `06_ABLATIONS_AND_ANALYSIS.md` — controls and alternative explanations.
9. `07_ABSTRACT_DEADLINE.md` — claim gates and submission checklist.

Do not begin causal intervention until `INTERMEDIATE_FINDINGS.md` identifies a
specific transition worth testing. Do not begin with sparse autoencoders; add
them only if simpler probes expose a transition that needs an interpretable
feature basis.

## Deadlines

Official ICLR 2027 deadlines are 18 September 2026 for abstracts and
25 September 2026 for papers, both 11:59 PM Anywhere on Earth.
