# 06 — Ablations and Representation Analysis

**Target completion: Sep 17**

## Goal

Answer the reviewer questions that are most likely to challenge the central mechanism.

Do not run peripheral ablations.

---

## Ablation A — Which layers contain useful evidence?

Compare:

- one earlier layer
- one middle/late layer
- final layer
- late-layer mean/persistence

Measure both:
- classifier performance
- semantic evidence quality/localization where possible

Desired conclusion should come from results, not assumption:

> semantic patch evidence emerges or stabilizes in later VLM layers.

---

## Ablation B — Patch budget K

Run:

```text
K ∈ {8, 16, 32, 64}
```

for at least:
- Random
- Attention
- Logit
- LGER

This checks whether the method only works by keeping many tokens.

---

## Ablation C — Evidence score

Compare:

1. max probability
2. top-1/top-2 logit margin
3. negative entropy
4. selected cross-layer persistence formulation

Hold K and classifier constant.

---

## Ablation D — Cross-layer aggregation

Compare:

```text
last layer only
mean across late layers
persistence-aware score
```

If the simplest method wins, use it. Do not force a complex contribution.

---

## Ablation E — Context patches

Compare:

```text
evidence only
evidence + random context
```

If context has negligible effect, remove it from the main method and keep it as an appendix result.

---

## Analysis A — Selection overlap

For each image calculate Jaccard overlap between:

- Attention Top-K
- Logit Top-K
- LGER Top-K

This quantitatively tests whether semantic routing is merely reproducing attention.

---

## Analysis B — Spatial localization

For CUB, where annotations allow it, measure whether selected patches fall inside:

- object bounding box
- annotated bird parts/regions if mapping is feasible

Important:

High overlap alone is not enough; attention may localize the bird broadly. The interesting question is whether selected patches are both localized and discriminative.

---

## Analysis C — Layer trajectory visualization

For selected spatial patches plot:

\[
e_i^{l_1}, e_i^{l_2}, ..., e_i^{l_m}
\]

Show:
- persistent high-evidence patch
- transient/noisy patch
- background patch

This provides the representation-learning interpretation of the method.

---

## Analysis D — Decoded token examples

For a small set of qualitative samples, show top decoded tokens from the same spatial patch across layers.

Do not claim that every token is a literal object label. Treat decoding as an interpretability probe of the VLM representation.

---

## Required paper artifacts

By the end of this phase generate:

```text
paper_assets/
  table_main.csv
  table_ablation_layers.csv
  table_ablation_k.csv
  table_ablation_score.csv
  fig_method.*
  fig_attention_vs_logit.*
  fig_low_data.*
  fig_layer_trajectory.*
```

---

## Exit criteria

- [ ] Every main paper claim has a corresponding experiment.
- [ ] Attention-vs-logit difference is quantified, not only visualized.
- [ ] Low-data claim has results.
- [ ] Layer-wise claim has results.
- [ ] We know which ablations belong in main paper vs appendix.
