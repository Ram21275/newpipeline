# 03 — Matched Measurement Baselines

## Principle

Hold data, pooling, regularization, and probe optimization fixed when comparing
representation stages. Hold the downstream representation fixed when comparing
localizers. These are two different experiments and must remain separate.

## A. Attribute accessibility by stage

For each compatible stage, train the same linear multi-label probe on selected
CUB attributes. Report per-attribute AUROC, macro-AUROC, macro-F1, and class
balance. Use training-set-only normalization and threshold selection.

Required controls:

- majority/prevalence baseline;
- shuffled-label probe;
- random Gaussian projection matched to probe dimension;
- global/CLS state versus mean of all patches;
- identical probe seeds and stopping rule.

Species accuracy may be reported as a secondary diagnostic, not the main proof
of fine-grained attribute information.

## B. Spatial localization on one fixed representation

Compare equal Top-K budgets:

- random patches;
- Vision-CLS attention;
- LLM visual attention from a precisely named query/answer token;
- dense concept-conditioned image–text similarity;
- Logit Lens for audited single tokens where valid.

Report bird-box metrics separately from visible-part metrics. Attribute-region
evaluation is valid only where CUB provides a defensible part/attribute mapping;
document that mapping before looking at results.

## C. Representation–utilization baseline

For the same attribute questions and images, compare:

- linear accessibility from frozen states;
- zero-shot VLM answer accuracy under a fixed prompt and parser;
- prompt-only and image-shuffled controls.

A probe/VLM gap is descriptive until a causal intervention is run.

## Statistical reporting

Use per-image paired differences and bootstrap confidence intervals where
possible. Probe-seed variance is not a substitute for sampling uncertainty.
Keep the official test split untouched until prompts, layer choices, attribute
subset, and metrics are frozen.
