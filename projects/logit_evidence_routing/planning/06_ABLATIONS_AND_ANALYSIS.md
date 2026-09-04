# 06 — Controls and Alternative Explanations

Run only controls that can change the interpretation of the central trajectory.

## Probe controls

- shuffled labels and prevalence baselines;
- learning curves versus examples per attribute;
- probe dimension and regularization held constant;
- linear versus one small nonlinear probe only if linear accessibility is low;
- report attributes with insufficient positives/negatives rather than hiding
  them in a macro average.

## Stage-comparison controls

- compare per-token and pooled representations;
- account for representation dimension and normalization;
- test whether a random projection preserves the same trend;
- verify exact vision-to-LLM token correspondence;
- report missing/incompatible measurements as N/A, not zero.

## Localization controls

- equal K for every selector;
- random selection uncertainty over multiple selection seeds;
- broad box and visible-part metrics reported separately;
- top-1 and Top-K metrics both shown;
- image-shuffled concept text and generic object words as semantic controls;
- no per-image species names in localization queries.

## Utilization controls

- prompt-only and image-shuffled answer baselines;
- intervention magnitude matched to random controls;
- remove versus mean-replace to separate deletion artifacts;
- repeat at one neighboring layer to test transition specificity.

## Alternative explanations to address

1. Vision-CLS selection may remove background noise without localizing the
   species-defining attribute.
2. A high-dimensional probe may memorize a small pilot; use learning curves,
   regularization, and the final untouched test split.
3. Apparent information loss may reflect a mismatched readout basis rather than
   destruction of information.
4. Apparent movement may result from token mixing or broken spatial alignment.
5. A probe/VLM gap may arise from prompt/parser failure rather than reasoning.

Sparse autoencoders remain optional. Use one only when simpler measurements show
a robust but semantically opaque transition.
