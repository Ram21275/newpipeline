# 05 — Main Experiments

## Experiment 1 — Attribute recoverability trajectory

Use a frozen, stage-aligned representation cache and the same linear multi-label
probe at vision early/middle/late/final, projector output, and LLM
early/middle/late/final. Plot macro-AUROC and macro-F1 versus depth, with
per-attribute results retained in CSV.

Primary question: where is the largest change in linear accessibility?

## Experiment 2 — Spatial and semantic localization

On the same images and selected attributes, compare Vision-CLS attention, LLM
attention, dense image–text similarity, and valid Logit Lens scores. Report:

- fraction/recall/IoU within the broad bird box;
- visible-part patch recall and any-part hit;
- top-1 distance to the nearest relevant visible part;
- overlap and disagreement between tools.

Primary question: are discriminative routing and semantic readability aligned?

## Experiment 3 — Representation versus normal VLM use

Ask fixed, parseable attribute questions and species questions. Compare VLM
answer performance with probe accessibility from the same images and stages.
Include prompt-only and image-shuffled controls.

Primary question: does a substantial, consistent accessibility–answer gap exist?

## Experiment 4 — Minimal causal intervention

Run only after `INTERMEDIATE_FINDINGS.md`. Remove or replace the smallest
supported evidence set at the identified stage. Match random controls by number
of tokens, spatial distribution where possible, and replacement magnitude.
Measure answer-logit and accuracy changes.

Primary question: does the final prediction depend causally on the identified
information?

## Required outputs

- `results/attribute_probe_by_stage.csv`
- `figures/attribute_probe_by_stage.png`
- `results/localization_by_tool.csv`
- `results/vlm_answer_vs_probe.csv`
- `INTERMEDIATE_FINDINGS.md`
- causal CSV/figure only after the transition report

Every table distinguishes development from final official-test evaluation.
