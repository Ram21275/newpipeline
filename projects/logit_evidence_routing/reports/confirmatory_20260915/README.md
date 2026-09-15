# Confirmatory development analysis handoff

Status: **PASS** on the locked CUB development cohort (834 decisions, 77
images, 26 attributes, 20 species). No local model forwards or official-test
image accesses were performed.

## Primary interpretation

The Vision-CLS-minus-Logit-Concept selector differential is positive in the
raw yes-minus-no direction for both targets:

| Stage | Target 0 | Target 1 | Raw target difference |
|---|---:|---:|---:|
| vision.late | 0.0719 | 0.0772 | 0.0053 |
| projector.output | 0.0822 | 0.0900 | 0.0078 |

The simultaneous 95% intervals for both raw target differences contain zero.
The apparent reversal on the correct-answer-margin scale is therefore the
expected consequence of multiplying negative-target rows by -1, not evidence
for separate positive and negative routing mechanisms. Attribute- and
species-blocked estimates, plus leave-one-group-out diagnostics, do not change
that conclusion.

Mean replacement establishes controlled sensitivity to the selected patch
set. It does not establish that the untouched VLM naturally uses the same
route, that information was erased, or that the effect generalizes beyond CUB.

## Absolute development behavior

| Model | Teacher-forced accuracy | Generated accuracy (all decisions) | Parse rate |
|---|---:|---:|---:|
| LLaVA-1.5-7B | 0.5504 | 0.5312 | 1.0000 |
| Qwen2.5-VL-7B | 0.5899 | 0.5168 | 0.8849 |

Point estimates are decision-weighted. Intervals resample whole image clusters
to respect the repeated decisions per image.

## Locked next experiments

1. Run exact spatial-and-four-bin norm matching at the largest common feasible
   selector size, K=34. The plan has 21,684 outcomes; 8,340 can be reused and
   13,344 require new forwards.
2. Run multimodal DeCo first, using fused language states conditioned on the
   same image and attribute question. For LLaVA, use layers 20-28 inclusive,
   alpha=0.6, top-p=0.9, and top-k=20. Extract DoLa on the identical forwards
   as a direct comparator.
3. Run the answer-vocabulary/order pilot. Promote only effects that agree for
   yes/no, true/false, and present/absent under counterbalancing.
4. On Qwen, require the merger/token-layout architecture gate before any
   internal measurement, then architecture, smoke, balanced pilot, and only
   then development-full.
5. Keep the guided multi-head fusion head secondary. It measures learned
   decodability from frozen stages and cannot by itself establish natural VLM
   routing.

The official-test cohort still requires one final feasible attribute-decision
lock. Do not access official-test images until that specification and every
development choice are committed.

## Modular extension point

Colleague-supplied results should be added as a separate analysis module with
their source hashes, cohort identity, and claim boundary. Do not rewrite these
locked primary tables; compare new estimates in an explicitly labeled
replication or extension table.
