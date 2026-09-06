# Current project handoff — 6 September 2026

Branch: `feat/iclr`. Project: `projects/logit_evidence_routing`.

## Current state (supersedes earlier smoke-only handoffs)

1. Phase 1: **PASS WITH ANOMALY**. Corrected bird/birds IDs 11199/17952; whitespace ID 29871 rejected. Twenty corrected figures reviewed; preserve Vision-CLS Top-K/top-1 mismatch.
2. Phase 2: **PASS**. 240 official-training-only CUB images, 160/80 development splits, 12 safetensors shards, nine stages, 576 patches. Cache digest `63cf0e80ec0a24533682467b6f3b23ccded8625ef25d2fb43d012aa0e72179d8`.
3. Phase 3: **full development PASS**, not just smoke. All nine stages, mean pooling, seeds 0/1/2, 300 epochs, CUDA, 256-dimensional projection, four controls. 90 summary rows and 2,340 attribute rows independently audited from the supplied raw reports.
4. Phase 4: **one-training-image cached-localizer reuse smoke PASS**, image 544, K=16/32, 14 metric rows, 42 agreement rows. Full Phase 4 is not complete.

Experiment commit for the supplied Phase 3/4 results: `1a6e0c96681f250659ce703207931289cd9112c2`. The commit adding this handoff is a later result/audit commit; do not relabel old experiments with its hash.

## Read first

- `reports/development_20260906/REVIEW.md`: verified current findings, source evidence, limitations and next gate.
- `reports/development_20260906/review/returned_bundle_audit.json`: independent returned-table/hash audit.
- `reports/development_20260906/review/primary_group_trajectory.csv` and `primary_attribute_transitions.csv`: all-group/all-attribute detail.
- `reports/PHASE_01B_TECHNICAL_REPORT.md`: earlier qualitative and localization audit.
- `planning/00_MASTER_PLAN.md`: overall research sequence; historical calendar is not a reset of completed gates.

## Most important observations

Primary macro-AUROC rises from 0.5561 at vision early to 0.6682 at vision late and 0.6718 at vision final, with projector 0.6516 and final LLM 0.6671. Shuffled-label means are near 0.5; random projection is a compression control, not a null baseline.

Macro stability hides heterogeneity: bill-shape group AUROC changes 0.8282 → 0.7533 from late vision to final LLM, while wing-pattern group changes 0.6506 → 0.7154. Do not claim all attributes are equally preserved. Attribute supports can be as small as three validation positives. Seed SD is not an image-sampling confidence interval.

Vision.late (hidden-state index 23) feeds the projector. Vision.final (24) is a separate diagnostic checkpoint. LLM indices are 8/16/31/32. Do not treat the vision.final→projector line in the display as an adjacent computation edge. No causal bottleneck, token redistribution, or answer-utilization gap has been established.

## Next authorized phase

Continue **Phase 4**, specifically the missing dense image–text similarity baseline and attribute-relevant localization protocol. Inspect existing code before adding features. Use a verified paired image/text embedding space with pinned heads, revision, normalization and preprocessing; do not compare arbitrary raw vision and text vectors. Define descriptions and relevant-part mappings for all existing 26 attributes, with positive/certain/visible/in-crop eligibility and explicit denominators. Generic bird/birds maps are object controls, not attribute-specific maps. Preserve compatible Logit Lens normalization and multi-token semantics.

Start with one deterministic development-training image on Kaggle. After that passes, run the matched development localization comparison. Then Phase 5 fixed-question utilization, Phase 6 combined findings, Phase 7 one supported causal intervention, and Phase 8 final frozen official-test evaluation. Do not jump ahead or select a hypothesis from the current macro curve.

## Operating constraints and paths

- Implement/inspect code and audit small result tables locally. Do not download models, datasets, checkpoints or large artifacts locally. Do not run actual VLM extraction or probe experiments locally.
- Kaggle must fetch/run the latest `feat/iclr`. Always run commands from `/kaggle/working/newpipeline/projects/logit_evidence_routing`, with `PYTHONPATH=src` as needed.
- Keep the model frozen, the 26-attribute selection unchanged, and the official CUB test split untouched until attributes, layers, prompts, parsers and metrics are frozen.
- The user explicitly requires repository changes to be committed and pushed to `feat/iclr`; preserve unrelated changes and verify the remote commit after pushing.
- Phase 1 corrected cache: `/kaggle/working/phase1b_corrected/cache`; gate: `/kaggle/working/phase1b_corrected/results/phase1_gate.json`.
- Phase 2 cache: `/kaggle/working/phase2_stage_cache`.
- Phase 3 full output: `/kaggle/working/phase3_development_1a6e0c96681f_20260906T171249251219Z`.
- Small source reports are checked into `reports/development_20260906/phase3_review_bundle` and `phase4_cached_localizer_smoke_bundle`; do not request them again.
- Original executable Kaggle cells are now in `notebooks/`. They reproduce completed runs; the next objective is the missing Phase 4 semantic comparison, not another Phase 3 rerun.

The source bundle audit uses only the Python standard library:

```bash
PYTHONPATH=src python scripts/review_development_bundles.py \
  --phase3-bundle reports/development_20260906/phase3_review_bundle \
  --phase4-bundle reports/development_20260906/phase4_cached_localizer_smoke_bundle \
  --output-dir /kaggle/working/verified_phase3_phase4_review
```

It verifies returned tables and hashes, not source tensor contents or original image split membership. Keep that provenance boundary in later claims.
