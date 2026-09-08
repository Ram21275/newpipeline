# Current project handoff — 6 September 2026

Branch: `feat/iclr`. Project: `projects/logit_evidence_routing`.

## Current state (supersedes earlier smoke-only handoffs)

1. Phase 1: **PASS WITH ANOMALY**. Corrected bird/birds IDs 11199/17952; whitespace ID 29871 rejected. Twenty corrected figures reviewed; preserve Vision-CLS Top-K/top-1 mismatch.
2. Phase 2: **PASS**. 240 official-training-only CUB images, 160/80 development splits, 12 safetensors shards, nine stages, 576 patches. Cache digest `63cf0e80ec0a24533682467b6f3b23ccded8625ef25d2fb43d012aa0e72179d8`.
3. Phase 3: the original full development computation passed, but its label-policy revalidation is now required. The Phase 4 full attempt exposed at least one selected-attribute cache row with a non-null target and certainty outside `probably`/`definitely`; the corrected Phase 3 loader masks such rows without re-extracting representations.
4. Phase 4: **paired-semantic one-image smoke PASS** on image 544. The first full attempt completed seven images and stopped safely at image 575 on the certainty inconsistency. The corrected runner masks and audits legacy target overrides; full Phase 4 must be rerun from the latest commit.

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

## Phase 4 implementation is ready; real semantic/full results are pending

The user requested completing Phase 4. The missing paired-semantic scorer and
matched development evaluator are now implemented, with synthetic-only local
validation. No real CLIP/model run or new Phase 4 result has been generated locally.

Run `notebooks/phase4_full_development_kaggle.ipynb` on Kaggle, both cells in order:

1. Fetch and verify latest `feat/iclr`; run `test_phase4.py`; execute the NEW paired
   CLIP semantic smoke on the smallest development-training image. This verifies
   global projection/normalization equivalence and all 27 query maps.
2. Run the full 240-image development localization only after its matching smoke
   passes, then validate/recompute summaries, show the validation tables/plots and
   export `phase4_development_results_bundle.zip` for review.

Inspect `unapproved_cached_targets_masked` and `phase3_revalidation_required` in
the returned Phase 4 report. Any override is excluded from Phase 4 eligibility.
Rerun Phase 3 with the corrected loader before Phase 5/6 comparisons; the existing
Phase 2 tensors remain reusable.

Implementation entry points:

- `configs/phase4_localization.json`: fixed 26 queries/attribute identities,
  explicit part proxies, K=16/32, random seeds 0/1/2, pinned paired CLIP revision.
- `src/lger/dense_clip.py`: frozen float32 CLIP paired image/text projections,
  contextual patch cosine, cached uint8 RGB normalization without spatial resampling.
- `src/lger/phase4.py`: metadata-only Phase 2 checks, corrected-map joining,
  positive/certain/visible/in-crop eligibility, matched metrics and aggregation.
- `scripts/run_phase4_localization.py`: smoke/development runner, same-protocol
  gate, per-image resume, source hashes, versions, qualitative panels and report.
- `scripts/validate_phase4_localization.py`: coverage/hash checks, recomputed
  summaries/paired deltas/agreement, readable results and small report ZIP.
- `planning/08_PHASE4_LOCALIZATION_PROTOCOL.md`: definitions, sources, exact
  artifact counts, claim boundaries and instructions.

The dense baseline is `openai/clip-vit-large-patch14-336` at
`ce19dc912ca5cd21c8a653c79e251e808ccabcd1`, using transformers 4.49.0. Its globally
trained head is applied to contextual patches as a diagnostic, not a dense
segmentation model. It is distinct from the quantized LLaVA vision tower.
Generic bird/birds and prompt-attention maps remain object-level controls;
dense-attribute maps use the named attribute query. Landmark proxies are not masks.

The full run keeps train and validation summaries separate. Random seeds are
averaged within image; macro attribute summaries weight evaluable attributes
equally and expose how many of the 26 attributes are evaluable. Do not change
queries or mappings based on the validation result. The six smallest validation
IDs supply qualitative examples, with the first eligible attribute by policy order.

Local validation includes an end-to-end synthetic smoke/resume, a synthetic
240-record coverage test (no pretrained inference), corruption/exclusion gates,
and visual inspection of rendered synthetic figures. Real Kaggle execution is
still necessary. Review its returned results before starting Phase 5, and do not
select a causal intervention from localization alone. Official test remains untouched.

## Operating constraints and paths

- Implement/inspect code and audit small result tables locally. Do not download models, datasets, checkpoints or large artifacts locally. Do not run actual VLM extraction or probe experiments locally.
- Kaggle must fetch/run the latest `feat/iclr`. Always run commands from `/kaggle/working/newpipeline/projects/logit_evidence_routing`, with `PYTHONPATH=src` as needed.
- Keep the model frozen, the 26-attribute selection unchanged, and the official CUB test split untouched until attributes, layers, prompts, parsers and metrics are frozen.
- The user explicitly requires repository changes to be committed and pushed to `feat/iclr`; preserve unrelated changes and verify the remote commit after pushing.
- Phase 1 corrected cache: `/kaggle/working/phase1b_corrected/cache`; gate: `/kaggle/working/phase1b_corrected/results/phase1_gate.json`.
- Phase 2 cache: `/kaggle/working/phase2_stage_cache`.
- Phase 3 full output: `/kaggle/working/phase3_development_1a6e0c96681f_20260906T171249251219Z`.
- Small source reports are checked into `reports/development_20260906/phase3_review_bundle` and `phase4_cached_localizer_smoke_bundle`; do not request them again.
- The new full Phase 4 notebook is in `notebooks/phase4_full_development_kaggle.ipynb`; the earlier notebooks preserve completed Phase 3/cached-map smoke runs.

The source bundle audit uses only the Python standard library:

```bash
PYTHONPATH=src python scripts/review_development_bundles.py \
  --phase3-bundle reports/development_20260906/phase3_review_bundle \
  --phase4-bundle reports/development_20260906/phase4_cached_localizer_smoke_bundle \
  --output-dir /kaggle/working/verified_phase3_phase4_review
```

It verifies returned tables and hashes, not source tensor contents or original image split membership. Keep that provenance boundary in later claims.
