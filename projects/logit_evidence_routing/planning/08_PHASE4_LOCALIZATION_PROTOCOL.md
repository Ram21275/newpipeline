# Complete Phase 4 development localization

Run the two code cells in order in the same GPU-enabled Kaggle kernel, with internet enabled for Git and the pinned CLIP checkpoint. Use the existing transformers==4.49.0 environment from Phases 1–3. All commands execute from `/kaggle/working/newpipeline/projects/logit_evidence_routing`.

Cell 1 fetches the latest `feat/iclr`, tests the new pipeline with synthetic inputs, and runs a NEW one-image paired-semantic smoke. This extends the already passed cached-map smoke by loading frozen CLIP on Kaggle, checking the paired projection/normalization against its global forward logits, and computing all 27 dense query maps. It does not load the 7B VLM or rerun Phase 2 extraction.

Cell 2 requires that matching smoke gate, scores all 240 development images, keeps train/validation summaries separate, verifies coverage and metrics, displays the validation tables and plots, and produces a ZIP containing reports and qualitative figures. It does not include model weights or dense score caches in the ZIP. The official CUB test split is never evaluated.

Outputs are commit-specific and resume using per-image score caches. If a cell is interrupted, rerun with the same checkout, protocol and input artifacts. Changed configuration or source hashes fail rather than silently mixing runs. A failed process stops the cell; do not bypass its gate. If the CUB input mount moved, specify only its small `parts/parts.txt` metadata path as described in Cell 1.

The fixed protocol is `configs/phase4_localization.json`. It contains the existing 26 attribute identities, text queries, explicit landmark-proxy mappings, K=16/32 and random seeds 0/1/2. These choices precede the Phase 4 full development result. Do not tune them using its validation outcomes.

# What the results mean

There are two matched evaluations. Object localization compares Vision-CLS attention, LLM attention, corrected generic bird/birds Logit Lens, fusion, dense CLIP bird-query similarity, and random selection. Attribute localization adds the dense query for each named attribute, evaluated only on sufficiently certain positive labels with relevant visible in-crop landmark proxies. Generic attention/bird maps remain labeled generic controls; they are not attribute-conditioned maps.

The dense model is `openai/clip-vit-large-patch14-336` at revision `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`. It computes cosine similarity between text embeddings and final patch states after CLIP's paired post-layer-normalization and visual projection. CLIP was trained with global image/text alignment; applying its head to contextual patches is a diagnostic, not dense-supervised segmentation. The RGB inputs are the already cropped uint8 images stored by Phase 1; no second resize/crop occurs.

Metrics include broad-box concentration, patch recall/IoU, top-1 pointing, visible/relevant-part patch recall, top-1 distance, selector agreement and paired differences versus random and the generic dense object query. Distinct random seeds are averaged within image. Macro attribute scores weight evaluable attributes equally, and the support table retains all 26 selected attributes, including those with zero eligible images. Missing/occluded parts are unavailable rather than zeros. SD describes image variability; no final-test confidence intervals are claimed.

The qualitative figures use fixed IDs: the smoke training image and the six smallest validation IDs. Each attribute panel uses the first eligible attribute in policy order, never the highest-scoring outcome. Heatmaps are independently scaled for display; cyan marks Top-32, yellow the top-1 patch, and green the ground-truth landmark proxies and bird box.

After both cells pass, attach `phase4_development_results_bundle.zip`. The numeric and qualitative results must be reviewed before Phase 5. Do not infer a causal bottleneck, evidence relocation, or VLM utilization from these localization measurements alone.

Implementation validation performed locally uses synthetic tensors/metadata and mocked pretrained scoring, plus rendered synthetic figures. No pretrained model weights, CUB data, actual VLM extraction, or actual probe experiments were downloaded/run locally. Real CLIP execution and Phase 4 measurements remain Kaggle work.


## Source/API verification

- [Pinned paired CLIP configuration](https://huggingface.co/openai/clip-vit-large-patch14-336/blob/ce19dc912ca5cd21c8a653c79e251e808ccabcd1/config.json): 336-pixel input, 14-pixel patches, 1024-dimensional vision states, 768-dimensional paired projection.
- [Transformers 4.49 CLIP implementation](https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/models/clip/modeling_clip.py): final vision hidden states are returned before post-layer normalization; the pooled CLS and text features pass through their paired heads before cosine similarity. The smoke compares manual global logits against this official forward path.

## Artifacts and exact coverage

- `phase4_run_report.json`, `phase4_validation_report.json`, `evaluation_config.json`: computational gates, immutable protocol, dependency versions, commits and input hashes.
- `object_metrics.csv`: 240 × 2 K values × (5 deterministic maps + 3 random seeds) = 3,840 rows.
- `attribute_eligibility.csv`: 240 × 26 = 6,240 rows, with explicit exclusion reasons.
- `attribute_metrics.csv`: 18 × E rows, where E is the eligible image/attribute pair count.
- `selector_agreement.csv`: 13,440 + 72 × E rows; all unordered equal-K pairs, including random-seed pairs.
- `localization_summary.csv`, `attribute_macro_summary.csv`, `attribute_support.csv`, `paired_selector_deltas.csv`, `selector_agreement_summary.csv`: separate development-train/validation summaries and denominators. No group or method pools train and validation results.
- `source_records.csv` and `dense_scores/`: per-image input/output hashes and resumable scores; score tensors stay on Kaggle.
- `PHASE4_RESULTS.md`, `localization_overview.png`, `selector_agreement.png`, `qualitative/`: readable validation results and fixed qualitative examples.

After execution, `quantitative_evaluation_complete=true` denotes the full development computation. `scientific_review_pending=true` remains until the returned reports and examples are inspected. The implementation does not fabricate or prefill a Phase 4 result and does not start Phase 5 automatically.
