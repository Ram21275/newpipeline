# Verified Phase 3 results and Phase 4 cached-localizer smoke

Review date: 6 September 2026. Experiment commit: `1a6e0c96681f250659ce703207931289cd9112c2`.

## Gate decision

- Phase 1: PASS WITH ANOMALY, previously audited. Preserve the Vision-CLS Top-K concentration versus poor top-1 pointing anomaly.
- Phase 2: PASS, 240 development images (160 train / 80 validation), 12 shards, nine stages, 576 patches, 1024-dimensional vision states and 4096-dimensional projector/LLM states. Frozen cache digest: `63cf0e80ec0a24533682467b6f3b23ccded8625ef25d2fb43d012aa0e72179d8`.
- Phase 3 full development trajectory: PASS after independent review of the returned raw CSV/JSON bundle.
- Phase 4 one-training-image cached-localizer reuse smoke: PASS. **Full Phase 4 remains incomplete.**
- Official CUB test use: reported zero throughout. The current review reads result files; it does not reload source image/tensor caches or independently repeat the original split-membership audit.

The returned bundle passes all 13 manifest file hashes/sizes, 90 unique run combinations, 2,340 per-attribute rows, fixed identities/supports for all 26 attributes, macro-versus-attribute consistency, finite metrics, finite fitted losses, feature dimensions, seed summaries, and paired difference means/SDs. Prevalence loss is NaN by design and its seed SD is undefined (n=1). The Phase 4 tables contain exactly 14 localization rows and 42 unordered pairwise comparisons for training image 544. Source reports and derived tables are checked in alongside this review; no weights, hidden-state tensors, original images, or datasets are included.

## Fixed Phase 3 protocol

Frozen LLaVA `llava-hf/llava-1.5-7b-hf`, pinned extraction revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`, 4-bit extraction, prompt `Describe the image briefly.` Nine cached stages, mean pooling, primary/prevalence/shuffled-label/random-projection controls, probe seeds 0/1/2, 300 epochs, learning rate 0.01, weight decay 0.0001, CUDA, random-projection dimension 256. Normalization and F1 thresholds use development training only. Attribute selection is unchanged and training-only; probably/definitely labels are observed and uncertain/missing states are masked.

Full primary macro-AUROC: vision early 0.5561, middle 0.5953, late 0.6682, final 0.6718; projector 0.6516; LLM early 0.6618, middle 0.6645, late 0.6608, final 0.6671. Shuffled means are near chance. Random projection preserves much of the signal because it compresses genuine features and is followed by a probe trained on genuine targets; it is not a null control. F1 is threshold-sensitive and does not follow an identical curve.

## Macro stability conceals attribute heterogeneity

The following group means weight the selected attributes equally within each group; they do not replace the 26-attribute macro metric. Group sizes differ. Values are means across seeds using full-precision source rows.

| Group | Attributes | Vision late | Projector | LLM final | LLM final − vision late |
|---|---:|---:|---:|---:|---:|
| has_bill_shape | 4 | 0.8282 | 0.8490 | 0.7533 | -0.0749 |
| has_breast_color | 5 | 0.5916 | 0.5546 | 0.5966 | +0.0050 |
| has_crown_color | 6 | 0.6884 | 0.6583 | 0.6947 | +0.0063 |
| has_head_pattern | 7 | 0.6244 | 0.5812 | 0.6171 | -0.0073 |
| has_wing_pattern | 4 | 0.6506 | 0.6883 | 0.7154 | +0.0648 |

Bill-shape accessibility decreases while wing-pattern accessibility increases. Thus “all attributes survive unchanged” is not supported by the nearly flat global macro curve. The projector decrease is also not uniform: bill and wing group averages increase at the projector, while breast, crown and head-pattern averages decrease.

Examples from the full, unfiltered attribute transition table: dagger-shaped bill decreases about 0.1343 AUROC; cone-shaped bill decreases 0.0863; spotted wing increases 0.1656; eyeline increases 0.1567. These are exploratory development contrasts, not selected discoveries or independent hypothesis tests. The full table retains all 26 attributes, including unchanged and contradictory cases. Attribute validation positive counts range from 3 to 41; rare-attribute extremes and the perfect blue-crown AUROC require caution. Do not change the frozen attribute set based on these results.

This supports **heterogeneous changes in measured linear accessibility**. It does not establish physical movement of evidence between tokens, a causal visual-language bottleneck, or a representation–utilization gap. Those require the planned localization and generated-answer measurements. Seed SD describes readout stability on the same images, not image-sampling uncertainty or confidence intervals.

## Correct stage topology

The returned smoke metadata confirms vision hidden-state indices 6/12/23/24 and LLM indices 8/16/31/32. The extractor defines vision.late from `model.config.vision_feature_layer`, so the projector input is vision layer 23; vision.final at layer 24 is an additional diagnostic, not the direct projector predecessor.

The adjacent projector comparison is 0.6682 → 0.6516 (about -0.0167 macro-AUROC). The supplied nine-point plot includes vision.final before the projector as a display order, not a literal computation edge. LLM final returns to about 0.6671, but this does not prove information equivalence or recovery of the same attributes. Probes observe visual-token states from the first full prompt pass, not information in generated answer tokens.

## Phase 4 smoke: engineering evidence only

Fixed development-training image 544 passed the corrected-cache identity, model/prompt/revision, patch geometry, mapped-box, finite-score, ranking, and layer-index checks on Kaggle. The returned table/metadata audit passes locally. It validates reuse of the existing maps with Phase 2 spatial metadata; it is not a comparative result over validation images.

At K=32 on this image, Vision-CLS selects 84.375% of patch centers inside the box and recalls 42.857% of the 14 visible-part patches, yet its top-1 is outside the box and 7.165 patch units from the nearest visible part. Corrected generic bird/birds Logit Lens has 81.25% box concentration, 21.429% part-patch recall, and top-1 distance 0.117 patches. LLM attention and fusion have top-1 distances about 7.202 patches. These numbers illustrate why broad boxes, Top-K coverage, and top-1 precision must be reported separately; they do not establish selector superiority from one training image.

Part-patch recall counts distinct patches containing visible annotated landmarks, not landmark instances. Generic bird/birds semantic scores locate object evidence, not specific crown colours, bill shapes, or eye stripes.

## Next gate: Phase 4 semantic/localization one-image smoke

1. Reuse the validated corrected attention/concept maps and Phase 2 geometry; no repeat of the full VLM stage extraction is needed for those maps.
2. Implement a dense image–text similarity baseline in a verified paired embedding space, pinning its model/revision, visual/text projection heads, normalization, patch scoring and preprocessing. Equal dimensions do not make arbitrary vision states and text embeddings comparable. Start on a deterministically selected development-training image in Kaggle before any full extraction.
3. Define descriptions and attribute-to-part mappings for the existing 26 attributes. Restrict positive-attribute localization to sufficiently certain positives with relevant landmarks visible and in crop; report missing/occluded/excluded counts and per-metric denominators. Head/eye landmarks are proxies, not annotated stripe masks. Broad-box and all-visible-part metrics remain separate.
4. Keep K=16/32 and random seeds 0/1/2. Evaluate selectors on the same eligible records for each attribute and metric. Do not use annotations to select evidence patches. Freeze descriptions, mappings and exclusions before examining full validation selector outcomes.
5. Preserve Logit Lens compatibility: no direct LM head on vision states; use the frozen final norm/head on compatible intermediate LLM states without double-normalizing final-normalized outputs. Multi-token attributes need dense similarity or an appropriate sequence score, not summed independent token marginals.

The dense baseline and attribute-part protocol are **not implemented by this result-recording change**. The current commit records completed work and makes its audit reproducible. After the Phase 4 semantic smoke and matched development comparison, proceed to Phase 5 fixed-question utilization. Do not select a causal intervention or write Phase 6 `INTERMEDIATE_FINDINGS.md` yet. Official test evaluation remains reserved for the frozen final protocol.

## Reproduce this review (no extraction or probe training)

Run from `/kaggle/working/newpipeline/projects/logit_evidence_routing` after fetching the latest `feat/iclr`:

```bash
PYTHONPATH=src python scripts/review_development_bundles.py \
  --phase3-bundle reports/development_20260906/phase3_review_bundle \
  --phase4-bundle reports/development_20260906/phase4_cached_localizer_smoke_bundle \
  --output-dir /kaggle/working/verified_phase3_phase4_review
```

The script uses only the Python standard library and produces the 234-row primary attribute trajectory, 45-row group trajectory, 52-row paired attribute transition table, and an audit report. Original Kaggle cells are versioned in `notebooks/phase3_development_kaggle.ipynb` and `notebooks/phase4_entry_cached_localizer_smoke.ipynb`; both stages have already run. Raw source results are immutable evidence; generated review tables do not overwrite them.
