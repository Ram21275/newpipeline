# Next-Chat Handoff Plan

**Project:** Fine-Grained Evidence Tracing in Vision-Language Models  
**Target:** ICLR 2027  
**Prepared:** 5 September 2026  
**Repository branch:** `feat/iclr`  
**Project directory:** `projects/logit_evidence_routing`  
**Current checked-out commit:** `623f9a7ad01ed813a991a2aacd8e688aec7bf9fb`

## Copy-paste prompt for the next chat

```text
Continue the ICLR 2027 fine-grained evidence-tracing project in this repository.

Repository root:
/Users/ramdabas/Downloads/newpipeline

Project directory:
projects/logit_evidence_routing

Branch:
feat/iclr

First read, in order:
1. projects/logit_evidence_routing/NEXT_CHAT_HANDOFF.md
2. projects/logit_evidence_routing/reports/PHASE_01B_TECHNICAL_REPORT.md
3. projects/logit_evidence_routing/planning/README.md
4. projects/logit_evidence_routing/planning/00_MASTER_PLAN.md
5. projects/logit_evidence_routing/planning/01_SIGNAL_VALIDATION.md
6. projects/logit_evidence_routing/planning/02_DATA_AND_CACHE_PIPELINE.md

Start with `git status --short --branch` and preserve all existing uncommitted
changes. Do not begin the stage-aligned representation cache until the corrected
Phase 01 sanity report has been generated and inspected. The currently downloaded
localization table is from the legacy cache and contains invalid token ID 29871;
do not use its logit_concept or attention_logit_fusion rows.

Immediate objective:
1. Inspect the corrected Kaggle result bundle, including metadata, quantitative
   localization, visible-part metrics, qualitative figures, and
   phase1_sanity_report.md.
2. Decide PASS, PASS WITH ANOMALY, or STOP / INVESTIGATE using the written gate.
3. Update the formal report with only verified corrected artifacts.
4. If and only if the gate passes, design and implement a one-image smoke test
   for the stage-aligned representation cache. Do not launch the full 240-image
   extraction until the schema, resumption checks, storage estimate, and tests
   pass locally.

Keep the VLM frozen. Treat attention, linear accessibility, semantic readability,
and causal utilization as distinct measurements. Logit Lens is one diagnostic,
not the proposed contribution. Do not use the official CUB test split until the
attribute set, prompts, layer choices, parser, and metrics are frozen.

At the end of the task, report files changed, tests run, exact experiment settings,
quantitative results, anomalies, remaining blockers, and the next gate. Stop before
starting another major experimental phase.
```

## 1. Current research objective

The project no longer aims to prove that Logit Lens is the best patch localizer.
It asks:

> Where does fine-grained visual evidence exist inside a frozen VLM, how does
> that evidence transform through the vision encoder, multimodal projector, and
> language model, and does normal answer generation use information that remains
> internally recoverable?

The same CUB images and attributes must be traced through:

```text
image -> vision layers -> projector -> LLM layers -> generated answer
```

Four measurements must remain separate:

1. spatial or discriminative importance;
2. linear accessibility;
3. direct semantic readability;
4. causal utilization.

The final contribution should be the strongest observed cross-stage phenomenon,
not a preselected tool or architecture.

## 2. Repository state at handoff

At preparation time:

```text
branch: feat/iclr
HEAD:   623f9a7 Clarify Kaggle imports and evidence tracing roadmap
remote: origin/feat/iclr at the same commit
```

The working tree contains intentional uncommitted work:

```text
M  projects/logit_evidence_routing/README.md
?? projects/logit_evidence_routing/reports/PHASE_01B_TECHNICAL_REPORT.md
?? projects/logit_evidence_routing/NEXT_CHAT_HANDOFF.md
```

The next chat must inspect and preserve these files. Do not reset or overwrite
them. The local test suite currently passes all 39 tests.

## 3. Completed implementation

The repository currently provides:

- official-split-aware CUB pilot preparation;
- frozen LLaVA 1.5 7B extraction on Kaggle;
- 4-bit `bitsandbytes` runtime validation;
- patch-aligned late-LLM states and score maps;
- Random, LLM-attention, Vision-CLS, vision-rollout, generic-logit,
  concept-logit, fusion, and global-pooling selectors;
- matched linear species probes;
- broad-box localization metrics;
- visible-part audit support;
- qualitative selector heatmaps;
- protection against legacy concept caches;
- non-destructive repair of the concept-tokenization error;
- automated Phase 01 sanity reporting.

Important implementation entry points:

- [`scripts/prepare_cub_pilot.py`](scripts/prepare_cub_pilot.py)
- [`scripts/extract_phase1b_localizers.py`](scripts/extract_phase1b_localizers.py)
- [`scripts/repair_phase1b_concepts.py`](scripts/repair_phase1b_concepts.py)
- [`scripts/run_phase1b_benchmark.py`](scripts/run_phase1b_benchmark.py)
- [`scripts/plot_phase1b_localizers.py`](scripts/plot_phase1b_localizers.py)
- [`scripts/write_phase1_sanity_report.py`](scripts/write_phase1_sanity_report.py)
- [`src/lger/hf_llava.py`](src/lger/hf_llava.py)
- [`src/lger/phase1b.py`](src/lger/phase1b.py)
- [`src/lger/probe.py`](src/lger/probe.py)
- [`src/lger/localization.py`](src/lger/localization.py)
- [`src/lger/scoring.py`](src/lger/scoring.py)

## 4. Current experimental evidence

### 4.1 Development-pilot protocol

- Dataset: CUB-200-2011.
- Pilot: 20 species classes.
- Development train: 160 images, 8 per class.
- Development validation: 80 images, 4 per class.
- Intended source: official CUB training partition only.
- Model: frozen `llava-hf/llava-1.5-7b-hf`.
- Representation: 4,096-dimensional visual-token states at LLM
  `layer_offset=-2`.
- Spatial positions: 576 patches.
- K: 16 and 32.
- Prompt: `Describe the image briefly.`
- Probe seeds: 0, 1, 2.
- Random-selection seeds: 0, 1, 2.
- Corrected fixed concepts: `bird`, `birds`.

Official-split and cache identity are intended by construction but remain part
of the pending Phase 01 audit.

### 4.2 Corrected probe output supplied from Kaggle

| Selector | K=16 accuracy | K=16 macro-F1 | K=32 accuracy | K=32 macro-F1 |
|---|---:|---:|---:|---:|
| Random | 40.8% | 38.1% | 54.4% | 53.4% |
| LLM attention | 70.0% | 68.9% | 68.8% | 68.7% |
| Vision-CLS attention | **92.5%** | **92.3%** | **95.0%** | 94.8% |
| Vision attention rollout | 50.0% | 50.4% | 55.0% | 54.9% |
| Logit maximum probability | 48.8% | 48.1% | 47.5% | 48.7% |
| Logit margin | 48.3% | 47.2% | 52.5% | 53.8% |
| Logit negative entropy | 49.6% | 49.3% | 53.8% | 53.4% |
| Corrected logit concept | 91.3% | 91.1% | **95.0%** | **95.1%** |
| Attention-logit fusion | 88.8% | 88.6% | **95.0%** | 94.8% |
| Global all-patch mean | 66.3% | 65.7% | 66.3% | 65.7% |

Bounded interpretation:

- Vision-CLS and corrected fixed-concept scores are strong routers for late-LLM
  states on this development pilot.
- Generic vocabulary confidence is not enough; semantic direction matters.
- Fusion has not improved on the strongest individual selector.
- These are species-accessibility results, not attribute localization or causal
  utilization results.

### 4.3 Valid unaffected localization observation

The concept repair changes only `logit_concept`, `attention_logit_fusion`, their
selections, and their derived pooled features. Random and Vision-CLS rows from
the earlier localization run are unaffected:

- Vision-CLS Top-16 inside-bird-box fraction: 73.7%, versus 47.4% random.
- Vision-CLS Top-32 inside-bird-box fraction: 80.8%, versus 47.1% random.
- Vision-CLS pointing game: 37.5%, versus 42.9% random.

The high Top-K box concentration and weak top-1 pointing rate form an explicit
anomaly to preserve.

### 4.4 Invalid/stale artifact warning

The result bundle previously inspected under local `Downloads/kaggle 2` was not
the corrected bundle. Its `evaluation_config.json` pointed to:

```text
/kaggle/working/phase1b/cache
```

and listed concept token IDs:

```text
[11199, 17952, 29871]
```

Token 29871 is a standalone SentencePiece whitespace marker. Therefore, do not
use that bundle's `logit_concept` or `attention_logit_fusion` localization rows.
The corrected probe console output is promising, but it does not replace the
missing corrected metadata and localization bundle.

## 5. Immediate Gate 1 plan

### Step 1: generate corrected qualitative figures on Kaggle

```bash
cd /kaggle/working/newpipeline/projects/logit_evidence_routing

PYTHONPATH=src python scripts/plot_phase1b_localizers.py \
  --cache-dir /kaggle/working/phase1b_corrected/cache \
  --output-dir /kaggle/working/phase1b_corrected/results/qualitative \
  --k 32 \
  --count 20
```

The current plotter selects the 20 validation images with the lowest
LLM-attention/concept Jaccard overlap. These are disagreement cases, not a
random representative sample.

### Step 2: generate the Phase 01 sanity report

```bash
PYTHONPATH=src python scripts/write_phase1_sanity_report.py \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --cache-dir /kaggle/working/phase1b_corrected/cache \
  --results-dir /kaggle/working/phase1b_corrected/results \
  --search-root /kaggle/input
```

### Step 3: package all audit artifacts

```bash
cd /kaggle/working
zip -r phase1b_corrected.zip \
  phase1b_corrected/results \
  phase1b_corrected/cache/extraction_config.json \
  phase1b_corrected/cache/correction_summary.json
```

Attach or copy `phase1b_corrected.zip` into the next chat/workspace.

### Step 4: validate the corrected bundle

The next chat should inspect, at minimum:

```text
phase1b_corrected/cache/extraction_config.json
phase1b_corrected/cache/correction_summary.json
phase1b_corrected/results/evaluation_config.json
phase1b_corrected/results/selector_metrics.csv
phase1b_corrected/results/selector_summary.csv
phase1b_corrected/results/validation_predictions.csv
phase1b_corrected/results/localization_metrics.csv
phase1b_corrected/results/localization_summary.csv
phase1b_corrected/results/qualitative/index.csv
phase1b_corrected/results/qualitative/*.png
phase1b_corrected/results/phase1_sanity_report.md
phase1b_corrected/results/vision_cls_part_localization.csv
```

Required correction checks:

- `concept_tokenization_policy == "single_lexical_token_v1"`;
- concept tokens are lexical `bird` and `birds` entries;
- concept token IDs exclude 29871;
- evaluation `cache_dir` points to `phase1b_corrected/cache`;
- cache and manifest IDs, paths, labels, classes, and splits match;
- train and validation image IDs and paths are disjoint;
- no official CUB test image appears in the development pilot;
- qualitative images exist;
- visible-part metrics cover every validation image;
- the sanity report records `PASS` or explains every anomaly.

### Gate 1 decision

- **PASS:** all integrity checks pass; freeze Phase 01 choices and continue.
- **PASS WITH ANOMALY:** integrity passes, but preserve the weak top-1 pointing
  result or other nonfatal anomaly in every later interpretation.
- **STOP / INVESTIGATE:** any split, manifest, cache identity, correction,
  duplicate-feature, or coverage check fails. Diagnose before implementing the
  stage cache.

### Gate 1 deliverables

- verified corrected result bundle;
- reviewed qualitative figures;
- `phase1_sanity_report.md`;
- `vision_cls_part_localization.csv`;
- corrected localization interpretation;
- updated [`reports/PHASE_01B_TECHNICAL_REPORT.md`](reports/PHASE_01B_TECHNICAL_REPORT.md).

## 6. Gate 2 plan: stage-aligned representation cache

Begin this section only after Gate 1 passes.

### Step 1: write the schema before the extractor

Create `representation_cache_schema.md` defining:

- image, class, official split, and dataset version identifiers;
- CUB attribute labels, presence/certainty, visible parts, and coordinates;
- original image size, processed size, crop transform, patch grid, and patch
  centers;
- exact vision, projector, and LLM stage names and resolved indices;
- tensor shape, dtype, and token correspondence at every stage;
- fixed prompts, generated answers, parsed answers, and answer-token logits;
- model revision, repository commit, extraction configuration, runtime, and
  peak memory;
- shard/index layout, atomic writes, resumption rules, and full-config
  validation.

Do not choose a storage format until one-image sizing is measured. Prefer a
small number of indexed shards using safetensors, HDF5, or chunked memmap. Avoid
one monolithic file and thousands of tiny per-layer files.

### Step 2: fill the current data-loader gap

The code already loads CUB classes, boxes, split metadata, and part locations.
It does not yet implement the official attribute-label pipeline. Inspect the
attached CUB distribution's exact attribute files, then add:

- an explicit attribute vocabulary;
- per-image presence labels and certainty;
- missing/uncertain-label handling;
- tests with a minimal synthetic CUB fixture;
- a documented attribute subset chosen using training data only.

Do not invent class descriptions or attribute-to-part mappings after looking at
validation results.

### Step 3: implement stage extraction

Resolve indices from the actual model configuration and initially store:

- vision early, middle, late, and final patch states;
- projector output before insertion into the LLM;
- LLM early, middle, late, and final visual-token states;
- fixed-prompt answer text and answer-token logits.

Required invariants:

- the entire VLM stays frozen;
- stable spatial patch identity is preserved or an exact mapping is stored;
- labels never influence feature extraction or patch selection;
- all tensors are detached, finite, and shape checked;
- resumption rejects config mismatches;
- incomplete records are written atomically and never mistaken for complete
  records.

### Step 4: local tests and one-image Kaggle smoke test

Local tests should cover schema validation, stage naming, layer resolution,
spatial mapping, atomic resumption, finite values, and mocked extraction.

Then run one image on Kaggle and report:

- resolved stage indices;
- shapes and dtypes;
- bytes per image and projected 240-image storage;
- runtime per image and projected runtime;
- peak GPU memory;
- successful reload and patch alignment across every stored stage.

### Step 5: decision before full extraction

Only after the smoke test passes should the next chat provide the command for
the 240-image development extraction. Stop and report if storage, alignment, or
memory is unsafe.

## 7. Main experiments after the stage cache

### Experiment 1: attribute recoverability by stage

Train the same linear multi-label probe at each compatible stage. Primary
metrics are per-attribute AUROC, macro-AUROC, and macro-F1. Required controls
include prevalence, shuffled labels, random projection, pooled versus
patch-local features, matched regularization, and learning curves.

Deliver:

```text
results/attribute_probe_by_stage.csv
figures/attribute_probe_by_stage.png
```

Primary question: where is the largest change in linear accessibility?

### Experiment 2: spatial versus semantic localization

Compare equal-K Vision-CLS attention, precisely defined LLM attention, dense
image-text similarity, and audited Logit Lens where compatible. Report broad
bird-box metrics separately from visible-part and attribute-relevant metrics.

Deliver:

```text
results/localization_by_tool.csv
```

Primary question: are discriminative routing and semantic readability aligned?

### Experiment 3: representation versus normal VLM use

Compare linear attribute accessibility with fixed, parseable VLM attribute and
species answers. Include prompt-only and image-shuffled controls.

Deliver:

```text
results/vlm_answer_vs_probe.csv
```

Primary question: does a consistent accessibility-answer gap exist?

### Gate 4: identify one transition

Write `INTERMEDIATE_FINDINGS.md` with the strongest supported observation,
exact numbers and uncertainty, alternative explanations, best-supported member
of H1-H4, and one minimal causal test. Allow a null or mixed result.

### Experiment 4: targeted causal intervention

Only after `INTERMEDIATE_FINDINGS.md`, remove or replace the smallest supported
evidence set at the identified stage. Compare with random controls matched by
token count, spatial distribution where possible, and replacement magnitude.
Measure answer-logit and accuracy changes and repeat at one neighboring layer.

## 8. Hypothesis decision table

| Hypothesis | Supporting pattern | Confound to rule out |
|---|---|---|
| H1: visual-to-language bottleneck | Attribute accessibility drops sharply at vision-to-projector or projector-to-LLM transition | Incompatible normalization or readout basis |
| H2: representation-utilization gap | Strong LLM-state probes but weak fixed-prompt VLM answers | Prompt or parser failure |
| H3: discriminative differs from semantic localization | Vision routing or attribute probes succeed while semantic-map overlap/part alignment is weak | Inappropriate semantic probe |
| H4: evidence transforms or moves | Local accessibility falls while pooled accessibility remains stable and verified cross-stage correspondence shifts | Broken patch/token alignment |

## 9. Claim and evaluation guardrails

- Attention is not a causal explanation.
- Probe accuracy measures accessibility, not model use.
- Whole-bird boxes do not establish part or attribute localization.
- Generic logit confidence does not establish semantic identity.
- A selector comparison cannot isolate a representational stage.
- The 240-image pilot is development evidence because it has informed method
  choices.
- Probe-seed variance is not sampling uncertainty; use paired per-image analysis
  and bootstrap confidence intervals.
- Do not use the official CUB test split until the analysis protocol is frozen.
- Do not add a sparse autoencoder until simpler measurements reveal a robust but
  semantically opaque transition.
- Do not begin causal interventions before `INTERMEDIATE_FINDINGS.md`.

## 10. Definition of done for the next chat

The next chat should complete one of these bounded outcomes:

### Preferred outcome

- corrected Phase 01 bundle inspected;
- correction and integrity checks pass;
- quantitative and qualitative anomalies documented;
- formal report updated;
- Phase 01 decision recorded;
- stage-cache schema and one-image implementation started only if the gate
  passes.

### Valid blocked outcome

- missing or invalid corrected artifacts identified precisely;
- exact Kaggle rerun/download command supplied;
- no stage-cache work started;
- blocker and required user artifact stated clearly.

### Valid stop outcome

- sanity report returns `STOP / INVESTIGATE`;
- root cause is diagnosed or narrowed;
- no later experiment is used to bypass the failed gate.

## 11. Canonical references

Read these files instead of reconstructing the project from chat history:

1. [`reports/PHASE_01B_TECHNICAL_REPORT.md`](reports/PHASE_01B_TECHNICAL_REPORT.md) — formal architecture, experiment, result, and roadmap report.
2. [`planning/README.md`](planning/README.md) — canonical research direction and evidence boundary.
3. [`planning/00_MASTER_PLAN.md`](planning/00_MASTER_PLAN.md) — hypotheses, gates, timeline, and claims.
4. [`planning/01_SIGNAL_VALIDATION.md`](planning/01_SIGNAL_VALIDATION.md) — immediate sanity-gate contract.
5. [`planning/01B_LOCALIZATION_SELECTOR_VALIDATION.md`](planning/01B_LOCALIZATION_SELECTOR_VALIDATION.md) — selector benchmark and tokenization correction.
6. [`planning/02_DATA_AND_CACHE_PIPELINE.md`](planning/02_DATA_AND_CACHE_PIPELINE.md) — stage-cache requirements.
7. [`planning/03_BASELINES.md`](planning/03_BASELINES.md) — matched probes and controls.
8. [`planning/04_METHOD_LOGIT_EVIDENCE_ROUTING.md`](planning/04_METHOD_LOGIT_EVIDENCE_ROUTING.md) — measurement framework; filename retained for compatibility.
9. [`planning/05_MAIN_EXPERIMENTS.md`](planning/05_MAIN_EXPERIMENTS.md) — main experimental program.
10. [`planning/06_ABLATIONS_AND_ANALYSIS.md`](planning/06_ABLATIONS_AND_ANALYSIS.md) — controls and alternative explanations.
11. [`planning/07_ABSTRACT_DEADLINE.md`](planning/07_ABSTRACT_DEADLINE.md) — paper claim gates.

## 12. Required end-of-task report format

Every future chat or experiment should finish with:

1. outcome and gate status;
2. files created or modified;
3. exact model, revision, data split, prompt, stages, seeds, and hyperparameters;
4. quantitative results and uncertainty;
5. qualitative observations;
6. anomalies and alternative explanations;
7. supported and unsupported claims;
8. tests and integrity checks run;
9. artifacts still required from Kaggle or the user;
10. the single next authorized phase.

