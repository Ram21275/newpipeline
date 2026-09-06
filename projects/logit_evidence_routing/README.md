# Fine-Grained Evidence Tracing in Vision-Language Models

This is the implementation workspace for the ICLR 2027 project in
[`planning/`](planning/README.md). The directory keeps its earlier
`logit_evidence_routing` name so existing Kaggle paths and caches remain valid.
The research question has changed: Logit Lens is now one diagnostic, not the
proposed method.

## Current question

Where does fine-grained visual evidence exist across LLaVA's vision encoder,
projector, and language model; how does its form change; and does answer
generation use information that remains internally recoverable?

The study separates:

- spatial/discriminative importance;
- linear accessibility of CUB attributes;
- direct semantic readability;
- causal use by the final answer.

## Current Phase 01 status

The valid development-pilot result is that Vision-CLS attention is a strong
router for late LLM patch states: Top-32 gives 95.0% species-probe accuracy,
versus 54.4% for Random-32 and 66.3% for all-patch mean pooling. Its Top-32 patch
centers are inside the broad bird box 80.8% of the time versus 47.1% for random.

Important boundaries:

- the classifier consumes late LLM states, not raw vision-layer features;
- three runs are probe seeds on one fixed development split;
- broad box overlap is not fine-grained part/attribute localization;
- old `logit_concept` and `attention_logit_fusion` rows are invalid because the
  concept set included standalone whitespace token 29871.

The corrected code rejects nonlexical and multi-token concept entries, records
decoded concept tokens, and refuses to benchmark legacy concept caches.

The corrected Kaggle probe run subsequently reached 91.3%/95.0% accuracy for
`logit_concept` at K=16/K=32 and 88.8%/95.0% for
`attention_logit_fusion`. The supplied corrected result bundle, sanity report,
and all 20 disagreement figures have now been audited. The gate is
`PASS WITH ANOMALY`; these remain development-pilot findings rather than held-out
test results. The full
architecture, result, evidence-boundary, and forward-test report is available
in [`reports/PHASE_01B_TECHNICAL_REPORT.md`](reports/PHASE_01B_TECHNICAL_REPORT.md).
To continue this work in a new chat, use the self-contained
[`NEXT_CHAT_HANDOFF.md`](NEXT_CHAT_HANDOFF.md).

## Local validation

```bash
cd projects/logit_evidence_routing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python scripts/run_synthetic_pilot.py
```

## Kaggle setup

Create a notebook with a GPU accelerator, enable Internet, and attach an official
CUB-200-2011 dataset containing `images/`, `images.txt`,
`image_class_labels.txt`, `train_test_split.txt`, `bounding_boxes.txt`, and
`parts/part_locs.txt`.

Clone the public feature branch:

```python
!git clone --branch feat/iclr --single-branch \
  https://github.com/Ram21275/newpipeline.git \
  /kaggle/working/newpipeline

%cd /kaggle/working/newpipeline/projects/logit_evidence_routing
!python -m pip install -r requirements-kaggle.txt
!python -m pip install -e . --no-deps
!python -c "import lger; print(lger.__file__)"
!PYTHONPATH=src python -m unittest discover -s tests -v
```

For an existing clone:

```python
%cd /kaggle/working/newpipeline
!git pull --ff-only origin feat/iclr
%cd projects/logit_evidence_routing
!python -m pip install -r requirements-kaggle.txt
!python -m pip install -e . --no-deps
!python -c "import lger; print(lger.__file__)"
!PYTHONPATH=src python -m unittest discover -s tests -v
```

If `unittest` reports `ModuleNotFoundError: No module named 'lger'`, the local
package was not installed into the Python process running the tests. The
explicit editable-install and `PYTHONPATH=src` commands above make both routes
unambiguous; the import check should print a path ending in `src/lger/__init__.py`.

## Correct the existing Phase 01B cache

If `/kaggle/working/phase1b/cache` still contains the 240 `.pt` records, reuse
their cached hidden states. Start with two records:

```python
!python scripts/repair_phase1b_concepts.py \
  --source-cache-dir /kaggle/working/phase1b/cache \
  --output-dir /kaggle/working/phase1b_corrected/cache \
  --fixed-concepts bird birds \
  --max-images 2
```

Then resume all records by running the same command without `--max-images 2`:

```python
!python scripts/repair_phase1b_concepts.py \
  --source-cache-dir /kaggle/working/phase1b/cache \
  --output-dir /kaggle/working/phase1b_corrected/cache \
  --fixed-concepts bird birds
```

This still loads LLaVA's frozen norm/head, but performs no image or full VLM
forward pass. It writes a new cache; the original is never edited in place.

If the old cache no longer exists, create a corrected cache from the images:

```python
!python scripts/extract_phase1b_localizers.py \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --search-root /kaggle/input \
  --output-dir /kaggle/working/phase1b_corrected/cache \
  --model llava-hf/llava-1.5-7b-hf \
  --layer-offset -2 \
  --fixed-concepts bird birds \
  --k 16 32 \
  --random-seeds 0 1 2
```

The config should report two lexical tokens and
`"concept_tokenization_policy": "single_lexical_token_v1"`. It must not contain
standalone token `▁` / ID 29871.

## Regenerate matched results

```python
!python scripts/run_phase1b_benchmark.py \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --cache-dir /kaggle/working/phase1b_corrected/cache \
  --output-dir /kaggle/working/phase1b_corrected/results \
  --k 16 32 \
  --selection-seeds 0 1 2 \
  --probe-seeds 0 1 2 \
  --device cuda

!python scripts/plot_phase1b_localizers.py \
  --cache-dir /kaggle/working/phase1b_corrected/cache \
  --output-dir /kaggle/working/phase1b_corrected/results/qualitative \
  --k 32 \
  --count 20
```

## Produce the Phase 01 gate report

```python
!python scripts/write_phase1_sanity_report.py \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --cache-dir /kaggle/working/phase1b_corrected/cache \
  --results-dir /kaggle/working/phase1b_corrected/results \
  --search-root /kaggle/input
```

This writes:

- `phase1_sanity_report.md`;
- `phase1_gate.json`, the machine-readable gate decision and audit findings;
- `vision_cls_part_localization.csv`.

The audit verifies that the benchmark points to the corrected cache, requires
the exact `bird`/`birds` lexical tokens, rejects token ID 29871, checks corrected
prediction/localization coverage, and cross-checks the qualitative index. It
returns exit code 2 and records `STOP / INVESTIGATE` if a blocking check fails.
`PASS WITH ANOMALY` is a successful gate that explicitly carries the strong
Top-K box concentration/weak top-1 pointing mismatch forward. Stop before
stage-wise representation extraction unless `phase1_gate.json` records
`"passed": true`.

## Moving forward after Phase 01

Do not run the next large experiment until `phase1_sanity_report.md` passes.
After it passes, proceed in this order:

1. Implement and smoke-test a stage-aligned cache for selected vision layers,
   projector output, selected LLM layers, spatial coordinates, CUB attributes,
   visible parts, and fixed-prompt answers.
2. Extract the existing 240-image development pilot once on Kaggle. Measure
   storage/runtime before expanding it.
3. Train the same linear multi-label CUB attribute probe at every stage. This is
   the first experiment that can test where attribute information is accessible.
4. Separately compare Vision-CLS attention, LLM attention, dense semantic
   similarity, and valid Logit Lens scores against part annotations.
5. Compare internal probe recoverability with fixed-prompt VLM answer accuracy.
   This tests for a representation–utilization gap without assuming one.
6. Write `INTERMEDIATE_FINDINGS.md`, select the strongest supported transition,
   and only then run one matched causal removal/replacement experiment.

The immediate development task after a passing report is therefore the
stage-aligned representation-cache implementation—not another selector sweep or
a sparse-autoencoder experiment.

## Phase 2 stage-aligned cache

Phase 1 passed with the documented Vision-CLS Top-K/top-1 pointing anomaly. The
Phase 2 begins with attribute-subset selection from development-training
annotations and one image of stage-aligned extraction. Its schema and acceptance
checks are documented in
[`representation_cache_schema.md`](representation_cache_schema.md).

After pulling the latest `feat/iclr` commit into a Kaggle GPU notebook, first
freeze the attribute subset without loading validation annotations:

```python
%cd /kaggle/working/newpipeline/projects/logit_evidence_routing

!PYTHONPATH=src python scripts/select_phase2_attributes.py \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --config configs/phase2_attribute_groups.json \
  --output /kaggle/working/representation_cache_smoke/attribute_subset.json \
  --search-root /kaggle/input
```

Then run exactly one frozen-model image:

```python
!PYTHONPATH=src python scripts/smoke_stage_cache.py \
  --phase1-gate /kaggle/working/phase1b_corrected/results/phase1_gate.json \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --attribute-subset /kaggle/working/representation_cache_smoke/attribute_subset.json \
  --output-dir /kaggle/working/representation_cache_smoke \
  --search-root /kaggle/input \
  --model llava-hf/llava-1.5-7b-hf \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --quantization 4bit \
  --prompt "Describe the image briefly." \
  --max-new-tokens 32
```

The smoke job writes `run_config.json`, one atomic `.pt` sizing record,
`index.json`, and `smoke_summary.json`. It validates a save/reload cycle and
reports resolved stages, shapes, dtypes, bytes per image, projected 240-image
storage/runtime, and peak GPU memory. The `.pt` layout is not the production
format. Stop after this command and review `smoke_summary.json` before choosing
the final indexed shard format or authorizing the 240-image extraction.

The reviewed smoke used 1,024-dimensional vision states, 4,096-dimensional
projector/LLM states, 2,051,509,760 peak allocated GPU bytes, and projected
7,300,734,000 bytes for all 240 images. This passes the development-pilot gate.
The production choice is 20-image safetensors shards with JSON reconstruction
metadata: twelve bounded shards instead of one monolith or 240 individual files.

Run the complete 160-train/80-validation development pilot with exactly the
smoke-tested model, revision, prompt, generation length, gate, and attribute
subset:

```python
!PYTHONPATH=src python scripts/extract_phase2_stage_cache.py \
  --phase1-gate /kaggle/working/phase1b_corrected/results/phase1_gate.json \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --attribute-subset /kaggle/working/phase2_smoke_6ea8e34/attribute_subset.json \
  --smoke-dir /kaggle/working/phase2_smoke_6ea8e34 \
  --output-dir /kaggle/working/phase2_stage_cache \
  --search-root /kaggle/input \
  --model llava-hf/llava-1.5-7b-hf \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --quantization 4bit \
  --prompt "Describe the image briefly." \
  --max-new-tokens 32 \
  --shard-size 20
```

Before loading the model, the command strictly parses the annotations for all
240 requested IDs. Thus the known damaged rows in the Kaggle dataset remain
ignored only when they are outside the pilot. Each completed shard is written
atomically, reloaded, schema-validated, hashed, and then added to `index.json`.
Rerunning the identical command resumes at the next whole shard; any changed
configuration requires a new output directory.

After all twelve shards complete, independently hash-check and reload all 240
records:

```python
!PYTHONPATH=src python scripts/validate_phase2_stage_cache.py \
  --cache-dir /kaggle/working/phase2_stage_cache
```

Proceed to Phase 3 only if `validation_report.json` reports `PASS`, 240 records,
160/80 development splits, nine stages per record, and zero official-test
images. The untouched official CUB test split remains reserved for Phase 8.

## Phase 3 attribute-probe smoke

After the full cache passes, smoke-test the attribute-recoverability pipeline on
the three boundary representations `vision.final`, `projector.output`, and
`llm.final`. This is an optimization and control check, not the frozen nine-stage
result:

```python
!PYTHONPATH=src python scripts/run_phase3_attribute_probes.py \
  --cache-dir /kaggle/working/phase2_stage_cache \
  --output-dir /kaggle/working/phase3_attribute_smoke \
  --stages vision.final projector.output llm.final \
  --pooling mean \
  --controls primary prevalence shuffled_labels random_projection \
  --seeds 0 \
  --epochs 50 \
  --learning-rate 0.01 \
  --weight-decay 0.0001 \
  --random-projection-dim 256 \
  --device cuda
```

The runner reads only the requested tensors, mean-pools the same 576 patches at
each stage, standardizes features using development-training statistics only,
masks `guess`/`not visible`/missing labels, and selects every F1 threshold using
training scores only. It reports per-attribute AUROC/F1 and macro summaries.
The prevalence, independently shuffled-label, and 256-dimensional Gaussian
projection runs test trivial class balance, label leakage, and feature-dimension
confounding respectively.

Inspect `attribute_probe_by_stage.csv` before starting the full nine-stage,
three-seed run. In particular, verify that outputs are finite, shuffled-label
performance is not suspiciously strong, and the primary results are not merely
the prevalence baseline. Do not interpret 50-epoch, one-seed smoke numbers as
scientific results.

## Outputs to download

```python
!cd /kaggle/working && zip -r phase1b_corrected.zip \
  phase1b_corrected/results \
  phase1b_corrected/cache/extraction_config.json \
  phase1b_corrected/cache/correction_summary.json
```

Download `phase1b_corrected.zip` from the Kaggle notebook Output/Files panel.

## Scientific reference

The original motivation is Arsh Naqvi's
[*Using Logit Space of VLMs for Attention to Detail*](https://www.arsh-naqvi.xyz/blog/logit-space-vlm-attention-to-detail).
The post describes a private trauma pipeline that combines attention-based
candidate localization with logit-lens filtering. This repository now uses that
idea as motivation for a public, controlled representation-tracing study rather
than presuming logit routing is the contribution.

## Verified development results (6 September 2026)

The full nine-stage Phase 3 development run and the one-image Phase 4
cached-localizer reuse smoke have passed. Read
[`reports/development_20260906/REVIEW.md`](reports/development_20260906/REVIEW.md)
for the independently audited source tables and attribute-level findings, and
[`NEXT_CHAT_HANDOFF.md`](NEXT_CHAT_HANDOFF.md) for current state. Earlier smoke
commands above are historical reproducibility instructions, not the next task.

The original Kaggle cells are versioned in
[`notebooks/phase3_development_kaggle.ipynb`](notebooks/phase3_development_kaggle.ipynb)
and [`notebooks/phase4_entry_cached_localizer_smoke.ipynb`](notebooks/phase4_entry_cached_localizer_smoke.ipynb).
No model weights, representation caches, or datasets are included in the result bundle.

The remaining Phase 4 implementation is now available in
[`notebooks/phase4_full_development_kaggle.ipynb`](notebooks/phase4_full_development_kaggle.ipynb).
Run both cells on Kaggle: new paired-semantic one-image smoke, then the full
240-image development comparison and result validator. See the
[fixed protocol](planning/08_PHASE4_LOCALIZATION_PROTOCOL.md) for scoring,
attribute-to-part proxies, exclusions, denominator rules and output counts.

Actual semantic/full-run results remain pending Kaggle execution. Local checks
use synthetic inputs only. The earlier cached-map smoke does not complete Phase 4,
and the new code does not invent results. Official test remains untouched; no
causal transition has been selected.
