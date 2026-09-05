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
`attention_logit_fusion`. These remain development-pilot findings until the
corrected result bundle and Phase 01 sanity report are audited. The full
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
