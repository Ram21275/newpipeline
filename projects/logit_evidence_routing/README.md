# Logit-Guided Evidence Routing

Initial implementation for the ICLR 2027 project described in [`planning/`](planning/README.md).

The research hypothesis is not considered validated yet. The current code establishes the leakage-safe and reproducible infrastructure needed for the Phase 01 signal-validation pilot.

## What is implemented

- Frozen final-normalization + LM-head projection for patch hidden states.
- Explicit visual-token gathering to prevent prompt/text-token contamination.
- Class-agnostic max-probability, logit-margin, and negative-entropy evidence scores.
- Last-layer, normalized late-layer mean, and persistence-aware aggregation.
- One API for Random-K, Attention-K, Logit-K, Attention+Logit, and LGER.
- Optional deterministic random context patches.
- Selected late-layer hidden-state representation with retained coordinates and layer IDs.
- A shared projection/Transformer/CLS classifier.
- Reproducibility records containing commit, config, seed, split, checkpoint, metrics, routing statistics, runtime, and GPU memory.
- Unit and synthetic smoke tests.

The defining contract is: **logits route; original hidden states represent**. The router never accepts a ground-truth label or class name.

## Local setup

```bash
cd projects/logit_evidence_routing
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
python scripts/run_synthetic_pilot.py
```

Without installing the package, the standard-library test runner also works from the repository root:

```bash
PYTHONPATH=projects/logit_evidence_routing/src \
  python -m unittest discover -s projects/logit_evidence_routing/tests -v
```

## Kaggle

Create a Kaggle notebook, select a GPU accelerator, enable Internet, and attach a
CUB-200-2011 dataset containing the official `CUB_200_2011` metadata and images.
The repository is public, so no GitHub token is required.

Clone the feature branch:

```python
!git clone --branch feat/iclr --single-branch \
  https://github.com/Ram21275/newpipeline.git \
  /kaggle/working/newpipeline
```

Install the Kaggle dependencies and run the local validation suite:

```python
%cd /kaggle/working/newpipeline/projects/logit_evidence_routing
!python -m pip install -r requirements-kaggle.txt
!python -m unittest discover -s tests -v
!python scripts/run_synthetic_pilot.py --output /kaggle/working/lger_synthetic_smoke.json
```

The Kaggle requirements pin `bitsandbytes==0.50.2`, whose Linux wheel includes
CUDA 12.8 support. The real extractor runs a tiny NF4 kernel before loading the
checkpoint, so a mismatched runtime fails before model shards are downloaded.

Create the deterministic Phase 01 split. The discovery code searches all attached
Kaggle inputs and refuses ambiguous or incomplete CUB layouts:

```python
!python scripts/prepare_cub_pilot.py \
  --search-root /kaggle/input \
  --output /kaggle/working/phase1/pilot_manifest.csv \
  --num-classes 20 \
  --train-per-class 8 \
  --val-per-class 4 \
  --seed 0
```

Run a two-image GPU smoke extraction before committing to the full job:

```python
!python scripts/extract_phase1_features.py \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --search-root /kaggle/input \
  --output-dir /kaggle/working/phase1/cache \
  --model llava-hf/llava-1.5-7b-hf \
  --layer-offset -2 \
  --k 16 32 \
  --random-seeds 0 1 2 \
  --max-images 2
```

If that succeeds, rerun the same command without `--max-images 2`. It resumes
from the two cached records instead of starting over.

Train the identical mean-pooled linear probe for all three selectors:

```python
!python scripts/run_phase1_probes.py \
  --manifest /kaggle/working/phase1/pilot_manifest.csv \
  --cache-dir /kaggle/working/phase1/cache \
  --output-dir /kaggle/working/phase1/results \
  --k 16 32 \
  --seeds 0 1 2
```

Generate the required lowest-overlap qualitative examples:

```python
!python scripts/plot_phase1_examples.py \
  --cache-dir /kaggle/working/phase1/cache \
  --output-dir /kaggle/working/phase1/results/qualitative \
  --k 32 \
  --count 20
```

Inspect `selector_summary.csv`, `selector_metrics.csv`, `patch_statistics.csv`,
and the qualitative figures. Save a Kaggle notebook version so everything under
`/kaggle/working/phase1` is preserved as notebook output.

For later code updates in the same Kaggle session:

```python
%cd /kaggle/working/newpipeline
!git pull --ff-only origin feat/iclr
%cd projects/logit_evidence_routing
!python -m pip install -r requirements-kaggle.txt
```

If this updates `bitsandbytes` in an already-running notebook, each `!python`
command starts a fresh process, so the extraction command can be retried directly.

## Next scientific milestone

Phase 01 asks only whether Logit-K contains useful signal beyond Random-K and how
it differs from Attention-K. Do not introduce cross-layer LGER into this table.
Proceed to the cache pipeline only if the measured selector metrics or qualitative
alignment satisfy the exit criteria in `planning/01_SIGNAL_VALIDATION.md`.
