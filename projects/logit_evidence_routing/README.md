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
- Final-layer hidden-state representation with retained coordinates and layer IDs.
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

Create a notebook, enable Internet, and run:

```python
!git clone --branch feat/iclr --single-branch https://github.com/Ram21275/newpipeline.git /kaggle/working/newpipeline
%cd /kaggle/working/newpipeline/projects/logit_evidence_routing
!pip install -e .
!python -m unittest discover -s tests -v
!python scripts/run_synthetic_pilot.py --output /kaggle/working/lger_synthetic_smoke.json
```

For later updates in the same Kaggle session:

```python
%cd /kaggle/working/newpipeline
!git pull --ff-only origin feat/iclr
%cd projects/logit_evidence_routing
!pip install -e .
```

## Next scientific milestone

Connect the existing LLaVA-Lens model code to these two model-facing boundaries:

1. `gather_visual_hidden_states(...)` using verified visual-token positions.
2. `FrozenLogitProjector(final_norm, lm_head)` using the exact frozen model modules.

Then run `configs/pilot.json` on a fixed 10–20% CUB training split for K=16 and K=32. Produce the machine-readable files required by `planning/01_SIGNAL_VALIDATION.md`; synthetic smoke output must never be reported as an experiment.
