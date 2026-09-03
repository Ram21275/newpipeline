# ICLR 2027 Execution Pack — Logit-Guided Evidence Routing

This folder is meant to be executed **in order**. Do not start the next phase until the current phase's exit criteria are satisfied.

## Deadline

Verified ICLR 2027 deadlines (all AoE):
- **Abstract:** 18 Sep 2026, 11:59 PM
- **Full paper:** 25 Sep 2026, 11:59 PM

## Core research question

> Can the vocabulary/logit space already learned by a frozen VLM be used as a semantic routing space to identify the small visual evidence that determines a fine-grained decision, especially in low-data regimes?

## Frozen core

The current project already provides the starting point:
1. Extract intermediate VLM hidden states.
2. Apply the frozen LM head as a Logit Lens.
3. Use logit-space information to select visual patches.
4. Preserve the original hidden representations of selected patches.
5. Train only a lightweight evidence aggregator/classifier.

The paper should not be framed as “we applied Logit Lens to images.” The intended contribution is **logit-space semantic evidence routing**.

## Execution order

1. `00_MASTER_PLAN.md`
2. `01_SIGNAL_VALIDATION.md`
3. `02_DATA_AND_CACHE_PIPELINE.md`
4. `03_BASELINES.md`
5. `04_METHOD_LOGIT_EVIDENCE_ROUTING.md`
6. `05_MAIN_EXPERIMENTS.md`
7. `06_ABLATIONS_AND_ANALYSIS.md`
8. `07_ABSTRACT_DEADLINE.md`
9. `08_FINAL_PAPER_AND_SUBMISSION.md`

## Hard rule

Do not add a new backbone, LoRA, VLM fine-tuning, segmentation head, DINO fusion, or another major method before the core hypothesis is validated.
