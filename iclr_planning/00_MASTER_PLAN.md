# 00 — Master ICLR Execution Plan

## Time window

Start: **3 Sep 2026**  
Abstract deadline: **18 Sep 2026 (AoE)**  
Paper deadline: **25 Sep 2026 (AoE)**

## Paper target

### Main claim
A frozen VLM already contains patch-level semantic evidence in its intermediate language-facing representations. The LM vocabulary space can be used to **route** fine-grained visual evidence to a lightweight downstream classifier.

### Three claims only

1. **Attention and semantic evidence are not equivalent.**
2. **Logit-space evidence is useful for selecting fine-grained visual tokens.**
3. **The advantage is strongest when labeled data are limited.**

Do not make more than these three primary claims.

---

## Calendar

| Dates | Phase | Required output |
|---|---|---|
| Sep 3–5 | Signal validation | Random vs attention vs logit pilot |
| Sep 5–7 | Data/cache pipeline | Reproducible public-data preprocessing |
| Sep 7–9 | Baselines | Frozen-VLM baseline table |
| Sep 9–11 | Final method | Cross-layer evidence routing implemented |
| Sep 11–14 | Main experiments | 2-dataset main result table |
| Sep 14–17 | Ablations/analysis | Layer, K, routing, low-data results |
| Sep 17–18 | Abstract | Genuine ICLR abstract submitted |
| Sep 19–22 | Paper | Complete 9-page draft + appendix |
| Sep 22–24 | Freeze | Re-runs, figures, anonymization |
| Sep 25 | Submission | Final PDF/OpenReview submission |

---

## Datasets

Use **two public fine-grained datasets** for the main paper:

- **CUB-200-2011** — local visual attributes/parts are useful for analysis.
- **FGVC-Aircraft** — fine-grained categories depend on small structural distinctions.

DARPA trauma data should be treated only as a motivating/qualitative application unless public release and evaluation are possible.

---

## Fixed model scope

Keep the VLM used by the existing LLava-Lens code unless it becomes technically impossible.

### Frozen
- Vision encoder
- Projector
- LLM
- LM head

### Trainable
- Optional 4096→512 projection
- Small Transformer/evidence aggregator
- Final classifier

---

## Main comparisons

1. Global frozen-VLM representation
2. Random-K patches
3. Attention Top-K
4. Logit-confidence Top-K
5. Attention + logit routing
6. **Ours: cross-layer semantic evidence routing**

Keep the downstream classifier architecture identical wherever possible.

---

## Metrics

Required:
- Top-1 accuracy
- Macro-F1
- Mean ± std over 3 seeds for main comparisons

Analysis when annotations permit:
- selected-patch overlap with object/part regions
- patch budget vs accuracy
- layer vs accuracy
- low-data performance

---

## Daily experiment discipline

Every run must save:
- git commit hash
- config YAML/JSON
- seed
- dataset split ID
- model checkpoint name
- metrics CSV
- selection statistics
- runtime and GPU memory

Never copy results manually into the paper without a machine-readable source file.

---

## Kill criteria / scope control

### By Sep 5
If logit routing is indistinguishable from random selection on the pilot, investigate scoring/calibration for **one day only**. Do not build the complete paper pipeline around a failed signal.

### By Sep 11
If cross-layer routing does not improve over single-layer logit selection, use the simpler single-layer method and turn cross-layer behavior into analysis rather than the main method.

### By Sep 14
If Dataset 2 is unstable, keep the method fixed and diagnose implementation/data issues. Do not introduce a new model family.

---

## Definition of “submission ready”

The project is ready only if all are true:

- [ ] Public reproducible dataset(s)
- [ ] Main baseline table
- [ ] 3-seed primary result
- [ ] Low-data experiment
- [ ] Layer/patch-budget analysis
- [ ] Clear distinction from attention-only and hallucination-focused Logit-Lens work
- [ ] 9-page anonymous paper
- [ ] Reproducible configs/code
- [ ] Genuine abstract submitted by Sep 18
