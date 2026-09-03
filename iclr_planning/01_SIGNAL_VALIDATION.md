# 01 — Signal Validation

**Target completion: Sep 5**

## Goal

Before building the full method, test the central hypothesis:

> Does Logit-Lens-derived patch evidence contain a useful signal for fine-grained recognition beyond random patch selection and ordinary attention?

This phase should be small and disposable.

---

## Task 1 — Reproduce the existing extraction path

For one frozen VLM and one small pilot split:

1. Run the RGB image through the VLM.
2. Extract intermediate hidden states for the chosen late layers.
3. Identify which sequence positions correspond to image patches.
4. Apply the model's final normalization + frozen LM head to those representations.
5. Save, for each patch/layer:
   - spatial patch index
   - layer index
   - attention score if available
   - max logit / max probability
   - top-5 decoded tokens
   - entropy or logit margin

### Sanity checks

- [ ] Visual-token indices are correct.
- [ ] Decoded patch logits change with image content.
- [ ] Different layers produce meaningfully different vocabulary distributions.
- [ ] No text/prompt token is accidentally treated as an image patch.

---

## Task 2 — Define three selectors

Use the same patch budget `K`.

### Random-K

Uniformly select K image patches.

### Attention-K

Select K patches with the largest visual attention score.

### Logit-K

Start with the simplest class-agnostic semantic score:

\[
S_{i,l}=\max_v p(v\mid h_i^l)
\]

Also record two alternatives, but do not optimize them yet:

\[
S^{margin}_{i,l}=z_{(1)}-z_{(2)}
\]

and

\[
S^{entropy}_{i,l}=-H[p(\cdot\mid h_i^l)].
\]

The point of this phase is to test whether semantic confidence contains signal at all.

---

## Task 3 — Avoid label leakage

For the public fine-grained task, **do not use the ground-truth class name to select patches**.

If a task vocabulary is later used, it must be the same vocabulary for every test image, e.g. the union of all class/attribute concepts.

---

## Task 4 — Tiny classifier probe

For each selector:

1. Select K patches.
2. Use the original hidden states, not the vocabulary probabilities, as features.
3. Mean-pool them for the first pilot.
4. Train the same linear classifier on top.

Do not build the full Transformer classifier yet.

Suggested pilot budgets:
- K = 16
- K = 32

Suggested pilot data:
- one dataset
- 10–20% of training data
- fixed validation split

---

## Task 5 — Produce the diagnostic figure

For 20–30 images save a 4-panel visualization:

1. Original image
2. Attention heatmap
3. Logit-evidence heatmap
4. Selected Logit-K patches with top decoded words

Also save examples where attention and logit evidence disagree.

---

## Required output files

```text
results/pilot/
  selector_metrics.csv
  patch_statistics.csv
  qualitative/
  config.yaml
  notes.md
```

`selector_metrics.csv` should contain at minimum:

```text
selector,K,seed,accuracy,macro_f1
random,16,...
attention,16,...
logit,16,...
```

---

## Exit criteria

Proceed to Phase 02 if **either** of these is true:

- Logit-K consistently improves downstream probe performance over Random-K, or
- Logit evidence provides clearly stronger localization/semantic alignment than Random-K and a plausible route exists to exploit it.

If Attention-K is better than Logit-K, that does **not** kill the project. Record the result; the next question becomes whether combining semantic evidence with attention helps.

If Logit-K behaves like random noise, stop and debug token indexing, normalization, LM-head application, and score calibration before continuing.
