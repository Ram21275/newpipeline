# 05 — Main Experiments

**Target completion: Sep 14**

## Goal

Generate the central paper table with enough statistical reliability to support the main claim.

---

## Experiment 1 — CUB-200-2011

Methods:

1. Global frozen-VLM
2. Random-K
3. Attention-K
4. Logit-K
5. Attention+Logit
6. LGER

Primary K:

```text
K = 32
```

Seeds:

```text
0, 1, 2
```

Metrics:
- top-1 accuracy
- macro-F1

---

## Experiment 2 — FGVC-Aircraft

Use the **same method definitions and primary hyperparameters** unless there is a genuine dataset-specific necessity.

Again run three seeds for the final methods.

---

## Experiment 3 — Low-data regime

This is a major paper claim.

Train with stratified fractions of the training set:

```text
5%
10%
25%
100%
```

Required comparison:
- Global
- Attention-K
- Logit-K
- LGER

At minimum run 3 seeds at 10% and 100%. If compute permits, use 3 seeds for every fraction.

Plot:

```text
x-axis: labeled training fraction
y-axis: accuracy / macro-F1
```

Hypothesis:

> pretrained semantic routing is most useful when downstream labels are scarce.

---

## Experiment 4 — Routing efficiency diagnostic

This is analysis, not the main claim.

For K ∈ {8,16,32,64}, record:
- accuracy
- macro-F1
- number of processed downstream tokens
- classifier FLOPs or runtime if easy to measure

Do not turn this into an efficiency paper.

---

## Statistical reporting

For the central table report:

\[
mean \pm std
\]

over seeds.

If two methods are very close, do not make strong significance claims without an appropriate statistical test.

---

## Table 1 — Main result

Generate programmatically:

| Dataset | Method | Acc ↑ | Macro-F1 ↑ |
|---|---|---:|---:|
| CUB | Global | | |
| CUB | Random-K | | |
| CUB | Attention-K | | |
| CUB | Logit-K | | |
| CUB | Attn+Logit | | |
| CUB | LGER | | |
| Aircraft | ... | | |

---

## Figure 1 — Method figure

Prepare assets for:

```text
RGB image
  ↓
Frozen VLM
  ↓
late-layer visual hidden states
  ↓ frozen LM head
layer-wise semantic evidence
  ↓ cross-layer aggregation
Top-K evidence patches
  ↓ original hidden states
small classifier
  ↓
fine-grained prediction
```

---

## Figure 2 — Attention vs semantic evidence

Choose examples illustrating:

A. high attention + high semantic evidence
B. high attention + low semantic evidence
C. lower attention + high semantic evidence

This figure should directly motivate why attention alone is insufficient.

---

## Exit criteria

By Sep 14:

- [ ] Dataset A complete, all main methods, 3 seeds.
- [ ] Dataset B has at least a complete single-seed table and final 3-seed jobs running/complete.
- [ ] Low-data experiment shows the shape of the result.
- [ ] Figures can be generated from saved outputs.
- [ ] No manual metric transcription.
