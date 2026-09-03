# 03 — Establish the Baselines

**Target completion: Sep 9**

## Goal

Build a trustworthy baseline table before implementing the final proposed method.

The classifier should be held as constant as possible so the experiment isolates the patch-routing signal.

---

## Shared classifier

Start with one simple architecture for all patch-based methods:

```text
selected hidden states [K × D]
        ↓
Linear projection D → 512
        ↓
CLS token + positional/layer encoding
        ↓
2–4 small self-attention blocks
        ↓
LayerNorm
        ↓
CLS
        ↓
Linear classifier
```

Do not tune a different head for each selector.

---

## Baseline B0 — Global frozen-VLM feature

Use a global representation from the same frozen VLM and train the same-sized classification head where practical.

Purpose: determine whether routing patches provides anything beyond global features.

---

## Baseline B1 — Random-K

Randomly select K image patches.

Use several random seeds.

This is the minimum control required to show that selection matters.

---

## Baseline B2 — Attention Top-K

Select K patches by visual attention.

Keep the definition of attention fixed and documented:
- exact layer/head aggregation
- query token used
- normalization

Do not cherry-pick attention definitions per dataset.

---

## Baseline B3 — Logit confidence Top-K

Select K using class-agnostic logit confidence.

Primary first implementation:

\[
S_i = \max_v p(v|h_i)
\]

Record margin and entropy scores for later ablations.

---

## Baseline B4 — Attention + logit

Use a simple non-learned combination, for example normalized sum or intersection/union of Top-K candidate sets.

The exact formulation must be fixed before viewing test results.

---

## Training protocol

Use the same:
- optimizer
- learning-rate schedule
- batch size
- epochs/early stopping
- augmentation
- train/val split
- seeds

for all selectors.

Recommended main-seed set:

```text
{0, 1, 2}
```

---

## Result table

Generate automatically:

| Method | K | Accuracy | Macro-F1 | Params trained |
|---|---:|---:|---:|---:|
| Global | – | | | |
| Random | 32 | | | |
| Attention | 32 | | | |
| Logit | 32 | | | |
| Attn+Logit | 32 | | | |

Also record mean ± std over seeds.

---

## Important diagnostic

For every method compute patch diversity:

- number of unique spatial patches
- spatial coverage
- overlap with attention selection
- overlap with logit selection

This prevents two supposedly different selectors from actually choosing nearly identical tokens.

---

## Exit criteria

- [ ] Baselines run from one common training script.
- [ ] Results are generated automatically from saved run files.
- [ ] Random selector uses multiple random seeds.
- [ ] No test-set tuning.
- [ ] We can state clearly whether logit selection is complementary to attention.
