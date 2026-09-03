# 04 — Final Method: Logit-Guided Evidence Routing

**Target completion: Sep 11**

## Goal

Implement exactly one proposed method beyond the baselines.

Working name:

> **Logit-Guided Evidence Routing (LGER)**

The name can change later. The mechanism should not.

---

## Core principle

For patch `i` at layer `l`:

\[
h_i^l \in \mathbb{R}^D
\]

Apply the frozen normalization/output head:

\[
z_i^l = W_{LM}\,LN(h_i^l)
\]

Convert this into a scalar semantic-evidence score:

\[
e_i^l = g(z_i^l)
\]

Then aggregate evidence across late layers:

\[
E_i = A(e_i^{l_1},\ldots,e_i^{l_m})
\]

Select:

\[
\mathcal{S}_K = TopK_i(E_i)
\]

and feed the **original hidden states** corresponding to selected evidence into the classifier.

> **Logits route. Hidden states represent.**

---

## Step 1 — Primary evidence score

Start with a class-agnostic score so test-time routing does not use the true label.

Candidate primary score:

\[
e_i^l=z_{(1)}-z_{(2)}
\]

where `z_(1)` and `z_(2)` are the largest two vocabulary logits.

Reason: a logit margin captures semantic decisiveness without depending strongly on softmax temperature.

Keep max probability and negative entropy as ablations.

---

## Step 2 — Cross-layer persistence

Implement the simplest defensible version first:

\[
E_i=\frac{1}{|L|}\sum_{l\in L}\tilde e_i^l
\]

where `tilde e` is normalized within each layer.

Optional second variant only if needed:

\[
E_i = mean_l(\tilde e_i^l) - \lambda\,std_l(\tilde e_i^l)
\]

which rewards evidence that remains strong across layers.

Do not introduce a learned router unless the non-learned formulation fails.

---

## Step 3 — Preserve spatial identity

For each selected patch retain:
- x/y patch coordinates
- layer ID or chosen aggregation layer
- hidden state
- evidence score

Add positional encoding before the lightweight Transformer.

---

## Step 4 — Representation choice

Primary implementation:

For each selected spatial patch, use the hidden state from the final selected layer, while the **selection score** is aggregated across late layers.

This keeps classifier input size fixed:

\[
K\times D
\]

rather than `K × number_of_layers × D`.

A multi-layer representation can be an ablation, not the default.

---

## Step 5 — Context injection

The original project adds random non-salient context patches. Preserve this as an optional training component.

Main experiment:
- selected evidence patches + small fixed number of context patches

Ablation:
- evidence only

Do not change context count per dataset without documenting it.

---

## Step 6 — Code interface

The router should expose something like:

```python
selected = router(
    candidate_hidden_states,
    candidate_logits,
    patch_coords,
    layer_ids,
    k=K,
)
```

Return:

```text
selected.hidden_states
selected.patch_indices
selected.scores
selected.layer_statistics
```

The classifier must not need to know how patches were selected.

---

## Required ablation switches

One config flag each for:

```text
score = maxprob | margin | negentropy
aggregate = last | mean_layers | persistent
context = none | random
K = 8 | 16 | 32 | 64
```

Do not create separate code paths for each paper table.

---

## Unit tests / sanity tests

- [ ] Same input/config produces same selected patches.
- [ ] K selected patches are unique.
- [ ] Layer aggregation aligns the same spatial index across layers.
- [ ] No ground-truth class enters the router.
- [ ] Classifier gradients do not update the frozen VLM/LM head.
- [ ] Random selector reproduces baseline behavior through the same API.

---

## Exit criteria

- [ ] LGER runs end-to-end on Dataset A.
- [ ] One config reproduces every main selector.
- [ ] Cross-layer score visualizations look sensible.
- [ ] Main method is no more complicated than necessary.

If cross-layer routing is worse than last-layer logit routing on validation, use last-layer routing as the main method and report cross-layer emergence as analysis.
