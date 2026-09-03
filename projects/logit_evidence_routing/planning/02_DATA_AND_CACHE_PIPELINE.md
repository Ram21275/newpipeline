# 02 — Public Data and Cache Pipeline

**Target completion: Sep 7**

## Goal

Turn the pilot into a reproducible pipeline on public datasets without creating an unmanageably large hidden-state cache.

---

## Dataset A — CUB-200-2011

Prepare:
- official train/test split
- class labels
- bounding boxes if available
- part/attribute annotations if available

Use classification labels for training; localization/part annotations are for analysis only.

## Dataset B — FGVC-Aircraft

Prepare:
- official train/validation/test protocol
- variant labels
- image paths

Do not tune on the official test set.

---

## Cache design

Do **not** blindly store `[all layers × all image patches × 4096]` for every public image. It can become hundreds of GB.

Use a two-stage cache.

### Stage A — routing metadata

For every image, store lightweight routing values:

```text
image_id
layer
patch_index
x,y
attention_score
max_logit
max_probability
entropy
logit_margin
top_token_ids
```

### Stage B — candidate hidden states

During preprocessing, retain only a generous candidate pool, for example Top-M patches per selected layer where `M > max K used in experiments`.

Example:
- experiment K ∈ {8,16,32,64}
- cache M = 96 or 128

Store hidden states in fp16/bfloat16.

This allows routing ablations without retaining every token.

---

## Layer scope

The original project observed useful information in late layers. Start with a fixed late-layer window, e.g. the last 8–12 layers supported by the current VLM.

Do not scan the entire network until the late-layer pipeline works.

Record exact layer IDs in config.

---

## Required preprocessing API

Implement functions equivalent to:

```python
extract_vlm_states(image, prompt, layers)
compute_patch_logits(hidden_states, lm_head, final_norm)
compute_routing_scores(logits, attention=None)
cache_candidates(image_id, metadata, hidden_states)
```

The exact code structure may differ, but these responsibilities must stay separate.

---

## Reproducibility rules

- Fix prompt template.
- Fix image preprocessing.
- Fix tokenizer/model revision.
- Save model commit/revision.
- Save patch-grid geometry.
- Save layer IDs.
- Use deterministic dataset splits.

---

## Dataset validation

Before full preprocessing, inspect 50 samples manually.

Check:
- [ ] correct class labels
- [ ] no train/test overlap
- [ ] image resolution/preprocessing correct
- [ ] patch coordinates map correctly to image coordinates
- [ ] bounding-box transforms remain correct after resize/crop

---

## Deliverables

```text
data/
  cub_manifest.csv
  aircraft_manifest.csv

cache/
  cub/
  aircraft/

configs/
  preprocessing_cub.yaml
  preprocessing_aircraft.yaml

reports/
  cache_stats.md
```

`cache_stats.md` must contain:
- number of images
- layers cached
- candidate M
- disk usage
- average preprocessing time/image
- GPU used
- any failures

---

## Exit criteria

- [ ] Both public datasets load reproducibly.
- [ ] At least Dataset A is fully cached.
- [ ] Dataset B preprocessing can run unattended.
- [ ] Patch coordinates have been visually verified.
- [ ] Cache size is acceptable for the available storage.
