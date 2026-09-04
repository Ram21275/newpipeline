# 02 — Stage-Aligned Representation Cache

## Goal

Cache the same fine-grained evidence at comparable points through one frozen
LLaVA pipeline without rerunning the model for every probe.

## Initial stage set

Resolve layer indices from model depth and record the exact indices:

- vision: early, middle, late, final patch states;
- projector: projected visual tokens before insertion into the LLM;
- LLM: early, middle, late, and final visual-token states;
- output: generated answer text and answer-token logits for fixed prompts.

Do not force LM-head Logit Lens onto a representation whose dimension or
normalization is incompatible with the language output head.

## Per-image schema

Store:

- dataset version, image ID/path, class ID/name, official split;
- CUB attributes and certainty/presence labels;
- visible part IDs and original coordinates;
- original and processed image sizes, crop transform, patch grid coordinates;
- stage name, exact layer index, tensor shape/dtype;
- patch-aligned representations for each selected stage;
- available CLS/LLM attention maps;
- fixed prompt and generated answer;
- model name, resolved revision, code commit, and extraction config.

Use safetensors, HDF5, or chunked memmap storage with an explicit index. Avoid a
single monolithic file and avoid thousands of tiny per-layer files.

## Required invariants

- one stable spatial patch index maps across vision, projector, and LLM visual
  tokens, or the schema records the exact mapping when token counts differ;
- resumption validates the full config rather than silently mixing runs;
- hidden states are detached and the entire VLM remains frozen;
- train/validation/test labels never influence extraction or token selection;
- an audit command checks IDs, shapes, finite values, and duplicate records.

## Pilot sizing

Estimate storage from one image before the full run. Start with the existing
240-image development pilot. Cache only selected layers initially. Expand to the
official test set only after the measurement protocol is frozen.

## Deliverables

- `representation_cache/`
- `representation_cache_schema.md`
- machine-readable extraction summary with bytes/image, runtime/image, and peak
  memory
- a smoke run showing one image can be loaded and aligned at every stage
