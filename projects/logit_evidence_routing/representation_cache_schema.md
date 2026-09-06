# Stage-Aligned Representation Cache Schema

**Schema version:** 1
**Status:** smoke accepted; production format frozen as indexed 20-image
safetensors shards with JSON reconstruction metadata.

## Purpose and evidence boundary

The cache traces the same spatial image-patch identity through a frozen classic
LLaVA-1.5 pipeline:

```text
vision early / middle / late / final
                  ↓
projector output before LLM insertion
                  ↓
LLM early / middle / late / final visual-token states
                  ↓
fixed-prompt generated answer and per-answer-token logits
```

It stores representations and raw CUB annotations. It does not select patches
using labels, train a probe, interpret attention causally, or claim that an
attribute is used by generation.

## Gate and run identity

Every run configuration records:

- a SHA-256 digest and status from `phase1_gate.json`; `passed` must be true;
- repository commit;
- CUB version plus hashes/sizes of identity metadata;
- pilot-manifest path and selected image ID;
- model name and requested/resolved immutable revision;
- Transformers, PyTorch, CUDA, and GPU versions;
- quantization mode, prompt, deterministic generation settings, and maximum
  answer length;
- validated resize, resampling, center-crop, rescaling, and normalization
  settings from the loaded image processor;
- exact resolved hidden-state indices and vision feature-selection strategy;
- annotation and uncertainty policy;
- smoke and production storage formats.

Resumption compares the entire canonical JSON configuration. Any mismatch
requires a new output directory. Existing incomplete or mismatched records are
never treated as complete.

## Stage naming and index semantics

Transformer hidden-state tuples contain the embedding output at index 0 and the
final model output at index `num_hidden_layers`. The initial policy resolves:

- `early`: the 25% depth boundary;
- `middle`: the 50% depth boundary;
- `late`: hidden-state index `-2`;
- `final`: hidden-state index `num_hidden_layers`.

For the vision tower, `late` is resolved from the model's actual
`vision_feature_layer`, so it is exactly the representation supplied to the
projector. Every resolved non-projector index is written to both the run config
and record metadata. `projector.output` has no transformer-layer index.

The required stage keys are:

```text
vision.early
vision.middle
vision.late
vision.final
projector.output
llm.early
llm.middle
llm.late
llm.final
```

Each value is a detached finite tensor shaped `[patch_count, hidden_size]`.
Smoke tensors are stored as float16. Vision dimensions may differ from language
dimensions; projector and all LLM dimensions must match.

## Stable spatial identity

The initial extractor supports classic LLaVA's `default` vision feature policy:
one CLS token is removed and the remaining row-major vision patches correspond
one-to-one with projector outputs and expanded LLM image-token positions.

Each record stores:

- original image size `(width, height)`;
- processed model size `(height, width)`;
- shortest-edge resized size, x/y scale, and integer center-crop offsets;
- patch grid `(rows, columns)`;
- row-major patch centers in processed-image pixels and normalized coordinates;
- expanded prompt positions occupied by visual tokens;
- original and crop-mapped CUB bounding box;
- original and crop-mapped visible-part coordinates.

The validator requires identical patch counts at all nine stages and verifies
that both coordinate arrays have shape `[patch_count, 2]`.

## CUB identity and labels

Per-image metadata contains:

- image ID and relative path;
- class ID and class name;
- official CUB split and development-pilot split;
- dataset version/identity;
- the complete official attribute vocabulary;
- raw binary presence, certainty ID/name, and annotation time for every
  available image attribute;
- explicit state: `present`, `absent`, `uncertain`, `not_visible`, or `missing`;
- `primary_target`, which is binary only for `probably` and `definitely` rows;
- raw and mapped part IDs, visibility, and coordinates.

`guess`, `not visible`, and missing annotations remain in the record but are
masked from the primary probe target. The cache never resolves uncertainty by
looking at validation performance.

The initial attribute-family configuration is
`configs/phase2_attribute_groups.json`. It predeclares crown-colour,
wing-pattern, bill-shape, breast-colour, and head-pattern families. The subset
generator uses only development-training image IDs and requires the configured
positive, negative, certainty, and missingness thresholds. Validation labels
are never read for attribute selection.

## Generation record

The generation section stores:

- fixed prompt and fully rendered chat prompt;
- prompt token count;
- decoded answer text;
- generated token IDs and token strings;
- one raw, pre-generation-processor full-vocabulary logit row from the language
  model for each generated token;
- deterministic greedy-decoding configuration.

Answer-token logits are captured by the language-model forward hook immediately
before greedy token selection. They are aligned row-for-row with generated token
IDs and must be finite. They measure the normal decoding trajectory; they are
not a causal intervention.

## Smoke layout and atomicity

The one-image smoke output is intentionally diagnostic:

```text
representation_cache_smoke/
  run_config.json
  attribute_subset.json
  records/
    <image_id>.pt
  index.json
  smoke_summary.json
```

`run_config.json`, the record, index, and summary are written through temporary
files followed by atomic rename. The `.pt` record is used only to obtain a real
one-image byte count before selecting the production shard format. Do not scale
this one-record-per-file layout to 240 images.

After the Kaggle smoke run, choose a small number of indexed safetensors, HDF5,
or chunked-memmap shards based on measured size and reload behavior. A single
monolithic file and thousands of per-layer files are prohibited.

## One-image acceptance checks

The smoke run passes only if:

1. the Phase 1 gate artifact has `passed: true`;
2. the model revision and stage indices resolve exactly;
3. all model parameters remain frozen;
4. all nine stage tensors are detached, finite, rank 2, and patch aligned;
5. crop geometry, patch centers, box, and visible parts are recorded;
6. attribute presence/certainty/missingness is complete and reloadable;
7. answer tokens and logits are aligned and finite;
8. the record survives save/reload validation;
9. a conflicting run config is rejected;
10. bytes/image, projected 240-image bytes, extraction time, projected runtime,
    and peak GPU memory are reported.

Stop after this smoke run. The 240-image extraction requires a reviewed smoke
summary and an explicit production shard-format decision.

## Production shard decision

The accepted Kaggle smoke measured:

- vision hidden size: 1,024;
- projector/LLM hidden size: 4,096;
- peak allocated GPU memory: 2,051,509,760 bytes;
- projected 240-image `.pt` sizing footprint: 7,300,734,000 bytes.

The production cache uses twelve 20-image safetensors shards. Each tensor key is
namespaced by image ID and its structural record path. A companion JSON file
contains all non-tensor metadata plus exact tensor references, allowing the
original validated nested record to be reconstructed without pickle. The
layout is:

```text
phase2_stage_cache/
  run_config.json
  index.json
  extraction_summary.json
  validation_report.json
  shards/
    stage-00000-of-00012.safetensors
    stage-00000-of-00012.json
    ...
```

The extractor audits all 240 requested CUB attribute rows before model loading,
rejects any official-test image, and requires the exact smoke-tested gate,
attribute subset, model revision, prompt, and generation settings. A shard is
saved through an adjacent temporary tensor file, reloaded, fully validated, and
hashed before its records enter the atomic index. Resumption is allowed only for
a deterministic prefix of the manifest under an identical canonical config.

The independent validator verifies shard hashes, reconstructs and validates all
240 records, checks the 160/80 development split, confirms all nine stages, and
asserts that the official CUB test split remains untouched.
