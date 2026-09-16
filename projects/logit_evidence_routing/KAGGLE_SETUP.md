# CUB final workflow on a fresh Kaggle session

Status: implementation and CPU tests are complete locally. Real LLaVA/Qwen3 GPU smoke tests and all new official-test inference remain unexecuted.

## Required Kaggle inputs

Attach one complete CUB-200-2011 dataset. The notebook clones the public
`feat/iclr` branch into `/kaggle/working/newpipeline`; Internet must therefore
be enabled. The CUB root must contain:

- `images/`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`, `classes.txt`
- `bounding_boxes.txt`
- `parts/parts.txt`, `parts/part_locs.txt`
- `attributes/attributes.txt`, `attributes/certainties.txt`, `attributes/image_attribute_labels.txt`

An images-only mirror is rejected. The code discovers the root below `/kaggle/input`; it does not assume a Kaggle slug. Enable a GPU and Internet for model downloads. For offline use, attach snapshots of both locked revisions plus compatible wheels, then pass `--local-snapshot` to each model command. Never paste or print an access token in the notebook.

Locked checkpoint revisions:

- `llava-hf/llava-1.5-7b-hf` at `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`
- `Qwen/Qwen3-VL-2B-Instruct` at `89644892e4d85e24eaac8bacfd4f463576704203`
- Partner audit: `Team-M3OW/vlm-hallu` at `1096a8642b2de9ba6a649a60a77f5641ba76db42`

The final requirements deliberately do not reinstall PyTorch. The Qwen3 environment is separate from the historical Qwen2.5 environment.

## Launch order

Clone or update the source first:

```bash
if [ -d /kaggle/working/newpipeline/.git ]; then
  git -C /kaggle/working/newpipeline pull --ff-only origin feat/iclr
else
  git clone --branch feat/iclr --single-branch \
    https://github.com/Ram21275/newpipeline.git \
    /kaggle/working/newpipeline
fi
```

Then set `PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing`
and run:

```bash
cd "$PROJECT"
python -m pip install --no-deps -e .
python -m pip install -r requirements-cub-final-kaggle.txt
export CUB_FINAL=/kaggle/working/cub_final
mkdir -p "$CUB_FINAL"/{environment,manifests,smoke,results,analysis,archive}
python -m cub_final inventory --output "$CUB_FINAL/environment/inventory.json"
python -m cub_final audit --output "$CUB_FINAL/manifests/cub_integrity_audit.json"
python -m cub_final manifests --output-dir "$CUB_FINAL/manifests"
```

Run the training-only real-model smoke checks sequentially. Start with unquantized bf16. Use `--quantization 4bit` only if the smoke establishes that memory requires it; then use the same precision for compared arms and retain that fact in the protocol.

```bash
python -m cub_final smoke --architecture llava --output "$CUB_FINAL/smoke/llava_smoke.json"
python -m cub_final smoke --architecture qwen3 --output "$CUB_FINAL/smoke/qwen3_smoke.json"
python -m cub_final lock --manifest-dir "$CUB_FINAL/manifests" --smoke-dir "$CUB_FINAL/smoke" --output "$CUB_FINAL/FINAL_PROTOCOL.yaml"
```

The lock command refuses to run if either smoke report is absent, failed, or used an official-test image. It never waits for manual approval.

Run the shared experiment one model at a time:

```bash
python -m cub_final run-shared --architecture llava --protocol "$CUB_FINAL/FINAL_PROTOCOL.yaml" --questions "$CUB_FINAL/manifests/question_manifest_expensive.jsonl" --output-dir "$CUB_FINAL/results"
python -m cub_final run-shared --architecture qwen3 --protocol "$CUB_FINAL/FINAL_PROTOCOL.yaml" --questions "$CUB_FINAL/manifests/question_manifest_expensive.jsonl" --output-dir "$CUB_FINAL/results"
```

Each item/condition is an atomic shard keyed by configuration hash. Re-running the same command skips complete keys; changed protocols do not reuse stale shards. Save `/kaggle/working/cub_final` as notebook output before the session ends. On a later session, attach that output and copy it back to `/kaggle/working/cub_final` before resuming.

## Resource and time estimate

The two smoke checks should usually finish in roughly 15–45 minutes total, dominated by downloads and LLaVA loading. The locked shared cohort can contain up to 1,024 attribute decisions. With single-example instrumented forwards, expect approximately 8–20 GPU-hours for LLaVA and 4–10 GPU-hours for Qwen3, plus download time; the actual smoke report's per-forward wall time is authoritative. The full probe/intervention suite is a multi-session run and may require another 10–30 GPU-hours. Do not silently shrink the cohort after reading outcomes.

Disk planning: allow roughly 20–30 GB for model cache plus 5–15 GB for atomic outputs/caches. Load models sequentially. Two GPUs are not treated as a single pooled-memory device.

## What the shared command records

It preserves complete yes/no candidate scores, ordinary generation, final-logit agreement, layer-first-token scores, visual-token counts, timing, memory, crop coordinates, and selector attention. Its DoLa arm is explicitly a yes/no-restricted first-token scoring adaptation; ordinary generated answers are retained separately. For Qwen3 it also runs an upscaled full-image compute bar and treats realized token counts and wall time—not requested pixel area—as authoritative.

The model-specific hidden-state probe and intervention stages continue to use the existing `lger` extraction/intervention modules. They must be migrated onto the generated locked manifests before they count as final-test evidence; the current local delivery does not claim those GPU stages are complete.
