# CUB confirmatory continuation: exact K=34 and multimodal DeCo/DoLa

This runbook is intentionally split by account. The retained-artifact account
runs the only job that needs the 72 MB selector metadata, development stage
means, and retained Phase 10 outcomes. The fresh account runs Qwen and
training-free multimodal DeCo with DoLa as a direct comparator from the
compact transfer package.

All development commands reject official-test images. Do not create an
official-test manifest in these sessions: the older frozen Phase 8 policy fixes
images but does not fully fix the attribute-decision cohort for a feasible
balanced evaluation. That missing choice must be committed before first test
access.

## Budget and stopping rules

- Retained account, five GPU hours: exact common K=34 LLaVA smoke/pilot/full
  (estimated 3.0–3.3 h), LLaVA DeCo/DoLa architecture/smoke/pilot (0.2–0.4 h),
  answer-token smoke/pilot (0.3–0.5 h), final audits/archive and contingency
  (0.8–1.3 h).
- Fresh account: Qwen layout gate and DeCo/DoLa smoke/pilot first (0.3–0.6 h
  including checkpoint load); Qwen development-full only after the pilot
  gate (roughly 1.0–1.5 h). A complete six-condition Qwen answer-token run is
  about 4.4 GPU h from retained timing and should be a separate session.
- Disk: reserve 25 GB for Qwen (the frozen config reports 16.6 GB checkpoint
  bytes, plus Hugging Face cache, environment, and outputs), 2 GB for CUB, and
  under 1 GB for results. The transfer archive deliberately contains no model
  weights, activation caches, test images, or record directories.
- Never advance from architecture → smoke → pilot → full unless the previous
  report says `PASS`. Rerunning a failed extraction resumes validated per-record
  JSON rather than starting again.

## Retained-artifact account

### Cell 1 — fetch the audited branch and check the environment

```bash
%%bash
set -Eeuo pipefail
REPO=/kaggle/working/newpipeline
git -C "$REPO" fetch origin feat/iclr
git -C "$REPO" checkout feat/iclr
git -C "$REPO" pull --ff-only origin feat/iclr
PROJECT="$REPO/projects/logit_evidence_routing"
python3 -m pip install -r "$PROJECT/requirements-kaggle.txt"
for path in \
  "$PROJECT/configs/phase10_exact_k34.json" \
  "$PROJECT/configs/phase10_exact_k34_analysis.json" \
  "$PROJECT/scripts/extract_multimodal_dola.py" \
  "$PROJECT/scripts/analyze_multimodal_layer_decoding.py" \
  "$PROJECT/scripts/extract_answer_token_robustness.py"; do
  test -f "$path"
done
PYTHONPYCACHEPREFIX=/tmp/lger_pycache PYTHONPATH="$PROJECT/src" \
  python3 -m pytest -q \
  "$PROJECT/tests/test_phase10_confirmatory.py" \
  "$PROJECT/tests/test_multimodal_dola.py" \
  "$PROJECT/tests/test_layer_decoding_analysis.py" \
  "$PROJECT/tests/test_answer_robustness.py" \
  "$PROJECT/tests/test_spatial_sweep.py"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h /kaggle/working
```

### Cell 2 — bind retained paths and validate identities

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/prepare_priority1_inputs.py"
python3 - <<'PY'
import csv, hashlib, json
from pathlib import Path

root=Path('/kaggle/working')
run=root/'multiphase_development_5b55fde6a51f'
followup=root/'logit_evidence_followup_complete'
required=[
    root/'phase10_selector_metadata/selector_token_metadata.csv',
    root/'phase10_llava_full/intervention_outcomes.csv',
    run/'phase7_bridge/development_train_stage_means.safetensors',
    followup/'cub_replication_manifest.csv',
    followup/'cub_llava_full/replication_vqa_decisions.csv',
    followup/'cub_qwen_full/replication_vqa_decisions.csv',
]
for path in required:
    assert path.is_file(), f'missing retained input: {path}'
rows=list(csv.DictReader(required[3].open()))
assert len(rows)==834
assert sum(int(r['target'])==0 for r in rows)==417
assert sum(int(r['target'])==1 for r in rows)==417
assert all(r['split']=='val' and r['official_split']=='train' for r in rows)
print('RETAINED INPUTS PASS; official_test_images_used=0')
PY
```

### Cell 3 — create and audit the exact common K=34 plan (CPU only)

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/run_phase10_priority1.py" plan \
  --token-metadata /kaggle/working/phase10_selector_metadata/selector_token_metadata.csv \
  --config "$PROJECT/configs/phase10_exact_k34.json" \
  --output /kaggle/working/phase10_exact_k34_plan.json \
  --seed 909 --norm-quantiles 4
python3 - <<'PY'
import csv, json
from pathlib import Path
p=json.loads(Path('/kaggle/working/phase10_exact_k34_plan.json').read_text())
assert p['status']=='PASS' and p['balanced_k']==34 and p['dose_k_values']==[32,34]
assert p['decision_count']==834 and p['target_counts']=={'0':417,'1':417}
assert p['matching_spatial_fallback_count']==0
for key,value in p['matching_details'].items():
    if key.endswith('_k_34'):
        assert value['spatial_mode']=='exact_bird_box'
        assert set(value['norm_quantiles_by_selector'].values())=={4}
old={(r['decision_id'],r['intervention_id']) for r in csv.DictReader(
    Path('/kaggle/working/phase10_llava_full/intervention_outcomes.csv').open())}
new={(r['decision_id'],r['intervention_id']) for r in p['records']}
assert len(new)==21684 and len(old & new)==8340 and len(new-old)==13344
print('PLAN PASS: 21,684 outcomes; 8,340 reused; 13,344 new forwards')
PY
```

### Cell 4 — exact K=34 smoke

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_ROOT="$(python3 -c 'import json; print(json.load(open("/kaggle/working/priority1_input_paths.json"))["cub_root"])')"
python3 -u "$PROJECT/scripts/extract_phase10_llava_interventions.py" \
  --mode smoke --experiment-config "$PROJECT/configs/phase10_exact_k34.json" \
  --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --behavior-decisions "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --means-file "$RUN/phase7_bridge/development_train_stage_means.safetensors" \
  --plan /kaggle/working/phase10_exact_k34_plan.json --cub-root "$CUB_ROOT" \
  --reuse-outcomes /kaggle/working/phase10_llava_full/intervention_outcomes.csv \
  --output-dir /kaggle/working/phase10_exact_k34_smoke
python3 -c "import json; r=json.load(open('/kaggle/working/phase10_exact_k34_smoke/phase10_extraction_report.json')); assert r['status']=='PASS' and r['executed_interventions']==32; print(r)"
```

### Cell 5 — exact K=34 balanced pilot

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_ROOT="$(python3 -c 'import json; print(json.load(open("/kaggle/working/priority1_input_paths.json"))["cub_root"])')"
python3 -u "$PROJECT/scripts/extract_phase10_llava_interventions.py" \
  --mode pilot --experiment-config "$PROJECT/configs/phase10_exact_k34.json" \
  --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --behavior-decisions "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --means-file "$RUN/phase7_bridge/development_train_stage_means.safetensors" \
  --plan /kaggle/working/phase10_exact_k34_plan.json --cub-root "$CUB_ROOT" \
  --reuse-outcomes /kaggle/working/phase10_llava_full/intervention_outcomes.csv \
  --smoke-dir /kaggle/working/phase10_exact_k34_smoke --pilot-decisions 20 \
  --output-dir /kaggle/working/phase10_exact_k34_pilot
python3 -c "import json; r=json.load(open('/kaggle/working/phase10_exact_k34_pilot/phase10_extraction_report.json')); assert r['status']=='PASS' and r['executed_interventions']==320; print(r)"
```

### Cell 6 — exact K=34 full development run and confirmatory analysis

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_ROOT="$(python3 -c 'import json; print(json.load(open("/kaggle/working/priority1_input_paths.json"))["cub_root"])')"
python3 -u "$PROJECT/scripts/extract_phase10_llava_interventions.py" \
  --mode full --experiment-config "$PROJECT/configs/phase10_exact_k34.json" \
  --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --behavior-decisions "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --means-file "$RUN/phase7_bridge/development_train_stage_means.safetensors" \
  --plan /kaggle/working/phase10_exact_k34_plan.json --cub-root "$CUB_ROOT" \
  --reuse-outcomes /kaggle/working/phase10_llava_full/intervention_outcomes.csv \
  --smoke-dir /kaggle/working/phase10_exact_k34_smoke \
  --pilot-dir /kaggle/working/phase10_exact_k34_pilot --pilot-decisions 20 \
  --output-dir /kaggle/working/phase10_exact_k34_full
python3 -u "$PROJECT/scripts/run_phase10_confirmatory_analysis.py" \
  --outcomes /kaggle/working/phase10_exact_k34_full/intervention_outcomes.csv \
  --plan /kaggle/working/phase10_exact_k34_plan.json \
  --decision-metadata /kaggle/working/phase10_selector_metadata/selector_token_metadata.csv \
  --behavior llava "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --behavior qwen "$FOLLOWUP/cub_qwen_full/replication_vqa_decisions.csv" \
  --config "$PROJECT/configs/phase10_exact_k34_analysis.json" \
  --output-dir /kaggle/working/phase10_exact_k34_analysis
python3 -c "import json; r=json.load(open('/kaggle/working/phase10_exact_k34_analysis/phase10_confirmatory_analysis.json')); assert r['status']=='PASS' and r['official_test_images_used']==0; print(r['claim_boundary'])"
```

### Cell 7 — LLaVA multimodal DeCo/DoLa architecture → smoke → pilot

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_ROOT="$(python3 -c 'import json; print(json.load(open("/kaggle/working/priority1_input_paths.json"))["cub_root"])')"
COMMON="--model-config $PROJECT/configs/llava_full_replication.json --experiment-config $PROJECT/configs/multimodal_dola.json --manifest $FOLLOWUP/cub_replication_manifest.csv --image-root $CUB_ROOT/images --candidate-start 20 --candidate-end 29 --candidate-stride 1"
python3 -u "$PROJECT/scripts/extract_multimodal_dola.py" --mode architecture $COMMON \
  --output-dir /kaggle/working/llava_dola_architecture
python3 -u "$PROJECT/scripts/extract_multimodal_dola.py" --mode smoke $COMMON \
  --architecture-dir /kaggle/working/llava_dola_architecture \
  --output-dir /kaggle/working/llava_dola_smoke
python3 -u "$PROJECT/scripts/extract_multimodal_dola.py" --mode pilot $COMMON \
  --architecture-dir /kaggle/working/llava_dola_architecture \
  --smoke-dir /kaggle/working/llava_dola_smoke --pilot-decisions 48 \
  --output-dir /kaggle/working/llava_dola_pilot
python3 -c "import json; r=json.load(open('/kaggle/working/llava_dola_pilot/multimodal_dola_report.json')); assert r['status']=='PASS' and len(r['observed_binary_deco_anchor_layers'])>1 and len(r['observed_binary_premature_layers'])>1; print(r)"
python3 -u "$PROJECT/scripts/analyze_multimodal_layer_decoding.py" \
  --input /kaggle/working/llava_dola_pilot/multimodal_dola_decisions.csv \
  --output-dir /kaggle/working/llava_dola_pilot_analysis \
  --bootstrap-samples 10000 --seed 20260915
python3 -c "import json; r=json.load(open('/kaggle/working/llava_dola_pilot_analysis/multimodal_layer_decoding_analysis.json')); print(r['primary_promotion_snapshot'])"
```

If the final assertion fails, retain the non-degenerate failure and do not run
development-full layer decoding. A single selected layer for every decision
means dynamic routing has not been demonstrated. DeCo is the primary method;
DoLa is retained on the identical examples and interval as its comparator.

### Cell 8 — LLaVA answer-token smoke → pilot

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_ROOT="$(python3 -c 'import json; print(json.load(open("/kaggle/working/priority1_input_paths.json"))["cub_root"])')"
COMMON="--model-config $PROJECT/configs/llava_full_replication.json --experiment-config $PROJECT/configs/answer_token_robustness.json --manifest $FOLLOWUP/cub_replication_manifest.csv --image-root $CUB_ROOT/images"
python3 -u "$PROJECT/scripts/extract_answer_token_robustness.py" --mode smoke $COMMON \
  --output-dir /kaggle/working/llava_answer_smoke
python3 -u "$PROJECT/scripts/extract_answer_token_robustness.py" --mode pilot $COMMON \
  --smoke-dir /kaggle/working/llava_answer_smoke --pilot-decisions 48 \
  --output-dir /kaggle/working/llava_answer_pilot
python3 -u "$PROJECT/scripts/analyze_answer_token_robustness.py" \
  --input /kaggle/working/llava_answer_pilot/answer_token_robustness_decisions.csv \
  --output-dir /kaggle/working/llava_answer_pilot_analysis \
  --bootstrap-samples 10000 --seed 20260915
python3 -c "import json; r=json.load(open('/kaggle/working/llava_answer_pilot/answer_token_robustness_report.json')); assert r['status']=='PASS'; print(r)"
```

### Cell 9 — deterministic retained-account result archive

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/build_submission_archive.py" \
  --root /kaggle/working \
  --include phase10_exact_k34_plan.json \
  --include phase10_exact_k34_smoke/evaluation_config.json \
  --include phase10_exact_k34_smoke/phase10_extraction_report.json \
  --include phase10_exact_k34_pilot/evaluation_config.json \
  --include phase10_exact_k34_pilot/phase10_extraction_report.json \
  --include phase10_exact_k34_full/evaluation_config.json \
  --include phase10_exact_k34_full/intervention_outcomes.csv \
  --include phase10_exact_k34_full/phase10_extraction_report.json \
  --include phase10_exact_k34_analysis \
  --include llava_dola_architecture/multimodal_dola_report.json \
  --include llava_dola_smoke/multimodal_dola_report.json \
  --include llava_dola_pilot/multimodal_dola_decisions.csv \
  --include llava_dola_pilot/multimodal_dola_report.json \
  --include llava_dola_pilot_analysis \
  --include llava_answer_smoke/answer_token_robustness_report.json \
  --include llava_answer_pilot/answer_token_robustness_decisions.csv \
  --include llava_answer_pilot/answer_token_robustness_report.json \
  --include llava_answer_pilot_analysis \
  --output /kaggle/working/lger_k34_dola_results.tar.gz
shasum -a 256 -c /kaggle/working/lger_k34_dola_results.tar.gz.sha256
cat /kaggle/working/lger_k34_dola_results.tar.gz.audit.json
```

## Fresh Kaggle account

Attach the compact `lger_kaggle_transfer_20260915.tar.gz` and CUB-200-2011
datasets. Internet is needed for the pinned Qwen checkpoint.

### Cell 10 — clone, install, unpack, and verify the transfer package

```bash
%%bash
set -Eeuo pipefail
git clone --branch feat/iclr --single-branch \
  https://github.com/Ram21275/newpipeline.git /kaggle/working/newpipeline
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 -m pip install -r "$PROJECT/requirements-qwen-kaggle.txt"
ARCHIVE="$(find /kaggle/input -type f -name lger_kaggle_transfer_20260915.tar.gz -print -quit)"
test -n "$ARCHIVE"
mkdir -p /kaggle/working/lger_transfer
tar -xzf "$ARCHIVE" -C /kaggle/working/lger_transfer
python3 - "$ARCHIVE" <<'PY'
import hashlib, json, tarfile, sys
archive=sys.argv[1]
with tarfile.open(archive,'r:gz') as tar:
    manifest=json.load(tar.extractfile('TRANSFER_MANIFEST.json'))
    for row in manifest['files']:
        payload=tar.extractfile(row['path']).read()
        assert len(payload)==row['bytes']
        assert hashlib.sha256(payload).hexdigest()==row['sha256']
assert not manifest['large_model_weights_included']
assert not manifest['large_activation_caches_included']
assert manifest['official_test_images_used']==0
print('TRANSFER PASS', manifest['file_count'], 'files')
PY
PYTHONPYCACHEPREFIX=/tmp/lger_pycache PYTHONPATH="$PROJECT/src" python3 -m pytest -q \
  "$PROJECT/tests/test_qwen_layout.py" \
  "$PROJECT/tests/test_multimodal_dola.py" \
  "$PROJECT/tests/test_layer_decoding_analysis.py"
```

### Cell 11 — discover CUB and run the mandatory Qwen layout gate

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
TRANSFER=/kaggle/working/lger_transfer
CUB_ROOT="$(PYTHONPATH="$PROJECT/src" python3 -c 'from pathlib import Path; from lger.cub import discover_cub_root; print(discover_cub_root(Path("/kaggle/input")))')"
REL="$(python3 -c 'import csv; print(next(csv.DictReader(open("/kaggle/working/lger_transfer/manifests/cub_replication_manifest.csv")))["relative_path"])')"
python3 -u "$PROJECT/scripts/validate_qwen_internal_layout.py" \
  --config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --protocol "$PROJECT/configs/qwen_internal_stages.json" \
  --image "$CUB_ROOT/images/$REL" \
  --output /kaggle/working/qwen_layout_report.json
python3 -c "import json; r=json.load(open('/kaggle/working/qwen_layout_report.json')); assert r['status']=='PASS' and all(r['assertions'].values()); print(r)"
```

### Cell 12 — Qwen DeCo/DoLa architecture → smoke → balanced pilot

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
TRANSFER=/kaggle/working/lger_transfer
CUB_ROOT="$(PYTHONPATH="$PROJECT/src" python3 -c 'from pathlib import Path; from lger.cub import discover_cub_root; print(discover_cub_root(Path("/kaggle/input")))')"
LAYERS="$(python3 -c 'import json; print(json.load(open("/kaggle/working/qwen_layout_report.json"))["language_layers"])')"
START=$(((LAYERS*5+7)/8))
END=$(((LAYERS*7)/8+1))
COMMON="--model-config $PROJECT/configs/qwen25vl_7b_replication.json --experiment-config $PROJECT/configs/multimodal_dola.json --manifest $TRANSFER/manifests/cub_replication_manifest.csv --image-root $CUB_ROOT/images --candidate-start $START --candidate-end $END --candidate-stride 1 --layout-report /kaggle/working/qwen_layout_report.json"
python3 -u "$PROJECT/scripts/extract_multimodal_dola.py" --mode architecture $COMMON \
  --output-dir /kaggle/working/qwen_dola_architecture
python3 -u "$PROJECT/scripts/extract_multimodal_dola.py" --mode smoke $COMMON \
  --architecture-dir /kaggle/working/qwen_dola_architecture \
  --output-dir /kaggle/working/qwen_dola_smoke
python3 -u "$PROJECT/scripts/extract_multimodal_dola.py" --mode pilot $COMMON \
  --architecture-dir /kaggle/working/qwen_dola_architecture \
  --smoke-dir /kaggle/working/qwen_dola_smoke --pilot-decisions 48 \
  --output-dir /kaggle/working/qwen_dola_pilot
python3 -c "import json; r=json.load(open('/kaggle/working/qwen_dola_pilot/multimodal_dola_report.json')); assert r['status']=='PASS' and len(r['observed_binary_deco_anchor_layers'])>1 and len(r['observed_binary_premature_layers'])>1; print(r)"
python3 -u "$PROJECT/scripts/analyze_multimodal_layer_decoding.py" \
  --input /kaggle/working/qwen_dola_pilot/multimodal_dola_decisions.csv \
  --output-dir /kaggle/working/qwen_dola_pilot_analysis \
  --bootstrap-samples 10000 --seed 20260915
python3 -c "import json; r=json.load(open('/kaggle/working/qwen_dola_pilot_analysis/multimodal_layer_decoding_analysis.json')); print(r['primary_promotion_snapshot'])"
```

### Cell 13 — optional Qwen development-full DeCo/DoLa, only after reviewing the pilot

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
TRANSFER=/kaggle/working/lger_transfer
CUB_ROOT="$(PYTHONPATH="$PROJECT/src" python3 -c 'from pathlib import Path; from lger.cub import discover_cub_root; print(discover_cub_root(Path("/kaggle/input")))')"
LAYERS="$(python3 -c 'import json; print(json.load(open("/kaggle/working/qwen_layout_report.json"))["language_layers"])')"
START=$(((LAYERS*5+7)/8))
END=$(((LAYERS*7)/8+1))
python3 -u "$PROJECT/scripts/extract_multimodal_dola.py" \
  --mode development_full \
  --model-config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --experiment-config "$PROJECT/configs/multimodal_dola.json" \
  --manifest "$TRANSFER/manifests/cub_replication_manifest.csv" \
  --image-root "$CUB_ROOT/images" --candidate-start "$START" \
  --candidate-end "$END" --candidate-stride 1 \
  --layout-report /kaggle/working/qwen_layout_report.json \
  --architecture-dir /kaggle/working/qwen_dola_architecture \
  --smoke-dir /kaggle/working/qwen_dola_smoke \
  --pilot-dir /kaggle/working/qwen_dola_pilot \
  --output-dir /kaggle/working/qwen_dola_full
python3 -u "$PROJECT/scripts/analyze_multimodal_layer_decoding.py" \
  --input /kaggle/working/qwen_dola_full/multimodal_dola_decisions.csv \
  --output-dir /kaggle/working/qwen_dola_full_analysis \
  --bootstrap-samples 10000 --seed 20260915
```

### Cell 14 — Qwen answer-token smoke/pilot; full belongs in another session

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
TRANSFER=/kaggle/working/lger_transfer
CUB_ROOT="$(PYTHONPATH="$PROJECT/src" python3 -c 'from pathlib import Path; from lger.cub import discover_cub_root; print(discover_cub_root(Path("/kaggle/input")))')"
COMMON="--model-config $PROJECT/configs/qwen25vl_7b_replication.json --experiment-config $PROJECT/configs/answer_token_robustness.json --manifest $TRANSFER/manifests/cub_replication_manifest.csv --image-root $CUB_ROOT/images"
python3 -u "$PROJECT/scripts/extract_answer_token_robustness.py" --mode smoke $COMMON \
  --output-dir /kaggle/working/qwen_answer_smoke
python3 -u "$PROJECT/scripts/extract_answer_token_robustness.py" --mode pilot $COMMON \
  --smoke-dir /kaggle/working/qwen_answer_smoke --pilot-decisions 48 \
  --output-dir /kaggle/working/qwen_answer_pilot
python3 -u "$PROJECT/scripts/analyze_answer_token_robustness.py" \
  --input /kaggle/working/qwen_answer_pilot/answer_token_robustness_decisions.csv \
  --output-dir /kaggle/working/qwen_answer_pilot_analysis \
  --bootstrap-samples 10000 --seed 20260915
python3 -c "import json; r=json.load(open('/kaggle/working/qwen_answer_pilot/answer_token_robustness_report.json')); assert r['status']=='PASS'; print(r)"
```

The guided multi-head fusion head remains a secondary experiment. Train it only
after a DeCo/DoLa pilot, with the frozen base VLM and the baselines in
`configs/guided_layer_fusion.json`; do not spend official-test access on it
unless its development promotion gate passes.
