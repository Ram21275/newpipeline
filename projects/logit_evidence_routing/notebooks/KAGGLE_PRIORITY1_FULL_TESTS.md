# Kaggle executable cells: Priority 0 robustness and Priority 1 interventions

Run these cells in order in a Kaggle GPU notebook with Internet enabled.  They
use only CUB development-validation decisions drawn from the official training
partition.  They do not create or inspect an official-test manifest, and they
do not modify the frozen Phase 8 protocol.

Cell 1 requires this continuation to have been committed and pushed to the
remote `feat/iclr` branch.  Its entry-point audit deliberately stops if Kaggle
has fetched an older commit.

Required retained inputs in `/kaggle/working` (or attach them as Kaggle datasets
and copy them to these paths before Cell 3):

```text
/kaggle/working/multiphase_development_5b55fde6a51f
/kaggle/working/phase1b_corrected/cache
/kaggle/working/phase2_stage_cache
/kaggle/working/reviewed_phase4_db1219637723
/kaggle/working/logit_evidence_followup_complete
```

The final directory can instead be reconstructed automatically from an attached
`logit_evidence_followup_complete.tar.gz` in Cell 3.  The CUB dataset must be
attached in its standard `CUB_200_2011` layout.

## Cell 1 - update the canonical branch and verify entry points

```bash
%%bash
set -euo pipefail
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR
REPO=/kaggle/working/newpipeline
if [ ! -e "$REPO" ]; then
  git clone --branch feat/iclr --single-branch \
    https://github.com/Ram21275/newpipeline.git "$REPO"
elif [ ! -d "$REPO/.git" ]; then
  echo "ERROR: $REPO exists but is not a Git checkout. Rename/remove it or set REPO correctly." >&2
  exit 1
else
  git -C "$REPO" fetch origin feat/iclr
  git -C "$REPO" checkout feat/iclr
  git -C "$REPO" pull --ff-only origin feat/iclr
fi
PROJECT="$REPO/projects/logit_evidence_routing"
for file in \
  scripts/repair_replication_generation_fields.py \
  scripts/prepare_priority1_inputs.py \
  scripts/run_paper_robustness.py \
  scripts/export_phase10_balanced_selector_metadata.py \
  scripts/run_phase10_priority1.py \
  scripts/extract_phase10_llava_interventions.py \
  scripts/plot_phase4_diverse_gallery.py \
  scripts/build_submission_archive.py \
  configs/phase10_priority1.json; do
  if [ ! -f "$PROJECT/$file" ]; then
    echo "ERROR: fetched commit is missing $PROJECT/$file" >&2
    echo "The Priority 1 continuation has not been pushed to origin/feat/iclr." >&2
    exit 1
  fi
done
git -C "$REPO" log -1 --oneline
```

## Cell 2 - install dependencies and run the complete CPU test suite

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
cd "$PROJECT"
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR
python3 -m pip install -r "$PROJECT/requirements-kaggle.txt"
PYTHONPYCACHEPREFIX=/tmp/lger_pycache python3 -m pytest -ra --tb=short
```

## Cell 3 - recover and validate every retained input

This replacement uses bounded metadata discovery and never walks through CUB
images or intervention-record directories. Attached Kaggle datasets are linked
read-only into the expected `/kaggle/working` paths. If an input is absent, the
error names the missing retained dataset immediately.

```bash
%%bash
set -euo pipefail
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/prepare_priority1_inputs.py"
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_ROOT="$(python3 -c 'import json; print(json.load(open("/kaggle/working/priority1_input_paths.json"))["cub_root"])')"
python3 - "$RUN" "$FOLLOWUP" "$CUB_ROOT" <<'PY'
import csv, json, sys
from pathlib import Path
run, followup, cub = map(Path, sys.argv[1:])
reports = {
    "multiphase": run / "multiphase_status.json",
    "phase5": run / "phase5/phase5_run_report.json",
    "phase7_bridge": run / "phase7_bridge/token_metadata_report.json",
    "stage_cache": Path("/kaggle/working/phase2_stage_cache/validation_report.json"),
    "localizer": Path("/kaggle/working/phase1b_corrected/cache/extraction_config.json"),
    "phase4": Path("/kaggle/working/reviewed_phase4_db1219637723/phase4_run_report.json"),
    "phase9": followup / "phase9_llava_full/phase9_extraction_report.json",
}
for label, path in reports.items():
    assert path.is_file(), f"missing {label}: {path}"
    report = json.loads(path.read_text())
    assert report.get("status", "PASS") == "PASS", (label, report.get("status"))
    assert report.get("official_test_images_used", report.get("official_test_images", 0)) == 0
    print(label, "PASS", path)
required = (
    run / "phase7_bridge/development_train_stage_means.safetensors",
    followup / "cub_replication_manifest.csv",
    followup / "cub_llava_full/replication_vqa_decisions.csv",
    followup / "cub_qwen_full/replication_vqa_decisions.csv",
    followup / "phase9_plan.json",
    followup / "phase9_llava_full/intervention_outcomes.csv",
    Path("/kaggle/working/phase1b_corrected/cache/records"),
    Path("/kaggle/working/phase2_stage_cache/index.json"),
    cub / "images",
)
for path in required:
    assert path.exists(), f"missing required input: {path}"
with (followup / "cub_replication_manifest.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 834
assert sum(int(row["target"]) == 0 for row in rows) == 417
assert sum(int(row["target"]) == 1 for row in rows) == 417
assert all(row["split"] == "val" and row["official_split"] == "train" for row in rows)
print("manifest PASS: 834 decisions, 417/417 targets, official_test_images_used=0")
print("CUB_ROOT", cub)
PY
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
df -h /kaggle/working
```

## Cell 4 - repair retained free-generation fields on CPU

This recomputes only `parsed_answer` and `generation_correct` from retained
`generated_text`.  It performs no model forward pass.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
OUT=/kaggle/working/priority0_results
python3 "$PROJECT/scripts/repair_replication_generation_fields.py" \
  --input llava "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --input qwen "$FOLLOWUP/cub_qwen_full/replication_vqa_decisions.csv" \
  --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --output-dir "$OUT/generation_reanalysis"
python3 "$PROJECT/scripts/analyze_replication_vqa.py" \
  --input llava "$OUT/generation_reanalysis/llava_replication_vqa_decisions.csv" \
  --input qwen "$OUT/generation_reanalysis/qwen_replication_vqa_decisions.csv" \
  --output-dir "$OUT/generation_analysis" \
  --bootstrap-samples 10000 --confidence-level 0.95 --seed 20260913
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/kaggle/working/priority0_results/generation_reanalysis/generation_reanalysis_report.json').read_text())
assert r['status']=='PASS' and r['boolean_no_preserved'] is True
assert r['official_test_images_used']==0 and r['model_extraction_performed'] is False
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 5 - run Priority 0 class, attribute, simultaneous, and influence checks

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
OUT=/kaggle/working/priority0_results
python3 "$PROJECT/scripts/run_paper_robustness.py" \
  --phase9-outcomes "$FOLLOWUP/phase9_llava_full/intervention_outcomes.csv" \
  --phase9-plan "$FOLLOWUP/phase9_plan.json" \
  --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --replication llava "$OUT/generation_reanalysis/llava_replication_vqa_decisions.csv" \
  --replication qwen "$OUT/generation_reanalysis/qwen_replication_vqa_decisions.csv" \
  --output-dir "$OUT/robustness" \
  --bootstrap-samples 10000 --confidence-level 0.95 --seed 20260914
python3 - <<'PY'
import csv, json
from pathlib import Path
root=Path('/kaggle/working/priority0_results/robustness')
report=json.loads((root/'priority0_robustness_report.json').read_text())
assert report['status']=='PASS' and report['phase9_family_size']==6
print('\nSIMULTANEOUS PHASE 9 FAMILY')
for row in csv.DictReader((root/'phase9_simultaneous_inference.csv').open()):
    print(row['estimand'], f"estimate={float(row['estimate']):.6f}",
          f"simultaneous CI=[{float(row['simultaneous_ci_low']):.6f}, {float(row['simultaneous_ci_high']):.6f}]")
print('\nINFLUENCE')
for row in csv.DictReader((root/'phase9_influence_summary.csv').open()):
    print(row['estimand'], f"positive={float(row['fraction_positive']):.3f}",
          f"LOIO=[{float(row['leave_one_image_out_min']):.6f}, {float(row['leave_one_image_out_max']):.6f}]")
PY
```

## Cell 6 - render a fixed diverse Phase 4 qualitative gallery

The selection rule takes two distinct eligible validation examples per
attribute group in policy order; it never reads a localization score.  Retained
dense scores are reused when present.  Otherwise only the ten selected gallery
images are scored with frozen CLIP.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/plot_phase4_diverse_gallery.py" \
  --stage-cache /kaggle/working/phase2_stage_cache \
  --localizer-cache /kaggle/working/phase1b_corrected/cache \
  --phase4-dir /kaggle/working/reviewed_phase4_db1219637723 \
  --output-dir /kaggle/working/phase4_diverse_gallery \
  --per-group 2
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/kaggle/working/phase4_diverse_gallery/gallery_report.json').read_text())
assert r['status']=='PASS' and r['official_test_images_used']==0
assert r['selection_uses_performance_metrics'] is False
assert r['attribute_groups']==5 and r['panels']==10
print(json.dumps(r,indent=2,sort_keys=True))
PY
ls -lh /kaggle/working/phase4_diverse_gallery/qualitative
```

## Cell 7 - export balanced selector metadata from retained caches

This cell performs no model extraction.  It expands the cached image-level
Vision-CLS and generic bird/birds Logit-Concept maps to all 834 decisions.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
python3 "$PROJECT/scripts/export_phase10_balanced_selector_metadata.py" \
  --stage-cache /kaggle/working/phase2_stage_cache \
  --localizer-cache /kaggle/working/phase1b_corrected/cache \
  --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --behavior-decisions "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --output-dir /kaggle/working/phase10_selector_metadata
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/kaggle/working/phase10_selector_metadata/selector_metadata_report.json').read_text())
assert r['status']=='PASS' and r['model_extraction_performed'] is False
assert r['decisions']==834 and r['target_counts']=={'0':417,'1':417}
assert r['official_test_images_used']==0
print('rows',r['token_rows'],'images',r['images'],'patches',r['patch_count'])
PY
```

## Cell 8 - build and audit the complete Priority 1 plan

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
python3 "$PROJECT/scripts/run_phase10_priority1.py" plan \
  --token-metadata /kaggle/working/phase10_selector_metadata/selector_token_metadata.csv \
  --config "$PROJECT/configs/phase10_priority1.json" \
  --output /kaggle/working/phase10_plan.json \
  --seed 909 --norm-quantiles 4
python3 - <<'PY'
import csv, json
from pathlib import Path
p=json.loads(Path('/kaggle/working/phase10_plan.json').read_text())
assert p['status']=='PASS' and p['decision_count']==834
assert p['target_counts']=={'0':417,'1':417}
assert p['dose_k_values']==[8,16,32,64] and p['balanced_k']==32
assert p['intervention_count']==35028
assert p['official_test_images_used']==0 and p['phase8_protocol_unchanged'] is True
old={(r['decision_id'],r['intervention_id']) for r in csv.DictReader(
    Path('/kaggle/working/logit_evidence_followup_complete/phase9_llava_full/intervention_outcomes.csv').open())}
new={(r['decision_id'],r['intervention_id']) for r in p['records']}
print('interventions',len(new),'exactly reusable retained rows',len(old & new),
      'new forwards',len(new-old))
assert len(old & new)>0
PY
```

## Cell 9 - Priority 1 LLaVA smoke

The smoke runs every K, selector, stage, and matched-random intervention for one
negative and one positive decision.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_phase10_llava_interventions.py" \
  --mode smoke --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --behavior-decisions "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --means-file "$RUN/phase7_bridge/development_train_stage_means.safetensors" \
  --plan /kaggle/working/phase10_plan.json --cub-root "$CUB_ROOT" \
  --reuse-outcomes "$FOLLOWUP/phase9_llava_full/intervention_outcomes.csv" \
  --output-dir /kaggle/working/phase10_llava_smoke
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/kaggle/working/phase10_llava_smoke/phase10_extraction_report.json').read_text())
assert r['status']=='PASS' and r['mode']=='smoke' and r['decisions']==2
assert r['target_counts']=={'0':1,'1':1} and r['interventions']==84
assert r['official_test_images_used']==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 10 - Priority 1 LLaVA stratified pilot

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_phase10_llava_interventions.py" \
  --mode pilot --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --behavior-decisions "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --means-file "$RUN/phase7_bridge/development_train_stage_means.safetensors" \
  --plan /kaggle/working/phase10_plan.json --cub-root "$CUB_ROOT" \
  --reuse-outcomes "$FOLLOWUP/phase9_llava_full/intervention_outcomes.csv" \
  --smoke-dir /kaggle/working/phase10_llava_smoke --pilot-decisions 20 \
  --output-dir /kaggle/working/phase10_llava_pilot
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/kaggle/working/phase10_llava_pilot/phase10_extraction_report.json').read_text())
assert r['status']=='PASS' and r['mode']=='pilot' and r['decisions']==20
assert r['interventions']==840 and r['official_test_images_used']==0
assert set(r['target_counts'])=={'0','1'} and min(r['target_counts'].values())>0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 11 - full Priority 1 LLaVA run

The output is resumable at one JSON record per intervention.  Re-running this
cell after an interruption resumes from the completed record files.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
FOLLOWUP=/kaggle/working/logit_evidence_followup_complete
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_phase10_llava_interventions.py" \
  --mode full --manifest "$FOLLOWUP/cub_replication_manifest.csv" \
  --behavior-decisions "$FOLLOWUP/cub_llava_full/replication_vqa_decisions.csv" \
  --means-file "$RUN/phase7_bridge/development_train_stage_means.safetensors" \
  --plan /kaggle/working/phase10_plan.json --cub-root "$CUB_ROOT" \
  --reuse-outcomes "$FOLLOWUP/phase9_llava_full/intervention_outcomes.csv" \
  --smoke-dir /kaggle/working/phase10_llava_smoke \
  --pilot-dir /kaggle/working/phase10_llava_pilot --pilot-decisions 20 \
  --output-dir /kaggle/working/phase10_llava_full
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/kaggle/working/phase10_llava_full/phase10_extraction_report.json').read_text())
assert r['status']=='PASS' and r['mode']=='full' and r['decisions']==834
assert r['target_counts']=={'0':417,'1':417} and r['interventions']==35028
assert r['reused_interventions']+r['executed_interventions']==r['interventions']
assert r['official_test_images_used']==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 12 - aggregate target interactions and K dose-response

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/run_phase10_priority1.py" aggregate \
  --outcomes /kaggle/working/phase10_llava_full/intervention_outcomes.csv \
  --plan /kaggle/working/phase10_plan.json \
  --output /kaggle/working/phase10_analysis.json \
  --bootstrap-samples 10000 --confidence-level 0.95 --seed 20260914
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/kaggle/working/phase10_analysis.json').read_text())
assert r['status']=='PASS' and r['decision_count']==834
assert r['official_test_images_used']==0 and r['phase8_protocol_unchanged'] is True
for section in ('selector_specific_contrasts','selector_target_interactions',
                'selector_by_target_interactions','dose_effects',
                'dose_simultaneous_intervals','dose_log2_trends'):
    print('\n'+section.upper())
    for row in r[section]:
        print(row)
PY
```

## Cell 13 - verify the old archive if it still exists, then build a new deterministic archive

If the original archive and sidecar are both still present, the first block
closes the historical outer-archive audit.  Regardless, the new archive has a
deterministic internal file manifest and a freshly verified SHA-256.

```bash
%%bash
set -euo pipefail
cd /kaggle/working
if [ -f logit_evidence_followup_complete.tar.gz ] && \
   [ -f logit_evidence_followup_complete.sha256 ]; then
  sha256sum -c logit_evidence_followup_complete.sha256
else
  echo 'Historical archive bytes are unavailable; the new deterministic archive will supersede that package.'
fi
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/build_submission_archive.py" \
  --root /kaggle/working \
  --include logit_evidence_followup_complete \
  --include priority0_results \
  --include phase4_diverse_gallery \
  --include phase10_selector_metadata \
  --include phase10_plan.json \
  --include phase10_llava_smoke \
  --include phase10_llava_pilot \
  --include phase10_llava_full \
  --include phase10_analysis.json \
  --output /kaggle/working/lger_submission_results.tar.gz
sha256sum -c /kaggle/working/lger_submission_results.tar.gz.sha256
python3 - <<'PY'
import json, tarfile
from pathlib import Path
archive=Path('/kaggle/working/lger_submission_results.tar.gz')
audit=json.loads(Path(str(archive)+'.audit.json').read_text())
assert audit['status']=='PASS' and audit['official_test_images_used']==0
assert audit['internal_manifest_verified'] is True
with tarfile.open(archive,'r:gz') as tar:
    manifest=json.load(tar.extractfile('ARCHIVE_MANIFEST.json'))
assert manifest['file_count']==audit['internal_file_count']
print(json.dumps(audit,indent=2,sort_keys=True))
PY
ls -lh /kaggle/working/lger_submission_results.tar.gz*
```

Download all three files from the Kaggle Output pane:

```text
/kaggle/working/lger_submission_results.tar.gz
/kaggle/working/lger_submission_results.tar.gz.sha256
/kaggle/working/lger_submission_results.tar.gz.audit.json
```

After download, provide the archive and its two audit sidecars for paper review.
Do not run any official-test cell before that review is complete.
