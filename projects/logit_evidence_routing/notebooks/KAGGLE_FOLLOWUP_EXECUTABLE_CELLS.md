# Kaggle executable cells: full LLaVA follow-up, then Qwen2.5-VL

Paste each fenced block below into **one Kaggle code cell** and run the cells in
order. Every cell is self-contained: it defines its own paths, stops on the
first error, and checks the report produced by the preceding scientific step.
Use a Kaggle GPU session with Internet enabled. These are development-only
experiments and do not change the frozen Phase 8 protocol.

## Cell 1 — update the repository and verify entry points

```bash
%%bash
set -euo pipefail
REPO=/kaggle/working/newpipeline
test -d "$REPO/.git"
git -C "$REPO" fetch origin feat/iclr
git -C "$REPO" checkout feat/iclr
git -C "$REPO" pull --ff-only origin feat/iclr
PROJECT="$REPO/projects/logit_evidence_routing"
test -f "$PROJECT/scripts/export_phase9_full_selector_metadata.py"
test -f "$PROJECT/scripts/extract_phase9_llava_interventions.py"
test -f "$PROJECT/scripts/extract_replication_vqa.py"
test -f "$PROJECT/scripts/analyze_replication_vqa.py"
test -f "$PROJECT/scripts/plot_followup_results.py"
git -C "$REPO" log -1 --oneline
```

## Cell 2 — inspect storage before deletion

This read-only cell shows retained outputs and downloaded model caches.

```bash
%%bash
set -euo pipefail
du -sh /kaggle/working/* 2>/dev/null | sort -h
du -sh /root/.cache/huggingface/hub/models--* 2>/dev/null | sort -h || true
df -h /kaggle/working
```

## Cell 3 — guarded cleanup of superseded outputs

The guards must pass before deletion. The cell keeps the multiphase run, repo,
corrected Phase 1B cache, Phase 2 cache, corrected Phase 3 run, and reviewed
Phase 4 bundle.

```bash
%%bash
set -euo pipefail
KEEP=/kaggle/working/multiphase_development_5b55fde6a51f
test -f "$KEEP/phase8_frozen_protocol.json"
test -f /kaggle/working/phase1b_corrected/cache/extraction_config.json
test -d /kaggle/working/phase1b_corrected/cache/records
test -f /kaggle/working/phase2_stage_cache/validation_report.json
test -d /kaggle/working/newpipeline/.git
rm -rf -- \
  /kaggle/working/phase1 \
  /kaggle/working/phase1b \
  /kaggle/working/phase1b_corrected_qualitative.zip \
  /kaggle/working/phase1b_corrected.zip \
  /kaggle/working/phase2_smoke_6ea8e34 \
  /kaggle/working/phase3_attribute_smoke_0b61b6d \
  /kaggle/working/phase3_review_bundle.zip \
  /kaggle/working/phase4_cached_localizer_smoke_1a6e0c96681f_20260906T172707539837Z \
  /kaggle/working/phase4_cached_localizer_smoke_9fcf1c38eab5_20260906T183026108961Z \
  /kaggle/working/phase4_cached_localizer_smoke_bundle.zip \
  /kaggle/working/phase4_localization_4836a3775d34 \
  /kaggle/working/phase4_localization_4be3f2a473ae \
  /kaggle/working/phase4_localization_9fcf1c38eab5 \
  /kaggle/working/phase4_localization_f1c5c9de024d
du -sh /kaggle/working/* 2>/dev/null | sort -h
df -h /kaggle/working
```

Do not delete these directories before downloading the final archive:

```text
/kaggle/working/multiphase_development_5b55fde6a51f
/kaggle/working/newpipeline
/kaggle/working/phase1b_corrected
/kaggle/working/phase2_stage_cache
/kaggle/working/phase3_development_1a6e0c96681f_20260906T171249251219Z
/kaggle/working/reviewed_phase4_db1219637723
```

## Cell 4 — install LLaVA dependencies and run all code tests

These tests check gates, manifests, exact tie handling, deterministic controls,
selector matching, clustered intervals, and the Qwen token-position adapter.
They do not load a model.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
cd "$PROJECT"
# The editable requirement (-e .[dev,kaggle]) resolves from this directory.
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR
python3 -m pip install -r "$PROJECT/requirements-kaggle.txt"
PYTHONPYCACHEPREFIX=/tmp/lger_pycache python3 -m pytest -ra --tb=short
```

## Cell 5 — validate scientific inputs and find CUB automatically

```bash
%%bash
set -euo pipefail
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"
CUB_ROOT="${CUB_MARKER%/images.txt}"
test -d "$CUB_ROOT/images"
python3 - "$RUN" "$CUB_ROOT" <<'PY'
import json, sys
from pathlib import Path
run, cub = Path(sys.argv[1]), Path(sys.argv[2])
checks = {
    "multiphase": run / "multiphase_status.json",
    "phase3r": run / "phase3r/phase3_run_report.json",
    "phase5": run / "phase5/phase5_run_report.json",
    "phase6": run / "phase6/phase6_run_report.json",
    "phase7_bridge": run / "phase7_bridge/token_metadata_report.json",
}
for label, path in checks.items():
    assert path.is_file(), f"missing {label}: {path}"
    report = json.loads(path.read_text())
    assert report["status"] == "PASS", (label, report.get("status"))
    assert report.get("official_test_images_used", 0) == 0, label
    print(label, "PASS", "official_test_images_used=0")
for relative in (
    "phase7_bridge/development_train_stage_means.safetensors",
    "phase5/decision_manifest.csv", "phase5/phase5_decisions.csv",
    "phase6/joint_decisions.csv",
):
    assert (run / relative).is_file(), relative
print("CUB_ROOT", cub)
PY
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
```

## Cell 6 — generate paper figures from the existing audited bundle

This cache-only cell verifies paper-package hashes and saves five main figures
plus the fixed Phase 4 qualitative gallery as PNG and PDF.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
MPLCONFIGDIR=/tmp/matplotlib python3 "$PROJECT/scripts/plot_multiphase_paper_figures.py" \
  --paper-package "$RUN/paper_package" \
  --phase4-dir /kaggle/working/reviewed_phase4_db1219637723 \
  --output-dir /kaggle/working/paper_figures_development
ls -lh /kaggle/working/paper_figures_development
```

## Cell 7 — export selector scores from saved caches

This aligns Vision-CLS and logit-concept patch scores to every positive
validation decision in Phase 6. It verifies cache digests and does not load a
model.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
python3 "$PROJECT/scripts/export_phase9_full_selector_metadata.py" \
  --stage-cache /kaggle/working/phase2_stage_cache \
  --localizer-cache /kaggle/working/phase1b_corrected/cache \
  --phase6-dir "$RUN/phase6" \
  --output-dir /kaggle/working/phase9_selector_metadata
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/kaggle/working/phase9_selector_metadata/selector_metadata_report.json")
r = json.loads(p.read_text())
assert r["status"] == "PASS" and r["official_test_images_used"] == 0
print(json.dumps(r, indent=2, sort_keys=True))
PY
```

## Cell 8 — build and validate the fixed Phase 9 plan

For each decision and selector, this chooses the top 32 patches and three
matched-random sets. The identical mask is applied at `vision.late` and
`projector.output`; global replacement is a manipulation check.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/run_phase9_mechanism.py" plan \
  --token-metadata /kaggle/working/phase9_selector_metadata/selector_token_metadata.csv \
  --output /kaggle/working/phase9_plan.json \
  --selected-stage vision.late \
  --neighbor-stage projector.output \
  --selectors vision_cls_attention logit_concept \
  --k 32 \
  --random-seeds 0 1 2
python3 - <<'PY'
import json
from collections import Counter, defaultdict
from pathlib import Path
p = json.loads(Path("/kaggle/working/phase9_plan.json").read_text())
assert p["status"] == "PASS"
assert p["development_only"] is True and p["phase8_protocol_unchanged"] is True
assert p["official_test_images_used"] == 0 and p["fixed_mask_across_stages"] is True
assert p["selector_methods"] == ["vision_cls_attention", "logit_concept"]
assert p["matched_random_seeds"] == [0, 1, 2]
by_design = defaultdict(dict)
for row in p["records"]:
    if row["selection_method"] != "global_control":
        key = (row["decision_id"], row["selection_method"],
               row["intervention_type"], row["replicate"])
        by_design[key][row["stage"]] = row["token_indices"]
assert all(set(v) == {"vision.late", "projector.output"} for v in by_design.values())
assert all(v["vision.late"] == v["projector.output"] for v in by_design.values())
print("decisions", p["decision_count"], "images", p["image_count"])
print("interventions", p["intervention_count"])
print("types", Counter(r["intervention_type"] for r in p["records"]))
PY
```

## Cell 9 — Phase 9 LLaVA smoke

Runs every planned intervention for one complete decision, exercising the real
hooks, replacement shapes, baseline reproduction, and resumable writes.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_phase9_llava_interventions.py" \
  --mode smoke --phase5-dir "$RUN/phase5" \
  --phase7-bridge-dir "$RUN/phase7_bridge" \
  --plan /kaggle/working/phase9_plan.json --cub-root "$CUB_ROOT" \
  --output-dir /kaggle/working/phase9_llava_smoke
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/phase9_llava_smoke/phase9_extraction_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="smoke" and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 10 — Phase 9 LLaVA pilot

Runs 20 complete decisions and requires the exact matching smoke protocol.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_phase9_llava_interventions.py" \
  --mode pilot --phase5-dir "$RUN/phase5" \
  --phase7-bridge-dir "$RUN/phase7_bridge" \
  --plan /kaggle/working/phase9_plan.json --cub-root "$CUB_ROOT" \
  --smoke-dir /kaggle/working/phase9_llava_smoke --pilot-decisions 20 \
  --output-dir /kaggle/working/phase9_llava_pilot
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/phase9_llava_pilot/phase9_extraction_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="pilot" and r["decisions"]==20
assert r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 11 — Phase 9 full LLaVA development cohort

Re-running this cell resumes from per-intervention records with digest checks.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_phase9_llava_interventions.py" \
  --mode full --phase5-dir "$RUN/phase5" \
  --phase7-bridge-dir "$RUN/phase7_bridge" \
  --plan /kaggle/working/phase9_plan.json --cub-root "$CUB_ROOT" \
  --smoke-dir /kaggle/working/phase9_llava_smoke \
  --pilot-dir /kaggle/working/phase9_llava_pilot --pilot-decisions 20 \
  --output-dir /kaggle/working/phase9_llava_full
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/phase9_llava_full/phase9_extraction_report.json").read_text())
p=json.loads(Path("/kaggle/working/phase9_plan.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="full" and r["official_test_images_used"]==0
assert r["interventions"]==p["intervention_count"] and r["decisions"]==p["decision_count"]
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 12 — aggregate Phase 9 confidence intervals

Estimates top-minus-random effects, fixed-mask stage interactions, selector
differences, failure-versus-success interactions, and global manipulation checks.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/run_phase9_mechanism.py" aggregate \
  --outcomes /kaggle/working/phase9_llava_full/intervention_outcomes.csv \
  --plan /kaggle/working/phase9_plan.json \
  --output /kaggle/working/phase9_llava_analysis.json \
  --bootstrap-samples 10000 --confidence-level 0.95 --seed 20260913
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/phase9_llava_analysis.json").read_text())
assert r["status"]=="PASS" and r["official_test_images_used"]==0
for section in ("selector_specific_contrasts","stage_interactions","selector_interactions",
                "failure_success_interactions","global_manipulation_checks"):
    print("\n"+section)
    for row in r[section]:
        print(row.get("stage",""),row.get("selection_method",""),
              f"estimate={row['estimate']:.6f}",
              f"CI=[{row['ci_low']:.6f}, {row['ci_high']:.6f}]",row["interpretation"])
PY
```

## Cell 13 — add an opposite-label CUB image control

Unlike an arbitrary shuffled image, this donor has the opposite value for the
queried attribute and prefers the same bird class when available.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
RUN=/kaggle/working/multiphase_development_5b55fde6a51f
python3 "$PROJECT/scripts/build_opposite_label_controls.py" \
  --input "$RUN/phase5/decision_manifest.csv" \
  --output /kaggle/working/cub_replication_manifest.csv --seed 31415
python3 - <<'PY'
import csv
from pathlib import Path
with Path("/kaggle/working/cub_replication_manifest.csv").open(newline="") as f:
    rows=list(csv.DictReader(f))
assert rows and {int(r["target"]) for r in rows}=={0,1}
assert all(int(r["opposite_label_target"])==1-int(r["target"]) for r in rows)
assert all(r["opposite_label_image_id"]!=r["image_id"] for r in rows)
print("decisions",len(rows),"same-class donors",sum(int(r["opposite_label_same_class"]) for r in rows))
PY
```

## Cell 14 — LLaVA four-control smoke

Evaluates one positive and one negative decision under correct image, prompt
only, shuffled image, and opposite-label image.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode smoke --config "$PROJECT/configs/llava_full_replication.json" \
  --manifest /kaggle/working/cub_replication_manifest.csv \
  --image-root "$CUB_ROOT/images" --output-dir /kaggle/working/cub_llava_smoke
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/cub_llava_smoke/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="smoke" and r["decisions"]==2
assert r["control_rows"]==8 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 15 — LLaVA four-control pilot

Round-robins 40 decisions across attribute/label strata and requires the exact
matching smoke report.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode pilot --config "$PROJECT/configs/llava_full_replication.json" \
  --manifest /kaggle/working/cub_replication_manifest.csv --image-root "$CUB_ROOT/images" \
  --smoke-dir /kaggle/working/cub_llava_smoke --pilot-decisions 40 \
  --output-dir /kaggle/working/cub_llava_pilot
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/cub_llava_pilot/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="pilot" and r["decisions"]==40
assert r["control_rows"]==160 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 16 — LLaVA full CUB four-control run

Evaluates every development decision and resumes safely from hashed records.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode full --config "$PROJECT/configs/llava_full_replication.json" \
  --manifest /kaggle/working/cub_replication_manifest.csv --image-root "$CUB_ROOT/images" \
  --smoke-dir /kaggle/working/cub_llava_smoke \
  --pilot-dir /kaggle/working/cub_llava_pilot --pilot-decisions 40 \
  --output-dir /kaggle/working/cub_llava_full
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/cub_llava_full/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="full"
assert r["control_rows"]==4*r["decisions"] and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 17 — analyze and archive LLaVA before clearing its checkpoint

Exact zero margins stay as incorrect abstentions. The analysis reports tie
rates and image-clustered correct-image-minus-control intervals.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/analyze_replication_vqa.py" \
  --input llava /kaggle/working/cub_llava_full/replication_vqa_decisions.csv \
  --output-dir /kaggle/working/cub_llava_analysis \
  --bootstrap-samples 10000 --confidence-level 0.95 --seed 20260913
cd /kaggle/working
tar -czf llava_phase9_and_cub_results.tar.gz \
  phase9_selector_metadata phase9_plan.json \
  phase9_llava_smoke phase9_llava_pilot phase9_llava_full \
  phase9_llava_analysis.json cub_replication_manifest.csv \
  cub_llava_smoke cub_llava_pilot cub_llava_full cub_llava_analysis \
  paper_figures_development
sha256sum llava_phase9_and_cub_results.tar.gz
ls -lh llava_phase9_and_cub_results.tar.gz
```

## Cell 18 — remove only LLaVA checkpoints and install Qwen

Qwen is pinned to an exact revision and is about 16.6 GB. Scientific LLaVA
outputs remain intact.

```bash
%%bash
set -euo pipefail
rm -rf -- \
  /root/.cache/huggingface/hub/models--llava-hf--llava-1.5-7b-hf \
  /kaggle/working/hf_cache/hub/models--llava-hf--llava-1.5-7b-hf
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 -m pip install -q -r "$PROJECT/requirements-qwen-kaggle.txt"
python3 - <<'PY'
import transformers
assert transformers.__version__=="4.51.3",transformers.__version__
print("transformers",transformers.__version__)
PY
df -h /kaggle/working
```

## Cell 19 — Qwen2.5-VL four-control smoke

Validates dynamic image-token expansion, eager attention, exact likelihoods,
generation, and 4-bit execution before any larger Qwen run.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode smoke --config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --manifest /kaggle/working/cub_replication_manifest.csv \
  --image-root "$CUB_ROOT/images" --output-dir /kaggle/working/cub_qwen_smoke
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/cub_qwen_smoke/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="smoke" and r["adapter"]=="qwen2_5_vl"
assert r["decisions"]==2 and r["control_rows"]==8 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 20 — Qwen2.5-VL four-control pilot

Runs the same balanced 40-decision pilot as LLaVA.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode pilot --config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --manifest /kaggle/working/cub_replication_manifest.csv --image-root "$CUB_ROOT/images" \
  --smoke-dir /kaggle/working/cub_qwen_smoke --pilot-decisions 40 \
  --output-dir /kaggle/working/cub_qwen_pilot
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/cub_qwen_pilot/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="pilot" and r["decisions"]==40
assert r["control_rows"]==160 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 21 — Qwen2.5-VL full CUB four-control run

This is the architecture-level replication of visual utilization. It does not
claim Qwen hidden-state localization or causality.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CUB_MARKER="$(find /kaggle/input -type f -path '*/CUB_200_2011/images.txt' -print -quit)"
test -n "$CUB_MARKER"; CUB_ROOT="${CUB_MARKER%/images.txt}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode full --config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --manifest /kaggle/working/cub_replication_manifest.csv --image-root "$CUB_ROOT/images" \
  --smoke-dir /kaggle/working/cub_qwen_smoke \
  --pilot-dir /kaggle/working/cub_qwen_pilot --pilot-decisions 40 \
  --output-dir /kaggle/working/cub_qwen_full
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/cub_qwen_full/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="full"
assert r["control_rows"]==4*r["decisions"] and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 22 — paired cross-model analysis and follow-up figures

Computes within-decision contrasts, image-clustered intervals, cross-model
differences on identical decisions, tie counts/rates, and PNG/PDF forest plots.

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/analyze_replication_vqa.py" \
  --input llava /kaggle/working/cub_llava_full/replication_vqa_decisions.csv \
  --input qwen /kaggle/working/cub_qwen_full/replication_vqa_decisions.csv \
  --output-dir /kaggle/working/cub_cross_model_analysis \
  --bootstrap-samples 10000 --confidence-level 0.95 --seed 20260913
MPLCONFIGDIR=/tmp/matplotlib python3 "$PROJECT/scripts/plot_followup_results.py" \
  --phase9-analysis /kaggle/working/phase9_llava_analysis.json \
  --replication-analysis /kaggle/working/cub_cross_model_analysis/replication_vqa_analysis.json \
  --output-dir /kaggle/working/followup_paper_figures
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/cub_cross_model_analysis/replication_vqa_analysis.json").read_text())
assert r["status"]=="PASS" and r["official_test_images_used"]==0
print("\nCONTROL SUMMARIES")
for x in r["summaries"]:
    print(x["model_label"],x["control"],f"accuracy={x['margin_accuracy']:.4f}",
          f"mean_margin={x['mean_correct_answer_margin']:.4f}",
          f"ties={x['margin_ties']} ({x['margin_tie_rate']:.4%})")
print("\nPAIRED CONTRASTS")
for x in r["paired_contrasts"]:
    print(x["model_label"],x["contrast"],x["metric"],f"estimate={x['estimate']:.6f}",
          f"CI=[{x['ci_low']:.6f}, {x['ci_high']:.6f}]",x["interpretation"])
print("\nCROSS-MODEL CONTRASTS")
for x in r["cross_model_contrasts"]:
    print(x["contrast"],f"estimate={x['estimate']:.6f}",
          f"CI=[{x['ci_low']:.6f}, {x['ci_high']:.6f}]",x["interpretation"])
PY
ls -lh /kaggle/working/followup_paper_figures
```

## Cell 23 — create and verify the final downloadable bundle

The archive includes reports, exact manifests, per-decision outputs, hashes,
old-result figures, and follow-up figures. It excludes checkpoints.

```bash
%%bash
set -euo pipefail
cd /kaggle/working
tar -czf logit_evidence_followup_complete.tar.gz \
  phase9_selector_metadata phase9_plan.json \
  phase9_llava_smoke phase9_llava_pilot phase9_llava_full \
  phase9_llava_analysis.json cub_replication_manifest.csv \
  cub_llava_smoke cub_llava_pilot cub_llava_full cub_llava_analysis \
  cub_qwen_smoke cub_qwen_pilot cub_qwen_full \
  cub_cross_model_analysis paper_figures_development followup_paper_figures
sha256sum logit_evidence_followup_complete.tar.gz \
  > logit_evidence_followup_complete.sha256
sha256sum -c logit_evidence_followup_complete.sha256
ls -lh logit_evidence_followup_complete.tar.gz \
  logit_evidence_followup_complete.sha256
```

Download these from the Kaggle **Output** pane:

```text
/kaggle/working/logit_evidence_followup_complete.tar.gz
/kaggle/working/logit_evidence_followup_complete.sha256
```

## Optional second-dataset preparation — CelebA development only

Attach the official **in-the-wild** CelebA layout containing `Img/img_celeba`,
`Anno`, and `Eval`. The builder excludes official-test filenames before parsing
attribute or landmark values, samples at most one image per identity, and makes
identity-disjoint development splits. It rejects aligned crops because their
coordinate system does not match the official in-the-wild boxes.

## Cell 24 — build the CelebA validation manifest

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CELEBA_IMAGES="$(find /kaggle/input -type d -path '*/Img/img_celeba' -print -quit)"
test -n "$CELEBA_IMAGES"
CELEBA_ROOT="${CELEBA_IMAGES%/Img/img_celeba}"
python3 "$PROJECT/scripts/prepare_celeba_replication.py" \
  --celeba-root "$CELEBA_ROOT" --config "$PROJECT/configs/celeba_replication.json" \
  --output-dir /kaggle/working/celeba_replication_development
python3 - <<'PY'
import csv
from pathlib import Path
source=Path("/kaggle/working/celeba_replication_development/celeba_development_decisions.csv")
target=Path("/kaggle/working/celeba_validation_decisions.csv")
with source.open(newline="",encoding="utf-8") as f:
    rows=[r for r in csv.DictReader(f) if r["development_split"]=="val"]
assert rows and all(r["official_test_image"]=="0" for r in rows)
with target.open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
print("validation decisions",len(rows),"identities",len({r["identity_id"] for r in rows}))
PY
python3 "$PROJECT/scripts/build_opposite_label_controls.py" \
  --input /kaggle/working/celeba_validation_decisions.csv \
  --output /kaggle/working/celeba_replication_manifest.csv --seed 31415
```

If CelebA is attached at the start, run Cell 24 immediately after Cell 13, run
Cells 25–27 after the CUB LLaVA full run in Cell 16, and run Cells 28–31 after
the CUB Qwen full run in Cell 21. This ordering downloads each checkpoint only
once.

## Cell 25 — CelebA LLaVA smoke

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CELEBA_IMAGES="$(find /kaggle/input -type d -path '*/Img/img_celeba' -print -quit)"
test -n "$CELEBA_IMAGES"; CELEBA_ROOT="${CELEBA_IMAGES%/Img/img_celeba}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode smoke --config "$PROJECT/configs/llava_full_replication.json" \
  --manifest /kaggle/working/celeba_replication_manifest.csv \
  --image-root "$CELEBA_ROOT" --output-dir /kaggle/working/celeba_llava_smoke
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/celeba_llava_smoke/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="smoke" and r["decisions"]==2
assert r["control_rows"]==8 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 26 — CelebA LLaVA pilot

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CELEBA_IMAGES="$(find /kaggle/input -type d -path '*/Img/img_celeba' -print -quit)"
test -n "$CELEBA_IMAGES"; CELEBA_ROOT="${CELEBA_IMAGES%/Img/img_celeba}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode pilot --config "$PROJECT/configs/llava_full_replication.json" \
  --manifest /kaggle/working/celeba_replication_manifest.csv --image-root "$CELEBA_ROOT" \
  --smoke-dir /kaggle/working/celeba_llava_smoke --pilot-decisions 72 \
  --output-dir /kaggle/working/celeba_llava_pilot
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/celeba_llava_pilot/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="pilot" and r["decisions"]==72
assert r["control_rows"]==288 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 27 — CelebA LLaVA full validation run

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CELEBA_IMAGES="$(find /kaggle/input -type d -path '*/Img/img_celeba' -print -quit)"
test -n "$CELEBA_IMAGES"; CELEBA_ROOT="${CELEBA_IMAGES%/Img/img_celeba}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode full --config "$PROJECT/configs/llava_full_replication.json" \
  --manifest /kaggle/working/celeba_replication_manifest.csv --image-root "$CELEBA_ROOT" \
  --smoke-dir /kaggle/working/celeba_llava_smoke \
  --pilot-dir /kaggle/working/celeba_llava_pilot --pilot-decisions 72 \
  --output-dir /kaggle/working/celeba_llava_full
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/celeba_llava_full/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="full"
assert r["control_rows"]==4*r["decisions"] and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 28 — CelebA Qwen smoke

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CELEBA_IMAGES="$(find /kaggle/input -type d -path '*/Img/img_celeba' -print -quit)"
test -n "$CELEBA_IMAGES"; CELEBA_ROOT="${CELEBA_IMAGES%/Img/img_celeba}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode smoke --config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --manifest /kaggle/working/celeba_replication_manifest.csv \
  --image-root "$CELEBA_ROOT" --output-dir /kaggle/working/celeba_qwen_smoke
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/celeba_qwen_smoke/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="smoke" and r["decisions"]==2
assert r["control_rows"]==8 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 29 — CelebA Qwen pilot

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CELEBA_IMAGES="$(find /kaggle/input -type d -path '*/Img/img_celeba' -print -quit)"
test -n "$CELEBA_IMAGES"; CELEBA_ROOT="${CELEBA_IMAGES%/Img/img_celeba}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode pilot --config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --manifest /kaggle/working/celeba_replication_manifest.csv --image-root "$CELEBA_ROOT" \
  --smoke-dir /kaggle/working/celeba_qwen_smoke --pilot-decisions 72 \
  --output-dir /kaggle/working/celeba_qwen_pilot
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/celeba_qwen_pilot/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="pilot" and r["decisions"]==72
assert r["control_rows"]==288 and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 30 — CelebA Qwen full validation run

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
CELEBA_IMAGES="$(find /kaggle/input -type d -path '*/Img/img_celeba' -print -quit)"
test -n "$CELEBA_IMAGES"; CELEBA_ROOT="${CELEBA_IMAGES%/Img/img_celeba}"
python3 "$PROJECT/scripts/extract_replication_vqa.py" \
  --mode full --config "$PROJECT/configs/qwen25vl_7b_replication.json" \
  --manifest /kaggle/working/celeba_replication_manifest.csv --image-root "$CELEBA_ROOT" \
  --smoke-dir /kaggle/working/celeba_qwen_smoke \
  --pilot-dir /kaggle/working/celeba_qwen_pilot --pilot-decisions 72 \
  --output-dir /kaggle/working/celeba_qwen_full
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path("/kaggle/working/celeba_qwen_full/replication_vqa_report.json").read_text())
assert r["status"]=="PASS" and r["mode"]=="full"
assert r["control_rows"]==4*r["decisions"] and r["official_test_images_used"]==0
print(json.dumps(r,indent=2,sort_keys=True))
PY
```

## Cell 31 — CelebA cross-model analysis and archive

```bash
%%bash
set -euo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/analyze_replication_vqa.py" \
  --input llava /kaggle/working/celeba_llava_full/replication_vqa_decisions.csv \
  --input qwen /kaggle/working/celeba_qwen_full/replication_vqa_decisions.csv \
  --output-dir /kaggle/working/celeba_cross_model_analysis \
  --bootstrap-samples 10000 --confidence-level 0.95 --seed 20260913
MPLCONFIGDIR=/tmp/matplotlib python3 "$PROJECT/scripts/plot_followup_results.py" \
  --phase9-analysis /kaggle/working/phase9_llava_analysis.json \
  --replication-analysis /kaggle/working/celeba_cross_model_analysis/replication_vqa_analysis.json \
  --output-dir /kaggle/working/celeba_followup_figures
cd /kaggle/working
tar -czf celeba_cross_model_development_results.tar.gz \
  celeba_replication_development celeba_validation_decisions.csv \
  celeba_replication_manifest.csv celeba_llava_smoke celeba_llava_pilot \
  celeba_llava_full celeba_qwen_smoke celeba_qwen_pilot celeba_qwen_full \
  celeba_cross_model_analysis celeba_followup_figures
sha256sum celeba_cross_model_development_results.tar.gz \
  > celeba_cross_model_development_results.sha256
sha256sum -c celeba_cross_model_development_results.sha256
ls -lh celeba_cross_model_development_results.tar.gz \
  celeba_cross_model_development_results.sha256
```

Freeze a separate CelebA test protocol before creating or inspecting an
official-test manifest.

## What the completed tests can establish

- **Phase 9 top minus matched random above zero:** selector-ranked patches
  causally support the correct-answer margin under this mean-replacement test.
- **Vision-CLS minus logit above zero:** the diagnostic localization mismatch
  corresponds to different causal relevance for answer production.
- **Vision-late minus projector interaction excluding zero:** the same patch
  mask has different causal susceptibility across that boundary. This does not
  prove literal information destruction.
- **Global replacement moves margins while selector effects are null:** the hook
  can move the answer, while the tested rankings do not isolate a selectively
  causal subset.
- **Correct image exceeds prompt-only, shuffled, and opposite-label controls:**
  the answer depends on query-relevant visual content rather than prompt priors
  or generic images alone.
- **The same correct-image advantage in LLaVA and Qwen:** visual utilization
  generalizes across the two tested model families on CUB.
- **The same pattern on identity-disjoint CelebA development:** the utilization
  result generalizes beyond birds, subject to CelebA label noise and landmark-
  proxy limitations.

Null-compatible intervals are legitimate findings. They do not prove absence
of information or of an effect. Development results can motivate the paper;
only the unchanged frozen official-test protocol supports the predeclared
confirmatory claim.
