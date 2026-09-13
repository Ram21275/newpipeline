# CUB-only Kaggle continuation: four cells

This is the short execution path for the retained CUB-200-2011 development
artifacts. It preserves every smoke, pilot, full-run, report, hash, and
official-test-use gate from the detailed runbook. Completed GPU outputs are
resumable and are skipped only after their identities, counts, and hashes pass.

Use a Kaggle GPU session with Internet enabled. Every command explicitly uses
`--celeba no`; this continuation never searches for, prepares, runs, analyzes,
or packages CelebA. The frozen Phase 8 protocol is unchanged and official-test
image use must remain zero.

## Cell 1 — update and verify the retained CUB preparation

```bash
%%bash
set -Eeuo pipefail
REPO=/kaggle/working/newpipeline
git -C "$REPO" fetch origin feat/iclr
git -C "$REPO" checkout feat/iclr
git -C "$REPO" pull --ff-only origin feat/iclr
PROJECT="$REPO/projects/logit_evidence_routing"
PYTHONUNBUFFERED=1 python3 -u \
  "$PROJECT/scripts/run_kaggle_followup.py" \
  --notebook-safe --celeba no preflight
```

The preflight requires these retained outputs and verifies their PASS status,
artifact hashes, counts, fixed masks across stages, frozen selector settings,
and shuffled/opposite-label donor controls:

- `/kaggle/working/phase9_selector_metadata/selector_metadata_report.json`
- `/kaggle/working/phase9_plan.json`
- `/kaggle/working/cub_replication_manifest.csv`

Do not run Cell 2 unless the last line includes `CUB PREPARATION PASS`.

## Cell 2 — run all gated CUB LLaVA work

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
PYTHONUNBUFFERED=1 python3 -u \
  "$PROJECT/scripts/run_kaggle_followup.py" \
  --notebook-safe --celeba no llava
```

This runs Phase 9 LLaVA smoke → pilot → full → image-clustered analysis,
followed by CUB four-condition LLaVA smoke → pilot → full → analysis. A full
run cannot start without passing, policy-matched smoke and pilot reports. Model
loading and downloads can be quiet; the runner prints a one-minute heartbeat.
At the start of every Kaggle session, this cell installs the exact pinned LLaVA,
Accelerate, and bitsandbytes versions and exercises a tiny NF4 quantize/dequantize
operation before accepting any retained stage or loading the model.
After a failure, fix the reported code or input issue and rerun this same cell;
completed stages are reused only when their hashes and counts still match.

## Cell 3 — switch to Qwen, run CUB, analyze, and package

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
PYTHONUNBUFFERED=1 python3 -u \
  "$PROJECT/scripts/run_kaggle_followup.py" \
  --notebook-safe --celeba no qwen \
  --clear-llava-checkpoint
```

LLaVA checkpoint deletion is guarded by a passing LLaVA analysis and verified
archive hash. Qwen then runs CUB smoke → pilot → full, computes the paired
cross-model image-clustered intervals, renders figures, and creates the final
CUB-only archive and SHA-256 sidecar.

## Cell 4 — audit every result and print download paths

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
PYTHONUNBUFFERED=1 python3 -u \
  "$PROJECT/scripts/run_kaggle_followup.py" \
  --notebook-safe --celeba no status
```

This audit checks all Phase 9 and CUB smoke/pilot/full reports, policy digests,
row and decision counts, four controls per decision, recorded artifact and
source hashes, archive hashes and members, 10,000-sample image-clustered 95%
confidence intervals, and official-test use. Exact teacher-forced likelihood
ties must remain incorrect abstentions.

Do not continue to the next cell unless the current cell ends in `PASS`.
Notebook-safe failures save `/kaggle/working/lger_compact_last_failure.json`
and end with `STOP HERE`. Interpret attention/localization only as diagnostic,
probes as evidence of accessibility, and controlled interventions as the only
support for causal claims. Keep null or mixed intervals explicitly
null-compatible; they do not establish absence.
