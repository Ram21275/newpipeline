# Compact Kaggle follow-up: four cells

This is the short execution path for the development-only Phase 9 and
cross-model follow-up. It preserves every smoke, pilot, full-run, report, and
official-test-use gate from the detailed 31-cell runbook. Completed outputs are
resumable, so rerun the same cell after fixing a failure.

Use a Kaggle GPU session with Internet enabled. The optional CelebA replication
runs automatically only when a valid in-the-wild CelebA layout is attached.

## Cell 1 — update, install, test, and prepare all cache-only inputs

```bash
%%bash
set -Eeuo pipefail
trap 'echo "FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR
REPO=/kaggle/working/newpipeline
git -C "$REPO" fetch origin feat/iclr
git -C "$REPO" checkout feat/iclr
git -C "$REPO" pull --ff-only origin feat/iclr
PROJECT="$REPO/projects/logit_evidence_routing"
cd "$PROJECT"
python3 scripts/run_kaggle_followup.py --celeba auto prepare
```

This cell prints storage availability, installs from the correct project
directory, runs the CPU suite, validates retained development-only artifacts,
exports the Phase 9 selector metadata, builds the fixed plan and opposite-label
controls, and prepares CelebA when attached. If the CPU suite already passed on
this exact commit, add `--skip-tests` after `prepare`.

## Cell 2 — run all gated LLaVA work

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/run_kaggle_followup.py" --celeba auto llava
```

This runs Phase 9 smoke → pilot → full → analysis, then CUB LLaVA smoke →
pilot → full → analysis. If CelebA was prepared, its LLaVA sequence runs here
while the checkpoint is available. A hashed LLaVA results archive is created
before this cell reports success.

## Cell 3 — switch to Qwen, run replications, analyze, and package

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/run_kaggle_followup.py" --celeba auto qwen \
  --clear-llava-checkpoint
```

The checkpoint removal is guarded: it occurs only after the LLaVA analysis and
hashed archive exist. This cell installs the pinned Qwen environment, runs CUB
smoke → pilot → full, runs CelebA when present, computes cross-model intervals,
renders figures, and creates the final archive and SHA-256 sidecar.

## Cell 4 — final audit and download paths

```bash
%%bash
set -Eeuo pipefail
PROJECT=/kaggle/working/newpipeline/projects/logit_evidence_routing
python3 "$PROJECT/scripts/run_kaggle_followup.py" --celeba auto status
```

Do not continue to the next cell unless the current cell ends in `PASS`. On a
failure, look for `FAILED STAGE` and `FAILED COMMAND` near the bottom of the
output; the underlying Python traceback appears immediately above them.

The detailed 31-cell notebook remains available for isolated debugging or
manual storage cleanup.
