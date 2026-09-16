# Split and exposure audit

Status: protocol implementation complete; final manifests are generated from the attached CUB copy on Kaggle; no new official-test inference has been run locally.

## Images used for fitting or tuning in this implementation

Only images marked `1` in CUB's official `train_test_split.txt` are eligible. The workflow creates an image-disjoint, class-stratified fit/development partition with seed `20260916`. Fit images may train diagnostic probes, replacement means, feature normalization, and donor policies. Development images may select layers, thresholds, crop padding, answer format checks, and DoLa parameters. The two real-model smoke examples must be from this official-training pool and the lock verifies `official_test_images_used = 0`.

## Images previously examined by either collaborator

This project's retained development artifacts contain 240 official-training images from 20 species, split into 160 development-training and 80 development-validation images. The retained intervention artifacts identify 77 development images. Exact IDs recoverable from the artifacts are retained in the historical result package.

The partner repository (pinned at `1096a8642b2de9ba6a649a60a77f5641ba76db42`) used the CUB official test split for species verification and part-geometry analyses in phases 18/19. An exact complete list of all partner-exposed CUB image IDs was not found in the audited narrative reports. Therefore the joint work must not describe the official test split as collectively untouched or as a fresh blind confirmation.

## Locked final evaluation images

The primary population is the recovered historical set of 20 species and 26 attributes, not all 200 CUB species. All eligible official-test images in those 20 species are used for inexpensive summaries. Before any model outputs are inspected, the manifest builder selects at most 256 official-test images and at most four eligible decisions per image for expensive arms, balanced across polarity as feasible. Exact IDs are written to:

- `manifests/image_manifest.jsonl`
- `manifests/question_manifest_all_test.jsonl`
- `manifests/question_manifest_expensive.jsonl`

These files are hashed into the automatically locked protocol. Official-test results may support a final benchmark evaluation, but not a claim that the collaborator pair had never previously examined the split. This protocol is locked prospectively at execution time; it is not called preregistered.

