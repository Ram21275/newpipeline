# Manuscript-ready text (implementation-stage version)

## Methods: split, locking, and exposure

We used the official CUB-200-2011 split as the top-level partition. Within official training images, we formed image-disjoint fit and development partitions stratified by species. Probe fitting, replacement statistics, normalization, crop-policy selection, answer-format checks, and decoder-layer choices were restricted to these two partitions. Before official-test inference, the pipeline wrote a configuration and cohort hash to a locked protocol. The primary population comprised the 20 historically studied species and 26 recovered attributes; it should not be interpreted as a 200-species CUB evaluation. The official test split had previously been examined by a collaborator for species verification and part geometry, so we describe the new run as a locked final benchmark evaluation rather than a jointly untouched or preregistered confirmation.

## Methods: answer evidence and lens validation

For each condition we scored complete candidate strings and defined the raw semantic margin as (m=\log P(\mathrm{yes})-\log P(\mathrm{no})). For an intervention, (\Delta_{raw}=m_{baseline}-m_{intervention}), and the correctness-oriented quantity was (\Delta_{correct}=(2y-1)\Delta_{raw}). We report both scales stratified by target polarity because a uniform shift toward “no” changes sign after correctness recoding. Intermediate answer-position states were passed through the frozen final normalization and output head. The mature distribution was always taken from the model's own logits, and the pipeline required dtype-appropriate numerical agreement with a direct reconstruction of the already-normalized final state.

## Methods: uncertainty

All inferential intervals use paired image-cluster resampling with 10,000 bootstrap draws, preserving every question and condition from a sampled image. Primary summaries weight images equally. Dose and layer families use simultaneous max-t bootstrap intervals; per-attribute analyses are exploratory and retain their support counts and undefined-metric exclusions.

## Results placeholder policy

No new test-set values are inserted until the corresponding per-example output, configuration hash, and clustered analysis are present. Negative or null outcomes are reported with their uncertainty and do not imply absence of encoded information or impossibility of every decoder intervention.

