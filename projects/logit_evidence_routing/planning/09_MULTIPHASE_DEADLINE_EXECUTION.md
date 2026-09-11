# 09 — Multiphase deadline execution

## Goal

Answer the paper question with four distinct measurements rather than treating
one heatmap as an explanation:

1. **Existence/accessibility:** can a fixed linear probe recover the small-detail
   attribute at each frozen stage?
2. **Localization:** do the highest-scoring patches overlap the relevant CUB
   landmark proxies, and how concentrated are they?
3. **Utilization/VQA:** does the frozen VLM use the image to answer the matching
   binary attribute question, relative to prompt-only and image-shuffled controls?
4. **Causal dependence:** does replacing top-evidence tokens damage the
   correct-answer margin more than spatial/norm-matched random or low-evidence
   tokens?

VQA is therefore Phase 5's behavioral component; it is not a separate project.

## Fast execution graph

```text
reviewed Phase 4 + validated Phase 2 cache
            |                         |
            |                         +--> Phase 5 VQA (GPU)
            +--> corrected Phase 3R probes (CPU)
                         \             /
                          Phase 6 joint analysis
                                  |
                       one unambiguous transition?
                           /                 \
                         yes                 no
                          |                   |
               Phase 7 matched causal    stop causal branch;
                    interventions          write null/mixed paper package
                          |
                 freeze Phase 8 protocol
                          |
                untouched official-test run
```

`scripts/run_multiphase_development.py` implements this graph. It starts Phase
3R in a background CPU process while Phase 5 exclusively owns the GPU. This is
the safe parallelism: two model-loading jobs are not placed on the same GPU.
Phase 6 is table-only. Phase 7 starts only when the predeclared Phase 6 rubric
selects exactly one transition.

## Phase definitions after realignment to the problem statement

### Phase 3R — where evidence is accessible

- Reuse the validated 240-image, nine-stage cache.
- Mask the 248 legacy certainty-policy overrides discovered by Phase 4.
- Export every decision-level probe score and the exact linear weights.
- Decompose mean-pooled margins into per-patch contributions with a numerical
  reconstruction check.
- Interpretation: linear accessibility, never causal use.

The cached LLM states were generated under the neutral Phase 2 prompt
`Describe the image briefly.`, whereas Phase 5 uses attribute questions.
Vision/projector states are unaffected by that text, but language-side
bottleneck/utilization labels are blocked by a Phase 6 context gate. A later
question-conditioned cache can lift this gate; the current deadline run does not
hide the mismatch by renaming the neutral prompt.

### Phase 4 — where evidence is localized

The returned development bundle remains descriptive-only. Its verified result
is selector-dependent localization: generic Logit Lens is sharp for object
semantics, while Vision-CLS provides broader landmark coverage. This motivates
the joint analysis but does not choose a causal layer by itself.

### Phase 5 — whether the VLM uses the image in VQA

- Fixed prompt: `Does the bird have {attribute_phrase}? Answer only yes or no.`
- Sequence-level teacher-forced yes/no log likelihood, so multi-token candidates
  are supported.
- Deterministic generation is secondary; unparseable answers count as incorrect
  rather than crashing the primary analysis.
- Grounded positive and negative validation decisions are sampled without
  replacement to the smaller grounded label count per attribute (417 + 417
  decisions in the reviewed development data).
- Prompt-only and attribute-stratified image-shuffled controls distinguish
  visual use from language prior.

### Phase 6 — where the trajectory changes

Join Phases 3R, 4 and 5 by the audited image/attribute decision identity. Use
image-clustered bootstrap intervals with a fixed-attribute macro primary
aggregation and retain micro/two-way robustness results. The rubric considers
visual-to-language loss, accessibility–answer gap, discriminative/semantic
localization mismatch, direct Top-K redistribution, and a valid mixed/null
outcome. It selects a Phase 7 target, not a causal conclusion.

### Phase 7 — whether selected evidence affects the answer

- Select at most five failures and five matched successes per attribute/target
  stratum to bound GPU time.
- Test the selected stage plus one adjacent causal stage.
- Replace Top-32, bottom-32, and three exact matched-random sets with the
  development-training stage mean.
- Match random patches jointly by bird-box membership and hidden-state norm
  quartile; insufficient strata fail closed.
- Primary effect: before-minus-after correct-answer teacher-forced margin.
- Primary causal contrast: top-evidence drop minus the mean matched-random drop,
  clustered by image with 10,000 bootstrap samples.
- A null-compatible confidence interval is a valid result.

### Phase 8 — held-out confirmation

`freeze_phase8_protocol.py` hashes every development artifact and freezes the
selected transition, stage pair, K, replacement, matched seeds, test sampling,
metric and uncertainty procedure. Only after that file exists may
`prepare_phase8_test_manifest.py` inspect the official CUB test split. The test
result cannot be used to change the protocol.

## One-run instructions

Use `notebooks/multiphase_development_kaggle.ipynb`. It fetches the latest
`feat/iclr`, installs pinned Kaggle dependencies, runs the complete local test
suite, discovers the reviewed Phase 4 bundle by its recorded SHA-256, and invokes
the orchestrator. The final cell displays Phase 6 findings and the generated
paper package.

If interrupted, rerun the same notebook against the same output directory.
Phase 5 and Phase 7 retain atomic per-decision/intervention records. Never delete
the run configuration to force an incompatible resume; use a new output
directory when code, configs, paths or source artifacts change.

## Paper outputs

The orchestrator always runs `build_paper_results_package.py` after Phase 6. It
writes machine-readable Phase 3–6 tables, a claim ledger, and an abstract
scaffold. When Phase 7 passes, it adds the causal table and freezes Phase 8.
Every development statement is labeled as development evidence; the abstract
keeps a held-out-number placeholder until Phase 8 is complete.
