# Development-only mechanism disambiguation after Phase 8 freeze

The current development results establish three different facts. Fine-grained
attribute labels are linearly recoverable, Vision-CLS and Logit-Lens maps differ
in landmark recall, and the correct image modestly improves answer accuracy over
prompt-only and shuffled controls. They do not yet establish that the localized
tokens selected by either diagnostic cause the answer, or that information is
destroyed at a particular boundary. The probe-ranked Phase 7 intervention was
null-compatible and did not directly test the two selectors that triggered the
Phase 6 gate.

This follow-up is development-only. It cannot modify the frozen Phase 8 sampling
frame, selected stage, intervention, effect definition, or allowed confirmatory
claim. Official-test usage remains zero.

## Test 1: selector-aligned replacement

For each decision in the complete Phase 6 joined development cohort, rank the
576 patch tokens at `vision.late` with the two saved selectors that triggered
the gate: `vision_cls_attention` and `logit_concept`. Retain the existing Phase
7 probe intervention as a separately reported reference without rerunning it. Replace
the top 32 vectors with the development-training stage mean. Compare the
correct-answer teacher-forced margin drop with three random selections matched
jointly on bird-box membership and hidden-state norm quartile.

Use exactly the same selected token indices at `vision.late` and
`projector.output`. This removes a confound in the original Phase 7 design,
where stage-specific rankings could make a stage difference reflect a changed
mask instead of changed susceptibility.

What it can show:

- A positive, interval-separated top-minus-random effect shows that the selector
  identifies tokens that causally support the correct-answer margin under this
  mean-replacement intervention.
- A larger Vision-CLS effect than Logit-Lens effect links the Phase 6 diagnostic
  mismatch to causal answer sensitivity.
- A null interval remains compatible with a weak selector, redundant evidence,
  insufficient intervention strength, or no utilization. It does not prove that
  evidence is absent.

## Test 2: global replacement manipulation check

Replace all 576 patch vectors at each tested stage with the same training mean.
This deliberately severe intervention checks that the hook and outcome can move
the answer margin. It is not a localization control and does not demonstrate
fine-grained causality.

- If global replacement changes the margin but no selector beats matched random,
  the intervention path works while the tested selectors lack spatially specific
  causal evidence under this manipulation.
- If even global replacement has a null-compatible effect, the selector nulls
  cannot distinguish robustness from an ineffective hook or outcome.

## Test 3: failure and stage interactions

Estimate selector-specific effects separately for Phase 5 successes and
failures, then report failure minus success with an image-cluster bootstrap.
Also compare the fixed-mask effect at `vision.late` with its effect at
`projector.output`.

- A larger effect on successes suggests that the selected evidence supports the
  answer when the model succeeds but is less decisive when it fails.
- A larger effect on failures shows that correct-answer evidence still affects
  failed decisions, although it was insufficient to determine the final answer.
- A stage interaction measures where the answer is more susceptible to the same
  patch mask. It cannot alone prove that information was erased between stages.

## Test 4: opposite-label image controls

The original shuffled-image control can pair a question with an image having the
same attribute label. Add a deterministic donor control in which the donor image
has the opposite label for the queried attribute. Match donors on class or
coarse appearance where support permits, and report the unmatched remainder.
This tests whether the answer margin tracks attribute-relevant visual content
more cleanly than an unrestricted shuffle.

## Test 5: question-conditioned layer trajectory

At the final query token, record the yes-minus-no correct-answer margin after
each language block with the image, prompt-only input, and opposite-label donor.
This is a diagnostic trajectory. A late divergence between correct-image and
controls localizes when answer-relevant image information becomes readable by
the vocabulary head; it does not establish causality without an intervention.

## Evidence rubric

| Result | Supported statement | Statement still unsupported |
|---|---|---|
| Frozen Phase 8 top-minus-random CI is above zero | The preregistered selected-stage token intervention has a larger held-out answer-margin effect than matched random | The model always uses the evidence; the exact layer where it is destroyed |
| Selector-aligned development CI is above zero | That selector identifies causally important tokens under mean replacement | General held-out causal confirmation |
| Global check moves margin; selectors do not beat random | Model answer is intervention-sensitive, but tested maps do not isolate a spatially specific causal subset | Evidence is absent |
| Success effect exceeds failure effect | Selected tokens are more consequential on decisions the model answers correctly | Failures are caused by a single bottleneck |
| Opposite-label donor shifts margin in the expected direction | Attribute-relevant image content affects the answer distribution | A particular internal route mediates the effect |
| All causal intervals include zero | No causal claim is supported at current precision/intervention strength | Evidence has been proven irrelevant or absent |

Run one complete-decision smoke, a 20-decision pilot, and then every eligible
Phase 6 joined development decision. The smoke and pilot only validate mechanics;
all reported development estimates come from the full run.

The smallest submission-critical experiment remains the already frozen Phase 8
held-out confirmation. The selector-aligned development follow-up is the smallest
mechanism experiment needed to explain why the current Phase 7 result is null.
