# 07 — Abstract and Submission Claim Gate

Official deadlines (11:59 PM AoE): abstract 18 September 2026; paper
25 September 2026.

## Before submitting the abstract

- [ ] Phase 01 sanity report passes.
- [ ] Invalid concept/fusion rows are removed or corrected.
- [ ] Stage-aligned cache is audited.
- [ ] Attribute recoverability by stage exists as machine-readable results.
- [ ] At least one quantitative spatial/semantic comparison is complete.
- [ ] The title and abstract describe an observed phenomenon, not H1–H4 as fact.
- [ ] All authors and anonymization requirements are settled.

## Allowed abstract structure

1. Fine-grained VLM decisions depend on localized attributes whose internal
   trajectory is poorly understood.
2. Introduce a controlled tracing protocol across the frozen vision encoder,
   projector, LLM, and answer.
3. State only the strongest measured transition with exact scope.
4. Distinguish accessibility, semantic readability, spatial importance, and
   causal use.
5. Report the main quantitative result and uncertainty.

## Claim templates—choose only after results

- “Linear accessibility of fine-grained attributes decreases most sharply
  between ___ and ___.”
- “Attribute information remains accessible through ___, but fixed-prompt VLM
  answers trail the probe by ___.”
- “Vision-side discriminative routing and text-readable semantic localization
  overlap by only ___ despite ___.”
- “Evidence becomes more distributed across tokens after ___ while pooled
  accessibility remains ___.”

## Forbidden claims without additional evidence

- attention explains or causes the answer;
- probe recoverability proves the VLM uses an attribute;
- bounding-box localization proves part/attribute localization;
- Logit Lens is a universal semantic localizer;
- the 240-image development pilot is final held-out evidence;
- a bottleneck exists because two selectors have different accuracy.

## Final paper gate

The paper needs an untouched official-test evaluation, paired uncertainty,
complete configs, an explicit limitations section, and at least one causal test
if it claims utilization rather than accessibility alone.
