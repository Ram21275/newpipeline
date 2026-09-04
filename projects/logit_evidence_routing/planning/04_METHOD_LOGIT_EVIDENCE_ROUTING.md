# 04 — Fine-Grained Evidence Tracing Framework

The filename is retained for compatibility with earlier links. The proposed
study is no longer Logit-Guided Evidence Routing.

## Objects being measured

For image patch `i` and stage `s`, let `h_i^s` be its aligned representation.
For attribute `a`, distinguish:

1. **Accessibility:** performance of a fixed linear probe `w_a^s h_i^s`.
2. **Semantic readability:** compatibility with text concept `a`, measured by a
   valid dense similarity or, on language-compatible states, an audited Logit
   Lens readout.
3. **Spatial importance:** attention or other routing weight assigned to patch
   `i`.
4. **Utilization:** change in the final output after a controlled intervention
   on the identified representation.

No single metric is used as a synonym for “evidence.”

## Logit Lens as one instrument

For an LLM-compatible hidden state:

`z_i^l = W_LM LN(h_i^l)`

The output estimates direct vocabulary readability at that layer. Generic
maximum probability and negative entropy measure confidence, not concept
identity. Concept token mass is limited to audited single lexical tokens. A
phrase such as “red crown” requires sequence-level scoring or an image–text
embedding probe.

The reference blog used attention to find candidates and logit probabilities to
filter trauma-relevant concepts in a private application. This project tests
which aspects transfer under public, quantitative CUB evaluation; it does not
assume the blog's pipeline is a benchmark baseline.

## Trajectory analysis

For each fixed attribute and image, assemble a stage trajectory rather than only
independent aggregate scores. Evaluate whether:

- accessibility rises, falls, or remains stable;
- localization shifts across spatial tokens;
- semantic readability appears later than linear accessibility;
- information becomes distributed, making patch-local probes weaker while
  pooled probes remain strong.

Represent redistribution with token-to-token correspondence or cross-stage
similarity only after verifying that the patch alignment is valid.

## Decision point

After layer-wise accessibility and localization are complete, write
`INTERMEDIATE_FINDINGS.md`. Name the largest supported transition and its
uncertainty. Only that transition receives a causal intervention.
