# Phase 4 development review

Decision: **PASS_DESCRIPTIVE_ONLY**. The full 240-image development computation,
all 6,240 eligibility rows, artifact hashes, aggregate tables, and every returned
qualitative panel were reviewed. No official CUB test image was used.

On development validation at K=32, corrected generic bird/birds Logit Lens is
the most concentrated object-semantic map (inside-box 0.8926, pointing 0.8000,
visible-part recall 0.2509, top-1 part distance 3.7215 patches). Vision-CLS
attention gives the broadest landmark coverage (object part recall 0.3931 and
attribute-proxy recall 0.5123) but weak top-1 pointing (0.3750). LLM attention is
approximately random on landmark recall, fusion mostly inherits the logit map,
and dense object/attribute patch similarity is weak and diffuse. Selector
agreement is generally low except for constructed or closely related pairs.

This supports complementary, selector-dependent localization. A
discriminative/semantic localization mismatch is the leading Phase 4 hypothesis,
not a selected transition. Phase 4 alone cannot establish causal use, answer
utilization, evidence destruction, or a visual-to-language bottleneck.

The corrected certainty policy masked 248 cached non-null targets whose certainty
was not `probably` or `definitely`. Therefore the old Phase 3 trajectory remains
invalid for cross-phase inference and must be rerun from the unchanged Phase 2
representation cache. Small validation supports (as low as n=3 for some
attributes) must remain visible; Phase 6 will use image-clustered bootstrap
intervals and report macro and robustness aggregations.
