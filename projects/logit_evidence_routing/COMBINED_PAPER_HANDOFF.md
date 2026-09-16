# Combined paper handoff

The common question is: when does access to visual evidence become a correct answer, and which intervention closes the gap?

## Common points

Both projects distinguish four quantities that are often collapsed: feature accessibility, spatial localization, causal influence, and end-task correctness. Both require the final model distribution as the mature readout, explicit compute accounting, paired condition comparisons, and honest null results. The shared CUB experiment places full image, predicted crop, random crop, oracle part crop, ordinary decoding, and restricted DoLa on the same attribute-question cohort for both LLaVA and Qwen3.

## Separate regimes and findings

This project studies fine-grained CUB attribute decisions on birds that often occupy much of the frame. The partner's central mechanism concerns under-resolved small evidence regions in V*Bench/HR-Bench; its phases 18/19 found that whole CUB birds and annotated parts do not occupy that sub-token regime. Those statements are compatible because the queried endpoint and resolution regime differ.

This project's Vision-CLS selector is image-salience based and LLaVA-specific. The partner's strongest proposer is question-conditioned decoder attention plus a trained depth-aware re-ranker. These are not interchangeable measurements. The shared cross-model selector is decoder attention; the learned proposer can be included only when its real checkpoint/provenance is available and must be labeled supervised.

The partner reports corrected null DoLa/DeCo scoring in a different multiple-choice task and retracts a prior late-layer degradation story caused by double normalization. The CUB implementation borrows the correctness check, not the result. New CUB DoLa outputs must stand on their own.

## Training disclosure

Both VLMs stay frozen. Attribute probes are supervised diagnostics fitted on official-training images. Training-derived means and donor policies are interventions, not learned answer models. A reused partner re-ranking head is trained even though the VLM is frozen. No new fusion network or answer reranker is added by default.

## Manuscript merge rule

Promote a common claim only if the exact endpoint is measured in both projects or both required CUB models. Otherwise label it model-specific, task-specific, historical development evidence, or unresolved. Do not use the partner's results as automatic replication of this project's attribute experiments.

