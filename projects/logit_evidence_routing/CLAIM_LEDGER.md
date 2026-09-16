# Claim ledger

| Claim | Current state | Evidence / required file |
|---|---|---|
| Species and attribute information are accessible at selected frozen stages in the historical cohort | Historical development support only | Existing paper probe tables |
| Vision-CLS and Logit-Concept differ in attribute-landmark localization | Historical development support only; LLaVA-specific | Existing phase 4/9 artifacts |
| Selective replacement has a positive pooled causal effect | Weakened / mostly null-compatible; K=64 confounded historically | Existing phase 9/10 artifacts |
| Polarity reversal proves different positive/negative mechanisms | Unsupported pending raw-margin interaction audit | New `analysis/paired_effects.jsonl` after GPU runs |
| Correct final logit-lens computation agrees with the model | Implemented as a fail-closed check; real checkpoints pending smoke | `smoke/*_smoke.json` |
| Central attribute-level intervention conclusion replicates on Qwen3 | Unreplicated | Qwen3 final intervention outputs required |
| Predicted or oracle re-encoding improves CUB attribute answers | Unresolved | Shared per-example outputs required |
| DoLa improves CUB attribute answers | Unresolved; partner's different MCQ setting was null | Shared DoLa outputs required |
| Localization, accessibility, causal influence, and answer correctness align on the same attribute decisions | Unresolved central combined claim | Final paired tables required |
| The official test split is jointly untouched | Rejected | Partner exposure audit |
| The study covers all 200 CUB species | Rejected | Locked scope is 20 recovered species |

Null findings will retain intervals and cannot establish that information is absent or that every possible decoder repair is impossible.

