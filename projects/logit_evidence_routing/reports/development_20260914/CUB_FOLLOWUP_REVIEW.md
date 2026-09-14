# CUB-only Phase 9 and cross-model follow-up review

Review date: 14 September 2026. Experiment commit:
`03c1e6c40dc1f7a14a5077d4132997be91510562`.

## Audit decision

The extracted scientific contents pass review. Outer-archive verification is
still pending because the downloaded `logit_evidence_followup_complete.tar.gz`
is not present locally; only its extracted directory and SHA-256 sidecar were
supplied. The sidecar reports
`a21943af76a6422d1fb44c86716377637a919a9bc03c9ab2fb1f0b3013f583b0`.
This digest has not been compared with the absent archive and must not be called
independently verified.

Within the extracted directory:

- all 10 smoke/pilot/full and selector reports are `PASS` and report zero
  official-test image use;
- all Phase 9 and replication analyses pass the repository's source-hash,
  artifact-hash, count, control-identity, interval, and conservative-tie gates;
- 9,661 JSON files parse, all 35 declared artifact hashes match, and the
  retained record counts are exactly 7,506 Phase 9 interventions, 834 LLaVA
  decisions, and 834 Qwen decisions;
- the CUB replication manifest contains 834 unique official-training decisions,
  balanced 417/417 by target, with valid shuffled and opposite-label donors;
- the extracted result tree contains no CelebA-named member.

The principal audited file hashes are:

| Artifact | SHA-256 |
|---|---|
| Phase 9 selector report | `4952a29671fa8e219ce66d17c8993feeda133e456f35f4cf5d9be0d048fd60ac` |
| Phase 9 fixed plan | `aefe725fa528dbe9610e466360af5416a966f100f029c3ea90b20c1eefc016b5` |
| CUB four-control manifest | `e8be5a756b4af85c1594cba4e86143e58c28b4bad7432402a92e16ccc44e0329` |
| Phase 9 full outcomes | `4094c87d3932737922e8a02244b860b1cd33437a07ccbf33821595b3e57ac3c0` |
| LLaVA full decisions | `074883d4855c2dd277209a195d82e13ae18034603972be786952f04bc8a043de` |
| Qwen full decisions | `b85d6e9e0181205c270c240cf104a03a86f366465a41beb9095496e846e787c2` |

## Frozen design and counts

Phase 9 is development-only and leaves Phase 8 unchanged. It covers 417
decisions on 76 images and 7,506 interventions. It uses the same top-32 mask at
`vision.late` and `projector.output`, three matched-random seeds, and the frozen
development-training stage-mean replacement. Smoke, pilot, and full counts are
1/18, 20/360, and 417/7,506 decisions/interventions, with one common policy
digest.

The cross-model study covers 834 balanced decisions on 77 CUB development
images. Each model has exactly four rows per decision: correct image,
prompt-only, shuffled image, and opposite-label image. Smoke, pilot, and full
runs contain 2/8, 40/160, and 834/3,336 decisions/control rows. All confidence
intervals use 10,000 bootstrap samples clustered by `image_id` with equal image
weighting.

Exact teacher-forced margin ties remain incorrect abstentions. The LLaVA full
run has 12 ties: four each for correct, shuffled, and opposite-label images.
The Qwen full run has two: one each for correct and shuffled images. Prompt-only
has no exact ties in either full run.

## Directly supported findings

Positive margin drop means the correct-answer teacher-forced margin fell after
replacement.

| Phase 9 selector and stage | Top-minus-matched-random effect | 95% image-cluster CI | Review |
|---|---:|---:|---|
| Vision-CLS, `vision.late` | 0.0715 | [0.0289, 0.1161] | positive selector-specific causal effect under mean replacement |
| Vision-CLS, `projector.output` | 0.0904 | [0.0470, 0.1347] | positive selector-specific causal effect under mean replacement |
| Logit-Concept, `vision.late` | -0.0057 | [-0.0306, 0.0226] | null-compatible |
| Logit-Concept, `projector.output` | 0.0004 | [-0.0250, 0.0287] | null-compatible |

Vision-CLS exceeds Logit-Concept at both stages: 0.0772 [0.0289, 0.1277] at
`vision.late` and 0.0900 [0.0409, 0.1424] at `projector.output`. The fixed-mask
Vision-CLS effect is 0.0189 larger at the projector output than at vision late
(reported interaction `vision.late - projector.output` = -0.0189,
[-0.0241, -0.0136]). This establishes greater downstream susceptibility, not
literal information destruction at the projector. The all-patch manipulation
checks are positive at both stages—0.3537 [0.2636, 0.4403] and 0.6675
[0.5904, 0.7397]—so the intervention path and margin outcome are responsive.

Both models show a positive correct-image advantage over every input control:

| Model | Control comparison | Correct-margin effect | 95% CI | Margin-accuracy effect | 95% CI |
|---|---|---:|---:|---:|---:|
| LLaVA | image − prompt-only | 0.1001 | [0.0256, 0.1735] | 0.0427 | [0.0090, 0.0768] |
| LLaVA | image − shuffled | 0.1050 | [0.0433, 0.1714] | 0.0591 | [0.0170, 0.1054] |
| LLaVA | image − opposite-label | 0.1907 | [0.1341, 0.2505] | 0.0834 | [0.0488, 0.1177] |
| Qwen | image − prompt-only | 0.6902 | [0.4743, 0.9076] | 0.0997 | [0.0457, 0.1547] |
| Qwen | image − shuffled | 0.7179 | [0.4339, 0.9983] | 0.0937 | [0.0407, 0.1471] |
| Qwen | image − opposite-label | 0.9898 | [0.7333, 1.2529] | 0.1415 | [0.0912, 0.1936] |

Thus, controlled image substitutions causally change the teacher-forced answer
distribution in both models, and the opposite-label control gives the largest
within-model shift. This supports visual utilization on this CUB development
cohort. It does not identify Qwen's internal route.

## Suggestive diagnostics

- Selector-specific effects are larger on Phase 5 successes than failures for
  every selector/stage comparison. This is consistent with selected evidence
  supporting decisions that already succeed; it does not establish a unique
  cause of failure or a repair mechanism.
- The three raw-margin difference-in-differences favor Qwen over LLaVA, with
  LLaVA-minus-Qwen estimates from -0.5901 to -0.7992 and intervals excluding
  zero. Raw logit-margin scales are not calibrated across architectures, so this
  supports a larger effect on the recorded Qwen margin scale, not the general
  claim that Qwen is intrinsically more visually reliant.
- The small Logit-Concept stage interaction excludes zero even though its
  selector-specific interval includes zero at both stages. It is a stage
  susceptibility diagnostic, not evidence that Logit-Concept patches are
  causally important.

## Unsupported claims

These results do not support official-test or out-of-dataset generalization,
CelebA replication, free-generation accuracy improvement, a unique
visual-to-language bottleneck, literal evidence erasure at the projector,
causal importance of Logit-Concept-selected patches, absence of evidence when a
confidence interval contains zero, or an internal causal route for Qwen.
Attention and localization remain diagnostics; probes establish accessibility;
only the controlled replacement and image interventions support causal claims.

