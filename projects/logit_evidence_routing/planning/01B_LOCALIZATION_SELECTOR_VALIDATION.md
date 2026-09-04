# 01B — Localization Selector Result and Correction

## What this experiment measured

Every selector ranks the same 576 spatial positions. The Top-K late LLM patch
states are mean-pooled and given to the same linear species probe. Therefore the
experiment measures the utility of a routing score over one fixed downstream
representation; it does not compare raw representation stages.

## Valid pilot observations

| Selector | K=16 accuracy | K=32 accuracy |
|---|---:|---:|
| Random | 40.8% | 54.4% |
| LLM attention | 70.0% | 68.8% |
| Vision CLS attention | **92.5%** | **95.0%** |
| Vision attention rollout | 50.0% | 55.0% |
| Logit max probability | 48.8% | 47.5% |
| Logit margin | 48.3% | 52.5% |
| Logit negative entropy | 49.6% | 53.8% |
| All 576 patches | 66.3% | 66.3% |

Vision-CLS selected-patch centers were inside the broad bird box 73.7% at K=16
and 80.8% at K=32, compared with 47.4% and 47.1% for random. However, its top-1
pointing-game rate was only 37.5%, below the random estimate in this pilot. The
ordered map and the Top-K set therefore tell different stories and both must be
reported.

These values use one 20-class development pilot (160 train, 80 validation) drawn
only from the official CUB training split. The three runs vary linear-probe
initialization, not the sampled images.

## Invalid rows

The original `logit_concept` token IDs were `[11199, 17952, 29871]`. In the
LLaMA tokenizer, 29871 is the standalone SentencePiece whitespace marker `▁`.
It entered because the code manually prefixed each concept with a space before
tokenization. Consequently:

- the reported `logit_concept` results are invalid;
- `attention_logit_fusion` is also invalid because it consumes that score;
- other selectors and their cached features are unaffected.

The corrected implementation encodes bare concepts, requires exactly one
non-special lexical token per concept, records decoded token strings, and blocks
benchmarking legacy concept caches. Multi-token attributes such as “red crown”
must use a sequence-aware score or dense semantic similarity, not a sum of
independent token marginals.

## Correction run

If the original Phase 01B cache still exists, run
`scripts/repair_phase1b_concepts.py`. It reuses cached late LLM patch states,
loads the frozen normalization and LM head, and recomputes only
`logit_concept`, `attention_logit_fusion`, and their derived features. It writes
a new cache and never modifies the source in place.

If only result CSVs remain, rerun `extract_phase1b_localizers.py` into a new
cache. Do not copy the invalid concept/fusion numbers into a paper or report.

## Phase decision

The original proposal that logit-space routing should be the method is rejected
by the valid pilot. Generic logit-confidence selectors are close to random for
species probing, while Vision-CLS routing is much stronger. Logit Lens remains a
semantic-readability diagnostic in the evidence-tracing study.

Before advancing, generate `phase1_sanity_report.md`, including official-split,
cache identity, qualitative, box, and visible-part checks.

## Original motivation

The reference blog is Arsh Naqvi,
[*Using Logit Space of VLMs for Attention to Detail*](https://www.arsh-naqvi.xyz/blog/logit-space-vlm-attention-to-detail).
It describes attention-based candidate localization and logit-lens filtering in
a private trauma dataset. Our CUB benchmark is a new controlled test, not a
reproduction of a public result from that post.
