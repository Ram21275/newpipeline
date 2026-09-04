# 01B — Localization-Selector Validation

## Why this rerun exists

The first pilot used unrestricted vocabulary confidence,
`max_v p(v | h_patch)`. The referenced Prisma demonstration applies a ViT's
ImageNet head to every patch and visualizes each patch's semantic prediction;
it does not establish that discarding the predicted identity and ranking only
by maximum confidence is a foreground localizer. Our decoded high-confidence
tokens also included punctuation and tokenizer fragments, while Logit-K
improved over Random-K only at `K=16`.

This rerun therefore retains unrestricted max-probability for a faithful
comparison and adds an explicit fixed-concept score suitable for testing bird
localization. The latter is an adaptation, not an exact reproduction of the
Prisma visualization.

Before Phase 02, run one bounded selector benchmark that separates three
questions:

1. Does a score localize the bird?
2. Do the selected original LLaVA hidden states retain class-discriminative
   information?
3. Is the signal complementary to decoder attention?

This is still Phase 01. It must not be presented as a new final method.

## Fixed experimental contract

- Keep LLaVA-1.5-7B, its vision tower, projector, LLM, and LM head frozen.
- Use the same CUB manifest as the first pilot.
- Use the same patch budgets for every Top-K selector.
- Rank with localization scores, but represent with the same late-layer LLaVA
  patch hidden states.
- Mean-pool and train the same linear probe for every selector.
- Never use an image's fine-grained ground-truth class to select its patches.
- Use CUB bounding boxes for evaluation only, never for selection or training.

## Selector suite

### Controls

- `random`: uniform spatial patches.
- `global_all`: mean of all visual tokens; not a localizer, but a coverage
  reference.
- `llm_attention`: the original final-prompt-token decoder attention score.
- `logit_maxprob`: the original unrestricted maximum vocabulary probability.

### Same-backbone localization signals

- `vision_cls_attention`: final-layer CLS-to-patch attention from LLaVA's frozen
  CLIP vision tower.
- `vision_attention_rollout`: residual-aware attention rollout through all
  vision-tower layers.
- `logit_margin`: top-1 minus top-2 vocabulary logit.
- `logit_negentropy`: negative vocabulary entropy.
- `logit_concept`: probability mass assigned to a fixed concept-token set. For
  CUB use `bird` and `birds` for every image. This uses dataset-level object
  knowledge, not the per-image species label, and must be reported as such.
- `attention_logit_fusion`: a fixed equal-weight sum of per-image standardized
  `llm_attention` and `logit_concept` scores.

Do not tune fusion weights after looking at validation results.

## Evaluation

### Downstream recognition

For every selector and `K`, save accuracy, macro-F1, per-image predictions, and
the selection seed separately from the probe initialization seed.

### Localization

Map the CUB bounding box through the exact resize-and-center-crop geometry used
by the model. Report:

- fraction of selected patch centers inside the box
- recall of all patch centers inside the box
- selected-mask/bounding-box-mask IoU on the patch grid
- pointing-game accuracy (whether the highest-scoring patch is inside)

The bounding box identifies the bird broadly. It does not measure whether the
selector found the fine-grained part that separates two species.

## Exit decision

Proceed to Phase 02 only if at least one semantic/localization selector:

1. beats Random-K consistently across adjacent patch budgets, and
2. localizes above the random baseline, and
3. adds useful patches beyond attention or improves the fixed fusion.

If only vision attention/rollout succeeds, the result validates localization-
guided routing but not the planned logit-space claim. At that point either keep
logit evidence as analysis or explicitly revise the paper's main hypothesis.

## Deferred external controls

DINO CLS attention, LOST/TokenCut, CLIP decomposition/CLIP Surgery, Grounding
DINO, and SAM are not part of this rerun. They introduce a second model,
additional supervision, or a different representation space. Add one as an
external baseline only after the same-backbone result is understood.

## Primary references

- Joseph and Nanda, *Laying the Foundations for Vision and Multimodal
  Mechanistic Interpretability & Open Problems*, 2024:
  https://www.lesswrong.com/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic%E9%93%BE%E6%8E%A5%E8%AF%A6%E6%83%85
- Abnar and Zuidema, *Quantifying Attention Flow in Transformers*, ACL 2020:
  https://aclanthology.org/2020.acl-main.385/
- Gandelsman, Efros, and Steinhardt, *Interpreting CLIP's Image Representation
  via Text-Based Decomposition*, ICLR 2024:
  https://openreview.net/forum?id=5Ca9sSzuDp
- Wang et al., *Self-Supervised Transformers for Unsupervised Object Discovery
  Using Normalized Cut*, CVPR 2022:
  https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Self-Supervised_Transformers_for_Unsupervised_Object_Discovery_Using_Normalized_Cut_CVPR_2022_paper.html
- Esmaeilkhani and Latecki, *Logit Lens Supervision for Patch-Level
  Explanations in Vision-Language Models*, 2026:
  https://arxiv.org/abs/2602.01530
