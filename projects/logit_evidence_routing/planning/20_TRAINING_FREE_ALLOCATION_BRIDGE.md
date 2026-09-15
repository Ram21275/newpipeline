# Training-free allocation: audit, alternatives, and smallest falsification pilot

Date: 2026-09-16. Status: implementation with synthetic validation; **no new CUB model results**.
This is a development extension. It does not amend Phase 8 or the locked Phase 10 tables.

## Recommendation

Test **query-anchored spectral depth projection** against random placement and
uniform encoding before pursuing an adaptive token threshold or a learned head.
Use Qwen2.5-VL-7B at the existing pinned revision, because its processor supports
variable token counts. First run two decisions as an architecture/smoke gate;
then at most 52 decisions, one for each available frozen attribute/target stratum.
This is a low-cost opportunity to falsify the method, not a powered positive study.

The first experiment can discriminate *operational* patterns supporting H1, H2,
or H3. It cannot establish that information was never encoded. The three
hypotheses are not mutually exclusive: spatial resolution, readout mismatch,
and answer-position transmission can all contribute on one item.

## Repository audit

Both remote heads were verified with `git ls-remote` on 2026-09-16:

| Repository | Verified commit | State at audit |
|---|---|---|
| `Ram21275/newpipeline`, `feat/iclr` | `10e329f103976160a7de33a79e4ff1f41b9ae122` | clean local checkout, remote agrees |
| `Team-M3OW/vlm-hallu`, `main` | `1096a8642b2de9ba6a649a60a77f5641ba76db42` | clean temporary clone, remote agrees |

The existing graph has 16 planning nodes and does not cover current phase code.
It supplied orientation only; implementation findings below come from direct reads.
The partner snapshot has phases 70, 71, 72, 73, 76, 77, 78, 79; **74/75 are absent**.
No partner datasets or model weights were downloaded. Reported partner results
were inspected in source/docs, not independently recomputed from raw GPU outputs.

### Findings that constrain this implementation

1. **DCR is supervised.** `scripts/phase70_rerank_head.py::build` derives a continuous
   coverage target from GT boxes. `oof` fits `HistGradientBoostingRegressor` under
   image-grouped folds. The signed linear distillation in
   `scripts/phase73_layer_structure.py` also fits GT coverage (`Ridge`). Neither
   is a training-free inference rule. The 52.9% localization and +12.0 pp end-task
   findings remain partner-reported benchmarks, not reproduced CUB results.
2. **Architecture dependence matters.** `PAPER_FLOW.md` and `phases/phase73.md`
   distinguish Qwen3 signed combinations from Qwen2's selected single layer.
   An out-of-fold label-selected layer still uses supervision to choose the
   rule. Our fixed quartile layers have no evaluation-label selection.
3. **The final-norm retraction supersedes old phases.** `FINDINGS.md §12D` retracts
   the final-layer readout advantage and original DoLa/DeCo comparison. New lens
   code takes mature scores from `outputs.logits`; intermediate states receive
   one final norm; the returned final hidden state receives none. It audits
   `head(h_final)` against model logits and logs the double-norm discrepancy.
4. **The ROI probe was not evidence of universal accessibility.**
   `FINDINGS.md §4Z` retracts positive-GT versus negative-random ROI selection.
   Our landmark evaluation uses the identical part rule for both labels.
5. **Attention statistics are not a validated budget controller.**
   `FINDINGS.md §14B` / `phases/phase69.md` tested 17 pass-1 features; their
   predictor/controller did not beat the relevant controls. Phase 78 reports
   crop-size headroom without an established predictor. Therefore the first
   CUB experiment fixes all budgets and crop sizes.
6. **The strongest partner claims exceed identification.** `METHOD.md` and
   phase 79 use phrases such as “nothing is encoded” and infer a reasoning
   mechanism from a late logit step. A logit trajectory alone cannot establish
   that distinction. Covered-versus-missed subgroups are also not randomized
   coverage assignments and can differ in difficulty. Retain the user's
   readout-bounded premise rather than these stronger interpretations.
7. **Local existing code is already cautious.** `hf_llava.py` rejects final
   hidden-state indices for double-normalized patch readouts. The multimodal
   DoLa extractor excludes the mature state from candidate projection and
   uses `outputs.logits` as mature. `hf_qwen.py` currently collects one language
   attention layer; it does not implement depth aggregation or visual
   reallocation. `guided_layer_fusion.py` contains trainable projections,
   MultiheadAttention and a classifier: frozen-backbone diagnostic learning,
   not a training-free allocation method.
8. **CUB evidence remains bounded.** Phase 4's review says descriptive
   complementary localization. Phase 10's README and inference/analysis files
   support controlled sensitivity, with null-compatible Logit-Concept effects.
   Neither establishes loss or natural utilization. No old result is relabeled
   as an allocation experiment.

Partner citations are pinned to the inspected revision:
[DCR implementation](https://github.com/Team-M3OW/vlm-hallu/blob/1096a8642b2de9ba6a649a60a77f5641ba76db42/scripts/phase70_rerank_head.py),
[updated paper flow](https://github.com/Team-M3OW/vlm-hallu/blob/1096a8642b2de9ba6a649a60a77f5641ba76db42/PAPER_FLOW.md),
[retractions and findings](https://github.com/Team-M3OW/vlm-hallu/blob/1096a8642b2de9ba6a649a60a77f5641ba76db42/FINDINGS.md).

## Mathematical alternatives

Let `A ∈ R^(L×N)` be head-averaged attention from the last prompt position to
visual tokens, with each layer normalized to sum one **over visual tokens**.
This is an operational question-conditioned map, not attention averaged over
all tokens spelling the question. Keep the same grid for real and generic
questions. Let `A0` be the generic map and `d = mean_l(A_l − A0_l)`.

### 1. Query-anchored spectral depth projection — recommended

Center each row spatially and normalize nonzero rows:

`X_l = (A_l − 1/N) / max(||A_l − 1/N||₂, ε)`.

Compute `C = XXᵀ = U diag(λ) Uᵀ`. Set
`J = {j : λ_j ≥ 0.95 λ_max}` and `P = U_J U_Jᵀ`.
The signed spatial score is

`s = Xᵀ P X d / λ_max`.

Equivalently, `w = PXd/λ_max`, `s=Xᵀw`. Both positive and negative depth
weights can arise. This expression is invariant to eigenvector sign changes
and orthogonal basis rotations within a repeated leading eigenspace. It also
does not require assigning semantic labels to the depth groups.

If there is no spatial variance, `||d||₂ < 1e-10`, or
`||s||₂ / ||d||₂ < 1e-6`, return the ordinary real-query mean and record why.
These numerical guards and the 5% eigenspace tolerance are fixed before runs.
For cropping use positive score mass; negative mass suppresses proposals.

**Identification assumption:** task-minus-generic attention must have a component
along the evidence-bearing depth mode. A dominant nuisance mode or a query prior
can violate this. A spectral decomposition identifies variance, not relevance.
Two equally plausible spatial regions cannot be semantically oriented from their
correlation matrix alone. The extra generic pass is the orientation instrument.

### 2. Direct query contrast — strong simplicity/prior-art baseline

`s_qc = mean_l(A_l − A0_l)`.

Same two glances and crop budget as the spectral rule, with no eigendecomposition.
This isolates whether spectral depth structure adds anything beyond query contrast.
It is identifiable only as question-specific change; changing a question also
changes language priors and prompt length. The generic prompt, answer vocabulary,
and selected crop policy are fixed independently of labels.

### 3. Agreement-medoid aggregation — cheapest robust alternative

With the same centered `X`, choose
`j* = argmax_j median_l(X_jᵀX_l)`.
Keep `G = {l : X_j*ᵀX_l ≥ 0}` and use
`s_med(i) = median_{l∈G} A_l(i)`.

This removes an oppositely pointing depth group without training or a null pass.
It is deterministic, with row-order tie breaking. It assumes the majority
agreement group is useful; a minority evidence group or shared sink defeats it.
It is implemented as a CPU alternative but is not added to the first GPU sweep.

| Alternative | Novelty potential | Inference cost before answer | Identifiability | Matched-control prospects |
|---|---|---|---|---|
| Spectral projection | Highest of these three, **unverified** | 2 glances; `O(L²N+L³)` CPU | Explicit query anchor; nuisance-mode risk | Worth testing only against direct contrast and uniform total budget |
| Direct query contrast | Low; close prior art | 2 glances; `O(LN)` | Query change is identifiable, evidence is not | Strong baseline, weak standalone contribution |
| Agreement medoid | Low/moderate; robust aggregation is established | 1 glance; `O(L²N)` | Majority-agreement assumption, no semantic orientation | Cheapest, but can retain the shared distractor |

Novelty ranking: spectral > medoid > direct contrast. Compute ranking: medoid <
direct contrast ≈ spectral (VLM passes dominate). Semantic orientation ranking:
spectral/direct contrast > medoid, conditional on a valid generic question.
Subjective likelihood of surviving localization/answer controls: direct contrast
first (closest to an established mechanism), spectral second (can denoise the
anchor but may select nuisance variance), medoid third (no semantic orientation).
This is a design judgment, not an empirical probability. For compute efficiency
the order may reverse because medoid needs one fewer glance. Direct contrast is
the most important simplicity control and uniform processing is the strongest
resource control. Spectral is recommended because its incremental contribution
can be cleanly falsified against direct contrast at the same pass budget.

### What averaging can destroy, mathematically

Under `X_l = a_l u + ε_l`, ordinary depth averaging retains
`mean(a_l)u`. Opposite-signed coefficients can make this term vanish even when
`Σ a_l²` is large. The rank-one component in `XXᵀ` is proportional to `aaᵀ`,
so a spectral rule can preserve the spatial direction when it dominates noise.
This is a conditional algebraic account, not proof that real layers obey this
model. Pairwise anticorrelation of layers is not the same as anticorrelation
with a GT evidence mask. The development controls test that missing connection.

### Prior-art check, 2026-09-16

[CARVE](https://geyuyao.com/carve/) already contrasts generic and task attention,
then masks/crops and re-infers in a training-free pipeline.
[Where Does Vision Meet Language?](https://arxiv.org/abs/2601.08151v2) studies
depth-dependent fusion and proposes training-free cross-depth contrastive attention.
[Segmentation From Attention](https://github.com/rayat137/Segmentation-From-Attention)
already provides automatic training-free layer ranking for BLIP segmentation.
These are direct novelty risks, not merely citations to add later. This initial
screen is not an exhaustive literature review. Do not claim a first training-free
attention crop, first depth contrast, or first automatic layer selector.

## Pilot protocol

### Cohort and split safety

Use the existing 240-image development manifest (160 train / 80 val), only its
`val` rows, intersected with CUB's official-training split. Keep the 26 attribute
identities and `probably`/`definitely` certainty policy. Select one candidate per
attribute/target by a fixed SHA-256 ordering of image ID and attribute ID.
No baseline correctness, margin, location quality, or token threshold enters
selection. Missing strata are reported, not substituted. The existing val cohort
has already informed earlier work; it is not a pristine confirmatory holdout.

Preparation reads metadata only. At execution all selected IDs, paths, split
membership, targets, certainty policy and input hashes are revalidated before
opening or hashing any image. Paths escaping the image directory are rejected.
The output directory and plan are immutable; a failure produces a FAIL report.
There is deliberately no resume path that could silently mix source revisions.

### Factorial and budget ledger

Here B0=144 merged tokens; final answer budget=432, divided into global=144 and
crop=288. Two crops divide the same 288 into 144 each. W=0.25 of each image
dimension, fixed for every selector. Full image context is always retained.
The model's default full-image processing is also measured separately.

| Arm | Visual encoding supplied to answer | Charged fresh encoded tokens, nominal |
|---|---|---:|
| `native_default` | processor-default full image | measured |
| `native` (controlled low uniform baseline) | whole image @144 | 144 |
| `uniform_answer` | whole image @432 | 432 |
| `uniform_total` | whole image @720 | 720 |
| spectral / direct query contrast | global144 + selected crop288 | 144 real + 144 generic + 432 = 720 |
| layer mean, fixed quartile layers, generic Logit-Concept | global144 + selected crop288 | 144 + 432 = 576 |
| random, seeds 0/1/2 | global144 + random crop288 | 432; no model glance is needed |
| landmark oracle, analysis only | global144 + oracle crop288 | 432; GT location explicitly privileged |
| spectral two-region | global144 + two crops144 each | 720 |
| random two-region, seeds 0/1/2 | global144 + two random crops144 each | 432 |
| internal attention, spectral/random/oracle regions | cached original144 merger tokens | original glance(s), then language forward only |
| mean replacement, same three regions | cached original144, selected tokens mean-replaced | original glance(s), then language forward only |

The fixed single layers are zero-based `floor(L/4)`, `floor(L/2)`, `floor(3L/4)`;
none is chosen by evaluation performance. Logit-Concept is a generic bird/birds
object control at the penultimate returned language state, with lexical IDs audited
from the pinned tokenizer. It is not an attribute-semantic probe.
Qwen's visual tower has no CLS token; **Vision-CLS is N/A**, rather than inventing
an architectural equivalent. Existing LLaVA Vision-CLS results remain Study A;
a later cross-architecture bridge needs their exact original-coordinate transforms.
Partner supervised DCR is a reported upper benchmark only, not fitted on this cohort.

Crop proposals maximize newly covered positive score mass on the native grid,
with a fixed overlap penalty for the second region. Pixel boundaries are rounded
and recorded. Random crops share candidate centers, width, and crop count, with
three reproducible seeds. Seed outcomes are averaged **within decision** before
cluster resampling. The second random crop is sampled without replacement but
may overlap; that is a spatial-random baseline, not a diversity-matched control.

The oracle crop is centered on the mean of visible relevant landmarks, using the
same rule for positive and negative labels. It is a **landmark oracle**, not a
true attribute-mask oracle, and may fail to cover separated parts. Report complete
visible-landmark coverage rather than assuming complete semantic evidence.

### Realized tokens and compute

Choose a merged grid within 2% of the requested token budget with minimum aspect
ratio distortion. Resize original pixels to that grid and require the processor
to return exactly the intended THW dimensions. Check inserted image tokens,
merger output length, and language sequence length on every forward. This changes
effective resolution; it does not invent pixels beyond the original image.

Record per-view grids, final tokens, total fresh encoded tokens, model/vision
forward counts, and synchronized forward latency. `native_default` uses the
unaltered processor policy. Internal arms temporarily bypass `visual.forward`
with the cached merger result; they do **not** execute a fresh vision forward.

**720 versus 144+144+432 matches cumulative visual tokens, not FLOPs.** Self-attention
cost, prompt lengths, vision windows and multiple forwards differ. Diagnostic
attention/state collection also changes runtime. These timings cannot establish
deployment efficiency. If the method survives the pilot, the next required step
is a clean inference-only latency/FLOP audit and an outcome-blind timing-calibrated
uniform grid, on separate development-training images. No success claim is
permitted until that baseline is measured. Do not pad uniform execution with
wasted passes and call it compute-matched.

### Readouts and interventions

* Primary answer endpoint: exact positive-minus-negative next-token logit margin
  for audited single-token alternatives; ties are incorrect and separately counted.
  Correct margin multiplies by `2y−1`. This is constrained binary discrimination,
  not unconstrained generated-answer accuracy. Multi-token alternatives fail the
  gate rather than silently using first-token scores.
* Layerwise query Logit Lens and fixed annotation-ROI patch readability are
  collected for every arm. Final projection parity is a hard gate. All logits
  must be finite. Layer numbers are architecture-specific.
* Compatible **pre-existing frozen diagnostic probes** may be supplied; the
  pipeline never fits them. Model revision, mean-pooling recipe, dimensions,
  positive scales and development-training-only source IDs are checked. Without
  compatible probes, record N/A and leave H2 unresolved. Existing LLaVA probes
  cannot be applied to Qwen hidden states. Pooled probes alone also cannot
  exclude distributed/nonlinear encodings.
* Internal guided attention adds `log(2)` to selected visual-key logits, only at
  the answer query in the prespecified `floor(2L/3)` language layer. The hook
  must run once. Random and oracle regions provide controls. A null rejects
  this intervention, not every possible H3 mechanism.
* Mean replacement uses the complement-token mean at the cached merger boundary.
  Regions use identical geometry rules and require a nonempty proper token set.
  Random crop regions have matching geometry but their discrete K can differ at
  edges; **this pilot does not replace Phase 10's exact-K/norm-matched causal
  estimate**. Record this limitation when interpreting sensitivity.

### Coverage and the proposed cliff

CUB parts are points; they do not give attribute area `a`. Therefore an empirical
0.15–0.25 **true attribute-token** cliff is not identifiable from the current
annotations. The runner reports per-landmark square-window exposures at fixed
normalized radii 0.005/0.01/0.02:

`t_proxy = Σ_views N_view × area(proxy ∩ view) / area(view)`.

These count repeated exposure across views; they are not unique visual information.
Fixed fractional windows can give nearly constant native exposure, so some
below/above strata will be empty. Empty strata mean “not identifiable,” not a
negative threshold finding. Radius sweeps are geometric sensitivity checks,
never threshold fitting. Full attribute masks or separately validated extent
annotations are needed before testing an actual CUB part-area cliff. Such
annotations may be used only for evaluation/oracle analysis, never inference.

### Inference and falsification

Bootstrap whole images, preserving all decisions and paired arms. Report raw
yes-direction effects as well as correct margins, target-specific accuracy,
positive response rates, localization and complete landmark-set coverage.
The analyzer uses multiplicity-adjusted intervals for its configured comparisons.
It never pools answer vocabularies or order conditions. Sparse image counts and
near-perfect AUROC require matched controls, not promotion.

Run a secondary robustness screen on the first 12 prespecified decisions for
yes/no, true/false and present/absent, each in both answer orders. Restrict this
screen to native, spectral, random and uniform-total arms. All choices remain
fixed if results are weak; expanding after a promising pilot requires a new
development protocol, not retrospective retuning.

| Observed pattern | Supported interpretation | What remains open |
|---|---|---|
| crop/upscale rescue, tested internal arms null | higher-resolution re-encoding is useful; consistent with H1 | hidden nonlinear/distributed detail, untested H3 routes, image-transform effects |
| fixed probe accesses native detail but lens fails | H2 readout mismatch remains plausible | probes may not match natural computation; diagnostic controls required |
| same cached visual tokens + internal attention rescues | new visual encoding is not necessary for those items; H3 viable | whether routing is the usual cause and whether rescue generalizes |
| localization improves, answers do not | reject the proposed method as an answer improvement | localization may remain a descriptive result |
| no gain over random or uniform at matched resources | reject/narrow allocation method | Study A–C can still form the paper |
| only yes-rate rises or vocabulary/order reverses gain | reject discrimination claim | prompt/readout bias |

The current pilot cannot honestly complete the H1/H2/H3 diagnosis without a
compatible accessibility instrument, and cannot validate a true threshold without
extent annotations. Those limits are explicit deliverables, not values filled
with zeros. Paper Study D remains conditional; Study A/B/C retain their separate
cohorts, instruments and claim boundaries.

## Execution

Entry points: `scripts/run_depth_allocation_bridge.py` and
`scripts/analyze_depth_allocation_bridge.py`. Run real inference only on Kaggle,
with `requirements-qwen-kaggle.txt`, after fetching the reviewed feature branch.
Keep all output outside existing locked report directories.

```bash
cd /kaggle/working/newpipeline/projects/logit_evidence_routing
export PYTHONPATH=src

# Supply the existing Phase 2 CSV with image_id, relative_path, split=train/val.
python scripts/run_depth_allocation_bridge.py --mode prepare \
  --cub-root /kaggle/input/YOUR_CUB_ROOT/CUB_200_2011 \
  --development-manifest /kaggle/working/YOUR_EXISTING_PHASE2_MANIFEST.csv \
  --plan /kaggle/working/depth_bridge_plan.json

python scripts/run_depth_allocation_bridge.py --mode smoke \
  --cub-root /kaggle/input/YOUR_CUB_ROOT/CUB_200_2011 \
  --development-manifest /kaggle/working/YOUR_EXISTING_PHASE2_MANIFEST.csv \
  --plan /kaggle/working/depth_bridge_plan.json \
  --output-dir /kaggle/working/depth_bridge_smoke

python scripts/run_depth_allocation_bridge.py --mode pilot \
  --cub-root /kaggle/input/YOUR_CUB_ROOT/CUB_200_2011 \
  --development-manifest /kaggle/working/YOUR_EXISTING_PHASE2_MANIFEST.csv \
  --plan /kaggle/working/depth_bridge_plan.json \
  --smoke-report /kaggle/working/depth_bridge_smoke/report.json \
  --output-dir /kaggle/working/depth_bridge_pilot

python scripts/analyze_depth_allocation_bridge.py \
  --run-dir /kaggle/working/depth_bridge_pilot \
  --output /kaggle/working/depth_bridge_analysis.json
```

Paths labeled `YOUR_...` must be replaced with existing Kaggle locations; this
audit did not establish their current mounted paths. For the robustness screen,
replace `--mode pilot` with `--mode robustness` and use a new output directory.
Optional `--frozen-probes` is a JSON with `model_revision`,
`pooling=all_visual_tokens_mean`, `diagnostic_only=true`, `training_image_ids`,
and per-attribute `weight`, `center`, `scale` arrays `[layers, hidden]` plus
`bias[layers]`. Do not fabricate a Qwen probe from LLaVA weights.
