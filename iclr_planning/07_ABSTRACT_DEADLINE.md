# 07 — ICLR Abstract Deadline

**Hard deadline: Sep 18, 2026, 11:59 PM AoE**

## Goal

Submit a **genuine** title + abstract that reflects the actual paper. Do not use placeholder text.

All intended authors must be added before the abstract deadline; no new authors can be added afterward under the ICLR 2027 rules.

---

## Sep 17 checklist

### Results
- [ ] Main claim supported on Dataset A.
- [ ] Dataset B status known.
- [ ] Low-data trend known.
- [ ] Final method name/mechanism frozen.

### Paper framing
- [ ] One-sentence problem statement.
- [ ] One-sentence failure of attention-only routing.
- [ ] One-sentence proposed method.
- [ ] One-sentence main empirical result.
- [ ] One-sentence analysis/interpretability result.

---

## Recommended abstract structure

### Sentence 1 — problem
Fine-grained visual decisions often depend on small local cues that are poorly characterized by global VLM representations or attention magnitude alone.

### Sentence 2 — observation
Intermediate visual-token representations of frozen VLMs can be projected through the language-model head, exposing patch-level semantic evidence in vocabulary space.

### Sentence 3 — method
Introduce Logit-Guided Evidence Routing, which aggregates this evidence across late layers to select a compact set of visual tokens while retaining their original hidden representations for downstream classification.

### Sentence 4 — experiments
Evaluate against global, random, attention-based, and single-layer logit-based routing on public fine-grained recognition benchmarks, including low-label regimes.

### Sentence 5 — result
Insert only results that are already measured by Sep 18.

### Sentence 6 — implication
Conclude that VLM logit space can function as a semantic routing signal rather than only an output or interpretability space.

---

## Do not claim

- state of the art unless directly established
- universal VLM behavior from one model
- medical reliability from qualitative trauma examples
- causal interpretation of attention/logits
- efficiency gains unless measured

---

## Submission administration

Before Sep 18:

- [ ] Every coauthor has an OpenReview account.
- [ ] Author list is final.
- [ ] Conflicts/profile information updated.
- [ ] Genuine abstract uploaded.
- [ ] Title is sensible but may still be edited before full-paper deadline.

---

## Freeze after abstract

After submission, do not change the core scientific question.

Permitted work:
- finish experiments
- simplify method
- improve analysis
- refine title/abstract before paper deadline if needed

Avoid introducing a new major contribution after Sep 18.
