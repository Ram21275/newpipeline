# Partner provenance audit

Repository: `https://github.com/Team-M3OW/vlm-hallu`  
Pinned commit: `1096a8642b2de9ba6a649a60a77f5641ba76db42`

Files audited: `REPLICATION_LEDGER.md`, `PAPER_FLOW.md`, and phase reports 18, 19, 60, 61/61b, 63, 64, 66, 70/70b, 71, 72, 73, 76, 77, and 79.

Safe correspondences for the combined paper:

- Both projects separate localization from answer correctness and require paired end-task validation.
- The partner's corrected lens takes the mature distribution from `model.logits`; the CUB final implementation asserts the same identity and never normalizes the final hidden state twice.
- The partner's corrected DoLa/DeCo comparison is null on its constrained MCQ setting. This is prior boundary evidence, not a result for the CUB attribute task.
- The partner's learned depth-aware proposer is supervised. It may be described as a learned proposer with a frozen VLM, never as training-free.
- Their whole-bird/species CUB result lies outside the small-object regime. This project's queried attribute/part decisions are a different task and do not erase that negative boundary.

Claims not imported as established for this project:

- A universal anti-correlated final layer: explicitly rejected by the partner's second-model replication.
- A universal "perception not reasoning" conclusion from a late lens jump.
- Any end-task gain from the partner's learned head on CUB attributes.
- Absence of information from a failed selective intervention.

