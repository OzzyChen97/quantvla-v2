# GDSQ-VLA draft review notes

## Compact paper outline

1. Explain why uniform PTQ is brittle for coupled language and diffusion-action stacks.
2. Introduce dual representation evidence: CKA for geometry and CS divergence for distribution overlap.
3. Convert representation evidence into a deployable mask using action weights, hard guards, a storage budget, and complete-configuration adjudication.
4. Freeze the 16:1 ratio on four declared development tasks.
5. Evaluate on 14 separate atomic tasks, then report the 18-task aggregate, ablations, storage, and runtime boundaries.

## Reverse outline and paragraph roles

### Abstract

- Challenge: VLA layers are heterogeneous under quantization.
- Method: dual similarity, action weighting, guards, and functional adjudication.
- Evidence: primary 14-task result, theoretical storage, and FP16 gap.

### Introduction

- Opening: VLA deployment motivates training-free PTQ.
- Challenge: errors move from multimodal features into the iterative action solver.
- Prior limitation: local metrics expose only one view of damage.
- Insight: complementary representation evidence needs a behavioral referee.
- Method: 116-layer probing and budgeted binary allocation.
- Evidence: primary improvement and deployment caveat.
- Contributions: formulation, procedure, and controlled evaluation.

### Method

- Overview: four-stage execution order.
- Setting: candidate scope and protected attention projections.
- Dual references: attribution versus deployment semantics.
- CKA: geometric preservation and scale blind spot.
- CS: distribution overlap and bandwidth limitation.
- Selection: action weighting, hard guards, and storage budget.
- Adjudication: cross-layer action behavior and ratio freeze.

### Experiments

- Setup: one benchmark, declared dev/primary split, paired protocol, task-level statistics.
- Unified result table: group the official Atomic (All 18), Composite-Seen, and Composite-Unseen tasks in one comparison, with GR00T N1.5 and planned $\pi_{0.5}$ policy blocks.
- Selection control: show the four-task ratio sweep first, then report Primary 14 separately as held-out evidence after tuning is frozen.
- Main result: gain over the fixed W4A8 layout on the held-out GR00T N1.5 Primary 14 tasks.
- Boundary: remaining gap to FP16.
- Ratio: closed-loop SR supersedes proxy ordering.
- Ablations: signal composition and ATM/OHB interaction.
- Runtime: theoretical storage is separated from eager implementation behavior.
- Missing evidence: $\pi_{0.5}$ cells remain TBD until complete frozen matrices are available.

### Conclusion

- Takeaway: geometry, distribution, and action behavior jointly improve allocation.
- Limitations: one model family, binary precision, a remaining FP16 gap, and no fused low-bit runtime.

## Claim--evidence map

- Claim: 16:1 is the frozen ratio. | Evidence: `runs/robocasa365_cs_loss/final_selection_official50.json`, 161/200, 80.5%. | Status: supported.
- Claim: GDSQ-VLA reaches 65.9% on the 14 tasks excluded from ratio selection. | Evidence: recomputation from the frozen official atomic summary after excluding the four `decision.selection_tasks`; 461/700. | Status: supported.
- Claim: GDSQ-VLA improves over QuantVLA W4A8+ATM/OHB by 20.3 points with CI [13.9, 26.9]. | Evidence: paired task-level recomputation on the same 14 tasks; Holm-adjusted p=0.0012. | Status: supported.
- Claim: GDSQ-VLA remains 8.0 points below FP16. | Evidence: paired task-level recomputation; CI [-12.1, -3.9], Holm-adjusted p=0.0073. | Status: supported.
- Claim: the final mask contains 100 W4A8 and 16 FP16 choices among 116 candidates. | Evidence: frozen final-plan layer map. | Status: supported.
- Claim: theoretical LLM+DiT component storage is 1.001 GiB, or 1.99x compression. | Evidence: `paper_style_memory` in the official atomic summary and the checkpoint/plan-based memory calculator. | Status: supported, theoretical only.
- Claim: CS and CKA are universally complementary. | Evidence: only a five-seed diagnostic screen and one checkpoint-specific ratio sweep. | Status: unsupported as a universal claim; paper uses checkpoint-scoped wording.
- Claim: the method transfers to seen composite tasks. | Evidence: complete frozen Composite-Seen summary, 40.6% versus 19.6% for W4A8+ATM/OHB; paired gain 21.0 points with CI [14.3, 28.0]. | Status: supported for Composite-Seen only.
- Claim: the atomic-selected allocation transfers better than the fixed W4A8 layout to unseen composite tasks. | Evidence: complete frozen Composite-Unseen summary, 40.4% versus 19.8%; paired gain 20.6 points with CI [14.5, 26.8] and Holm-adjusted p=0.0002. | Status: supported for the GR00T N1.5 Composite-Unseen split; not lossless versus FP16 (45.3%).
- Claim: the improvement persists across all official RoboCasa365 task groups. | Evidence: complete 50-task aggregate, 50.8% versus 30.4% for W4A8+ATM/OHB; paired gain 20.3 points with CI [16.9, 24.0]. | Status: supported for the frozen GR00T N1.5 checkpoints.
- Claim: the current implementation accelerates inference or lowers live GPU memory. | Evidence: eager runtime shows the opposite. | Status: rejected; paper explicitly limits the claim to theoretical component storage.
- Claim: \method transfers to $\pi_{0.5}$. | Evidence: complete frozen matrices are unavailable. | Status: needs evidence; the unified table uses TBD and the paper makes no cross-policy claim.

## Frozen provenance

- Ratio selection SHA256: `52f0c15fd24e9eeb0fbaf9edfc51f0f86a54867d8694cc056b01d90b8feb43b6`.
- Official atomic summary SHA256: `9f06db12444b6c582f451c2cc096bcb5ba5738cf888558955acc039aedfed29f`.
- Official Composite-Seen summary SHA256: `372688e3ac4e20eb83f2b752b2bf357d190e91d75b6cb0629e02eaf5296aa58e`.
- Official Composite-Unseen summary SHA256: `147c400a18689d27b625a943123d8948824fd7a9b4f6095fc13a8b4cfa8a5213`.
- Official 50-task aggregate summary SHA256: `243162a79a57fa9bf85402364e2cb6bf50c69baa5320fc2256e5821dfd0ef810`.
- Five-seed diagnostic summary SHA256: `da948a0b16976e41a54900305e88ee6e86b95493d8115b871980b3bc2ce0a577`.
- CVPR template tag: `CVPR2026-v1(latex)` at commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`.
- QuantVLA arXiv v4 source SHA256: `a719a574b3ed8a58533c75f2273288a479b900d385db0a6aef47dfca25893228`.
- RoboCasa365 arXiv v1 source SHA256: `f10bbc07c60b72d81332b5d1c796690b87613ab82304565e30daccc600b8e568`.
- Full citation and result-placeholder provenance: `reference_audit.md`.

## Architecture figure provenance

- Generator: bundled `imagegen` CLI with `gpt-image-2`, high quality, through the user-supplied OpenAI-compatible Images API endpoint.
- Compared candidates: `figures/candidates/gdsq-a-balanced.png`, `figures/candidates/gdsq-b-layer-map.png`, and `figures/candidates/gdsq-c-teacher-probe.png`.
- Selected base: `figures/candidates/gdsq-a-balanced.png`; it had the cleanest full-width hierarchy and the most complete claim-aligned labels.
- Targeted edit: `figures/candidates/gdsq-a-paired-edit.png`, changing only the adjudication label to `Paired Top-K Functional Adjudication`.
- Prompt records: `figures/prompts/variant-a-balanced.txt` and `figures/prompts/variant-a-paired-edit.txt`; the general production specification remains in `figures/gdsq_pipeline_prompt.txt`.
- Final asset: `figures/gdsq_pipeline.png`, normalized to 2048x1152 PNG; SHA256 `5ab0a3c61afcb72a0c4ccc9b035d0a3c4d7ec5d6fcf1383e69de305162360188`.
- Visual audit: the figure contains the three declared stages, 116 candidates, separate CKA/CS evidence, 16:1 weighting, action weight, RMS/saturation guards, the W6-equivalent budget, paired Top-K adjudication, 100 W4A8 plus 16 FP16 choices, 64 protected FP16 attention projections, and the qualified 1.99x theoretical storage claim. It contains no ATM/OHB or runtime-speed/live-memory claim.

## Five-dimension adversarial self-review

### 1. Contribution

- Pass: the paper defines a concrete dual-reference, dual-similarity selection pipeline rather than presenting another fixed layout.
- Pass: configuration-level functional adjudication addresses a clear limitation of additive layer proxies.
- Risk: novelty is layer allocation over a known quantization substrate; the paper must emphasize the controlled formulation and behavioral evidence, not claim a new low-bit operator.

### 2. Writing clarity

- Pass: the candidate scope, references, metrics, guards, budget, ratio rule, and final mask are defined.
- Pass: each paragraph has one role and terminology is stable.
- Pass: inspected the generated architecture figure on page 4 of the final review PDF at full-page resolution; labels and arrows remain readable at full paper width.

### 3. Experimental strength

- Pass: primary tasks are separated from ratio-selection tasks.
- Pass: paired uncertainty and corrected significance are reported.
- Risk: the method remains below FP16; this is framed as an explicit compression--accuracy trade-off.

### 4. Evaluation completeness

- Pass: strong fixed-layout, FP16, ratio, signal, and calibration comparisons are present.
- Pass: Atomic, Composite-Seen, and Composite-Unseen matrices are complete and strictly validated for GR00T N1.5.
- Needs new experiment: $\pi_{0.5}$ matrices must be complete before any cross-policy claim.

### 5. Method design soundness

- Pass: hard guards prevent similarity metrics from overriding activation feasibility.
- Pass: the final ratio is selected by closed-loop SR rather than proxy score alone.
- Risk: common random numbers differ from independent native GPU noise; the protocol is labeled paired and not presented as identical to an independent-noise leaderboard setting.

## Final draft checks

- [x] Replace the pipeline placeholder with a validated GPT-Image 2 PNG.
- [x] Replace Composite-Seen TBD cells from its complete frozen summary.
- [x] Replace Composite-Unseen TBD cells only from its complete frozen summary.
- [x] Re-run claim/evidence audit after inserting Composite-Unseen and the complete 50-task aggregate.
- [x] Confirm main-text page count is at most eight excluding references (references begin on page 7; the complete review PDF is nine pages including references).
- [x] Confirm no missing citations, references, or anonymization leaks.
- [x] Confirm FP16 is labeled uncompressed and excluded from best-compressed highlighting; \method (Ours) is the final row in direct configuration comparisons.
- [x] Cross-check all 43 cited BibTeX entries; consolidate official RoboCasa365 task groups in one table and report the internal Primary 14 split separately after ratio selection.
