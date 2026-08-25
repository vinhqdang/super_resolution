# CHASR: Causal Hallucination Attribution for Extreme Super-Resolution

**Status:** Approved for planning
**Target venue:** Signal Processing: Image Communication, Special Issue "Extreme Super-Resolution: Pushing the Boundaries of Image and Video Enhancement" (article type "VSI: Extreme Super-Resolution")
**Submission deadline:** 2026-12-15 (final manuscript), 2026-12-31 (special issue submission window close)
**Scope:** Image xSR (8x/16x+). Video xSR explicitly scoped out as future work.

## 1. Motivation and Gap

Extreme super-resolution (xSR, 8x/16x+) is severely ill-posed: massive information loss makes plausible-but-false detail generation ("hallucination") unavoidable to some degree, but current work only measures *how much* hallucination occurred, not *why*. This matters directly for the special issue's stated concerns: forensic/legal reliability, task-driven trust (OCR, detection), and the special issue's explicit call for "evaluation metrics ... beyond PSNR/SSIM."

**Closest prior art:**

| Work | Citation | What it does | Gap vs. this proposal |
|---|---|---|---|
| HalluGen | Kim, Tregidgo, Jin, Figini, Alexander. arXiv:2512.03345 (2025-12-03) | 2-way taxonomy: intrinsic (data-consistency violation) vs. extrinsic (null-space content wrong); synthetic controllable-hallucination benchmark via diffusion posterior sampling; patch-level SHAFE metric | Only 2 causes, not 4; tested at ≤4x, not extreme factors; medical/industrial domain, not natural-image xSR |
| Hallucination Score (HS) | Ren, Goyal, Hu, Aumentado-Armstrong, Mohomed, Levinshtein. arXiv:2507.14367 (2025-07-18, v2 2026-01-08) | MLLM-based scalar severity score (P1: implausibility under any degradation; P2: semantic deviation); used as differentiable fine-tuning reward | Measures severity, not cause; single scalar, not attributable/actionable |
| DDNM | Wang, Yu, Zhang. arXiv:2212.00490, ICLR 2023 | Formal range-null space decomposition for zero-shot linear inverse problems (diffusion prior fills null-space, data consistency fixes range-space) | Assumes known degradation (not blind); no attribution/diagnosis framing |
| Chain-of-Zoom | Kim, Kim, Ye (KAIST AI). arXiv:2505.18600 (2025-05-24) | Model-agnostic autoregressive scale-state framework pushing 4x diffusion SR backbones to 16x-256x via VLM text guidance | Generation-side mitigation, zero uncertainty/causal machinery |
| Aithal et al. (OOD hallucination) | cited via HS paper's related work | Defines generative-model hallucination as OOD content relative to training data | Generic to generative models, not SR/restoration-specific |
| MANet / KernelGAN / DASR | blind degradation estimation lineage | Estimate unknown blur kernels/noise for blind SR | Used here as a *component* (degradation re-estimation signal), not as prior art for attribution |

**Gap this manuscript fills:** no existing work (a) jointly attributes hallucination to four distinct causes, (b) validates at extreme (8x/16x+) factors specifically — all hallucination-measurement work above tops out at 4x, (c) turns attribution into an actionable per-region reliability signal rather than a single scalar.

Note for related-work framing: the special issue's guest editors (Larabi, Sendjasni — XLIM/Univ. Poitiers) have prior work on task-driven xSR for license plates (arXiv:2501.01483, "Embedding Similarity Guided License Plate Super Resolution"), and there is an active ICIP 2026 XLPSR Grand Challenge (arXiv:2607.08896) in the same space — cite for context, not as core prior art.

## 2. Architecture

**Frozen backbone B:** Chain-of-Zoom-style autoregressive diffusion xSR model (wraps an existing pretrained 4x diffusion SR checkpoint, e.g. SeeSR/StableSR-class, to reach 8x/16x). Not fine-tuned — keeps all training compute on the attribution module, which fits a single local GPU. B must support K stochastic degradation-consistent samples {x̂_1...x̂_K} (different seeds/DDIM eta) for the null-space signal below — verify Chain-of-Zoom's public code/checkpoints support this before implementation; if unavailable, fall back to a DDNM-wrapped SeeSR/StableSR backbone run autoregressively for the extreme factor (equivalent capability, more implementation work).

**Causal Attribution Module (CAM):** given LR input y and B's output(s), computes four per-pixel signal maps at output resolution, then a small trained fusion head combines them.

### Signal 1 — Null-space uncertainty (ill-posedness)
Pixelwise variance across K degradation-consistent stochastic samples: `S1(p) = Var_k[x̂_k(p)]`. Range-space is fixed by data-consistency projection (DDNM-style); samples differ only in null-space content. High variance = large null-space = hallucination risk inherent to the ill-posedness itself, not a model defect.

### Signal 2 — Generative-prior over-reliance
Local sensitivity of output to LR input evidence vs. to the prior's latent/noise code, estimated via finite-difference perturbation or backprop through a few diffusion steps:
`g_evidence(p) = ||∂x̂(p)/∂y||`, `g_prior(p) = ||∂x̂(p)/∂z||`
`S2(p) = g_prior(p) / (g_prior(p) + g_evidence(p) + eps)`
High S2 = output driven by the learned prior rather than available input evidence, even in regions where the evidence isn't fully absent — the "model ignored a usable clue" pathology, distinct from S1 (a region can have small null-space but still show prior over-reliance if the model doesn't use the evidence that is there).

### Signal 3 — Degradation-model mismatch
Run a blind kernel/noise estimator (MANet-style) on local patches of y to get estimated degradation params `θ̂(p)`. Compare to backbone B's assumed/training degradation distribution via re-degradation consistency error `||A_θ̂(x̂) - y||` using the estimated kernel. High mismatch = hallucination attributable to B assuming the wrong degradation model.

### Signal 4 — Distribution shift / OOD familiarity
Frozen self-supervised encoder (DINOv2) extracts patch features for LR input and SR output; compute k-NN or Mahalanobis distance to a reference bank of training-distribution patch features (built once from B's training data, e.g. DIV2K/Flickr2K). High distance = content the model hasn't seen = hallucination attributable to distribution shift.

### Fusion head F
Small CNN (4-6 conv layers) taking [S1, S2, S3, S4, shallow image features] → outputs:
- Dominant-cause map: per-pixel 5-way softmax (4 causes + "reliable/not hallucinated")
- Calibrated reliability/hallucination-risk scalar map

Trained on synthetic supervision (Section 3) — this is the only trained component in the whole pipeline.

## 3. Synthetic Supervision Construction ("xSR-CausalBench")

Extends HalluGen's diffusion-posterior-sampling controllable-hallucination trick from 2-way to 4-way. For each HR source image, apply one of four controlled injection procedures, producing a per-pixel weak "dominant cause" label mask plus a "reliable" class for well-conditioned regions:

1. **Ill-posedness-dominant:** aggressively increase effective scale factor / mask large regions before degrading; keep degradation model correctly specified and content in-distribution.
2. **Prior-over-reliance-dominant:** suppress LR conditioning signal in specific patches beyond nominal degradation (e.g. targeted heavy blur/zeroing) while leaving the diffusion prior unconstrained; moderate null-space elsewhere.
3. **Degradation-mismatch-dominant:** degrade with an out-of-training-distribution kernel/noise model while keeping content in-distribution and null-space moderate.
4. **Distribution-shift-dominant:** paste/blend rare or held-out-domain texture patches into otherwise normal, standard-degradation images.

Source images: held-out subset of DIV2K validation + Urban100 + Manga109 (structure/texture diversity). Target size: ~500-1000 images x 4 factors.

**Known risk — factor entanglement:** the four causes are not fully separable in practice (e.g. large ill-posedness naturally increases prior reliance). The spec requires this to be addressed explicitly in the paper: (a) report a correlation matrix among S1-S4 on the synthetic benchmark, (b) report attribution accuracy with confusion analysis rather than claiming clean separation, (c) discuss the entanglement honestly in Section 5 (Discussion/Limitations) rather than hiding it.

## 4. Datasets

- **Backbone context (not retrained):** B is used as published/pretrained; no fine-tuning.
- **Fusion-head training:** xSR-CausalBench (Section 3), synthetic.
- **Real-world evaluation:** RealSR and/or DRealSR (genuine camera LR-HR pairs) — tests generalization of attribution to real unknown degradations. No synthetic cause labels available here; evaluated via HS-style severity correlation and qualitative analysis only.
- **Downstream case study:** small forensic-style test set (e.g. license-plate or text-patch crops) to correlate high-S3/high-S4 regions with actual OCR/readability failure — ties to the special issue's task-driven framing without becoming the paper's whole focus.

## 5. Evaluation Protocol

1. **Fidelity (context only):** PSNR/SSIM/LPIPS/DISTS vs. bicubic, Real-ESRGAN, and backbone-alone. Not the headline result — the backbone is frozen and unchanged by this work.
2. **Hallucination severity:** reproduce or approximate the HS metric (MLLM-based, or a lighter perceptual-deviation proxy if the full HS pipeline is unavailable); correlate CHASR's fused risk score against it.
3. **Attribution accuracy (headline result):** per-pixel 5-way classification accuracy / mIoU of predicted dominant cause vs. synthetic ground truth on xSR-CausalBench. Ablate against: (a) each signal alone, (b) equal-weight hand-combination (no learned fusion), (c) full learned fusion (ours).
4. **Downstream utility case study:** correlate flagged high-S3/S4 regions with actual task failure (OCR/readability) on the forensic-style test set.
5. **Disentanglement analysis:** correlation matrix among S1-S4; ablation removing each signal from the fusion head.

## 6. Paper Structure (6 sections)

1. **Introduction** — xSR + hallucination risk + the causal-attribution gap
2. **Related Work** — blind SR, diffusion xSR (Chain-of-Zoom, StableSR/SeeSR/PASD), hallucination measurement (HS, HalluGen), uncertainty/conformal SR, null-space methods (DDNM)
3. **Method** — CHASR: four causal signals, fusion head, xSR-CausalBench construction
4. **Experiments** — datasets, baselines, metrics, main results, ablations, downstream case study
5. **Discussion / Limitations** — factor entanglement (honest treatment, not hidden), synthetic-to-real gap, backbone dependency, forensic-use caveats
6. **Conclusion**

## 7. Open Risks / To-Verify Before Implementation

- Verify Chain-of-Zoom public code/checkpoints support multi-seed stochastic sampling needed for S1; fallback is a DDNM-wrapped SeeSR/StableSR backbone run autoregressively.
- HS/HalluGen reference implementations may not be publicly reproducible in full — may need to reimplement metric proxies from the papers' descriptions.
- DINOv2 feature-bank construction (S4) is inference-only, computationally light, but needs a representative training-distribution sample from B's actual training set (or a documented proxy if B's exact training set isn't published).
- Single-blind vs. double-blind submission format for this journal must be verified before deciding whether to include a GitHub repo link / code release in the manuscript (per Elsevier SPIC author guidelines) — check at manuscript-writing time, not now.
- Factor entanglement (Section 3) is a methodological risk, not just an implementation detail — must be reported honestly with quantitative evidence, not smoothed over.

## 8. Out of Scope

- Video/multi-frame xSR (future work, one paragraph in Discussion)
- Fine-tuning or retraining the SR backbone itself
- Full task-driven optimization loop (OCR/detection accuracy as a training objective) — used only as a downstream case study, not the paper's core contribution
