# CHASR Progress Log

Tracks implementation progress for the Phase 1 plan so work can resume from any machine after `git pull`.

- **Design spec:** `docs/superpowers/specs/2026-08-25-chasr-causal-hallucination-attribution-design.md`
- **Implementation plan:** `docs/superpowers/plans/2026-08-25-chasr-phase1-core-pipeline.md`
- **Execution method:** superpowers `subagent-driven-development` skill — a fresh implementer subagent per task, followed by a task reviewer, followed by a fix loop if needed. All work happens directly on `main` (no feature branch, per project decision).

## How to resume

1. Open Claude Code in this repo.
2. Say: "Continue subagent-driven-development execution of docs/superpowers/plans/2026-08-25-chasr-phase1-core-pipeline.md — see PROGRESS.md for status, resume at the first task marked ⬜."
3. The `.superpowers/sdd/2026-08-25-chasr-phase1-core-pipeline/` ledger is git-ignored scratch state (per-task briefs, reports, diffs) — it only exists on the machine that ran each task. This file is the durable cross-machine record; the git history (`git log --oneline`) is the source of truth for exactly what code exists.

## Task status (14 tasks total)

| # | Task | Status | Commits |
|---|------|--------|---------|
| 1 | Repo scaffolding & environment setup | ✅ done | `7af974a..889cc0e` |
| 2 | Standard degradation pipeline | ✅ done (1 fix round — sigma ground-truth bug) | `889cc0e..d3a7b20` |
| 3 | Mismatch (OOD) degradation pipeline | ✅ done | `d3a7b20..36ffc72` |
| 4 | DDNM range-null space projection | ✅ done | `36ffc72..44b6f55` |
| 5 | Chained diffusion backbone (16x, tiled, multi-sample) + `estimate_prior_reliance_map` | ✅ done — real GPU run verified, not just stub tests | see below |
| 6 | Signal S1 — null-space variance | ✅ done | `269e095..2916971` |
| 7 | Signal S2 — generative-prior over-reliance (standalone utility; real pipeline uses Task 5's estimator) | ✅ done | `485fe96..9516b22` |
| 8 | Blind degradation kernel estimator | ⬜ not started | |
| 9 | Signal S3 — degradation-model mismatch | ⬜ not started | |
| 10 | Signal S4 — distribution shift via DINOv2 | ⬜ not started | |
| 11 | xSR-CausalBench synthetic benchmark | ⬜ not started | |
| 12 | Fusion head model + training (with signal-stack caching) | ⬜ not started | |
| 13 | Evaluation metrics (fidelity, attribution accuracy, disentanglement) | ⬜ not started | |
| 14 | End-to-end inference, downstream OCR case study, orchestration scripts | ⬜ not started | |

## Notable fixes made during execution (beyond the original plan)

Caught in the pre-flight conflict scan before any code was written (see plan's revision history in `git log -- docs/superpowers/plans/`):
- HR/LR size mismatch between the backbone's fixed 64x64→1024x1024 (16x) assumption and the original 256x256 dataset defaults — standardized on 1024x1024 throughout.
- Signal S2 was wired up with a no-op stand-in that made it mathematically zero everywhere — replaced with `estimate_prior_reliance_map` (Task 5), a real LR/seed-perturbation estimator.
- Per-epoch recomputation of signal stacks would have made one epoch of fusion-head training take days — added `precompute_signal_stacks` caching.
- The disentanglement metric (spec Section 5, item 5) was imported but never called in the evaluation script — wired up via a new `signal_stack` field on `run_chasr`'s return dict.

Caught during Task 2's review:
- `degrade_standard` recorded a `sigma` in its ground-truth params that didn't match the sigma actually used to build the blur kernel (double-sampled from the RNG stream). Fixed by having `sample_standard_kernel` return `(kernel, sigma)` instead of just the kernel.

Caught during Task 4's review:
- `ddnm_project` used `.view()` on `x_hat`/`y` instead of `.reshape()`, which would raise `RuntimeError` on non-contiguous tensors — plausible for real diffusion sampler outputs (post-`permute`/batched-index results) once wired into Task 5's backbone wrapper. Fixed by switching to `.reshape()`, matching the pattern already used for the output; added a regression test with a non-contiguous input tensor plus tests for the general (non-identity) null-space invariant and the `scale=1` degenerate case.

Environment note (2026-08-26, second machine): `opencv-python`, `diffusers`, `lpips`, and `easyocr` from `requirements.txt` were not yet installed in this machine's `py313` conda env — installed via `pip install -r requirements.txt` (split into two calls after a mid-download connection reset). No code changes; flagging in case other machines hit the same gap.

Caught during Task 5's review (code-reviewer agent, before any real-model run) and confirmed live on GPU:
- `ddnm_project(hop1, lr_patch, ...)` (and the tile/final-stitch calls) did plain tensor arithmetic between a CUDA fp16 pipeline output and a CPU fp32 `lr_patch`/`stitched` buffer — would raise a device-mismatch `RuntimeError`. Fixed by casting the reference tensor to the pipeline output's actual device/dtype at each stage instead of assuming CPU fp32.
- `estimate_prior_reliance_map`'s S2 formula divided `g_evidence` by `eps` (a true finite-difference derivative) but left `g_prior` (a discrete reseed, no natural step size) un-normalized — biases every real, non-degenerate pipeline toward S2≈0 regardless of actual behavior, since the two terms aren't on comparable scales. Fixed by comparing raw (non-eps-normalized) output-magnitude changes for both terms instead; added a `_MixedSensitivityPipeline` test exercising a non-degenerate case (both stub tests before this only covered the two 0%/100%-sensitivity extremes, which don't distinguish the buggy formula from the fixed one).
- `scripts/smoke_test_backbone.py` (plan's Step 5 deliverable) hadn't been written yet — added, and extended beyond the plan's shape-only assertion to also save a visual comparison (LR input, bicubic baseline, 2 stochastic SR samples, and a null-space-variance preview) to `outputs/smoke_test_backbone/` (git-ignored).

Caught only by actually running the real model on GPU (none of the above were catchable from stub-based unit tests or static review alone):
- `StableDiffusionUpscalePipeline.__call__` requires `prompt` or `prompt_embeds` — the plan's original `_upscale_4x` passed neither, so the very first real call raised `ValueError`. Fixed by passing `prompt=""` (text-free upscaling, matching the backbone's blind/domain-agnostic framing). The stub pipelines in the test file didn't declare a `prompt` parameter at all, so this drift was invisible to pytest — added an `assert prompt is not None` guard in the stub as a regression trip-wire.
- The pipeline also requires a **batched** `(1, C, H, W)` image tensor — passing the documented-shape `(C, H, W)` tensor made `check_inputs` read the channel dimension as a batch of 3 single-channel images and raise a batch-size-vs-prompt mismatch. Fixed with `patch.unsqueeze(0)` in `_upscale_4x`. (The installed `diffusers==0.31.0`'s `VaeImageProcessor.preprocess` — inspected during Task 5's code review — appeared to auto-unsqueeze a 3D input, but the actual installed/working version, 0.40.0 per the next bullet, does not; static source-reading of a library is not a substitute for running it.)
- `diffusers==0.31.0` (pinned in `requirements.txt`) segfaults on import against the `transformers>=4.57.0`-resolved `5.15.1` (a `FLAX_WEIGHTS_NAME` symbol diffusers 0.31.0 expects was removed in transformers 5.x). Fixed by upgrading to `diffusers>=0.40.0`, which imports cleanly. Also added `variant="fp16"` to `from_pretrained`, which wasn't in the plan's original code — without it, diffusers downloads the full fp32 checkpoint (~5GB) and casts locally instead of fetching the ~4x-smaller pre-quantized fp16 files (~1.7GB total), which matters a lot under the network conditions below.
- `huggingface_hub`'s own downloader (both its Xet fast-transfer backend and its plain-HTTP fallback via `HF_HUB_DISABLE_XET=1`) stalled indefinitely on this machine's network — near-zero throughput for many minutes — while plain `curl` to the identical resolve URLs ran at several MB/s (with the underlying network itself periodically dropping long-lived connections, requiring resumable/retrying downloads). Worked around by `curl`-downloading the three fp16 weight files by hand into `checkpoints/stable-diffusion-x4-upscaler/` (git-ignored) and adding a `model_id` constructor parameter to `DiffusionSRBackbone` (defaults to the hub repo id, unchanged for normal use) so a local snapshot directory can be substituted — exposed via `CHASR_MODEL_ID` in the smoke script. Root cause not diagnosed further (likely a proxy/CDN-path difference between `huggingface_hub`'s client and raw HTTP); worth re-testing on a different network before assuming this is permanent.
- **Must pass an absolute path** to `model_id` when using the local-checkpoint override — a relative path (`checkpoints/stable-diffusion-x4-upscaler`) contains exactly one `/`, which `from_pretrained` can misparse as a `org/repo`-shaped hub id rather than a local directory (depending on `cwd` at call time), silently falling back to hub resolution and hanging with zero GPU activity instead of raising a clear error.

**Known cosmetic limitation, not yet fixed:** the 2x2 tile-stitching in `sample_k_16x` has no overlap/blending, so real outputs show a faint grid-seam artifact at the tile boundaries (visible in the Task 5 demo images). Not blocking — DDNM projection still enforces exact data consistency across the seams — but worth revisiting (e.g. overlapping tiles + linear blend) if later signal quality is affected.

Task 6 (`compute_null_space_variance`) matched the plan verbatim and passed code review with no HIGH/CRITICAL findings — only a MEDIUM (no test pinned the unbiased-variance estimator choice against a silent scale-changing regression; added `test_variance_matches_unbiased_formula` with hand-computed expected values) and a LOW (moved a nested `import pytest` to module scope). Note for Task 12: production usage calls this with `k=2` (per the plan's `build_signal_stack`), which is a very small sample for a variance estimate — noted as a design consideration for that task, not a defect here.

Task 7 (`compute_prior_reliance`) matched the plan verbatim and passed code review with no HIGH/CRITICAL findings. The review specifically checked whether this file's symmetric eps-normalization (both `g_evidence` and `g_prior` divided by the same `eps`) has an analogous bug to Task 5's real asymmetric-eps bug (`estimate_prior_reliance_map`, logged above) — confirmed it does not; the `eps` division is algebraically inert to the ratio here, since both terms scale by the same factor. Two MEDIUM findings addressed: (1) documented in the module docstring that a `forward_fn` insensitive to both `lr` and `z` produces a misleading `0` ("fully evidence-driven") rather than the more accurate "indeterminate" — the `1e-8` guard only prevents NaN/Inf, it doesn't distinguish the cases — and pinned that documented behavior with `test_fully_insensitive_forward_fn_returns_documented_zero_not_nan`; (2) added `test_result_is_invariant_to_eps_choice`, a regression guard specifically against reintroducing a Task-5-style asymmetric-eps bug in this file (the intended "known-correct" reference implementation), plus `test_equal_sensitivity_gives_mid_range_reliance` strengthening the existing valid-range test to also check calibration at the midpoint.

## Next up

Task 8: Blind degradation kernel estimator (`docs/superpowers/plans/2026-08-25-chasr-phase1-core-pipeline.md`, line ~878) — a MANet-style estimator for Signal S3 (degradation-model mismatch); likely the first task in this batch needing an actual trained/trainable component rather than pure signal-processing math, so check its scope carefully before starting.
