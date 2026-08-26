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
| 5 | Chained diffusion backbone (16x, tiled, multi-sample) + `estimate_prior_reliance_map` | ⬜ not started | |
| 6 | Signal S1 — null-space variance | ⬜ not started | |
| 7 | Signal S2 — generative-prior over-reliance (standalone utility; real pipeline uses Task 5's estimator) | ⬜ not started | |
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

## Next up

Task 5: Chained diffusion backbone wrapper (16x, tiled, multi-sample) + `estimate_prior_reliance_map` (`docs/superpowers/plans/2026-08-25-chasr-phase1-core-pipeline.md`, line 454) — first GPU-heavy task; verify `stabilityai/stable-diffusion-x4-upscaler` availability and unit-test-with-stub-object strategy per the plan's Global Constraints before starting.
