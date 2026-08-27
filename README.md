# super_resolution

CHASR: Causal Hallucination Attribution for Extreme Super-Resolution.
Manuscript target: https://www.sciencedirect.com/special-issue/335613/extreme-super-resolution-pushing-the-boundaries-of-image-and-video-enhancement

## Design and Plan

- Design spec: `docs/superpowers/specs/2026-08-25-chasr-causal-hallucination-attribution-design.md`
- Phase 1 implementation plan: `docs/superpowers/plans/2026-08-25-chasr-phase1-core-pipeline.md`
- Progress log: `PROGRESS.md`

## Setup

```bash
conda activate py313
pip install -r requirements.txt
```

## Phase 1 pipeline (run in order)

```bash
conda run -n py313 python scripts/download_datasets.py --out-dir data/
conda run -n py313 python scripts/train_kernel_estimator.py --data-dir data/DIV2K_train_HR --out checkpoints/kernel_estimator.pt
conda run -n py313 python scripts/build_feature_bank.py --data-dir data/DIV2K_train_HR --out checkpoints/feature_bank.pt
conda run -n py313 python scripts/train_fusion_head.py
conda run -n py313 python scripts/run_evaluation.py
```

Real experiment runs (everything above except the tests) require the downloaded datasets and will take substantial wall-clock time on an 8GB-class laptop GPU — budget accordingly. See `scripts/train_fusion_head.py`'s docstring for a rough cost estimate.

## Tests

```bash
conda run -n py313 pytest tests/ -v
```

## Manual smoke tests (not run in CI/pytest)

```bash
conda run -n py313 python scripts/smoke_test_backbone.py
```
