# CHASR Phase 1: Core Causal-Attribution Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and validate the core CHASR pipeline (Section 5, spec) at the 16x extreme-SR factor: a frozen chained diffusion backbone produces stochastic degradation-consistent SR samples, four causal signals are computed from them, a small fusion head learns to attribute hallucinated pixels to one of four causes on a synthetic benchmark (xSR-CausalBench), and the headline attribution-accuracy result plus fidelity/disentanglement metrics are produced.

**Architecture:** Patch-based pipeline (64x64 LR -> 1024x1024 HR, 16x) built from independently testable modules: degradation simulators, a DDNM range-null projection utility, a tiled/chained `stabilityai/stable-diffusion-x4-upscaler` backbone wrapper, four signal-computation modules (null-space variance, prior-reliance sensitivity, degradation-mismatch via a trained blind kernel estimator, distribution-shift via DINOv2 feature-bank distance), a synthetic controllable-hallucination benchmark builder, a small trained fusion CNN, and evaluation metrics. GPU-heavy components (the diffusion backbone, DINOv2) are wrapped so unit tests inject stub objects and run fast/offline; real-model integration is verified via separate smoke scripts run manually.

**Tech Stack:** Python 3.13 (conda env `py313`, already has CUDA-enabled torch 2.11 + transformers, scikit-image, opencv, scipy, sklearn, easyocr). New dependencies: `diffusers`, `lpips`. All commands run via `conda run -n py313 ...` per project convention.

**Spec:** `docs/superpowers/specs/2026-08-25-chasr-causal-hallucination-attribution-design.md` — this plan implements Sections 2-5 at the 16x factor only; 8x, video, the full downstream utility study beyond a minimal OCR case, and full HS-metric reproduction are explicitly deferred (Section 8 of the spec, and user instruction to extend later).

## Global Constraints

- Use `conda run -n py313 <cmd>` for every Python execution (per project convention — do not create a new conda env).
- GPU is an 8GB laptop RTX A2000 — all diffusion/DINOv2 work must run at fp16, patch sizes capped at 128x128 input to any single diffusion call, and unit tests must not require downloading or running the real diffusion/DINOv2 models (inject stub objects instead; real-model runs live in `scripts/smoke_test_*.py`, run manually).
- Downsampling operator `A` is always average-pooling by the given integer scale factor — this must stay consistent between the degradation pipeline (Tasks 2-3) and the DDNM projection (Task 4), since the projection's correctness depends on it.
- No claude identity/co-authorship in any git commit (already enforced by global git config).
- Push to `origin main` after each commit (per user's global instruction to push after every edit).
- New source code lives under `src/`, tests under `tests/`, one-off/manual scripts under `scripts/`, in the existing `C:\work\super_resolution` repo.

---

## Task 1: Repo Scaffolding and Environment Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/degradation/__init__.py`
- Create: `src/backbone/__init__.py`
- Create: `src/signals/__init__.py`
- Create: `src/causalbench/__init__.py`
- Create: `src/fusion/__init__.py`
- Create: `src/eval/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_environment.py`

**Interfaces:**
- Produces: package skeleton (`src.degradation`, `src.backbone`, `src.signals`, `src.causalbench`, `src.fusion`, `src.eval`) that every later task imports from.

- [ ] **Step 1: Install new dependencies into the py313 env**

Run: `conda run -n py313 pip install diffusers==0.31.0 lpips==0.1.4`
Expected: both install without error (transformers, torch, skimage, cv2, scipy, sklearn, easyocr are already present per environment check).

- [ ] **Step 2: Write requirements.txt documenting the full dependency set**

```
torch>=2.11.0
diffusers==0.31.0
transformers>=4.57.0
lpips==0.1.4
scikit-image>=0.26.0
opencv-python>=4.13.0
scipy>=1.17.0
scikit-learn>=1.8.0
easyocr>=1.7.2
pytest>=9.1.0
numpy>=2.4.0
Pillow>=12.1.0
```

- [ ] **Step 3: Create package skeleton**

Create each `__init__.py` above as an empty file (just a package marker — no content needed yet since each task fills in its own module).

- [ ] **Step 4: Write the environment smoke test**

```python
# tests/test_environment.py
import importlib

REQUIRED_MODULES = [
    "torch", "diffusers", "transformers", "lpips",
    "skimage", "cv2", "scipy", "sklearn", "easyocr", "numpy", "PIL",
]


def test_required_modules_importable():
    missing = []
    for name in REQUIRED_MODULES:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    assert not missing, f"Missing modules: {missing}"


def test_cuda_available():
    import torch
    assert torch.cuda.is_available(), "CUDA GPU not detected — required for backbone/signal tasks"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `conda run -n py313 pytest tests/test_environment.py -v`
Expected: 2 passed.

- [ ] **Step 6: Commit and push**

```bash
git add requirements.txt src/ tests/__init__.py tests/test_environment.py
git commit -m "chore: scaffold CHASR package structure and verify environment"
git push origin main
```

---

## Task 2: Standard Degradation Pipeline

**Files:**
- Create: `src/degradation/real_esrgan_degrade.py`
- Test: `tests/degradation/test_real_esrgan_degrade.py`

**Interfaces:**
- Produces: `sample_standard_kernel(rng: np.random.Generator) -> np.ndarray` (2D kernel), `degrade_standard(hr: np.ndarray, scale: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]` where the returned dict has keys `sigma_x, sigma_y, theta, noise_sigma` (ground-truth degradation params, consumed by Task 8's kernel estimator training and Task 11's benchmark).
- Consumes: nothing (base module).

- [ ] **Step 1: Write the failing test for kernel sampling and degradation shape/consistency**

```python
# tests/degradation/test_real_esrgan_degrade.py
import numpy as np
from src.degradation.real_esrgan_degrade import sample_standard_kernel, degrade_standard


def test_sample_standard_kernel_is_normalized_2d():
    rng = np.random.default_rng(0)
    kernel = sample_standard_kernel(rng)
    assert kernel.ndim == 2
    assert kernel.shape[0] == kernel.shape[1]
    assert np.isclose(kernel.sum(), 1.0, atol=1e-5)


def test_degrade_standard_output_shape_and_downsample_consistency():
    rng = np.random.default_rng(0)
    hr = rng.uniform(0, 1, size=(256, 256, 3)).astype(np.float32)
    lr, params = degrade_standard(hr, scale=4, rng=rng)
    assert lr.shape == (64, 64, 3)
    assert lr.min() >= -0.05 and lr.max() <= 1.05  # allow small noise overshoot
    for key in ("sigma_x", "sigma_y", "theta", "noise_sigma"):
        assert key in params


def test_degrade_standard_kernel_params_within_standard_pool():
    rng = np.random.default_rng(1)
    hr = rng.uniform(0, 1, size=(128, 128, 3)).astype(np.float32)
    for _ in range(20):
        _, params = degrade_standard(hr, scale=4, rng=rng)
        assert 0.2 <= params["sigma_x"] <= 1.5
        assert 0.2 <= params["sigma_y"] <= 1.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/degradation/test_real_esrgan_degrade.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.degradation.real_esrgan_degrade'`

- [ ] **Step 3: Implement the standard degradation pipeline**

```python
# src/degradation/real_esrgan_degrade.py
"""Standard (in-training-distribution) degradation pipeline.

Isotropic Gaussian blur -> average-pool downsample -> Gaussian noise -> JPEG
recompression. The downsample step is deliberately average-pooling so it
matches the operator assumed by the DDNM projection in
src/backbone/ddnm_projection.py — the two must stay consistent.
"""
import cv2
import numpy as np

STANDARD_SIGMA_RANGE = (0.2, 1.5)
STANDARD_NOISE_RANGE = (0.0, 0.03)
STANDARD_JPEG_QUALITY_RANGE = (70, 100)
KERNEL_SIZE = 21


def sample_standard_kernel(rng: np.random.Generator) -> np.ndarray:
    """Isotropic Gaussian blur kernel (sigma_x == sigma_y, theta == 0)."""
    sigma = rng.uniform(*STANDARD_SIGMA_RANGE)
    ax = np.arange(KERNEL_SIZE) - KERNEL_SIZE // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _blur(hr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    return cv2.filter2D(hr, ddepth=-1, kernel=kernel, borderType=cv2.BORDER_REFLECT)


def _average_pool_downsample(img: np.ndarray, scale: int) -> np.ndarray:
    h, w, c = img.shape
    return img.reshape(h // scale, scale, w // scale, scale, c).mean(axis=(1, 3))


def _add_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    return img + rng.normal(0, sigma, size=img.shape).astype(np.float32)


def _jpeg_recompress(img: np.ndarray, quality: int) -> np.ndarray:
    img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", img_u8, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec.astype(np.float32) / 255.0


def degrade_standard(hr: np.ndarray, scale: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """hr: (H, W, 3) float32 in [0, 1]. Returns (lr, ground_truth_params)."""
    sigma = rng.uniform(*STANDARD_SIGMA_RANGE)
    kernel = sample_standard_kernel(rng)
    blurred = _blur(hr, kernel)
    lr = _average_pool_downsample(blurred, scale)
    noise_sigma = rng.uniform(*STANDARD_NOISE_RANGE)
    lr = _add_noise(lr, noise_sigma, rng)
    quality = int(rng.integers(*STANDARD_JPEG_QUALITY_RANGE))
    lr = _jpeg_recompress(lr, quality)
    params = {"sigma_x": float(sigma), "sigma_y": float(sigma), "theta": 0.0, "noise_sigma": float(noise_sigma)}
    return lr, params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/degradation/test_real_esrgan_degrade.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/degradation/real_esrgan_degrade.py tests/degradation/test_real_esrgan_degrade.py
git commit -m "feat: add standard degradation pipeline for xSR training pairs"
git push origin main
```

---

## Task 3: Mismatch (Out-of-Distribution) Degradation Pipeline

**Files:**
- Create: `src/degradation/mismatch_degrade.py`
- Test: `tests/degradation/test_mismatch_degrade.py`

**Interfaces:**
- Consumes: `_average_pool_downsample`, `_add_noise`, `_jpeg_recompress` from `src.degradation.real_esrgan_degrade` (shared helpers, imported not duplicated).
- Produces: `sample_mismatch_kernel(rng) -> np.ndarray`, `degrade_mismatch(hr: np.ndarray, scale: int, rng) -> tuple[np.ndarray, dict]` — same return shape as `degrade_standard`, but params drawn from a distribution disjoint from `STANDARD_SIGMA_RANGE`. Consumed by Task 9 (S3 degradation-mismatch signal) and Task 11 (benchmark's mismatch-dominant procedure).

- [ ] **Step 1: Write the failing test asserting the mismatch pool is disjoint from the standard pool**

```python
# tests/degradation/test_mismatch_degrade.py
import numpy as np
from src.degradation.real_esrgan_degrade import STANDARD_SIGMA_RANGE
from src.degradation.mismatch_degrade import sample_mismatch_kernel, degrade_mismatch


def test_mismatch_kernel_is_anisotropic():
    rng = np.random.default_rng(0)
    kernel = sample_mismatch_kernel(rng)
    assert kernel.ndim == 2
    assert np.isclose(kernel.sum(), 1.0, atol=1e-5)


def test_mismatch_params_outside_standard_pool():
    rng = np.random.default_rng(2)
    hr = rng.uniform(0, 1, size=(128, 128, 3)).astype(np.float32)
    outside_count = 0
    for _ in range(20):
        _, params = degrade_mismatch(hr, scale=4, rng=rng)
        max_sigma = max(params["sigma_x"], params["sigma_y"])
        if max_sigma > STANDARD_SIGMA_RANGE[1] or abs(params["theta"]) > 1e-6:
            outside_count += 1
    assert outside_count == 20  # every sample must be outside the standard pool


def test_degrade_mismatch_output_shape():
    rng = np.random.default_rng(3)
    hr = rng.uniform(0, 1, size=(256, 256, 3)).astype(np.float32)
    lr, params = degrade_mismatch(hr, scale=4, rng=rng)
    assert lr.shape == (64, 64, 3)
    for key in ("sigma_x", "sigma_y", "theta", "noise_sigma"):
        assert key in params
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/degradation/test_mismatch_degrade.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.degradation.mismatch_degrade'`

- [ ] **Step 3: Implement the mismatch degradation pipeline**

```python
# src/degradation/mismatch_degrade.py
"""Out-of-training-distribution degradation pipeline: anisotropic/rotated
Gaussian blur with a wider sigma range than the standard pool. Used to
synthesize the degradation-mismatch-dominant slice of xSR-CausalBench and
as negative-distribution training data for the blind kernel estimator.
"""
import cv2
import numpy as np

from src.degradation.real_esrgan_degrade import (
    KERNEL_SIZE,
    STANDARD_JPEG_QUALITY_RANGE,
    STANDARD_NOISE_RANGE,
    _add_noise,
    _average_pool_downsample,
    _blur,
    _jpeg_recompress,
)

MISMATCH_SIGMA_RANGE = (1.6, 4.0)  # disjoint from STANDARD_SIGMA_RANGE = (0.2, 1.5)
MISMATCH_THETA_RANGE = (0.0, np.pi)


def sample_mismatch_kernel(rng: np.random.Generator) -> np.ndarray:
    """Anisotropic, rotated Gaussian blur kernel — outside the standard pool."""
    sigma_x = rng.uniform(*MISMATCH_SIGMA_RANGE)
    sigma_y = rng.uniform(*MISMATCH_SIGMA_RANGE)
    theta = rng.uniform(*MISMATCH_THETA_RANGE)
    ax = np.arange(KERNEL_SIZE) - KERNEL_SIZE // 2
    xx, yy = np.meshgrid(ax, ax)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)
    kernel = np.exp(-(xr ** 2 / (2 * sigma_x ** 2) + yr ** 2 / (2 * sigma_y ** 2)))
    kernel /= kernel.sum()
    return kernel.astype(np.float32), sigma_x, sigma_y, theta


def degrade_mismatch(hr: np.ndarray, scale: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    kernel, sigma_x, sigma_y, theta = sample_mismatch_kernel(rng)
    blurred = _blur(hr, kernel)
    lr = _average_pool_downsample(blurred, scale)
    noise_sigma = rng.uniform(*STANDARD_NOISE_RANGE)
    lr = _add_noise(lr, noise_sigma, rng)
    quality = int(rng.integers(*STANDARD_JPEG_QUALITY_RANGE))
    lr = _jpeg_recompress(lr, quality)
    params = {"sigma_x": float(sigma_x), "sigma_y": float(sigma_y), "theta": float(theta), "noise_sigma": float(noise_sigma)}
    return lr, params
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/degradation/test_mismatch_degrade.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/degradation/mismatch_degrade.py tests/degradation/test_mismatch_degrade.py
git commit -m "feat: add out-of-distribution degradation pipeline for mismatch signal"
git push origin main
```

---

## Task 4: DDNM Range-Null Space Projection

**Files:**
- Create: `src/backbone/ddnm_projection.py`
- Test: `tests/backbone/test_ddnm_projection.py`

**Interfaces:**
- Produces: `ddnm_project(x_hat: torch.Tensor, y: torch.Tensor, scale: int) -> torch.Tensor` where `x_hat` is `(C, H, W)`, `y` is `(C, H//scale, W//scale)`. Consumed by Task 5 (backbone wrapper, applied after every stochastic sample) and Task 6 (S1 signal operates on projected samples).
- Consumes: nothing (pure tensor math, no model dependency — deliberately GPU/download-free so it's fast to test).

- [ ] **Step 1: Write the failing test for exact data-consistency after projection**

```python
# tests/backbone/test_ddnm_projection.py
import torch
from src.backbone.ddnm_projection import ddnm_project


def _average_pool(x: torch.Tensor, scale: int) -> torch.Tensor:
    c, h, w = x.shape
    return x.view(c, h // scale, scale, w // scale, scale).mean(dim=(2, 4))


def test_ddnm_project_enforces_exact_data_consistency():
    torch.manual_seed(0)
    scale = 4
    x_hat = torch.rand(3, 16, 16)
    y = torch.rand(3, 4, 4)  # arbitrary target observation, unrelated to x_hat
    x_proj = ddnm_project(x_hat, y, scale)
    reprojected = _average_pool(x_proj, scale)
    assert torch.allclose(reprojected, y, atol=1e-5)


def test_ddnm_project_preserves_null_space_detail():
    torch.manual_seed(1)
    scale = 4
    x_hat = torch.rand(3, 16, 16)
    y = _average_pool(x_hat, scale)  # already consistent
    x_proj = ddnm_project(x_hat, y, scale)
    assert torch.allclose(x_proj, x_hat, atol=1e-5)


def test_ddnm_project_output_shape():
    x_hat = torch.rand(3, 32, 32)
    y = torch.rand(3, 8, 8)
    x_proj = ddnm_project(x_hat, y, scale=4)
    assert x_proj.shape == x_hat.shape
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/backbone/test_ddnm_projection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backbone.ddnm_projection'`

- [ ] **Step 3: Implement the projection**

```python
# src/backbone/ddnm_projection.py
"""Range-null space projection for the average-pool downsampling operator A.

For A = average-pool by `scale`, the pseudo-inverse correction has a closed
form: within every scale x scale block, replace the block by
(block - block_mean + y_pixel). This exactly enforces A(x_proj) == y (the
range-space is fixed to the observation) while leaving the null-space
(within-block detail, i.e. anything A maps to zero) untouched. This is the
DDNM-style projection referenced in spec Section 2, Signal 1.
"""
import torch


def ddnm_project(x_hat: torch.Tensor, y: torch.Tensor, scale: int) -> torch.Tensor:
    """x_hat: (C, H, W). y: (C, H//scale, W//scale). Returns (C, H, W)."""
    c, h, w = x_hat.shape
    blocks = x_hat.view(c, h // scale, scale, w // scale, scale)
    block_mean = blocks.mean(dim=(2, 4), keepdim=True)
    y_expanded = y.view(c, h // scale, 1, w // scale, 1)
    x_proj = blocks - block_mean + y_expanded
    return x_proj.reshape(c, h, w)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/backbone/test_ddnm_projection.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/backbone/ddnm_projection.py tests/backbone/test_ddnm_projection.py
git commit -m "feat: add DDNM range-null space projection for average-pool degradation"
git push origin main
```

---

## Task 5: Chained Diffusion Backbone Wrapper (16x, Tiled, Multi-Sample)

**Files:**
- Create: `src/backbone/diffusion_backbone.py`
- Create: `scripts/smoke_test_backbone.py`
- Test: `tests/backbone/test_diffusion_backbone.py`

**Interfaces:**
- Consumes: `ddnm_project` from `src.backbone.ddnm_projection`.
- Produces: `class DiffusionSRBackbone` with `__init__(self, pipeline=None, device="cuda")` (lazy-loads `diffusers.StableDiffusionUpscalePipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler")` in fp16 if `pipeline is None`; accepts an injected stub for testing), and `sample_k_16x(self, lr_patch: torch.Tensor, k: int, base_seed: int) -> list[torch.Tensor]` returning `k` tensors of shape `(3, 1024, 1024)` for a `(3, 64, 64)` input, each DDNM-projected against the true observation at every hop; plus module-level `estimate_prior_reliance_map(backbone, lr_patch, base_seed=0, eps=1e-2) -> torch.Tensor` returning `(1024, 1024)` in `[0, 1]` — the real-pipeline S2 estimator (see Step 3 docstring for why this doesn't reuse `compute_prior_reliance` from Task 7 directly: that function assumes a continuously-perturbable latent, but this backbone's stochasticity is a discrete seed). Consumed by Task 6 (S1), Task 12/Task 14 (S2 in the real pipeline), Task 11 (benchmark).

- [ ] **Step 1: Write the failing test using a stub pipeline (no real model download)**

```python
# tests/backbone/test_diffusion_backbone.py
import torch
from src.backbone.diffusion_backbone import DiffusionSRBackbone


class _StubPipeline:
    """Returns a deterministic-shape random image; ignores the prompt/seed logic
    beyond producing a torch generator-seeded tensor, standing in for the real
    diffusers pipeline so this test needs no network/GPU download."""

    def __call__(self, image, generator, num_inference_steps=20, output_type="pt"):
        # image: PIL.Image or tensor sized (h, w); we only need the output 4x bigger.
        h, w = image.shape[-2], image.shape[-1]
        g = generator
        out = torch.rand(1, 3, h * 4, w * 4, generator=g)
        return type("Result", (), {"images": out})()


def test_sample_k_16x_output_shape_and_count():
    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    samples = backbone.sample_k_16x(lr_patch, k=2, base_seed=0)
    assert len(samples) == 2
    for s in samples:
        assert s.shape == (3, 1024, 1024)


def test_sample_k_16x_is_degradation_consistent_at_each_hop():
    from src.backbone.ddnm_projection import ddnm_project

    def _average_pool(x, scale):
        c, h, w = x.shape
        return x.view(c, h // scale, scale, w // scale, scale).mean(dim=(2, 4))

    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    samples = backbone.sample_k_16x(lr_patch, k=1, base_seed=0)
    final = samples[0]
    reprojected_full = _average_pool(final, scale=16)
    assert torch.allclose(reprojected_full, lr_patch, atol=1e-4)


def test_sample_k_16x_different_seeds_give_different_samples():
    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    samples = backbone.sample_k_16x(lr_patch, k=2, base_seed=0)
    assert not torch.allclose(samples[0], samples[1])


class _ContentOnlyPipeline:
    """Output depends only on the input image (nearest-neighbor 4x upsample)
    and completely ignores the generator/seed — the opposite extreme from
    _StubPipeline. Used to verify estimate_prior_reliance_map reports near-
    zero prior reliance when the seed provably doesn't matter."""

    def __call__(self, image, generator, num_inference_steps=20, output_type="pt"):
        out = torch.nn.functional.interpolate(image.unsqueeze(0), scale_factor=4, mode="nearest")
        return type("Result", (), {"images": out})()


def test_estimate_prior_reliance_map_output_shape():
    from src.backbone.diffusion_backbone import estimate_prior_reliance_map

    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    s2 = estimate_prior_reliance_map(backbone, lr_patch)
    assert s2.shape == (1024, 1024)
    assert (s2 >= 0).all() and (s2 <= 1).all()


def test_estimate_prior_reliance_map_high_when_only_seed_matters():
    from src.backbone.diffusion_backbone import estimate_prior_reliance_map

    backbone = DiffusionSRBackbone(pipeline=_StubPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    s2 = estimate_prior_reliance_map(backbone, lr_patch)
    assert s2.mean() > 0.9


def test_estimate_prior_reliance_map_low_when_only_content_matters():
    from src.backbone.diffusion_backbone import estimate_prior_reliance_map

    backbone = DiffusionSRBackbone(pipeline=_ContentOnlyPipeline(), device="cpu")
    lr_patch = torch.rand(3, 64, 64)
    s2 = estimate_prior_reliance_map(backbone, lr_patch)
    assert s2.mean() < 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/backbone/test_diffusion_backbone.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backbone.diffusion_backbone'`

- [ ] **Step 3: Implement the backbone wrapper**

```python
# src/backbone/diffusion_backbone.py
"""Frozen extreme-SR backbone B: two chained 4x diffusion super-resolution
hops (64x64 -> 256x256 -> 1024x1024, i.e. 16x total), matching the
Chain-of-Zoom autoregressive-zoom idea (arXiv:2505.18600) but built on the
public, directly downloadable stabilityai/stable-diffusion-x4-upscaler
checkpoint for reproducibility on a single 8GB GPU. Each hop is DDNM-
projected (src.backbone.ddnm_projection) against its true observation so
every returned sample is exactly degradation-consistent — required for the
Signal 1 (null-space variance) computation in src/signals/null_space.py.

The second hop is applied per-tile (2x2 tiling of the 256x256 intermediate
into four 128x128 tiles, each upscaled to 512x512 and stitched back to
1024x1024) to stay within the 8GB VRAM budget — see Global Constraints.
"""
import torch

from src.backbone.ddnm_projection import ddnm_project


class DiffusionSRBackbone:
    def __init__(self, pipeline=None, device: str = "cuda"):
        self.device = device
        self._pipeline = pipeline  # lazy-loaded below if None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from diffusers import StableDiffusionUpscalePipeline

            self._pipeline = StableDiffusionUpscalePipeline.from_pretrained(
                "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
            ).to(self.device)
            self._pipeline.enable_attention_slicing()
        return self._pipeline

    def _upscale_4x(self, patch: torch.Tensor, seed: int) -> torch.Tensor:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = self.pipeline(image=patch, generator=generator, num_inference_steps=20, output_type="pt")
        return result.images[0]

    def sample_k_16x(self, lr_patch: torch.Tensor, k: int, base_seed: int) -> list[torch.Tensor]:
        """lr_patch: (3, 64, 64). Returns k tensors of shape (3, 1024, 1024)."""
        samples = []
        for i in range(k):
            seed = base_seed + i
            hop1 = self._upscale_4x(lr_patch, seed=seed)  # (3, 256, 256)
            hop1 = ddnm_project(hop1, lr_patch, scale=4)

            tiles = []
            for ty in range(2):
                for tx in range(2):
                    tile_lr = hop1[:, ty * 128 : (ty + 1) * 128, tx * 128 : (tx + 1) * 128]
                    tile_hr = self._upscale_4x(tile_lr, seed=seed + 1000 + ty * 2 + tx)  # (3, 512, 512)
                    tile_hr = ddnm_project(tile_hr, tile_lr, scale=4)
                    tiles.append((ty, tx, tile_hr))

            stitched = torch.zeros(3, 1024, 1024)
            for ty, tx, tile_hr in tiles:
                stitched[:, ty * 512 : (ty + 1) * 512, tx * 512 : (tx + 1) * 512] = tile_hr

            final = ddnm_project(stitched, lr_patch, scale=16)
            samples.append(final)
        return samples


def estimate_prior_reliance_map(backbone: DiffusionSRBackbone, lr_patch: torch.Tensor, base_seed: int = 0, eps: float = 1e-2) -> torch.Tensor:
    """Signal S2 (spec Section 2), real-pipeline implementation. Generic
    finite-difference sensitivity (src/signals/prior_reliance.py) assumes a
    continuously-perturbable forward_fn and latent code; this backbone's
    stochasticity is a discrete integer seed, not a continuous latent, so
    S2 is estimated directly here instead: sensitivity of the raw (pre-
    DDNM-projection) hop-1 output to a small perturbation of the LR
    evidence at fixed seed (g_evidence) vs. to a different seed at fixed LR
    evidence (g_prior, a discrete perturbation — no eps normalization).
    Computed once at hop-1 resolution (256x256) — only 2 extra single-hop
    diffusion calls, not full 16x chains — then nearest-upsampled to
    1024x1024 to align with S1/S3/S4.
    """
    base_hop1 = backbone._upscale_4x(lr_patch, seed=base_seed)  # (3, 256, 256)

    lr_perturbed = lr_patch + eps * torch.randn_like(lr_patch)
    hop1_lr_perturbed = backbone._upscale_4x(lr_perturbed, seed=base_seed)
    g_evidence = (hop1_lr_perturbed - base_hop1).abs().mean(dim=0) / eps  # (256, 256)

    hop1_seed_perturbed = backbone._upscale_4x(lr_patch, seed=base_seed + 1)
    g_prior = (hop1_seed_perturbed - base_hop1).abs().mean(dim=0)  # (256, 256)

    s2_hop1 = g_prior / (g_prior + g_evidence + 1e-8)
    return torch.nn.functional.interpolate(s2_hop1.unsqueeze(0).unsqueeze(0), size=(1024, 1024), mode="nearest").squeeze(0).squeeze(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/backbone/test_diffusion_backbone.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the manual smoke script for the real model (not part of pytest — run once by hand)**

```python
# scripts/smoke_test_backbone.py
"""Manual integration check: downloads stabilityai/stable-diffusion-x4-upscaler
(~5GB) and runs one real 16x chained upscale. Not run in CI/pytest — run by
hand with: conda run -n py313 python scripts/smoke_test_backbone.py
"""
import torch

from src.backbone.diffusion_backbone import DiffusionSRBackbone


def main():
    backbone = DiffusionSRBackbone(device="cuda")
    lr_patch = torch.rand(3, 64, 64)
    samples = backbone.sample_k_16x(lr_patch, k=1, base_seed=0)
    print("Output shape:", samples[0].shape)
    assert samples[0].shape == (3, 1024, 1024)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit and push**

```bash
git add src/backbone/diffusion_backbone.py scripts/smoke_test_backbone.py tests/backbone/test_diffusion_backbone.py
git commit -m "feat: add chained diffusion backbone for 16x extreme SR with DDNM consistency"
git push origin main
```

---

## Task 6: Signal S1 — Null-Space Variance

**Files:**
- Create: `src/signals/null_space.py`
- Test: `tests/signals/test_null_space.py`

**Interfaces:**
- Produces: `compute_null_space_variance(samples: list[torch.Tensor]) -> torch.Tensor` returning `(H, W)` (channel-averaged pixelwise variance). Consumed by Task 12 (fusion head input) and Task 14 (end-to-end inference).
- Consumes: nothing beyond `torch` (operates on the output of `DiffusionSRBackbone.sample_k_16x`).

- [ ] **Step 1: Write the failing test**

```python
# tests/signals/test_null_space.py
import torch
from src.signals.null_space import compute_null_space_variance


def test_identical_samples_give_zero_variance():
    x = torch.rand(3, 8, 8)
    samples = [x.clone() for _ in range(4)]
    s1 = compute_null_space_variance(samples)
    assert s1.shape == (8, 8)
    assert torch.allclose(s1, torch.zeros(8, 8), atol=1e-6)


def test_variable_region_has_higher_variance_than_stable_region():
    torch.manual_seed(0)
    base = torch.rand(3, 8, 8)
    samples = []
    for _ in range(6):
        noisy = base.clone()
        noisy[:, :4, :] += torch.randn(3, 4, 8) * 0.5  # only top half varies
        samples.append(noisy)
    s1 = compute_null_space_variance(samples)
    assert s1[:4, :].mean() > s1[4:, :].mean()


def test_raises_on_fewer_than_two_samples():
    import pytest
    with pytest.raises(ValueError):
        compute_null_space_variance([torch.rand(3, 4, 4)])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/signals/test_null_space.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.signals.null_space'`

- [ ] **Step 3: Implement**

```python
# src/signals/null_space.py
"""Signal S1 (spec Section 2): pixelwise variance across K degradation-
consistent stochastic samples, channel-averaged. High variance = large
null-space = hallucination risk inherent to the ill-posedness itself.
"""
import torch


def compute_null_space_variance(samples: list[torch.Tensor]) -> torch.Tensor:
    if len(samples) < 2:
        raise ValueError("Need at least 2 samples to compute variance")
    stacked = torch.stack(samples, dim=0)  # (K, C, H, W)
    var = stacked.var(dim=0, unbiased=True)  # (C, H, W)
    return var.mean(dim=0)  # (H, W)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/signals/test_null_space.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/signals/null_space.py tests/signals/test_null_space.py
git commit -m "feat: add S1 null-space variance signal"
git push origin main
```

---

## Task 7: Signal S2 — Generative-Prior Over-Reliance

**Files:**
- Create: `src/signals/prior_reliance.py`
- Test: `tests/signals/test_prior_reliance.py`

**Interfaces:**
- Produces: `compute_prior_reliance(forward_fn, lr: torch.Tensor, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor` returning `(H, W)` in `[0, 1]`, where `forward_fn(lr, z) -> torch.Tensor` (shape `(C, H, W)`) is any differentiable-or-perturbable stand-in for the backbone. This is a general, standalone-tested utility for any forward_fn with a continuous latent `z` — it is deliberately NOT wired into the real CHASR pipeline, because `DiffusionSRBackbone`'s stochasticity is a discrete integer seed, not a continuous latent that `z + eps*randn` can perturb. The real pipeline's S2 uses `estimate_prior_reliance_map` from Task 5 instead (see that task's docstring). Not consumed elsewhere in this plan — kept as a documented, tested building block for a future differentiable backbone.
- Consumes: nothing beyond `torch` — takes `forward_fn` as a parameter so tests use a simple synthetic function instead of the real backbone.

- [ ] **Step 1: Write the failing test with a synthetic forward function of known sensitivity**

```python
# tests/signals/test_prior_reliance.py
import torch
from src.signals.prior_reliance import compute_prior_reliance


def test_evidence_dominant_region_gives_low_reliance():
    def forward_fn(lr, z):
        # Output driven entirely by lr (upsampled), z ignored -> evidence-dominant.
        return torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=4, mode="nearest").squeeze(0)

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert s2.shape == (16, 16)
    assert s2.mean() < 0.3


def test_prior_dominant_region_gives_high_reliance():
    def forward_fn(lr, z):
        # Output driven entirely by z, lr ignored -> prior-dominant.
        return z * 1.0

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert s2.mean() > 0.7


def test_output_in_valid_range():
    def forward_fn(lr, z):
        up = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=4, mode="nearest").squeeze(0)
        return 0.5 * up + 0.5 * z

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert (s2 >= 0).all() and (s2 <= 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/signals/test_prior_reliance.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.signals.prior_reliance'`

- [ ] **Step 3: Implement**

```python
# src/signals/prior_reliance.py
"""Signal S2 (spec Section 2): local sensitivity of the output to LR input
evidence vs. to the prior's latent/noise code, via finite-difference
perturbation so it works against any black-box forward_fn (the real
diffusion backbone is not cleanly differentiable end-to-end through
sampling steps).
"""
import torch


def compute_prior_reliance(forward_fn, lr: torch.Tensor, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    base = forward_fn(lr, z)  # (C, H, W)

    lr_perturbed = lr + eps * torch.randn_like(lr)
    out_lr_perturbed = forward_fn(lr_perturbed, z)
    g_evidence = (out_lr_perturbed - base).abs().mean(dim=0) / eps  # (H, W)

    z_perturbed = z + eps * torch.randn_like(z)
    out_z_perturbed = forward_fn(lr, z_perturbed)
    g_prior = (out_z_perturbed - base).abs().mean(dim=0) / eps  # (H, W)

    return g_prior / (g_prior + g_evidence + 1e-8)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/signals/test_prior_reliance.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/signals/prior_reliance.py tests/signals/test_prior_reliance.py
git commit -m "feat: add S2 generative-prior over-reliance signal"
git push origin main
```

---

## Task 8: Blind Degradation Kernel Estimator

**Files:**
- Create: `src/signals/kernel_estimator.py`
- Create: `scripts/train_kernel_estimator.py`
- Test: `tests/signals/test_kernel_estimator.py`

**Interfaces:**
- Consumes: `degrade_standard` (Task 2), `degrade_mismatch` (Task 3) as synthetic training-data sources.
- Produces: `class KernelEstimator(torch.nn.Module)` with `forward(self, lr_patch: torch.Tensor) -> torch.Tensor` returning `(4,)` = `[sigma_x, sigma_y, theta, noise_sigma]` per 64x64 patch; and `KernelEstimatorDataset` for training. Consumed by Task 9 (S3 signal).

- [ ] **Step 1: Write the failing test — model shape and one-step overfit sanity check**

```python
# tests/signals/test_kernel_estimator.py
import numpy as np
import torch
from src.degradation.real_esrgan_degrade import degrade_standard
from src.signals.kernel_estimator import KernelEstimator


def test_forward_output_shape():
    model = KernelEstimator()
    lr = torch.rand(1, 3, 64, 64)
    out = model(lr)
    assert out.shape == (1, 4)


def test_model_can_overfit_a_tiny_synthetic_batch():
    rng = np.random.default_rng(0)
    hr = rng.uniform(0, 1, size=(256, 256, 3)).astype(np.float32)
    lrs, targets = [], []
    for _ in range(4):
        lr, params = degrade_standard(hr, scale=4, rng=rng)
        lrs.append(torch.from_numpy(lr).permute(2, 0, 1))
        targets.append(torch.tensor([params["sigma_x"], params["sigma_y"], params["theta"], params["noise_sigma"]]))
    lr_batch = torch.stack(lrs).float()
    target_batch = torch.stack(targets).float()

    model = KernelEstimator()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    losses = []
    for _ in range(50):
        optim.zero_grad()
        pred = model(lr_batch)
        loss = torch.nn.functional.mse_loss(pred, target_batch)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/signals/test_kernel_estimator.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.signals.kernel_estimator'`

- [ ] **Step 3: Implement the estimator and its training dataset**

```python
# src/signals/kernel_estimator.py
"""Lightweight blind degradation estimator (simplified MANet-style): a
small CNN regressing degradation parameters (sigma_x, sigma_y, theta,
noise_sigma) from a 64x64 LR patch. Used by Signal S3 (degradation
mismatch) in src/signals/degradation_mismatch.py to compare the estimated
degradation against the backbone's assumed training distribution.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from src.degradation.mismatch_degrade import degrade_mismatch
from src.degradation.real_esrgan_degrade import degrade_standard


class KernelEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),  # 64 -> 32
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),  # 32 -> 16
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 16 -> 8
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, 4)

    def forward(self, lr_patch: torch.Tensor) -> torch.Tensor:
        feats = self.net(lr_patch).flatten(1)
        return self.head(feats)


class KernelEstimatorDataset(Dataset):
    """Mixes standard- and mismatch-pool degradations 50/50 so the estimator
    learns to regress kernel params across both distributions."""

    def __init__(self, hr_images: list[np.ndarray], scale: int = 4, seed: int = 0):
        self.hr_images = hr_images
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.hr_images)

    def __getitem__(self, idx):
        hr = self.hr_images[idx]
        if self.rng.random() < 0.5:
            lr, params = degrade_standard(hr, self.scale, self.rng)
        else:
            lr, params = degrade_mismatch(hr, self.scale, self.rng)
        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float()
        target = torch.tensor([params["sigma_x"], params["sigma_y"], params["theta"], params["noise_sigma"]]).float()
        return lr_t, target
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/signals/test_kernel_estimator.py -v`
Expected: 2 passed.

- [ ] **Step 5: Write the training script (run manually against real dataset images, not part of pytest)**

```python
# scripts/train_kernel_estimator.py
"""Trains the blind kernel estimator on real HR images (e.g. DIV2K train
patches). Run manually: conda run -n py313 python scripts/train_kernel_estimator.py
--data-dir data/DIV2K_train_HR --epochs 20 --out checkpoints/kernel_estimator.pt
"""
import argparse
import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.signals.kernel_estimator import KernelEstimator, KernelEstimatorDataset


def load_hr_images(data_dir: str, patch_size: int = 256, max_images: int = 500) -> list[np.ndarray]:
    paths = sorted(glob.glob(f"{data_dir}/*.png"))[:max_images]
    images = []
    for p in paths:
        img = np.asarray(Image.open(p).convert("RGB").resize((patch_size, patch_size))).astype(np.float32) / 255.0
        images.append(img)
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default="checkpoints/kernel_estimator.pt")
    args = parser.parse_args()

    hr_images = load_hr_images(args.data_dir)
    dataset = KernelEstimatorDataset(hr_images)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = KernelEstimator().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        total_loss = 0.0
        for lr_batch, target_batch in loader:
            lr_batch, target_batch = lr_batch.to(device), target_batch.to(device)
            optim.zero_grad()
            pred = model(lr_batch)
            loss = torch.nn.functional.mse_loss(pred, target_batch)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"epoch {epoch}: loss {total_loss / len(loader):.4f}")

    import os

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit and push**

```bash
git add src/signals/kernel_estimator.py scripts/train_kernel_estimator.py tests/signals/test_kernel_estimator.py
git commit -m "feat: add blind kernel estimator for degradation-mismatch signal"
git push origin main
```

---

## Task 9: Signal S3 — Degradation-Model Mismatch

**Files:**
- Create: `src/signals/degradation_mismatch.py`
- Test: `tests/signals/test_degradation_mismatch.py`

**Interfaces:**
- Consumes: `KernelEstimator` (Task 8), `STANDARD_SIGMA_RANGE` from `src.degradation.real_esrgan_degrade`.
- Produces: `compute_degradation_mismatch(estimator: KernelEstimator, lr_patch: torch.Tensor) -> float` — a single scalar per patch (broadcast to a full-resolution map by the fusion head input builder in Task 12, since the estimator operates per-patch, not per-pixel). Consumed by Task 12 and Task 14.

- [ ] **Step 1: Write the failing test**

```python
# tests/signals/test_degradation_mismatch.py
import torch
from src.signals.degradation_mismatch import compute_degradation_mismatch
from src.signals.kernel_estimator import KernelEstimator


class _FixedEstimator:
    def __init__(self, sigma_x, sigma_y, theta, noise_sigma):
        self._out = torch.tensor([[sigma_x, sigma_y, theta, noise_sigma]])

    def __call__(self, lr_patch):
        return self._out

    def eval(self):
        return self


def test_in_distribution_estimate_gives_low_mismatch():
    estimator = _FixedEstimator(sigma_x=0.8, sigma_y=0.8, theta=0.0, noise_sigma=0.01)
    lr = torch.rand(1, 3, 64, 64)
    score = compute_degradation_mismatch(estimator, lr)
    assert score < 0.3


def test_out_of_distribution_estimate_gives_high_mismatch():
    estimator = _FixedEstimator(sigma_x=3.5, sigma_y=3.0, theta=0.9, noise_sigma=0.01)
    lr = torch.rand(1, 3, 64, 64)
    score = compute_degradation_mismatch(estimator, lr)
    assert score > 0.7


def test_real_kernel_estimator_produces_valid_range():
    estimator = KernelEstimator()
    estimator.eval()
    lr = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        score = compute_degradation_mismatch(estimator, lr)
    assert 0.0 <= score <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/signals/test_degradation_mismatch.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.signals.degradation_mismatch'`

- [ ] **Step 3: Implement**

```python
# src/signals/degradation_mismatch.py
"""Signal S3 (spec Section 2): re-estimate the LR patch's degradation
kernel with the blind estimator, then score how far the estimate falls
outside the backbone's assumed (standard-pool) training distribution.
Normalized to [0, 1] via a fixed max-distance scale so it composes cleanly
with the other three signals in the fusion head.
"""
import torch

from src.degradation.real_esrgan_degrade import STANDARD_SIGMA_RANGE

_STANDARD_CENTER = sum(STANDARD_SIGMA_RANGE) / 2  # 0.85
_MAX_EXPECTED_DISTANCE = 4.0  # calibrated against MISMATCH_SIGMA_RANGE upper bound


def compute_degradation_mismatch(estimator, lr_patch: torch.Tensor) -> float:
    out = estimator(lr_patch)
    sigma_x, sigma_y, theta, _noise_sigma = out[0].tolist()
    sigma_distance = max(0.0, max(sigma_x, sigma_y) - STANDARD_SIGMA_RANGE[1])
    theta_distance = abs(theta)  # standard pool always has theta == 0
    raw = sigma_distance + theta_distance
    return float(min(1.0, raw / _MAX_EXPECTED_DISTANCE))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/signals/test_degradation_mismatch.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit and push**

```bash
git add src/signals/degradation_mismatch.py tests/signals/test_degradation_mismatch.py
git commit -m "feat: add S3 degradation-mismatch signal"
git push origin main
```

---

## Task 10: Signal S4 — Distribution Shift via DINOv2 Feature Bank

**Files:**
- Create: `src/signals/distribution_shift.py`
- Create: `scripts/build_feature_bank.py`
- Test: `tests/signals/test_distribution_shift.py`

**Interfaces:**
- Produces: `class FeatureBank` with `__init__(self, reference_features: torch.Tensor)` (`(N, D)`) and `knn_distance(self, query_features: torch.Tensor, k: int = 5) -> torch.Tensor` (`(M,)`); `extract_dinov2_features(encoder, patches: torch.Tensor) -> torch.Tensor` where `encoder` is injectable (real one: `transformers.AutoModel.from_pretrained("facebook/dinov2-small")`). Consumed by Task 12 and Task 14.
- Consumes: `sklearn.neighbors.NearestNeighbors` for the kNN distance (already installed).

- [ ] **Step 1: Write the failing test — feature-bank distance logic with a stub encoder**

```python
# tests/signals/test_distribution_shift.py
import torch
from src.signals.distribution_shift import FeatureBank, extract_dinov2_features


class _StubEncoder:
    """Maps each patch to its own mean-pooled RGB value as a fake 3-D
    'feature' so distance-to-bank behavior is easy to reason about without
    downloading DINOv2."""

    def __call__(self, pixel_values):
        feats = pixel_values.mean(dim=(2, 3))  # (B, 3)
        return type("Out", (), {"pooler_output": feats})()


def test_extract_dinov2_features_shape():
    encoder = _StubEncoder()
    patches = torch.rand(5, 3, 32, 32)
    feats = extract_dinov2_features(encoder, patches)
    assert feats.shape == (5, 3)


def test_in_distribution_query_has_low_knn_distance():
    reference = torch.rand(100, 8)
    bank = FeatureBank(reference)
    in_dist_query = reference[:3] + 0.001 * torch.randn(3, 8)
    dist = bank.knn_distance(in_dist_query, k=5)
    assert dist.shape == (3,)
    assert dist.mean() < 0.1


def test_ood_query_has_high_knn_distance():
    reference = torch.rand(100, 8)  # roughly in [0, 1]
    bank = FeatureBank(reference)
    ood_query = torch.full((3, 8), 10.0)  # far outside [0, 1]
    dist = bank.knn_distance(ood_query, k=5)
    assert dist.mean() > 5.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/signals/test_distribution_shift.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.signals.distribution_shift'`

- [ ] **Step 3: Implement**

```python
# src/signals/distribution_shift.py
"""Signal S4 (spec Section 2): distance-to-training-distribution in a
frozen DINOv2 feature space. `encoder` is injected so tests avoid
downloading the real facebook/dinov2-small checkpoint — production callers
pass transformers.AutoModel.from_pretrained("facebook/dinov2-small").
"""
import torch
from sklearn.neighbors import NearestNeighbors


def extract_dinov2_features(encoder, patches: torch.Tensor) -> torch.Tensor:
    """patches: (B, 3, H, W). Returns (B, D). interpolate_pos_encoding=True is
    required for the real model because CHASR patches (e.g. 1024x1024 SR
    output) are not the fixed size DINOv2 was pretrained at; omitting it
    raises a position-embedding size mismatch at inference time."""
    if _accepts_kwarg(encoder):
        out = encoder(pixel_values=patches, interpolate_pos_encoding=True)
    else:
        out = encoder(patches)
    return out.pooler_output


def _accepts_kwarg(encoder) -> bool:
    # transformers models take pixel_values=...; the test stub takes a positional arg.
    return hasattr(encoder, "config")


class FeatureBank:
    def __init__(self, reference_features: torch.Tensor):
        self.reference = reference_features.detach().cpu().numpy()
        self._nn = NearestNeighbors(n_neighbors=min(5, len(self.reference)))
        self._nn.fit(self.reference)

    def knn_distance(self, query_features: torch.Tensor, k: int = 5) -> torch.Tensor:
        query_np = query_features.detach().cpu().numpy()
        k = min(k, len(self.reference))
        distances, _ = self._nn.kneighbors(query_np, n_neighbors=k)
        return torch.from_numpy(distances.mean(axis=1)).float()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/signals/test_distribution_shift.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the feature-bank build script (run manually against real training data)**

```python
# scripts/build_feature_bank.py
"""Builds the DINOv2 reference feature bank from real HR training patches
(e.g. DIV2K/Flickr2K). Run manually:
conda run -n py313 python scripts/build_feature_bank.py --data-dir data/DIV2K_train_HR --out checkpoints/feature_bank.pt
"""
import argparse
import glob

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out", default="checkpoints/feature_bank.pt")
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--max-images", type=int, default=800)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()

    paths = sorted(glob.glob(f"{args.data_dir}/*.png"))[: args.max_images]
    features = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("RGB").resize((args.patch_size, args.patch_size))
            tensor = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(device)
            out = encoder(pixel_values=tensor, interpolate_pos_encoding=True)
            features.append(out.pooler_output.cpu())

    bank = torch.cat(features, dim=0)
    import os

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(bank, args.out)
    print(f"Saved feature bank of shape {bank.shape} to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit and push**

```bash
git add src/signals/distribution_shift.py scripts/build_feature_bank.py tests/signals/test_distribution_shift.py
git commit -m "feat: add S4 distribution-shift signal via DINOv2 feature bank"
git push origin main
```

---

## Task 11: xSR-CausalBench — Synthetic Controllable-Hallucination Benchmark

**Files:**
- Create: `src/causalbench/build_benchmark.py`
- Create: `src/causalbench/dataset.py`
- Create: `scripts/download_datasets.py`
- Test: `tests/causalbench/test_build_benchmark.py`
- Test: `tests/causalbench/test_dataset.py`

**Interfaces:**
- Consumes: `degrade_standard` (Task 2), `degrade_mismatch` (Task 3).
- Produces: `CAUSE_LABELS = {"ILL_POSED": 0, "PRIOR_RELIANCE": 1, "DEGRADATION_MISMATCH": 2, "DISTRIBUTION_SHIFT": 3, "RELIABLE": 4}`; `build_ill_posed_sample`, `build_prior_reliance_sample`, `build_mismatch_sample`, `build_distribution_shift_sample` — each `(hr: np.ndarray, ood_patch: np.ndarray | None, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]` returning `(hr, lr, label_map)` where `label_map` is `(H, W)` int array of `CAUSE_LABELS` values; `class CausalBenchDataset(torch.utils.data.Dataset)` yielding `(lr, hr, label_map)`. Consumed by Task 12 (fusion head training).

- [ ] **Step 1: Write the failing test for the four injection procedures**

```python
# tests/causalbench/test_build_benchmark.py
import numpy as np
from src.causalbench.build_benchmark import (
    CAUSE_LABELS,
    build_distribution_shift_sample,
    build_ill_posed_sample,
    build_mismatch_sample,
    build_prior_reliance_sample,
)


def _random_hr(size=256):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(size, size, 3)).astype(np.float32)


def test_ill_posed_sample_label_map():
    rng = np.random.default_rng(1)
    hr, lr, label_map = build_ill_posed_sample(_random_hr(), rng)
    assert label_map.shape == hr.shape[:2]
    assert (label_map == CAUSE_LABELS["ILL_POSED"]).any()


def test_prior_reliance_sample_label_map():
    rng = np.random.default_rng(2)
    hr, lr, label_map = build_prior_reliance_sample(_random_hr(), rng)
    assert (label_map == CAUSE_LABELS["PRIOR_RELIANCE"]).any()
    assert (label_map == CAUSE_LABELS["RELIABLE"]).any()  # untouched region stays reliable


def test_mismatch_sample_label_map():
    rng = np.random.default_rng(3)
    hr, lr, label_map = build_mismatch_sample(_random_hr(), rng)
    assert (label_map == CAUSE_LABELS["DEGRADATION_MISMATCH"]).any()


def test_distribution_shift_sample_label_map():
    rng = np.random.default_rng(4)
    ood_patch = rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32)
    hr, lr, label_map = build_distribution_shift_sample(_random_hr(), ood_patch, rng)
    assert (label_map == CAUSE_LABELS["DISTRIBUTION_SHIFT"]).any()
    assert (label_map == CAUSE_LABELS["RELIABLE"]).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/causalbench/test_build_benchmark.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.causalbench.build_benchmark'`

- [ ] **Step 3: Implement the four controlled injection procedures**

```python
# src/causalbench/build_benchmark.py
"""xSR-CausalBench construction (spec Section 3): four controlled
injection procedures, each producing a per-pixel weak "dominant cause"
label mask alongside the (HR, LR) pair. Extends HalluGen's diffusion-
posterior-sampling controllable-hallucination idea (arXiv:2512.03345) from
a 2-way to a 4-way taxonomy.
"""
import numpy as np

from src.degradation.mismatch_degrade import degrade_mismatch
from src.degradation.real_esrgan_degrade import degrade_standard

CAUSE_LABELS = {
    "ILL_POSED": 0,
    "PRIOR_RELIANCE": 1,
    "DEGRADATION_MISMATCH": 2,
    "DISTRIBUTION_SHIFT": 3,
    "RELIABLE": 4,
}


def build_ill_posed_sample(hr: np.ndarray, rng: np.random.Generator, scale: int = 16):
    """Whole-image extreme downsample under a correctly-specified (standard)
    degradation model: null-space is large everywhere by construction."""
    lr, _ = degrade_standard(hr, scale, rng)
    label_map = np.full(hr.shape[:2], CAUSE_LABELS["ILL_POSED"], dtype=np.int64)
    return hr, lr, label_map


def build_prior_reliance_sample(hr: np.ndarray, rng: np.random.Generator, scale: int = 16, patch_frac: float = 0.3):
    """Standard degradation, but a random rectangular sub-region has its LR
    evidence additionally zeroed out beyond the nominal degradation model —
    forcing the backbone to fall back on the generative prior there."""
    h, w = hr.shape[:2]
    ph, pw = int(h * patch_frac), int(w * patch_frac)
    y0, x0 = rng.integers(0, h - ph), rng.integers(0, w - pw)

    hr_evidence_suppressed = hr.copy()
    hr_evidence_suppressed[y0 : y0 + ph, x0 : x0 + pw] = 0.0
    lr, _ = degrade_standard(hr_evidence_suppressed, scale, rng)

    label_map = np.full(hr.shape[:2], CAUSE_LABELS["RELIABLE"], dtype=np.int64)
    label_map[y0 // scale * scale : (y0 + ph) // scale * scale, x0 // scale * scale : (x0 + pw) // scale * scale] = CAUSE_LABELS[
        "PRIOR_RELIANCE"
    ]
    return hr, lr, label_map


def build_mismatch_sample(hr: np.ndarray, rng: np.random.Generator, scale: int = 16):
    """Whole image degraded with an out-of-training-distribution kernel
    while content stays in-distribution."""
    lr, _ = degrade_mismatch(hr, scale, rng)
    label_map = np.full(hr.shape[:2], CAUSE_LABELS["DEGRADATION_MISMATCH"], dtype=np.int64)
    return hr, lr, label_map


def build_distribution_shift_sample(hr: np.ndarray, ood_patch: np.ndarray, rng: np.random.Generator, scale: int = 16):
    """Standard degradation, but a rare/OOD texture patch is blended into
    an otherwise normal image before degrading."""
    h, w = hr.shape[:2]
    ph, pw = ood_patch.shape[:2]
    y0, x0 = rng.integers(0, h - ph), rng.integers(0, w - pw)

    hr_blended = hr.copy()
    hr_blended[y0 : y0 + ph, x0 : x0 + pw] = ood_patch
    lr, _ = degrade_standard(hr_blended, scale, rng)

    label_map = np.full(hr.shape[:2], CAUSE_LABELS["RELIABLE"], dtype=np.int64)
    label_map[y0:y0 + ph, x0:x0 + pw] = CAUSE_LABELS["DISTRIBUTION_SHIFT"]
    return hr_blended, lr, label_map
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/causalbench/test_build_benchmark.py -v`
Expected: 4 passed.

- [ ] **Step 5: Write the failing test for the PyTorch Dataset wrapper**

```python
# tests/causalbench/test_dataset.py
import numpy as np
import torch
from src.causalbench.dataset import CausalBenchDataset


def test_dataset_getitem_shapes_and_dtypes(tmp_path):
    rng = np.random.default_rng(0)
    hr_images = [rng.uniform(0, 1, size=(256, 256, 3)).astype(np.float32) for _ in range(4)]
    ood_patches = [rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32) for _ in range(2)]

    dataset = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=0)
    assert len(dataset) == len(hr_images) * 4  # 4 procedures per source image

    lr, hr, label_map = dataset[0]
    assert isinstance(lr, torch.Tensor) and lr.shape[0] == 3
    assert isinstance(hr, torch.Tensor) and hr.shape[0] == 3
    assert isinstance(label_map, torch.Tensor) and label_map.dtype == torch.int64
    assert label_map.shape == hr.shape[1:]


def test_dataset_covers_all_four_procedures():
    rng = np.random.default_rng(1)
    hr_images = [rng.uniform(0, 1, size=(256, 256, 3)).astype(np.float32) for _ in range(1)]
    ood_patches = [rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32)]

    dataset = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=0)
    seen_labels = set()
    for i in range(len(dataset)):
        _, _, label_map = dataset[i]
        seen_labels.update(torch.unique(label_map).tolist())
    assert seen_labels == {0, 1, 2, 3, 4}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/causalbench/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.causalbench.dataset'`

- [ ] **Step 7: Implement the Dataset wrapper**

```python
# src/causalbench/dataset.py
"""PyTorch Dataset over xSR-CausalBench: for each source HR image, all four
controlled-injection procedures are generated deterministically per index
so the benchmark is reproducible given a fixed seed.
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from src.causalbench.build_benchmark import (
    build_distribution_shift_sample,
    build_ill_posed_sample,
    build_mismatch_sample,
    build_prior_reliance_sample,
)

_PROCEDURES = ["ill_posed", "prior_reliance", "mismatch", "distribution_shift"]


class CausalBenchDataset(Dataset):
    def __init__(self, hr_images: list[np.ndarray], ood_patches: list[np.ndarray], scale: int = 16, seed: int = 0):
        self.hr_images = hr_images
        self.ood_patches = ood_patches
        self.scale = scale
        self.seed = seed

    def __len__(self):
        return len(self.hr_images) * len(_PROCEDURES)

    def __getitem__(self, idx):
        image_idx, procedure_idx = divmod(idx, len(_PROCEDURES))
        hr_source = self.hr_images[image_idx]
        procedure = _PROCEDURES[procedure_idx]
        rng = np.random.default_rng(self.seed + idx)

        if procedure == "ill_posed":
            hr, lr, label_map = build_ill_posed_sample(hr_source, rng, self.scale)
        elif procedure == "prior_reliance":
            hr, lr, label_map = build_prior_reliance_sample(hr_source, rng, self.scale)
        elif procedure == "mismatch":
            hr, lr, label_map = build_mismatch_sample(hr_source, rng, self.scale)
        else:
            ood_patch = self.ood_patches[idx % len(self.ood_patches)]
            hr, lr, label_map = build_distribution_shift_sample(hr_source, ood_patch, rng, self.scale)

        lr_t = torch.from_numpy(lr).permute(2, 0, 1).float()
        hr_t = torch.from_numpy(hr).permute(2, 0, 1).float()
        label_t = torch.from_numpy(label_map).long()
        return lr_t, hr_t, label_t
```

- [ ] **Step 8: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/causalbench/test_dataset.py -v`
Expected: 2 passed.

- [ ] **Step 9: Write the dataset acquisition helper script**

```python
# scripts/download_datasets.py
"""Downloads/documents the source datasets used to build xSR-CausalBench
and for backbone/estimator training and evaluation. DIV2K has a stable
public direct-download URL; the other sets require manual steps due to
registration/access-agreement requirements — verify URLs are still live
before relying on them, hosting can change.

Run: conda run -n py313 python scripts/download_datasets.py --out-dir data/
"""
import argparse
import os
import zipfile

import requests

DIV2K_URLS = {
    "DIV2K_train_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

MANUAL_DATASETS = {
    "Urban100": "https://github.com/jbhuang0604/SelfExSR (request access per repo instructions)",
    "Manga109": "http://www.manga109.org/en/ (requires registration + usage agreement)",
    "RealSR": "https://github.com/csjcai/RealSR (request access per repo instructions)",
    "DRealSR": "https://github.com/xiezw5/Component-Divide-and-Conquer-for-Real-World-Image-Super-Resolution (request access per repo instructions)",
}


def download_and_extract(name: str, url: str, out_dir: str):
    zip_path = os.path.join(out_dir, f"{name}.zip")
    print(f"Downloading {name} from {url} ...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    os.remove(zip_path)
    print(f"Extracted {name} to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for name, url in DIV2K_URLS.items():
        try:
            download_and_extract(name, url, args.out_dir)
        except Exception as e:
            print(f"Failed to auto-download {name}: {e}. Download manually from {url}")

    print("\nThe following datasets require manual download (registration/access agreement):")
    for name, info in MANUAL_DATASETS.items():
        print(f"  {name}: {info}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Commit and push**

```bash
git add src/causalbench/ scripts/download_datasets.py tests/causalbench/
git commit -m "feat: add xSR-CausalBench 4-way controllable-hallucination benchmark"
git push origin main
```

---

## Task 12: Fusion Head — Model and Training

**Files:**
- Create: `src/fusion/model.py`
- Create: `src/fusion/train.py`
- Test: `tests/fusion/test_model.py`

**Interfaces:**
- Consumes: `CAUSE_LABELS` (Task 11).
- Produces: `class FusionHead(torch.nn.Module)` with `forward(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` where `signals` is `(B, 5, H, W)` (S1, S2, S3-broadcast, S4-broadcast, plus one shallow image-luminance feature channel) and the return is `(cause_logits (B, 5, H, W), reliability (B, 1, H, W))`. Consumed by Task 14 (end-to-end inference).

- [ ] **Step 1: Write the failing test**

```python
# tests/fusion/test_model.py
import torch
from src.fusion.model import FusionHead


def test_forward_output_shapes():
    model = FusionHead()
    signals = torch.rand(2, 5, 32, 32)
    cause_logits, reliability = model(signals)
    assert cause_logits.shape == (2, 5, 32, 32)
    assert reliability.shape == (2, 1, 32, 32)


def test_reliability_output_in_valid_range():
    model = FusionHead()
    signals = torch.rand(2, 5, 32, 32)
    _, reliability = model(signals)
    assert (reliability >= 0).all() and (reliability <= 1).all()


def test_model_can_overfit_a_tiny_synthetic_batch():
    torch.manual_seed(0)
    model = FusionHead()
    signals = torch.rand(4, 5, 16, 16)
    labels = torch.randint(0, 5, (4, 16, 16))
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(50):
        optim.zero_grad()
        cause_logits, _ = model(signals)
        loss = torch.nn.functional.cross_entropy(cause_logits, labels)
        loss.backward()
        optim.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0] * 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/fusion/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.fusion.model'`

- [ ] **Step 3: Implement**

```python
# src/fusion/model.py
"""Fusion head (spec Section 2): the only trained component in the CHASR
pipeline. Combines the four causal signals (plus a shallow image feature)
into a calibrated per-pixel 5-way cause classification (4 causes +
"reliable") and an overall reliability score.
"""
import torch
import torch.nn as nn


class FusionHead(nn.Module):
    def __init__(self, in_channels: int = 5, num_classes: int = 5):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.cause_head = nn.Conv2d(32, num_classes, 1)
        self.reliability_head = nn.Conv2d(32, 1, 1)

    def forward(self, signals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(signals)
        cause_logits = self.cause_head(feats)
        reliability = torch.sigmoid(self.reliability_head(feats))
        return cause_logits, reliability
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/fusion/test_model.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the training script (run manually against the full CausalBench)**

```python
# src/fusion/train.py
"""Trains the fusion head on xSR-CausalBench. Each dataset item's signal
stack requires ~5 diffusion sampling calls per k (1 hop-1 call + 4 tiled
hop-2 calls) through the backbone (Task 5) — expensive enough that
recomputing it on every training epoch would make Phase 1 training take
days on an 8GB laptop GPU. `precompute_signal_stacks` therefore builds
every (signal_stack, label_map) pair ONCE, optionally caching to disk, and
`train` iterates epochs over that cheap, cached tensor set — only the
small fusion head does repeated forward/backward passes.
"""
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.backbone.diffusion_backbone import estimate_prior_reliance_map
from src.causalbench.dataset import CausalBenchDataset
from src.fusion.model import FusionHead
from src.signals.degradation_mismatch import compute_degradation_mismatch
from src.signals.distribution_shift import FeatureBank, extract_dinov2_features
from src.signals.null_space import compute_null_space_variance


def build_signal_stack(lr, backbone, kernel_estimator, feature_bank, dino_encoder, k=2):
    samples = backbone.sample_k_16x(lr, k=k, base_seed=0)
    s1 = compute_null_space_variance(samples)  # (H, W)

    s2 = estimate_prior_reliance_map(backbone, lr, base_seed=0)  # (H, W) — see Task 5 for why this, not compute_prior_reliance

    s3_scalar = compute_degradation_mismatch(kernel_estimator, lr.unsqueeze(0))
    s3 = torch.full_like(s1, s3_scalar)

    query_feats = extract_dinov2_features(dino_encoder, samples[0].unsqueeze(0))
    s4_scalar = feature_bank.knn_distance(query_feats).item()
    s4 = torch.full_like(s1, s4_scalar)

    luminance = samples[0].mean(dim=0)  # (H, W), shallow image feature
    return torch.stack([s1, s2, s3, s4, luminance], dim=0)  # (5, H, W)


def precompute_signal_stacks(dataset: CausalBenchDataset, backbone, kernel_estimator, feature_bank, dino_encoder, k: int = 2, cache_path: str | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Returns a list of (signal_stack (5,H,W), label_map (H,W)) pairs,
    computed once. If cache_path is given and already exists, loads from
    it instead of recomputing (e.g. across separate training script runs)."""
    if cache_path and os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    cached_items = []
    for idx in range(len(dataset)):
        lr, _hr, label_map = dataset[idx]
        signal_stack = build_signal_stack(lr, backbone, kernel_estimator, feature_bank, dino_encoder, k=k)
        cached_items.append((signal_stack, label_map))

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(cached_items, cache_path)
    return cached_items


class _CachedSignalDataset(Dataset):
    def __init__(self, cached_items: list[tuple[torch.Tensor, torch.Tensor]]):
        self.cached_items = cached_items

    def __len__(self):
        return len(self.cached_items)

    def __getitem__(self, idx):
        return self.cached_items[idx]


def train(cached_items: list[tuple[torch.Tensor, torch.Tensor]], epochs: int = 30, batch_size: int = 4, device: str = "cuda") -> FusionHead:
    loader = DataLoader(_CachedSignalDataset(cached_items), batch_size=batch_size, shuffle=True)
    model = FusionHead().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        total_loss = 0.0
        for signal_stack, label_map in loader:
            signal_stack, label_map = signal_stack.to(device), label_map.to(device)

            optim.zero_grad()
            cause_logits, _ = model(signal_stack)
            loss = nn.functional.cross_entropy(cause_logits, label_map)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f"epoch {epoch}: loss {total_loss / len(loader):.4f}")

    return model
```

- [ ] **Step 6: Commit and push**

```bash
git add src/fusion/model.py src/fusion/train.py tests/fusion/test_model.py
git commit -m "feat: add fusion head model and training loop"
git push origin main
```

---

## Task 13: Evaluation Metrics — Fidelity, Attribution Accuracy, Disentanglement

**Files:**
- Create: `src/eval/fidelity_metrics.py`
- Create: `src/eval/attribution_metrics.py`
- Create: `src/eval/disentanglement.py`
- Test: `tests/eval/test_fidelity_metrics.py`
- Test: `tests/eval/test_attribution_metrics.py`
- Test: `tests/eval/test_disentanglement.py`

**Interfaces:**
- Produces: `compute_psnr(pred, target) -> float`, `compute_ssim(pred, target) -> float`, `compute_lpips(pred, target, lpips_model) -> float` (fidelity_metrics.py); `pixel_accuracy(pred_labels, gt_labels) -> float`, `mean_iou(pred_labels, gt_labels, num_classes=5) -> float` (attribution_metrics.py); `signal_correlation_matrix(signals: torch.Tensor) -> torch.Tensor` where `signals` is `(4, N)` flattened per-signal samples, returns `(4, 4)` (disentanglement.py). Consumed by Task 14 (reporting) and the evaluation run script.

- [ ] **Step 1: Write the failing tests for fidelity metrics**

```python
# tests/eval/test_fidelity_metrics.py
import numpy as np
import torch
from src.eval.fidelity_metrics import compute_psnr, compute_ssim


def test_psnr_identical_images_is_very_high():
    img = np.random.default_rng(0).uniform(0, 1, size=(64, 64, 3)).astype(np.float32)
    psnr = compute_psnr(img, img)
    assert psnr > 80  # effectively infinite for identical images, clipped for stability


def test_psnr_decreases_with_more_noise():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32)
    small_noise = img + rng.normal(0, 0.01, img.shape).astype(np.float32)
    large_noise = img + rng.normal(0, 0.2, img.shape).astype(np.float32)
    assert compute_psnr(img, small_noise) > compute_psnr(img, large_noise)


def test_ssim_identical_images_is_one():
    img = np.random.default_rng(0).uniform(0, 1, size=(64, 64, 3)).astype(np.float32)
    assert compute_ssim(img, img) > 0.999
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/eval/test_fidelity_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.fidelity_metrics'`

- [ ] **Step 3: Implement fidelity metrics**

```python
# src/eval/fidelity_metrics.py
"""Fidelity metrics (spec Section 5, item 1) — context metrics, not the
headline result, since the backbone is frozen and unchanged by this work.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    value = peak_signal_noise_ratio(target, pred, data_range=1.0)
    return float(min(value, 100.0))  # clip infinite/very large values for identical images


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, pred, data_range=1.0, channel_axis=-1))


def compute_lpips(pred, target, lpips_model) -> float:
    """pred, target: torch tensors (3, H, W) in [0, 1]. lpips_model: lpips.LPIPS instance."""
    import torch

    pred_n = pred.unsqueeze(0) * 2 - 1
    target_n = target.unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(lpips_model(pred_n, target_n).item())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/eval/test_fidelity_metrics.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write the failing tests for attribution accuracy**

```python
# tests/eval/test_attribution_metrics.py
import torch
from src.eval.attribution_metrics import mean_iou, pixel_accuracy


def test_pixel_accuracy_perfect_match():
    pred = torch.tensor([[0, 1], [2, 3]])
    gt = torch.tensor([[0, 1], [2, 3]])
    assert pixel_accuracy(pred, gt) == 1.0


def test_pixel_accuracy_no_match():
    pred = torch.tensor([[0, 0], [0, 0]])
    gt = torch.tensor([[1, 1], [1, 1]])
    assert pixel_accuracy(pred, gt) == 0.0


def test_mean_iou_perfect_match_is_one():
    pred = torch.randint(0, 5, (16, 16))
    gt = pred.clone()
    assert mean_iou(pred, gt, num_classes=5) == 1.0


def test_mean_iou_no_overlap_is_zero():
    pred = torch.zeros(4, 4, dtype=torch.long)
    gt = torch.ones(4, 4, dtype=torch.long)
    assert mean_iou(pred, gt, num_classes=5) == 0.0
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/eval/test_attribution_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.attribution_metrics'`

- [ ] **Step 7: Implement attribution metrics**

```python
# src/eval/attribution_metrics.py
"""Attribution accuracy metrics (spec Section 5, item 3 — the headline
result): does the fusion head's predicted dominant cause match the
synthetic ground truth on xSR-CausalBench?
"""
import torch


def pixel_accuracy(pred_labels: torch.Tensor, gt_labels: torch.Tensor) -> float:
    return float((pred_labels == gt_labels).float().mean().item())


def mean_iou(pred_labels: torch.Tensor, gt_labels: torch.Tensor, num_classes: int = 5) -> float:
    ious = []
    for cls in range(num_classes):
        pred_mask = pred_labels == cls
        gt_mask = gt_labels == cls
        union = (pred_mask | gt_mask).sum().item()
        if union == 0:
            continue  # class absent from both — skip rather than penalize
        intersection = (pred_mask & gt_mask).sum().item()
        ious.append(intersection / union)
    return float(sum(ious) / len(ious)) if ious else 0.0
```

- [ ] **Step 8: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/eval/test_attribution_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 9: Write the failing test for disentanglement analysis**

```python
# tests/eval/test_disentanglement.py
import torch
from src.eval.disentanglement import signal_correlation_matrix


def test_correlation_matrix_shape_and_diagonal_is_one():
    torch.manual_seed(0)
    signals = torch.rand(4, 1000)
    corr = signal_correlation_matrix(signals)
    assert corr.shape == (4, 4)
    assert torch.allclose(torch.diag(corr), torch.ones(4), atol=1e-4)


def test_identical_signals_have_correlation_one():
    base = torch.rand(1000)
    signals = torch.stack([base, base, torch.rand(1000), torch.rand(1000)])
    corr = signal_correlation_matrix(signals)
    assert torch.isclose(corr[0, 1], torch.tensor(1.0), atol=1e-4)


def test_matrix_is_symmetric():
    torch.manual_seed(1)
    signals = torch.rand(4, 500)
    corr = signal_correlation_matrix(signals)
    assert torch.allclose(corr, corr.T, atol=1e-5)
```

- [ ] **Step 10: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/eval/test_disentanglement.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.disentanglement'`

- [ ] **Step 11: Implement disentanglement analysis**

```python
# src/eval/disentanglement.py
"""Signal disentanglement analysis (spec Section 3 risk / Section 5, item
5): correlation matrix among S1-S4 must be reported honestly rather than
assuming clean separation — see spec Section 7 open risks.
"""
import torch


def signal_correlation_matrix(signals: torch.Tensor) -> torch.Tensor:
    """signals: (4, N) flattened per-signal samples. Returns (4, 4) Pearson
    correlation matrix."""
    centered = signals - signals.mean(dim=1, keepdim=True)
    cov = centered @ centered.T / (signals.shape[1] - 1)
    std = torch.sqrt(torch.diag(cov))
    denom = std.unsqueeze(0) * std.unsqueeze(1)
    return cov / denom
```

- [ ] **Step 12: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/eval/test_disentanglement.py -v`
Expected: 3 passed.

- [ ] **Step 13: Commit and push**

```bash
git add src/eval/fidelity_metrics.py src/eval/attribution_metrics.py src/eval/disentanglement.py tests/eval/
git commit -m "feat: add fidelity, attribution-accuracy, and disentanglement evaluation metrics"
git push origin main
```

---

## Task 14: End-to-End Inference, Minimal Downstream OCR Case Study, Orchestration Scripts

**Files:**
- Create: `src/fusion/infer.py`
- Create: `src/eval/downstream_ocr_case_study.py`
- Create: `scripts/run_evaluation.py`
- Create: `scripts/train_fusion_head.py`
- Create: `README.md` (extend existing one-line README with a usage section)
- Test: `tests/fusion/test_infer.py`
- Test: `tests/eval/test_downstream_ocr_case_study.py`

**Interfaces:**
- Consumes: everything from Tasks 4-13.
- Produces: `run_chasr(lr_patch: torch.Tensor, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head) -> dict` returning `{"sr_image": Tensor(3,1024,1024), "cause_map": Tensor(1024,1024) long, "reliability_map": Tensor(1024,1024), "signal_stack": Tensor(4,1024,1024)}` (the last is S1-S4 only, for the disentanglement metric — Task 13). This is the top-level entry point the manuscript's experiments section reports numbers from.

- [ ] **Step 1: Write the failing test for end-to-end inference plumbing with stubbed components**

```python
# tests/fusion/test_infer.py
import torch
from src.fusion.infer import run_chasr
from src.fusion.model import FusionHead


class _StubBackbone:
    def sample_k_16x(self, lr_patch, k, base_seed):
        return [torch.rand(3, 1024, 1024) for _ in range(k)]

    def _upscale_4x(self, patch, seed):
        # estimate_prior_reliance_map (Task 5) calls this directly; stub it
        # so run_chasr's plumbing is testable without a real backbone.
        h, w = patch.shape[-2], patch.shape[-1]
        return torch.rand(3, h * 4, w * 4)


class _StubKernelEstimator:
    def __call__(self, lr_patch):
        return torch.tensor([[0.8, 0.8, 0.0, 0.01]])


class _StubFeatureBank:
    def knn_distance(self, query_features, k=5):
        return torch.tensor([0.5])


class _StubDinoEncoder:
    def __call__(self, pixel_values):
        return type("Out", (), {"pooler_output": pixel_values.mean(dim=(2, 3))})()


def test_run_chasr_returns_expected_keys_and_shapes():
    lr_patch = torch.rand(3, 64, 64)
    backbone = _StubBackbone()
    kernel_estimator = _StubKernelEstimator()
    feature_bank = _StubFeatureBank()
    dino_encoder = _StubDinoEncoder()
    fusion_head = FusionHead()

    result = run_chasr(lr_patch, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head, k=2)

    assert result["sr_image"].shape == (3, 1024, 1024)
    assert result["cause_map"].shape == (1024, 1024)
    assert result["reliability_map"].shape == (1024, 1024)
    assert result["cause_map"].dtype == torch.int64
    assert result["signal_stack"].shape == (4, 1024, 1024)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/fusion/test_infer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.fusion.infer'`

- [ ] **Step 3: Implement end-to-end inference**

```python
# src/fusion/infer.py
"""Top-level CHASR entry point: LR patch in, SR image + per-pixel cause
attribution + reliability map out. Wires together Tasks 4-13. This is what
the manuscript's Experiments section (spec Section 5) calls to produce
every reported number.
"""
import torch

from src.backbone.diffusion_backbone import estimate_prior_reliance_map
from src.signals.degradation_mismatch import compute_degradation_mismatch
from src.signals.distribution_shift import extract_dinov2_features
from src.signals.null_space import compute_null_space_variance


def run_chasr(lr_patch, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head, k: int = 4) -> dict:
    samples = backbone.sample_k_16x(lr_patch, k=k, base_seed=0)
    sr_image = samples[0]

    s1 = compute_null_space_variance(samples)

    s2 = estimate_prior_reliance_map(backbone, lr_patch, base_seed=0)  # see Task 5 for why this, not compute_prior_reliance

    s3_scalar = compute_degradation_mismatch(kernel_estimator, lr_patch.unsqueeze(0))
    s3 = torch.full_like(s1, s3_scalar)

    query_feats = extract_dinov2_features(dino_encoder, sr_image.unsqueeze(0))
    s4_scalar = feature_bank.knn_distance(query_feats).item()
    s4 = torch.full_like(s1, s4_scalar)

    luminance = sr_image.mean(dim=0)
    signal_stack = torch.stack([s1, s2, s3, s4, luminance], dim=0).unsqueeze(0)

    with torch.no_grad():
        cause_logits, reliability = fusion_head(signal_stack)
    cause_map = cause_logits.argmax(dim=1).squeeze(0)
    reliability_map = reliability.squeeze(0).squeeze(0)

    return {
        "sr_image": sr_image,
        "cause_map": cause_map,
        "reliability_map": reliability_map,
        "signal_stack": signal_stack.squeeze(0)[:4],  # (4, H, W): S1-S4 only, luminance (index 4) dropped — for disentanglement analysis (Task 13)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/fusion/test_infer.py -v`
Expected: 1 passed.

- [ ] **Step 5: Write the failing test for the downstream OCR case study helper**

```python
# tests/eval/test_downstream_ocr_case_study.py
import numpy as np
from src.eval.downstream_ocr_case_study import correlate_ocr_failure_with_cause


def test_correlate_ocr_failure_with_cause_returns_per_cause_rates():
    cause_map = np.zeros((32, 32), dtype=np.int64)
    cause_map[:16, :] = 2  # DEGRADATION_MISMATCH region
    ocr_failure_mask = np.zeros((32, 32), dtype=bool)
    ocr_failure_mask[:16, :] = True  # OCR fails exactly where mismatch is flagged

    rates = correlate_ocr_failure_with_cause(cause_map, ocr_failure_mask, num_classes=5)
    assert rates[2] == 1.0  # 100% failure rate in the mismatch region
    assert rates[4] == 0.0  # 0% failure rate in the (implicitly reliable) rest
```

- [ ] **Step 6: Run test to verify it fails**

Run: `conda run -n py313 pytest tests/eval/test_downstream_ocr_case_study.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.eval.downstream_ocr_case_study'`

- [ ] **Step 7: Implement the downstream case study helper**

```python
# src/eval/downstream_ocr_case_study.py
"""Minimal downstream utility case study (spec Section 5, item 4): does
flagging a region as degradation-mismatch or distribution-shift correlate
with actual OCR failure? Uses easyocr (already installed) against a small
set of text-patch crops; this module holds the correlation logic, kept
separate from the OCR call itself so it's unit-testable without running
the OCR engine.
"""
import numpy as np


def correlate_ocr_failure_with_cause(cause_map: np.ndarray, ocr_failure_mask: np.ndarray, num_classes: int = 5) -> dict:
    """Returns {cause_id: failure_rate} — fraction of pixels flagged as that
    cause where OCR also failed. A cause with no pixels present is omitted."""
    rates = {}
    for cls in range(num_classes):
        cls_mask = cause_map == cls
        if cls_mask.sum() == 0:
            continue
        rates[cls] = float(ocr_failure_mask[cls_mask].mean())
    return rates


def run_easyocr_readability_check(reader, image: np.ndarray, ground_truth_text: str) -> bool:
    """reader: easyocr.Reader instance. Returns True if OCR read matches
    ground truth (case-insensitive, whitespace-stripped)."""
    results = reader.readtext(image, detail=0)
    read_text = " ".join(results).strip().lower()
    return read_text == ground_truth_text.strip().lower()
```

- [ ] **Step 8: Run test to verify it passes**

Run: `conda run -n py313 pytest tests/eval/test_downstream_ocr_case_study.py -v`
Expected: 1 passed.

- [ ] **Step 9: Write the orchestration scripts**

```python
# scripts/train_fusion_head.py
"""Full Phase 1 fusion-head training entry point. Requires:
- checkpoints/kernel_estimator.pt (Task 8's scripts/train_kernel_estimator.py)
- checkpoints/feature_bank.pt (Task 10's scripts/build_feature_bank.py)
- data/DIV2K_train_HR/, data/DIV2K_valid_HR/ (scripts/download_datasets.py)

HR images are resized to exactly 1024x1024 — this must match
DiffusionSRBackbone.sample_k_16x's fixed 16x factor (Task 5): at
scale=16, CausalBenchDataset then produces exactly 64x64 LR patches,
which is what the backbone expects as input. Do not change one without
the other. TRAIN_MAX_IMAGES/TRAIN_K are kept small deliberately — each
dataset item costs ~5 diffusion calls per k during the one-time
precompute pass below; 40 images x 4 procedures x k=2 x 5 calls = 1600
diffusion calls, roughly 2-3 hours one-time on an 8GB laptop GPU. Raise
these only once Phase 1 numbers exist and you're deliberately scaling up.

Run: conda run -n py313 python scripts/train_fusion_head.py
"""
import glob
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from src.backbone.diffusion_backbone import DiffusionSRBackbone
from src.causalbench.dataset import CausalBenchDataset
from src.fusion.train import precompute_signal_stacks, train
from src.signals.distribution_shift import FeatureBank
from src.signals.kernel_estimator import KernelEstimator

HR_PATCH_SIZE = 1024  # must match DiffusionSRBackbone's fixed 16x output size
TRAIN_MAX_IMAGES = 40
TRAIN_K = 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hr_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[:TRAIN_MAX_IMAGES]
    hr_images = [np.asarray(Image.open(p).convert("RGB").resize((HR_PATCH_SIZE, HR_PATCH_SIZE))).astype(np.float32) / 255.0 for p in hr_paths]
    ood_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[TRAIN_MAX_IMAGES : TRAIN_MAX_IMAGES + 10]
    ood_patches = [np.asarray(Image.open(p).convert("RGB").resize((64, 64))).astype(np.float32) / 255.0 for p in ood_paths]

    dataset = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=0)

    kernel_estimator = KernelEstimator().to(device)
    kernel_estimator.load_state_dict(torch.load("checkpoints/kernel_estimator.pt", weights_only=True))
    kernel_estimator.eval()

    feature_bank = FeatureBank(torch.load("checkpoints/feature_bank.pt", weights_only=True))
    dino_encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    backbone = DiffusionSRBackbone(device=device)

    cached_items = precompute_signal_stacks(
        dataset, backbone, kernel_estimator, feature_bank, dino_encoder, k=TRAIN_K, cache_path="checkpoints/causalbench_signal_cache.pt"
    )
    model = train(cached_items, epochs=30, device=device)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/fusion_head.pt")
    print("Saved checkpoints/fusion_head.pt")


if __name__ == "__main__":
    main()
```

```python
# scripts/run_evaluation.py
"""Runs the full Phase 1 evaluation protocol (spec Section 5) and prints a
results table: fidelity metrics, attribution accuracy/mIoU on a held-out
xSR-CausalBench split, and the disentanglement correlation matrix.

HR images are resized to 1024x1024, matching scripts/train_fusion_head.py
(see that file's docstring for why — must match the backbone's fixed 16x
factor). The held-out image range [50:70] is deliberately disjoint from
train_fusion_head.py's [0:40] + [40:50] ood range so evaluation never
sees a training image.

Run: conda run -n py313 python scripts/run_evaluation.py
"""
import glob

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from src.backbone.diffusion_backbone import DiffusionSRBackbone
from src.causalbench.dataset import CausalBenchDataset
from src.eval.attribution_metrics import mean_iou, pixel_accuracy
from src.eval.disentanglement import signal_correlation_matrix
from src.eval.fidelity_metrics import compute_psnr, compute_ssim
from src.fusion.infer import run_chasr
from src.fusion.model import FusionHead
from src.signals.distribution_shift import FeatureBank
from src.signals.kernel_estimator import KernelEstimator

HR_PATCH_SIZE = 1024  # must match DiffusionSRBackbone's fixed 16x output size
EVAL_MAX_IMAGES = 20


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    hr_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[50 : 50 + EVAL_MAX_IMAGES]
    hr_images = [np.asarray(Image.open(p).convert("RGB").resize((HR_PATCH_SIZE, HR_PATCH_SIZE))).astype(np.float32) / 255.0 for p in hr_paths]
    ood_paths = sorted(glob.glob("data/DIV2K_valid_HR/*.png"))[50 + EVAL_MAX_IMAGES : 50 + EVAL_MAX_IMAGES + 5]
    ood_patches = [np.asarray(Image.open(p).convert("RGB").resize((64, 64))).astype(np.float32) / 255.0 for p in ood_paths]

    held_out = CausalBenchDataset(hr_images, ood_patches, scale=16, seed=100)

    kernel_estimator = KernelEstimator().to(device)
    kernel_estimator.load_state_dict(torch.load("checkpoints/kernel_estimator.pt", weights_only=True))
    kernel_estimator.eval()
    feature_bank = FeatureBank(torch.load("checkpoints/feature_bank.pt", weights_only=True))
    dino_encoder = AutoModel.from_pretrained("facebook/dinov2-small").to(device).eval()
    backbone = DiffusionSRBackbone(device=device)
    fusion_head = FusionHead().to(device)
    fusion_head.load_state_dict(torch.load("checkpoints/fusion_head.pt", weights_only=True))
    fusion_head.eval()

    psnrs, ssims, accuracies, ious, all_signals = [], [], [], [], []
    for i in range(len(held_out)):
        lr, hr, gt_label_map = held_out[i]
        result = run_chasr(lr, backbone, kernel_estimator, feature_bank, dino_encoder, fusion_head)

        sr_np = result["sr_image"].permute(1, 2, 0).clamp(0, 1).numpy()
        hr_np = hr.permute(1, 2, 0).clamp(0, 1).numpy()
        psnrs.append(compute_psnr(sr_np, hr_np))
        ssims.append(compute_ssim(sr_np, hr_np))

        accuracies.append(pixel_accuracy(result["cause_map"], gt_label_map))
        ious.append(mean_iou(result["cause_map"], gt_label_map))

        all_signals.append(result["signal_stack"].reshape(4, -1))  # (4, H*W) per image

    print(f"PSNR: {np.mean(psnrs):.2f}")
    print(f"SSIM: {np.mean(ssims):.4f}")
    print(f"Attribution pixel accuracy: {np.mean(accuracies):.4f}")
    print(f"Attribution mIoU: {np.mean(ious):.4f}")

    flattened_signals = torch.cat(all_signals, dim=1)  # (4, N * H * W) across all held-out images
    corr = signal_correlation_matrix(flattened_signals)
    print("S1-S4 disentanglement correlation matrix (spec Section 5, item 5):")
    print(corr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 10: Extend the README with a usage section**

```markdown
# super_resolution

CHASR: Causal Hallucination Attribution for Extreme Super-Resolution.
Manuscript target: https://www.sciencedirect.com/special-issue/335613/extreme-super-resolution-pushing-the-boundaries-of-image-and-video-enhancement

## Design and Plan

- Design spec: `docs/superpowers/specs/2026-08-25-chasr-causal-hallucination-attribution-design.md`
- Phase 1 implementation plan: `docs/superpowers/plans/2026-08-25-chasr-phase1-core-pipeline.md`

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

## Tests

```bash
conda run -n py313 pytest tests/ -v
```
```

- [ ] **Step 11: Commit and push**

```bash
git add src/fusion/infer.py src/eval/downstream_ocr_case_study.py scripts/run_evaluation.py scripts/train_fusion_head.py README.md tests/fusion/test_infer.py tests/eval/test_downstream_ocr_case_study.py
git commit -m "feat: add end-to-end inference, downstream OCR case study, and orchestration scripts"
git push origin main
```

---

## After This Plan

Once Task 14 is complete, run the full test suite (`conda run -n py313 pytest tests/ -v`) and the manual smoke scripts (`scripts/smoke_test_backbone.py`) before relying on the pipeline for real experiments. Real experiment runs (`scripts/train_kernel_estimator.py`, `scripts/build_feature_bank.py`, `scripts/train_fusion_head.py`, `scripts/run_evaluation.py`) require the downloaded datasets and will take substantial wall-clock time on the 8GB laptop GPU — budget accordingly before the manuscript-writing phase begins.

Deferred to later plans, per spec Section 8 and user instruction to extend after the core experiment: 8x factor support, video/multi-frame xSR, the full HS-metric reproduction (vs. the fidelity/attribution-accuracy metrics implemented here), a larger downstream utility study beyond the minimal OCR case, and the manuscript-writing phase itself.
