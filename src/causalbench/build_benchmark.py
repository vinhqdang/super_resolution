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


def build_ill_posed_sample(hr: np.ndarray, rng: np.random.Generator, scale: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole-image extreme downsample under a correctly-specified (standard)
    degradation model: null-space is large everywhere by construction."""
    lr, _ = degrade_standard(hr, scale, rng)
    label_map = np.full(hr.shape[:2], CAUSE_LABELS["ILL_POSED"], dtype=np.int64)
    return hr, lr, label_map


def build_prior_reliance_sample(hr: np.ndarray, rng: np.random.Generator, scale: int = 16, patch_frac: float = 0.3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standard degradation, but a random rectangular sub-region has its LR
    evidence additionally zeroed out beyond the nominal degradation model —
    forcing the backbone to fall back on the generative prior there."""
    h, w = hr.shape[:2]
    ph, pw = int(h * patch_frac), int(w * patch_frac)
    y0, x0 = rng.integers(0, h - ph), rng.integers(0, w - pw)

    hr_evidence_suppressed = hr.copy()
    hr_evidence_suppressed[y0 : y0 + ph, x0 : x0 + pw] = 0.0
    lr, _ = degrade_standard(hr_evidence_suppressed, scale, rng)

    # Label the exact suppressed rectangle — matching where evidence was
    # actually destroyed. A prior (floor-aligned-to-scale) version could
    # shift/shrink this region, in the worst case (small ph/pw, unlucky
    # y0/x0) to an EMPTY region despite evidence genuinely being zeroed,
    # silently mislabeling the whole patch RELIABLE.
    label_map = np.full(hr.shape[:2], CAUSE_LABELS["RELIABLE"], dtype=np.int64)
    label_map[y0 : y0 + ph, x0 : x0 + pw] = CAUSE_LABELS["PRIOR_RELIANCE"]
    return hr, lr, label_map


def build_mismatch_sample(hr: np.ndarray, rng: np.random.Generator, scale: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Whole image degraded with an out-of-training-distribution kernel
    while content stays in-distribution."""
    lr, _ = degrade_mismatch(hr, scale, rng)
    label_map = np.full(hr.shape[:2], CAUSE_LABELS["DEGRADATION_MISMATCH"], dtype=np.int64)
    return hr, lr, label_map


def build_distribution_shift_sample(hr: np.ndarray, ood_patch: np.ndarray, rng: np.random.Generator, scale: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
