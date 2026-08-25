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


def sample_standard_kernel(rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Isotropic Gaussian blur kernel (sigma_x == sigma_y, theta == 0).
    Returns (kernel, sigma) — the sigma is returned alongside the kernel
    (matching src.degradation.mismatch_degrade.sample_mismatch_kernel's
    pattern) so callers can record the *actual* sigma used as ground truth
    instead of re-sampling a second, inconsistent one."""
    sigma = rng.uniform(*STANDARD_SIGMA_RANGE)
    ax = np.arange(KERNEL_SIZE) - KERNEL_SIZE // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel.astype(np.float32), sigma


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
    kernel, sigma = sample_standard_kernel(rng)
    blurred = _blur(hr, kernel)
    lr = _average_pool_downsample(blurred, scale)
    noise_sigma = rng.uniform(*STANDARD_NOISE_RANGE)
    lr = np.clip(_add_noise(lr, noise_sigma, rng), 0.0, 1.0)
    quality = int(rng.integers(*STANDARD_JPEG_QUALITY_RANGE))
    lr = _jpeg_recompress(lr, quality)
    params = {"sigma_x": float(sigma), "sigma_y": float(sigma), "theta": 0.0, "noise_sigma": float(noise_sigma)}
    return lr, params
