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


def sample_mismatch_kernel(rng: np.random.Generator) -> tuple[np.ndarray, float, float, float]:
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
    lr = np.clip(_add_noise(lr, noise_sigma, rng), 0.0, 1.0)
    quality = int(rng.integers(*STANDARD_JPEG_QUALITY_RANGE))
    lr = _jpeg_recompress(lr, quality)
    params = {"sigma_x": float(sigma_x), "sigma_y": float(sigma_y), "theta": float(theta), "noise_sigma": float(noise_sigma)}
    return lr, params
