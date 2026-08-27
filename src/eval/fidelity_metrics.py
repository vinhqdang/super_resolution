"""Fidelity metrics (spec Section 5, item 1) — context metrics, not the
headline result, since the backbone is frozen and unchanged by this work.
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    if np.array_equal(pred, target):
        return 100.0  # avoid relying on skimage's div-by-zero -> inf -> clip behavior
    value = peak_signal_noise_ratio(target, pred, data_range=1.0)
    return float(min(value, 100.0))  # clip very large (but non-identical) values too


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(target, pred, data_range=1.0, channel_axis=-1))


def compute_lpips(pred, target, lpips_model) -> float:
    """pred, target: torch tensors (3, H, W) in [0, 1]. lpips_model: lpips.LPIPS instance.
    torch is imported locally (not at module top like the rest of this
    package) so a caller who only needs compute_psnr/compute_ssim isn't
    forced to import torch."""
    import torch

    pred_n = pred.unsqueeze(0) * 2 - 1
    target_n = target.unsqueeze(0) * 2 - 1
    with torch.no_grad():
        return float(lpips_model(pred_n, target_n).item())
