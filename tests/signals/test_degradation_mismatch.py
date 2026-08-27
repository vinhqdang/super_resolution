import math

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
    # Near the mismatch pool's own upper bounds (sigma close to 4.0, theta
    # close to pi) — unambiguously severe mismatch, not just moderate.
    estimator = _FixedEstimator(sigma_x=4.0, sigma_y=3.8, theta=2.5, noise_sigma=0.01)
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


def test_exact_standard_boundary_gives_zero_mismatch():
    # sigma at the exact standard-pool upper bound, theta exactly 0 — the
    # in-distribution boundary should score as fully consistent.
    estimator = _FixedEstimator(sigma_x=1.5, sigma_y=1.5, theta=0.0, noise_sigma=0.01)
    lr = torch.rand(1, 3, 64, 64)
    score = compute_degradation_mismatch(estimator, lr)
    assert score == 0.0


def test_true_worst_case_mismatch_saturates_to_one():
    # sigma and theta both at the mismatch pool's own upper bounds — the
    # calibrated normalizer should make this land exactly at the 1.0 cap,
    # not saturate early (the bug the calibration fix addressed).
    estimator = _FixedEstimator(sigma_x=4.0, sigma_y=4.0, theta=math.pi, noise_sigma=0.01)
    lr = torch.rand(1, 3, 64, 64)
    score = compute_degradation_mismatch(estimator, lr)
    assert math.isclose(score, 1.0, abs_tol=1e-6)


def test_negative_estimator_output_does_not_crash_or_go_negative():
    # An untrained/adversarial estimator can emit values outside any
    # physically-meaningful range (no output activation constrains it) —
    # the score should clamp gracefully rather than error or go negative.
    estimator = _FixedEstimator(sigma_x=-5.0, sigma_y=-5.0, theta=-1.0, noise_sigma=-0.01)
    lr = torch.rand(1, 3, 64, 64)
    score = compute_degradation_mismatch(estimator, lr)
    assert 0.0 <= score <= 1.0
