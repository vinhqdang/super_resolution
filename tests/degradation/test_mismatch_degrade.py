import numpy as np
from src.degradation.real_esrgan_degrade import STANDARD_SIGMA_RANGE
from src.degradation.mismatch_degrade import sample_mismatch_kernel, degrade_mismatch


def test_mismatch_kernel_is_anisotropic():
    rng = np.random.default_rng(0)
    kernel, sigma_x, sigma_y, theta = sample_mismatch_kernel(rng)
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
