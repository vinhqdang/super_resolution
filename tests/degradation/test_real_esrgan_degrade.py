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
