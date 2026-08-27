import numpy as np
import torch

from src.eval.fidelity_metrics import compute_lpips, compute_psnr, compute_ssim


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


class _FakeLpipsModel:
    """Stub standing in for lpips.LPIPS: records the tensors it was called
    with so the test can verify compute_lpips's [0,1]->[-1,1] rescale and
    batch-dim/no_grad/item() unwrap contract without instantiating the
    real (expensive) LPIPS network."""

    def __init__(self):
        self.last_call_args = None

    def __call__(self, pred_n, target_n):
        self.last_call_args = (pred_n, target_n)
        return torch.tensor(0.25)


def test_compute_lpips_rescales_to_minus_one_one_and_unwraps_scalar():
    model = _FakeLpipsModel()
    pred = torch.full((3, 4, 4), 0.75)
    target = torch.full((3, 4, 4), 0.25)

    result = compute_lpips(pred, target, model)

    assert result == 0.25
    pred_n, target_n = model.last_call_args
    assert pred_n.shape == (1, 3, 4, 4)
    assert target_n.shape == (1, 3, 4, 4)
    assert torch.allclose(pred_n, torch.full((1, 3, 4, 4), 0.5))  # 0.75*2-1
    assert torch.allclose(target_n, torch.full((1, 3, 4, 4), -0.5))  # 0.25*2-1
