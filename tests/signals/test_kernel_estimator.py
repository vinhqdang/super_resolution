import numpy as np
import torch

from src.degradation.real_esrgan_degrade import degrade_standard
from src.signals.kernel_estimator import KernelEstimator, KernelEstimatorDataset


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


def _hr_images(n, size=64):
    rng = np.random.default_rng(0)
    return [rng.uniform(0, 1, size=(size, size, 3)).astype(np.float32) for _ in range(n)]


def test_dataset_mixes_standard_and_mismatch_pools_roughly_evenly():
    # theta is exactly 0 for every standard-pool sample and (near-certainly,
    # continuous uniform over (0, pi)) nonzero for every mismatch-pool
    # sample, so it's a reliable proxy for which pool an item came from.
    dataset = KernelEstimatorDataset(_hr_images(200), scale=4, seed=0)
    standard_count = sum(1 for i in range(len(dataset)) if dataset[i][1][2].item() == 0.0)
    fraction = standard_count / len(dataset)
    assert 0.35 < fraction < 0.65


def test_dataset_is_deterministic_per_item():
    dataset = KernelEstimatorDataset(_hr_images(5), scale=4, seed=0)
    lr_a, target_a = dataset[2]
    lr_b, target_b = dataset[2]
    assert torch.allclose(lr_a, lr_b)
    assert torch.allclose(target_a, target_b)


def test_dataset_different_indices_give_different_draws():
    dataset = KernelEstimatorDataset(_hr_images(5), scale=4, seed=0)
    _, target_0 = dataset[0]
    _, target_1 = dataset[1]
    assert not torch.allclose(target_0, target_1)
