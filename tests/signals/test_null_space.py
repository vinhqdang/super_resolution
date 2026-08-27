import pytest
import torch

from src.signals.null_space import compute_null_space_variance, normalized_null_space_variance


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
    with pytest.raises(ValueError):
        compute_null_space_variance([torch.rand(3, 4, 4)])


def test_variance_matches_unbiased_formula():
    a = torch.tensor([[[1.0]]])  # (1, 1, 1)
    b = torch.tensor([[[3.0]]])
    s1 = compute_null_space_variance([a, b])
    assert torch.allclose(s1, torch.tensor([[2.0]]))  # unbiased: (1-2)^2 + (3-2)^2 = 2


def test_normalized_variance_is_bounded_to_zero_one():
    a = torch.tensor([[[0.0]]])
    b = torch.tensor([[[1.0]]])  # worst-case pixel-value spread for K=2
    s1_norm = normalized_null_space_variance([a, b])
    assert (s1_norm >= 0).all() and (s1_norm <= 1).all()
    assert torch.allclose(s1_norm, torch.tensor([[1.0]]))  # saturates at the theoretical max


def test_normalized_variance_preserves_ordering():
    x = torch.rand(3, 8, 8)
    low_var_samples = [x.clone(), x.clone() + 0.01]
    high_var_samples = [x.clone(), x.clone() + 0.5]
    low = normalized_null_space_variance(low_var_samples)
    high = normalized_null_space_variance(high_var_samples)
    assert low.mean() < high.mean()
