import torch

from src.eval.disentanglement import signal_correlation_matrix


def test_correlation_matrix_shape_and_diagonal_is_one():
    torch.manual_seed(0)
    signals = torch.rand(4, 1000)
    corr = signal_correlation_matrix(signals)
    assert corr.shape == (4, 4)
    assert torch.allclose(torch.diag(corr), torch.ones(4), atol=1e-4)


def test_identical_signals_have_correlation_one():
    base = torch.rand(1000)
    signals = torch.stack([base, base, torch.rand(1000), torch.rand(1000)])
    corr = signal_correlation_matrix(signals)
    assert torch.isclose(corr[0, 1], torch.tensor(1.0), atol=1e-4)


def test_matrix_is_symmetric():
    torch.manual_seed(1)
    signals = torch.rand(4, 500)
    corr = signal_correlation_matrix(signals)
    assert torch.allclose(corr, corr.T, atol=1e-5)


def test_anti_correlated_signals_have_correlation_minus_one():
    base = torch.rand(1000)
    signals = torch.stack([base, -base, torch.rand(1000), torch.rand(1000)])
    corr = signal_correlation_matrix(signals)
    assert torch.isclose(corr[0, 1], torch.tensor(-1.0), atol=1e-4)


def test_raises_on_fewer_than_two_samples():
    import pytest
    with pytest.raises(ValueError):
        signal_correlation_matrix(torch.rand(4, 1))
