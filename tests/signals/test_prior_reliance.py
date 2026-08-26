import torch

from src.signals.prior_reliance import compute_prior_reliance


def test_evidence_dominant_region_gives_low_reliance():
    def forward_fn(lr, z):
        # Output driven entirely by lr (upsampled), z ignored -> evidence-dominant.
        return torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=4, mode="nearest").squeeze(0)

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert s2.shape == (16, 16)
    assert s2.mean() < 0.3


def test_prior_dominant_region_gives_high_reliance():
    def forward_fn(lr, z):
        # Output driven entirely by z, lr ignored -> prior-dominant.
        return z * 1.0

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert s2.mean() > 0.7


def test_output_in_valid_range():
    def forward_fn(lr, z):
        up = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=4, mode="nearest").squeeze(0)
        return 0.5 * up + 0.5 * z

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert (s2 >= 0).all() and (s2 <= 1).all()


def test_equal_sensitivity_gives_mid_range_reliance():
    def forward_fn(lr, z):
        up = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=4, mode="nearest").squeeze(0)
        return 0.5 * up + 0.5 * z

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert abs(s2.mean().item() - 0.5) < 0.15


def test_fully_insensitive_forward_fn_returns_documented_zero_not_nan():
    """Known limitation (see module docstring): when forward_fn is
    insensitive to both lr and z, the ratio is 0/0-guarded to 0, read as
    "fully evidence-driven" rather than the more accurate "indeterminate."
    This pins that documented behavior so a future change to the epsilon
    guard doesn't silently start returning NaN/Inf instead."""

    def forward_fn(lr, z):
        return torch.zeros(3, 16, 16)

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)
    s2 = compute_prior_reliance(forward_fn, lr, z)
    assert torch.isfinite(s2).all()
    assert torch.allclose(s2, torch.zeros(16, 16))


def test_result_is_invariant_to_eps_choice():
    """Regression guard for the Task-5-style bug (asymmetric eps
    normalization biasing the ratio toward 0): since both g_evidence and
    g_prior are divided by the same eps here, the ratio is algebraically
    independent of eps. Seed the RNG identically before each call so the
    perturbation directions match and only their scale differs."""

    def forward_fn(lr, z):
        up = torch.nn.functional.interpolate(lr.unsqueeze(0), scale_factor=4, mode="nearest").squeeze(0)
        return 0.5 * up + 0.5 * z

    lr = torch.rand(3, 4, 4)
    z = torch.rand(3, 16, 16)

    torch.manual_seed(0)
    s2_small_eps = compute_prior_reliance(forward_fn, lr, z, eps=1e-3)
    torch.manual_seed(0)
    s2_large_eps = compute_prior_reliance(forward_fn, lr, z, eps=1e-2)

    assert torch.allclose(s2_small_eps, s2_large_eps, atol=1e-3)
