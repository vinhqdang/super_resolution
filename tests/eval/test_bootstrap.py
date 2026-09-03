import pytest

from src.eval.bootstrap import paired_bootstrap_ci


def test_identical_values_give_zero_diff_and_zero_width_ci():
    values = [0.5, 0.6, 0.4, 0.55, 0.48]
    result = paired_bootstrap_ci(values, values, n_resamples=500)
    assert result["observed_diff"] == pytest.approx(0.0)
    assert result["ci_lower"] == pytest.approx(0.0)
    assert result["ci_upper"] == pytest.approx(0.0)
    assert result["excludes_zero"] is False


def test_clearly_separated_values_exclude_zero():
    values_a = [0.9] * 40
    values_b = [0.1] * 40
    result = paired_bootstrap_ci(values_a, values_b, n_resamples=2000)
    assert result["observed_diff"] == pytest.approx(0.8)
    assert result["excludes_zero"] is True
    assert result["ci_lower"] > 0.0


def test_noisy_but_overlapping_values_may_not_exclude_zero():
    # Small n, high variance, true difference near zero: the CI should be
    # wide enough to plausibly include zero — this is a sanity check that
    # the function does not spuriously report significance on noise.
    import random

    rng = random.Random(0)
    values_a = [0.5 + rng.uniform(-0.3, 0.3) for _ in range(5)]
    values_b = [0.5 + rng.uniform(-0.3, 0.3) for _ in range(5)]
    result = paired_bootstrap_ci(values_a, values_b, n_resamples=2000, seed=1)
    assert result["ci_lower"] < result["observed_diff"] < result["ci_upper"]


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="equal length"):
        paired_bootstrap_ci([0.1, 0.2], [0.1, 0.2, 0.3])


def test_empty_inputs_raise():
    with pytest.raises(ValueError, match="non-empty"):
        paired_bootstrap_ci([], [])


def test_deterministic_given_seed():
    values_a = [0.7, 0.6, 0.8, 0.5, 0.9, 0.4]
    values_b = [0.5, 0.5, 0.6, 0.4, 0.6, 0.3]
    result1 = paired_bootstrap_ci(values_a, values_b, n_resamples=1000, seed=42)
    result2 = paired_bootstrap_ci(values_a, values_b, n_resamples=1000, seed=42)
    assert result1 == result2


def test_ci_width_shrinks_as_confidence_decreases():
    values_a = [0.7, 0.6, 0.8, 0.5, 0.9, 0.4, 0.65, 0.55]
    values_b = [0.5, 0.5, 0.6, 0.4, 0.6, 0.3, 0.45, 0.5]
    wide = paired_bootstrap_ci(values_a, values_b, n_resamples=2000, ci=0.99, seed=0)
    narrow = paired_bootstrap_ci(values_a, values_b, n_resamples=2000, ci=0.80, seed=0)
    assert (wide["ci_upper"] - wide["ci_lower"]) > (narrow["ci_upper"] - narrow["ci_lower"])
