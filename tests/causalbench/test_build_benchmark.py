import numpy as np

from src.causalbench.build_benchmark import (
    CAUSE_LABELS,
    build_distribution_shift_sample,
    build_ill_posed_sample,
    build_mismatch_sample,
    build_prior_reliance_sample,
)


def _random_hr(size=256):
    rng = np.random.default_rng(0)
    return rng.uniform(0, 1, size=(size, size, 3)).astype(np.float32)


def test_ill_posed_sample_label_map():
    rng = np.random.default_rng(1)
    hr, lr, label_map = build_ill_posed_sample(_random_hr(), rng)
    assert label_map.shape == hr.shape[:2]
    assert (label_map == CAUSE_LABELS["ILL_POSED"]).any()


def test_prior_reliance_sample_label_map():
    rng = np.random.default_rng(2)
    hr, lr, label_map = build_prior_reliance_sample(_random_hr(), rng)
    assert (label_map == CAUSE_LABELS["PRIOR_RELIANCE"]).any()
    assert (label_map == CAUSE_LABELS["RELIABLE"]).any()  # untouched region stays reliable


def test_prior_reliance_label_region_exactly_matches_suppressed_evidence():
    # Regression test: the label region must match the actual
    # evidence-suppressed rectangle exactly, not a scale-grid-floor-aligned
    # approximation that can shrink it (in the worst case to empty) or
    # shift it relative to where evidence was genuinely destroyed.
    size = 256
    patch_frac = 0.3
    hr = _random_hr(size)
    rng = np.random.default_rng(2)
    _, _, label_map = build_prior_reliance_sample(hr, rng, scale=16, patch_frac=patch_frac)

    expected_ph = expected_pw = int(size * patch_frac)
    labeled_count = int((label_map == CAUSE_LABELS["PRIOR_RELIANCE"]).sum())
    assert labeled_count == expected_ph * expected_pw


def test_prior_reliance_label_region_nonempty_even_for_small_unaligned_patch():
    # The bug this guards against: a small patch at an unlucky offset could
    # floor-align to an EMPTY label region despite evidence genuinely being
    # suppressed there. Use a tiny patch_frac so the region is much smaller
    # than the scale grid (16), which would have triggered the bug.
    hr = _random_hr(256)
    rng = np.random.default_rng(2)
    _, _, label_map = build_prior_reliance_sample(hr, rng, scale=16, patch_frac=0.05)
    assert (label_map == CAUSE_LABELS["PRIOR_RELIANCE"]).any()


def test_mismatch_sample_label_map():
    rng = np.random.default_rng(3)
    hr, lr, label_map = build_mismatch_sample(_random_hr(), rng)
    assert (label_map == CAUSE_LABELS["DEGRADATION_MISMATCH"]).any()


def test_distribution_shift_sample_label_map():
    rng = np.random.default_rng(4)
    ood_patch = rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32)
    hr, lr, label_map = build_distribution_shift_sample(_random_hr(), ood_patch, rng)
    assert (label_map == CAUSE_LABELS["DISTRIBUTION_SHIFT"]).any()
    assert (label_map == CAUSE_LABELS["RELIABLE"]).any()


def test_distribution_shift_label_region_matches_ood_patch_size():
    rng = np.random.default_rng(4)
    ood_patch = rng.uniform(0, 1, size=(64, 64, 3)).astype(np.float32)
    _, _, label_map = build_distribution_shift_sample(_random_hr(), ood_patch, rng)
    labeled_count = int((label_map == CAUSE_LABELS["DISTRIBUTION_SHIFT"]).sum())
    assert labeled_count == 64 * 64
