"""Paired bootstrap confidence interval over per-item metric values (closes
the gap the manuscript itself named: "the margins above come from one run
on 20 held-out images with no per-image variance or significance test
reported, so they should be read as this pilot's point estimate rather
than as a result shown to be robust to resampling" — Section 4.5). Resamples
held-out items with replacement rather than pixels, since items (not
individual pixels) are the independent units xSR-CausalBench actually
draws at random; this also directly answers the peer-review question of
whether the fusion head's margin over a baseline is distinguishable from
sampling noise at this pilot's held-out-set size.
"""
import numpy as np


def paired_bootstrap_ci(values_a: list[float], values_b: list[float], n_resamples: int = 10000, ci: float = 0.95, seed: int = 0) -> dict:
    """values_a, values_b: per-item metric values for two methods computed
    over the SAME items in the same order (e.g. per-image pixel accuracy
    from the same held-out set). Resamples item indices with replacement
    n_resamples times and computes mean(a) - mean(b) for each resample,
    returning the observed difference and a percentile confidence interval
    on that difference. `excludes_zero` is True only when the CI lies
    strictly on one side of zero, i.e. the margin is distinguishable from
    no difference at the given confidence level under this resampling.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError(f"values_a and values_b must have equal length, got {len(a)} and {len(b)}")
    if len(a) == 0:
        raise ValueError("values_a and values_b must be non-empty")

    n = len(a)
    rng = np.random.default_rng(seed)
    observed_diff = float(a.mean() - b.mean())

    idx = rng.integers(0, n, size=(n_resamples, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.quantile(diffs, alpha))
    upper = float(np.quantile(diffs, 1.0 - alpha))

    return {
        "observed_diff": observed_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "excludes_zero": (lower > 0.0) or (upper < 0.0),
        "n_items": n,
        "n_resamples": n_resamples,
        "ci": ci,
    }
