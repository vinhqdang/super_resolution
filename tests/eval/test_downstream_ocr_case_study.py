import numpy as np

from src.eval.downstream_ocr_case_study import correlate_ocr_failure_with_cause


def test_correlate_ocr_failure_with_cause_returns_per_cause_rates():
    # np.zeros leaves the "rest" of the map at class 0, not 4 — the plan's
    # original test asserted rates[4] without ever putting class 4 into
    # cause_map, which raises KeyError (correlate_ocr_failure_with_cause
    # correctly omits classes absent from the map, per its own docstring).
    # Explicitly set the rest to RELIABLE (4) so the test matches its
    # stated intent ("0% failure rate in the ... reliable rest").
    cause_map = np.full((32, 32), 4, dtype=np.int64)  # RELIABLE by default
    cause_map[:16, :] = 2  # DEGRADATION_MISMATCH region
    ocr_failure_mask = np.zeros((32, 32), dtype=bool)
    ocr_failure_mask[:16, :] = True  # OCR fails exactly where mismatch is flagged

    rates = correlate_ocr_failure_with_cause(cause_map, ocr_failure_mask, num_classes=5)
    assert rates[2] == 1.0  # 100% failure rate in the mismatch region
    assert rates[4] == 0.0  # 0% failure rate in the reliable rest


def test_correlate_ocr_failure_with_cause_omits_absent_classes():
    cause_map = np.zeros((32, 32), dtype=np.int64)  # only class 0 present
    ocr_failure_mask = np.zeros((32, 32), dtype=bool)
    rates = correlate_ocr_failure_with_cause(cause_map, ocr_failure_mask, num_classes=5)
    assert set(rates.keys()) == {0}
