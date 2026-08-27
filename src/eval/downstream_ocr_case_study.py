"""Minimal downstream utility case study (spec Section 5, item 4): does
flagging a region as degradation-mismatch or distribution-shift correlate
with actual OCR failure? Uses easyocr (already installed) against a small
set of text-patch crops; this module holds the correlation logic, kept
separate from the OCR call itself so it's unit-testable without running
the OCR engine.
"""
import numpy as np


def correlate_ocr_failure_with_cause(cause_map: np.ndarray, ocr_failure_mask: np.ndarray, num_classes: int = 5) -> dict:
    """Returns {cause_id: failure_rate} — fraction of pixels flagged as that
    cause where OCR also failed. A cause with no pixels present is omitted."""
    rates = {}
    for cls in range(num_classes):
        cls_mask = cause_map == cls
        if cls_mask.sum() == 0:
            continue
        rates[cls] = float(ocr_failure_mask[cls_mask].mean())
    return rates


def run_easyocr_readability_check(reader, image: np.ndarray, ground_truth_text: str) -> bool:
    """reader: easyocr.Reader instance. Returns True if OCR read matches
    ground truth (case-insensitive, whitespace-stripped)."""
    results = reader.readtext(image, detail=0)
    read_text = " ".join(results).strip().lower()
    return read_text == ground_truth_text.strip().lower()
