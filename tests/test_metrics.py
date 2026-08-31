"""Checks for src/eval/metrics.py (shared by Features 1, 3, 4, 5)."""

from __future__ import annotations

import numpy as np

from src.eval.metrics import binary_metrics, expected_calibration_error, fpr_at_tpr


def test_perfect_separation():
    y = [0, 0, 0, 1, 1, 1]
    s = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    m = binary_metrics(y, s, threshold=0.5)
    assert m["auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["accuracy"] == 1.0
    assert m["fpr"] == 0.0
    assert m["tpr"] == 1.0
    assert m["fp"] == 0 and m["fn"] == 0


def test_confusion_counts_sum_to_n():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    s = rng.random(50)
    m = binary_metrics(y, s)
    assert m["tp"] + m["tn"] + m["fp"] + m["fn"] == 50
    assert m["num_samples"] == 50


def test_single_class_auc_is_nan():
    m = binary_metrics([1, 1, 1], [0.2, 0.6, 0.9])
    assert np.isnan(m["auc"])


def test_threshold_shifts_fpr():
    y = [0, 0, 1, 1]
    s = [0.4, 0.6, 0.55, 0.9]
    hi = binary_metrics(y, s, threshold=0.5)
    lo = binary_metrics(y, s, threshold=0.95)
    assert hi["fpr"] >= lo["fpr"]


def test_ece_zero_for_calibrated():
    # scores exactly equal to empirical accuracy in each bin
    y = np.array([0, 1] * 50)
    s = np.array([0.5] * 100)
    assert abs(expected_calibration_error(y, s)) < 1e-9


def test_fpr_at_tpr_monotone_target():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(100), np.ones(100)].astype(int)
    s = np.r_[rng.normal(0.4, 0.15, 100), rng.normal(0.6, 0.15, 100)].clip(0, 1)
    f_low, _ = fpr_at_tpr(y, s, 0.5)
    f_high, _ = fpr_at_tpr(y, s, 0.95)
    assert f_high >= f_low


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
