"""Pure-function checks for the robustness benchmark report/summary layer (Feature 1).

Does not need trained artifacts — feeds synthetic per-(model, condition) rows through the
aggregation + reporting code.
"""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd

from src.eval.robustness_bench import build_summary, write_degradation_curves, write_report
from src.eval.scoring import STATIC_ENSEMBLE_WEIGHTS, DetectorBank, RawScores


def _synthetic_df():
    rows = []
    conds = [
        ("clean", "clean", 0.0, 0),
        ("jpeg_q50", "compression", 0.5, 4),
        ("jpeg_q30", "compression", 0.7, 5),
        ("gauss_blur_s2.0", "blur", 0.67, 3),
        ("resize_50pct", "resolution", 0.5, 3),
        ("center_crop_50", "crop", 0.5, 3),
        ("screenshot_typical", "screenshot", 0.6, 2),
        ("resize75_jpeg70", "compound", 0.55, 1),
    ]
    rng = np.random.default_rng(0)
    for model in ("cnn", "forensic", "clip", "static_ensemble", "existing_fusion"):
        for name, fam, sev, rank in conds:
            rows.append({
                "model": model, "transformation": name, "family": fam,
                "severity": sev, "severity_rank": rank,
                "compound_id": name if fam == "compound" else "", "seed": 0,
                "auc": float(rng.uniform(0.6, 0.95)), "pr_auc": 0.8, "accuracy": 0.7,
                "precision": 0.7, "recall": 0.7, "f1": 0.7, "fpr": 0.3, "tpr": 0.7,
                "brier": 0.2, "ece": 0.1, "tp": 5, "fp": 2, "fn": 2, "tn": 5, "num_samples": 14,
            })
    return pd.DataFrame(rows)


def test_build_summary_has_expected_columns():
    df = _synthetic_df()
    s = build_summary(df)
    assert set(s.index) == {"cnn", "forensic", "clip", "static_ensemble", "existing_fusion"}
    for col in ("clean_auc", "robust_auc_mean", "compound_auc_mean", "final_score"):
        assert col in s.columns
    # final_score is the documented 0.5/0.5 blend
    row = s.loc["cnn"]
    assert abs(row["final_score"] - (0.5 * row["clean_auc"] + 0.5 * row["robust_auc_mean"])) < 1e-9


def test_report_and_curves_write(tmp_path=None):
    d = tmp_path or tempfile.mkdtemp()
    d = str(d)
    df = _synthetic_df()
    s = build_summary(df)
    figs = write_degradation_curves(df, d)
    assert figs, "expected at least one degradation curve"
    meta = {"git_sha": "test", "timestamp": "now", "split": "test", "n_real": 7, "n_fake": 7,
            "tiers_available": {}, "thresholds_source": "raw 0.5 (untuned)"}
    path = write_report(df, s, meta, d, figs)
    text = open(path).read()
    assert "Robustness Benchmark" in text
    assert "ROBUST-MEAN" in text


def test_static_ensemble_weights_sum_to_one():
    assert abs(sum(STATIC_ENSEMBLE_WEIGHTS.values()) - 1.0) < 1e-9


def test_static_ensemble_falls_back_when_tier_missing():
    bank = DetectorBank.__new__(DetectorBank)  # skip __post_init__ / artifact loading
    raw = RawScores(
        tier1=np.array([np.nan, np.nan]),
        tier2=np.array([0.2, 0.8]),
        tier3=np.array([0.4, 0.6]),
        degradation=np.array([0.1, 0.1]),
    )
    out = DetectorBank.static_ensemble(bank, raw)
    # only tier2 + tier3 available -> weighted mean of those two, renormalised
    w2, w3 = STATIC_ENSEMBLE_WEIGHTS["tier2"], STATIC_ENSEMBLE_WEIGHTS["tier3"]
    expected0 = (w2 * 0.2 + w3 * 0.4) / (w2 + w3)
    assert abs(out[0] - expected0) < 1e-9


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
