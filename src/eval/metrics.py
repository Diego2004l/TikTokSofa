"""Shared classification-metric helpers for every evaluation in the project (Features 1, 3, 4, 5).

`binary_metrics()` returns the full set the spec asks for — never accuracy alone: ROC-AUC,
PR-AUC, accuracy, precision, recall, F1, FPR, TPR, the 2x2 confusion matrix, and optional
calibration metrics (Brier score + expected calibration error).

Convention throughout the repo: label 1 = AI/fake (the positive class), label 0 = real.
`score` is P(AI). A prediction is positive when `score >= threshold`.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def df_to_markdown(df, floatfmt: str = "{:.4f}", index: bool = True) -> str:
    """Minimal DataFrame -> GitHub Markdown table (avoids the optional `tabulate` dependency)."""
    import pandas as pd

    d = df.copy()
    if index:
        d = d.reset_index()

    def cell(x):
        if isinstance(x, float):
            return "—" if not np.isfinite(x) else floatfmt.format(x)
        return str(x)

    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, row in d.iterrows():
        lines.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(lines)


def _safe(fn, *a, **k) -> float:
    try:
        v = float(fn(*a, **k))
        return v if np.isfinite(v) else float("nan")
    except ValueError:
        return float("nan")


def expected_calibration_error(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Standard equal-width-bin ECE: mean |confidence - accuracy| weighted by bin population."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_score >= lo) & (y_score < hi if hi < 1.0 else y_score <= hi)
        if not mask.any():
            continue
        conf = y_score[mask].mean()
        acc = y_true[mask].mean()
        ece += mask.mean() * abs(conf - acc)
    return float(ece)


def binary_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
    with_calibration: bool = True,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    n_pos = tp + fn
    n_neg = tn + fp
    tpr = tp / n_pos if n_pos else float("nan")          # recall / sensitivity
    fpr = fp / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tpr
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and np.isfinite(precision) and np.isfinite(recall) else float("nan"))
    accuracy = (tp + tn) / len(y_true) if len(y_true) else float("nan")

    out = {
        "auc": _safe(roc_auc_score, y_true, y_score) if len(set(y_true.tolist())) == 2 else float("nan"),
        "pr_auc": _safe(average_precision_score, y_true, y_score) if len(set(y_true.tolist())) == 2 else float("nan"),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": threshold,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "num_samples": int(len(y_true)),
        "num_pos": n_pos, "num_neg": n_neg,
    }
    if with_calibration:
        out["brier"] = _safe(lambda: np.mean((y_score - y_true) ** 2))
        out["ece"] = expected_calibration_error(y_true, y_score)
    return out


def confusion_dict(m: dict) -> dict:
    return {"tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]}


def fpr_at_tpr(y_true, y_score, target_tpr: float = 0.9) -> tuple[float, float]:
    """Lowest FPR achievable while holding TPR >= target, plus the threshold that gets there.
    Returns (fpr, threshold). NaN if the target TPR is unreachable."""
    from sklearn.metrics import roc_curve

    y_true = np.asarray(y_true, dtype=int)
    if len(set(y_true.tolist())) != 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    ok = tpr >= target_tpr
    if not ok.any():
        return float("nan"), float("nan")
    i = np.argmax(ok)  # first index where tpr >= target
    return float(fpr[i]), float(thr[i])
