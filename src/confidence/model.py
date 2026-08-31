"""Confidence model + abstention policy (Feature 4).

`ConfidenceModel`  -- calibrated logistic regression predicting P(the final prediction is
correct) from `signals.SIGNAL_NAMES`. Confidence is this probability, NOT the raw detector score.
Bucketed to HIGH / MEDIUM / LOW by two thresholds tuned alongside the policy.

`AbstentionPolicy` -- decides AI / REAL / UNKNOWN from (final_score, confidence):
    * AI    if final_score >= hi  and confidence >= conf_min
    * REAL  if final_score <= lo  and confidence >= conf_min
    * UNKNOWN otherwise
`fit()` grid-searches (lo, hi, conf_min) on a VALIDATION split to minimise the abstention rate
subject to a target false-positive rate and a minimum selective accuracy -- no hard-coded cuts.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.confidence.signals import SIGNAL_NAMES

CONFIDENCE_BUCKETS = ("LOW", "MEDIUM", "HIGH")


class ConfidenceModel:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.estimator = None
        self.feature_names_ = list(SIGNAL_NAMES)

    def fit(self, signals: np.ndarray, correct: np.ndarray) -> "ConfidenceModel":
        X = np.asarray(signals, dtype=float)
        y = np.asarray(correct, dtype=int)
        base = Pipeline([("scale", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000, random_state=self.seed))])
        if min(int(y.sum()), int(len(y) - y.sum())) >= 20:
            self.estimator = CalibratedClassifierCV(base, method="isotonic", cv=3).fit(X, y)
        else:
            self.estimator = base.fit(X, y)
        return self

    def predict_confidence(self, signals: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(signals, dtype=float))
        return self.estimator.predict_proba(X)[:, 1]


def bucket(conf: float, med: float, high: float) -> str:
    if conf >= high:
        return "HIGH"
    if conf >= med:
        return "MEDIUM"
    return "LOW"


@dataclass
class AbstentionParams:
    lo: float = 0.35
    hi: float = 0.65
    conf_min: float = 0.5
    conf_med: float = 0.5
    conf_high: float = 0.75
    target_fpr: float = 0.1
    min_selective_accuracy: float = 0.9


class AbstentionPolicy:
    def __init__(self, params: AbstentionParams | None = None):
        self.params = params or AbstentionParams()
        self.fit_report_: dict = {}

    # ------------------------------------------------------------------
    def decide(self, final_score: float, confidence: float) -> dict:
        p = self.params
        if confidence >= p.conf_min and final_score >= p.hi:
            label, abstained = "AI", False
        elif confidence >= p.conf_min and final_score <= p.lo:
            label, abstained = "REAL", False
        else:
            label, abstained = "UNKNOWN", True
        return {"label": label, "score": float(final_score), "abstained": abstained,
                "confidence": bucket(confidence, p.conf_med, p.conf_high),
                "confidence_score": float(confidence)}

    def decide_batch(self, scores: np.ndarray, confidences: np.ndarray) -> list[dict]:
        return [self.decide(float(s), float(c)) for s, c in zip(scores, confidences)]

    # ------------------------------------------------------------------
    def fit(self, scores: np.ndarray, labels: np.ndarray, confidences: np.ndarray,
            target_fpr: float = 0.1, min_selective_accuracy: float = 0.9,
            lo_grid=None, hi_grid=None, conf_grid=None) -> "AbstentionPolicy":
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)
        conf = np.asarray(confidences, dtype=float)
        lo_grid = np.round(np.arange(0.20, 0.51, 0.05), 2) if lo_grid is None else lo_grid
        hi_grid = np.round(np.arange(0.50, 0.86, 0.05), 2) if hi_grid is None else hi_grid
        conf_grid = np.round(np.arange(0.30, 0.81, 0.05), 2) if conf_grid is None else conf_grid

        best = None
        for lo, hi, cmin in itertools.product(lo_grid, hi_grid, conf_grid):
            if lo >= hi:
                continue
            answered = ((conf >= cmin) & ((scores >= hi) | (scores <= lo)))
            if answered.sum() == 0:
                continue
            pred = (scores[answered] >= hi).astype(int)
            y = labels[answered]
            tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
            tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            sel_acc = (tp + tn) / answered.sum()
            coverage = answered.mean()
            feasible = fpr <= target_fpr and sel_acc >= min_selective_accuracy
            # maximise coverage among feasible; if none feasible, minimise fpr then maximise coverage
            key = (feasible, coverage if feasible else -fpr, coverage)
            if best is None or key > best[0]:
                best = (key, AbstentionParams(
                    lo=float(lo), hi=float(hi), conf_min=float(cmin),
                    conf_med=float(np.quantile(conf, 0.33)), conf_high=float(np.quantile(conf, 0.66)),
                    target_fpr=target_fpr, min_selective_accuracy=min_selective_accuracy),
                    {"coverage": float(coverage), "selective_accuracy": float(sel_acc),
                     "fpr": float(fpr), "feasible": bool(feasible)})
        if best is None:
            raise SystemExit("No policy answered any sample — check the score/confidence inputs.")
        self.params = best[1]
        self.fit_report_ = best[2]
        return self

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"params": asdict(self.params), "fit_report": self.fit_report_}

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        with open(path.replace(".joblib", ".json"), "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_policy(path: str) -> AbstentionPolicy:
    return joblib.load(path)
