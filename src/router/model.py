"""The adaptive router model (Feature 3).

Lightweight by design:
  * `model="logreg"` -> standardised logistic regression (a calibrated linear model; the
    coefficients are directly inspectable as learned evidence weights).
  * `model="gbdt"`   -> small `HistGradientBoostingClassifier` (max_depth 3, few iters) for when
    the routing rule is non-linear (e.g. "trust Tier 2 only when degradation is low").

Either way it is wrapped in `CalibratedClassifierCV` (isotonic) when data permits, so the final
router score is a calibrated P(AI).
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.router.features import Evidence, RouterConfig, build_matrix, build_vector, feature_names

__all__ = ["AdaptiveRouter", "RouterConfig", "load_router"]


def _base_estimator(config: RouterConfig):
    if config.model == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=config.seed)),
        ])
    if config.model == "gbdt":
        return HistGradientBoostingClassifier(
            max_depth=3, max_iter=150, learning_rate=0.06,
            l2_regularization=1.0, random_state=config.seed,
        )
    raise ValueError(f"unknown router model {config.model!r}")


class AdaptiveRouter:
    def __init__(self, config: RouterConfig | None = None):
        self.config = config or RouterConfig()
        self.estimator = None
        self.feature_names_ = feature_names(self.config)
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    def fit(self, evidence: list[Evidence], labels, sample_weight=None) -> "AdaptiveRouter":
        X = build_matrix(evidence, self.config)
        y = np.asarray(labels, dtype=int)
        base = _base_estimator(self.config)
        pos, neg = int(y.sum()), int(len(y) - y.sum())
        if self.config.calibrate and pos >= 20 and neg >= 20:
            self.estimator = CalibratedClassifierCV(base, method="isotonic", cv=3)
            self.estimator.fit(X, y)
        else:
            base.fit(X, y, **({"lr__sample_weight": sample_weight} if sample_weight is not None
                              and self.config.model == "logreg" else {}))
            self.estimator = base
        return self

    def oof_fit_predict(self, evidence: list[Evidence], labels, n_splits: int = 5):
        """Fit on out-of-fold splits and return (router, oof_scores) so the router's own
        selection/calibration never sees a sample it was trained on (spec anti-leakage)."""
        from sklearn.model_selection import StratifiedKFold

        X = build_matrix(evidence, self.config)
        y = np.asarray(labels, dtype=int)
        oof = np.full(len(y), np.nan)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.config.seed)
        for tr, te in skf.split(X, y):
            fold = _base_estimator(self.config)
            fold.fit(X[tr], y[tr])
            oof[te] = fold.predict_proba(X[te])[:, 1]
        self.fit(evidence, labels)  # final model on all data for deployment
        return oof

    # ------------------------------------------------------------------
    def predict_proba(self, evidence: list[Evidence]) -> np.ndarray:
        X = build_matrix(evidence, self.config)
        return self.estimator.predict_proba(X)[:, 1]

    def predict_one(self, ev: Evidence) -> float:
        return float(self.estimator.predict_proba(build_vector(ev, self.config).reshape(1, -1))[0, 1])

    # ------------------------------------------------------------------
    def coefficients(self) -> dict | None:
        """Learned linear weights (logreg only), for the write-up's 'which evidence is trusted'."""
        est = self.estimator
        if isinstance(est, CalibratedClassifierCV):
            est = est.calibrated_classifiers_[0].estimator
        if isinstance(est, Pipeline) and hasattr(est.named_steps.get("lr", None), "coef_"):
            return dict(zip(self.feature_names_, est.named_steps["lr"].coef_[0].round(4).tolist()))
        return None

    def save(self, path: str) -> None:
        joblib.dump(self, path)


def load_router(path: str) -> AdaptiveRouter:
    return joblib.load(path)
