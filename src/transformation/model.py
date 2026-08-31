"""The transformation profiler model (Feature 2).

Per label: a `HistGradientBoostingClassifier` (lightweight, no GPU, handles the mixed-scale
forensic features well) wrapped in isotonic `CalibratedClassifierCV` so the output is a
**calibrated** confidence in [0, 1]. Plus one `HistGradientBoostingRegressor` for
`overall_degradation` (a degradation score, NOT a probability).

`TransformationProfiler.predict(img)` returns the dict documented in `__init__.py`. It accepts a
precomputed Tier 2 `forensic` dict so inference does not recompute the expensive features.
"""

from __future__ import annotations

import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from src.transformation.features import profiler_features
from src.transformation.synthesize import PROFILE_LABELS, overall_severity

__all__ = ["PROFILE_LABELS", "TransformationProfiler", "load_profiler"]


class TransformationProfiler:
    def __init__(self, calibrate: bool = True, random_state: int = 0):
        self.calibrate = calibrate
        self.random_state = random_state
        self.classifiers: dict[str, object] = {}
        self.regressor: HistGradientBoostingRegressor | None = None
        self.feature_dim: int | None = None
        self.metadata: dict = {}

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, Y: np.ndarray, S: np.ndarray) -> "TransformationProfiler":
        self.feature_dim = X.shape[1]
        for i, label in enumerate(PROFILE_LABELS):
            y = Y[:, i].astype(int)
            base = HistGradientBoostingClassifier(
                max_depth=4, max_iter=200, learning_rate=0.08, random_state=self.random_state
            )
            if self.calibrate and y.sum() >= 15 and (len(y) - y.sum()) >= 15:
                clf = CalibratedClassifierCV(base, method="isotonic", cv=3)
            else:
                clf = base
            clf.fit(X, y)
            self.classifiers[label] = clf

        y_overall = np.array([overall_severity(row) for row in S], dtype=np.float32)
        self.regressor = HistGradientBoostingRegressor(
            max_depth=4, max_iter=250, learning_rate=0.08, random_state=self.random_state
        )
        self.regressor.fit(X, y_overall)
        return self

    # ------------------------------------------------------------------
    def _score_matrix(self, X: np.ndarray) -> dict[str, np.ndarray]:
        out = {}
        for label, clf in self.classifiers.items():
            out[label] = clf.predict_proba(X)[:, 1]
        out["overall_degradation"] = np.clip(self.regressor.predict(X), 0.0, 1.0)
        return out

    def predict_batch(self, imgs: list[Image.Image], forensics: list[dict] | None = None) -> list[dict]:
        feats = np.stack([
            profiler_features(im, forensics[i] if forensics else None) for i, im in enumerate(imgs)
        ])
        cols = self._score_matrix(feats)
        return [
            {k: float(v[i]) for k, v in cols.items()}
            for i in range(len(imgs))
        ]

    def predict(self, img: Image.Image, forensic: dict | None = None) -> dict:
        return self.predict_batch([img], [forensic] if forensic is not None else None)[0]

    def predict_from_features(self, X: np.ndarray) -> list[dict]:
        cols = self._score_matrix(np.atleast_2d(X))
        return [{k: float(v[i]) for k, v in cols.items()} for i in range(len(cols["overall_degradation"]))]

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @property
    def output_keys(self) -> list[str]:
        return PROFILE_LABELS + ["overall_degradation"]


def load_profiler(path: str) -> TransformationProfiler:
    return joblib.load(path)
