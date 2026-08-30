"""Degradation-aware adaptive fusion (spec section 4, Fusion).

This is the project's main architectural differentiator (spec section 7): instead of a static
stacked ensemble with fixed per-tier weights, Tier 2's block-grid/double-JPEG degradation
estimate is used to reweight Tier 2 vs. Tier 3 BEFORE they reach the meta-model. Rationale: a
heavily re-compressed/re-processed image has already had its forensic (Tier 2) signal degraded,
but CLIP's (Tier 3) semantic signal is comparatively transform-robust — so as estimated
degradation rises, we should trust Tier 2 less and Tier 3 more.

Final input to the logistic-regression meta-model: [tier0, tier1, tier2_reweighted,
tier3_reweighted, degradation_score].
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def reweight(tier2_score: float, tier3_score: float, degradation_score: float) -> tuple[float, float]:
    """degradation_score in [0, 1] (see src/frequency/features.py). As it rises, Tier 2's
    weight decays and Tier 3's weight is boosted above its baseline."""
    tier2_weight = 1.0 - degradation_score
    tier3_weight = 0.5 + 0.5 * degradation_score
    return tier2_score * tier2_weight, tier3_score * tier3_weight


def build_features(tier0: float, tier1: float, tier2: float, tier3: float, degradation_score: float) -> np.ndarray:
    tier2_adj, tier3_adj = reweight(tier2, tier3, degradation_score)
    return np.array([tier0, tier1, tier2_adj, tier3_adj, degradation_score], dtype=np.float32)


def build_feature_matrix(rows: list[dict]) -> np.ndarray:
    """`rows` — dicts with keys tier0, tier1, tier2, tier3, degradation_score."""
    return np.stack(
        [build_features(r["tier0"], r["tier1"], r["tier2"], r["tier3"], r["degradation_score"]) for r in rows]
    )


def train_fusion(rows: list[dict], labels: list[int], seed: int = 0) -> LogisticRegression:
    X = build_feature_matrix(rows)
    y = np.array(labels)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(X, y)
    return clf


def predict_fusion(clf: LogisticRegression, tier0: float, tier1: float, tier2: float, tier3: float, degradation_score: float) -> float:
    x = build_features(tier0, tier1, tier2, tier3, degradation_score).reshape(1, -1)
    return float(clf.predict_proba(x)[0, 1])
