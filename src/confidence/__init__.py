"""Feature 4 -- confidence, abstention, and detector disagreement.

The final system is not forced to answer AI/REAL for every image. It may return:

    {"label": "UNKNOWN", "score": 0.57, "confidence": "LOW", "abstained": true}

meaning "evidence is insufficient or contradictory -- human review recommended".

Pieces:
  * `signals.py`   -- turn an evidence bundle (+ final score, + optional multi-crop pass) into a
    fixed signal vector: decision margin, detector disagreement stats, transformation severity,
    missing-tier count, provenance strength, cross-crop consistency.
  * `multicrop.py` -- deterministic multi-crop re-scoring, run ONLY for hard cases (high
    disagreement / near the boundary / high transformation severity), preserving the cost-aware
    cascade.
  * `model.py`     -- `ConfidenceModel` (a calibrated linear model predicting P(prediction is
    correct) from the signals -- confidence is NOT the raw probability) and `AbstentionPolicy`
    (score dead-zone + confidence gate, thresholds TUNED on validation for a target FPR, never
    hard-coded).
  * `tune.py` / `evaluate.py` -- fit the policy on validation; report coverage, accuracy @
    coverage, FPR, TPR, abstention rate, and the risk-coverage curve on test.
"""

from __future__ import annotations

from src.confidence.model import (
    CONFIDENCE_BUCKETS,
    AbstentionPolicy,
    ConfidenceModel,
    load_policy,
)

__all__ = ["CONFIDENCE_BUCKETS", "AbstentionPolicy", "ConfidenceModel", "load_policy"]
