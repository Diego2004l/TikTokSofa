"""Checks for confidence + abstention (Feature 4)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.confidence.model import AbstentionPolicy, ConfidenceModel, bucket
from src.confidence.multicrop import make_crops, multi_crop_eval
from src.confidence.signals import (
    SIGNAL_NAMES,
    confidence_signals,
    disagreement_stats,
    is_hard_case,
)
from src.router.features import Evidence


def _ev(t1=0.9, t2=0.85, t3=0.92, deg=0.1, profile_ov=0.1):
    return Evidence(tier1=t1, tier2=t2, tier3=t3, degradation=deg,
                    profile={"overall_degradation": profile_ov, "screenshot_like": 0.0})


def test_signal_vector_shape():
    vec, detail = confidence_signals(_ev(), 0.88)
    assert vec.shape == (len(SIGNAL_NAMES),)
    assert set(detail) == set(SIGNAL_NAMES)
    assert np.isfinite(vec).all()


def test_disagreement_stats():
    d = disagreement_stats([0.9, 0.1, 0.8])
    assert abs(d["range"] - 0.8) < 1e-9
    assert d["max_pair"] >= d["std"]
    assert disagreement_stats([0.5])["std"] == 0.0


def test_is_hard_case_triggers():
    assert is_hard_case(0.52, _ev())                       # near boundary
    assert is_hard_case(0.95, _ev(t1=0.95, t2=0.1, t3=0.9))  # disagreement
    assert is_hard_case(0.95, _ev(profile_ov=0.8))         # high severity
    assert not is_hard_case(0.95, _ev())                   # confident + agreeing + clean


def test_multicrop_consistency_bounds_and_direction():
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8))
    assert len(make_crops(img, 5)) == 5
    stable = multi_crop_eval(lambda cs: [0.9] * len(cs), img, 5)
    swingy = multi_crop_eval(lambda cs: [0.1, 0.9, 0.2, 0.8, 0.5][: len(cs)], img, 5)
    assert stable["consistency"] == 1.0
    assert 0.0 <= swingy["consistency"] < stable["consistency"]


def test_bucket_thresholds():
    assert bucket(0.9, 0.4, 0.75) == "HIGH"
    assert bucket(0.5, 0.4, 0.75) == "MEDIUM"
    assert bucket(0.2, 0.4, 0.75) == "LOW"


def test_confidence_model_learns_correctness():
    rng = np.random.default_rng(0)
    # margin (col 0) correlates with correctness
    n = 400
    margin = rng.random(n)
    correct = (rng.random(n) < 0.3 + 0.6 * margin).astype(int)
    sig = np.column_stack([margin] + [rng.random(n) for _ in range(len(SIGNAL_NAMES) - 1)])
    cm = ConfidenceModel().fit(sig, correct)
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(correct, cm.predict_confidence(sig)) > 0.7


def test_abstention_policy_respects_target_fpr():
    rng = np.random.default_rng(1)
    n = 600
    y = (rng.random(n) < 0.5).astype(int)
    scores = np.clip(np.where(y == 1, rng.normal(0.65, 0.2, n), rng.normal(0.35, 0.2, n)), 0, 1)
    conf = np.clip(rng.random(n) * 0.5 + 2 * np.abs(scores - 0.5), 0, 1)
    pol = AbstentionPolicy().fit(scores, y, conf, target_fpr=0.05, min_selective_accuracy=0.8)
    answered = np.array([not pol.decide(s, c)["abstained"] for s, c in zip(scores, conf)])
    assert answered.sum() > 0
    pred = (scores[answered] >= pol.params.hi).astype(int)
    yy = y[answered]
    fp = ((pred == 1) & (yy == 0)).sum(); tn = ((pred == 0) & (yy == 0)).sum()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    # feasible fit should hit roughly the target on its own tuning data
    assert fpr <= 0.15


def test_decide_unknown_when_low_confidence():
    pol = AbstentionPolicy()
    pol.params.conf_min = 0.6
    d = pol.decide(0.95, 0.2)
    assert d["label"] == "UNKNOWN" and d["abstained"] is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
