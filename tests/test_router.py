"""Checks for the adaptive router (Feature 3)."""

from __future__ import annotations

import numpy as np

from src.router.features import (
    Evidence,
    RouterConfig,
    build_matrix,
    build_vector,
    feature_names,
)
from src.router.model import AdaptiveRouter


def _ev(t1=0.9, t2=0.2, t3=0.8, deg=0.3, profile=None, **kw):
    return Evidence(tier1=t1, tier2=t2, tier3=t3, degradation=deg,
                    profile=profile or {k: 0.1 for k in
                        ["jpeg_compression", "resize_degradation", "blur", "noise",
                         "sharpening", "crop", "screenshot_like", "overall_degradation"]}, **kw)


CONFIGS = [
    RouterConfig(use_disagreement=False, use_profile=False),
    RouterConfig(use_disagreement=True, use_profile=False),
    RouterConfig(use_disagreement=True, use_profile=True),
    RouterConfig(use_disagreement=True, use_profile=True, use_uncertainty=True),
]


def test_vector_length_matches_feature_names():
    for cfg in CONFIGS:
        v = build_vector(_ev(), cfg)
        assert v.shape == (len(feature_names(cfg)),), cfg.tag()
        assert np.isfinite(v).all()


def test_missing_tier_sets_indicator_and_neutral_fill():
    cfg = RouterConfig(use_disagreement=True, use_profile=False)
    names = feature_names(cfg)
    v = build_vector(Evidence(tier1=None, tier2=0.3, tier3=0.7, degradation=0.1), cfg)
    d = dict(zip(names, v))
    assert d["tier1_missing"] == 1.0
    assert d["tier1"] == 0.5           # neutral fill
    assert d["n_tiers_available"] == 2.0


def test_disagreement_high_vs_low():
    cfg = RouterConfig(use_disagreement=True, use_profile=False)
    names = feature_names(cfg)
    hi = dict(zip(names, build_vector(_ev(t1=0.95, t2=0.1, t3=0.9), cfg)))
    lo = dict(zip(names, build_vector(_ev(t1=0.9, t2=0.87, t3=0.94), cfg)))
    assert hi["dis_std"] > lo["dis_std"]
    assert hi["dis_range"] > lo["dis_range"]


def test_router_fit_predict_and_oof():
    rng = np.random.default_rng(0)
    ev, y = [], []
    for _ in range(200):
        label = int(rng.random() < 0.5)
        base = 0.7 if label else 0.3
        ev.append(_ev(t1=np.clip(base + rng.normal(0, 0.15), 0, 1),
                      t2=np.clip(base + rng.normal(0, 0.15), 0, 1),
                      t3=np.clip(base + rng.normal(0, 0.15), 0, 1),
                      deg=rng.random()))
        y.append(label)
    cfg = RouterConfig(use_disagreement=True, use_profile=True, model="logreg")
    router = AdaptiveRouter(cfg)
    oof = router.oof_fit_predict(ev, y, n_splits=4)
    assert np.isfinite(oof).all()
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, oof) > 0.8   # signal is learnable
    p = router.predict_proba(ev[:10])
    assert p.shape == (10,) and ((p >= 0) & (p <= 1)).all()


def test_coefficients_exposed_for_logreg():
    rng = np.random.default_rng(1)
    ev = [_ev(t1=rng.random(), t2=rng.random(), t3=rng.random()) for _ in range(80)]
    y = [int(rng.random() < 0.5) for _ in range(80)]
    router = AdaptiveRouter(RouterConfig(model="logreg", calibrate=False)).fit(ev, y)
    coefs = router.coefficients()
    assert coefs is not None
    assert set(coefs) == set(router.feature_names_)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
