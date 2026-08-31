"""Checks for the transformation profiler (Feature 2)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.transformation.features import (
    PROFILER_FEATURE_DIM,
    feature_names,
    profiler_features,
)
from src.transformation.model import PROFILE_LABELS, TransformationProfiler
from src.transformation.synthesize import (
    SEVERITY_GRID,
    SynthConfig,
    overall_severity,
    synth_labels_for_image,
)
import random


def _img(seed=0, size=(80, 80)):
    rng = np.random.default_rng(seed)
    # low-freq base + texture so forensic features are non-degenerate
    base = rng.integers(60, 200, (4, 4, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize(size, Image.BICUBIC)
    noise = rng.integers(-12, 12, (*size[::-1], 3))
    return Image.fromarray(np.clip(np.asarray(img).astype(int) + noise, 0, 255).astype(np.uint8))


def test_feature_vector_shape_and_finite():
    v = profiler_features(_img())
    assert v.shape == (PROFILER_FEATURE_DIM,)
    assert len(feature_names()) == PROFILER_FEATURE_DIM
    assert np.isfinite(v).all()


def test_feature_extraction_deterministic():
    img = _img(1)
    assert np.allclose(profiler_features(img), profiler_features(img))


def test_synth_labels_match_applied_families():
    img = _img(2)
    rng = random.Random(0)
    cfg = SynthConfig(variants_per_image=20, max_ops=3, clean_fraction=0.0, grid="train")
    rows = synth_labels_for_image(img, cfg, rng)
    assert len(rows) == 20
    for _, binary, sev in rows:
        assert binary.shape == (len(PROFILE_LABELS),)
        assert set(np.unique(binary)) <= {0.0, 1.0}
        # severity is only non-zero where the label fired
        assert np.all((sev > 0) <= (binary > 0))
        assert 1 <= binary.sum() <= 4  # up to 3 sampled + possible screenshot->jpeg


def test_clean_variant_has_no_labels():
    img = _img(3)
    cfg = SynthConfig(variants_per_image=40, max_ops=2, clean_fraction=1.0, grid="train")
    rows = synth_labels_for_image(img, cfg, random.Random(0))
    assert all(binary.sum() == 0 for _, binary, _ in rows)


def test_severity_grids_disjoint():
    for label, grids in SEVERITY_GRID.items():
        train_params = {p for _, p in grids["train"]}
        eval_params = {p for _, p in grids["eval"]}
        assert not (train_params & eval_params), label


def test_overall_severity_monotone():
    assert overall_severity(np.zeros(7)) == 0.0
    one = np.zeros(7); one[0] = 0.5
    two = np.zeros(7); two[0] = 0.5; two[1] = 0.5
    assert overall_severity(two) > overall_severity(one) > 0.0
    assert overall_severity(np.ones(7)) <= 1.0


def test_profiler_fit_predict_roundtrip():
    rng = np.random.default_rng(0)
    imgs = [_img(i) for i in range(8)]
    X, Y, S = [], [], []
    for k, img in enumerate(imgs):
        for _, b, s in synth_labels_for_image(
            img, SynthConfig(variants_per_image=12, max_ops=2, grid="train"), random.Random(k)
        ):
            X.append(profiler_features(img if b.sum() == 0 else img))  # features not the point here
            Y.append(b); S.append(s)
    X = np.stack(X); Y = np.stack(Y); S = np.stack(S)
    prof = TransformationProfiler(calibrate=False).fit(X, Y, S)
    out = prof.predict(imgs[0])
    assert set(out) == set(PROFILE_LABELS) | {"overall_degradation"}
    assert all(0.0 <= v <= 1.0 for v in out.values())


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
