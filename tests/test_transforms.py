"""Fast, dependency-light checks for the transform library (Feature 1).

Runnable either with pytest (`pytest tests/`) or standalone (`python -m tests.test_transforms`).
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

from src.transforms import build_conditions, random_compound_chain, screenshot_sim
from src.transforms.registry import DEFAULT_SEED


def _sample_image(size=(96, 96), seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (*size[::-1], 3), dtype=np.uint8))


def test_every_condition_returns_rgb_image():
    img = _sample_image()
    for cond in build_conditions(seed=DEFAULT_SEED):
        out = cond(img)
        assert isinstance(out, Image.Image), cond.name
        assert out.mode == "RGB", cond.name
        assert min(out.size) >= 1, cond.name


def test_conditions_are_deterministic():
    img = _sample_image()
    for cond in build_conditions(seed=DEFAULT_SEED):
        a = np.asarray(cond(img).convert("RGB"))
        b = np.asarray(cond(img).convert("RGB"))
        assert np.array_equal(a, b), f"{cond.name} is not deterministic"


def test_random_compound_chain_reproducible_by_seed():
    img = _sample_image()
    c1 = random_compound_chain(seed=123, n_ops=3)
    c2 = random_compound_chain(seed=123, n_ops=3)
    c3 = random_compound_chain(seed=124, n_ops=3)
    assert c1.labels == c2.labels
    assert np.array_equal(np.asarray(c1(img)), np.asarray(c2(img)))
    assert c1.labels != c3.labels or not np.array_equal(np.asarray(c1(img)), np.asarray(c3(img)))


def test_clean_is_identity_pixels():
    img = _sample_image().convert("RGB")
    clean = next(c for c in build_conditions() if c.name == "clean")
    assert np.array_equal(np.asarray(clean(img)), np.asarray(img))


def test_jpeg_actually_compresses():
    img = _sample_image((128, 128))
    cond = next(c for c in build_conditions() if c.name == "jpeg_q30")
    buf_before = io.BytesIO(); img.save(buf_before, format="PNG")
    out = cond(img)
    assert np.asarray(out).shape == np.asarray(img.convert("RGB")).shape
    # q30 on noise should differ substantially from the source
    assert np.abs(np.asarray(out).astype(int) - np.asarray(img.convert("RGB")).astype(int)).mean() > 1.0


def test_screenshot_sim_changes_size_and_recompresses():
    img = _sample_image((200, 200))
    out = screenshot_sim(img, scale=0.85, border=10, jpeg_quality=70)
    assert out.mode == "RGB"
    assert out.size != img.size


def test_registry_metadata_wellformed():
    conds = build_conditions()
    names = [c.name for c in conds]
    assert len(names) == len(set(names)), "duplicate condition names"
    fams = {c.family for c in conds}
    assert {"compression", "resolution", "blur", "noise", "crop", "color", "screenshot",
            "compound", "random_compound"} <= fams
    for c in conds:
        assert 0.0 <= c.severity <= 1.5, (c.name, c.severity)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
