"""Compound transformation chains (Feature 1).

Two kinds:
  1. Named, fixed chains (e.g. ``resize -> crop -> jpeg``) enumerated in ``registry.py``.
  2. Deterministic *random* compound chains: given a fixed seed, sample an ordered subset of
     transform families and their severities. Same seed => byte-identical output, so the
     benchmark stays reproducible while still probing transform combinations no fixed list
     would enumerate.

``Chain`` is a dataclass (not a closure) so it stays picklable for ``DataLoader`` workers and
for caching, matching the rationale in ``src/augmentations.py``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

from PIL import Image

from src.transforms import catalog
from src.transforms.screenshot import screenshot_sim

Transform = Callable[[Image.Image], Image.Image]


@dataclass
class Chain:
    """An ordered, picklable sequence of transforms applied left-to-right."""

    fns: tuple[Transform, ...]
    labels: tuple[str, ...] = field(default=())

    def __call__(self, img: Image.Image) -> Image.Image:
        for fn in self.fns:
            img = fn(img)
        return img

    def describe(self) -> str:
        return " -> ".join(self.labels) if self.labels else f"chain[{len(self.fns)}]"


def chain(*pairs: tuple[str, Transform]) -> Chain:
    """`chain(("crop80", fn), ("jpeg50", fn))` -> Chain with human-readable labels."""
    labels, fns = zip(*pairs) if pairs else ((), ())
    return Chain(tuple(fns), tuple(labels))


# ---------------------------------------------------------------------------
# Named fixed chains (the spec's explicit list)
# ---------------------------------------------------------------------------

NAMED_CHAINS: dict[str, Chain] = {
    "resize75_jpeg70": chain(
        ("resize75", partial(catalog.resize_scale, scale=0.75)),
        ("jpeg70", partial(catalog.jpeg_compress, quality=70)),
    ),
    "crop80_jpeg70": chain(
        ("crop80", partial(catalog.center_crop, frac=0.8)),
        ("jpeg70", partial(catalog.jpeg_compress, quality=70)),
    ),
    "blur1_jpeg70": chain(
        ("blur1.0", partial(catalog.gaussian_blur, sigma=1.0)),
        ("jpeg70", partial(catalog.jpeg_compress, quality=70)),
    ),
    "resize75_crop80_jpeg50": chain(
        ("resize75", partial(catalog.resize_scale, scale=0.75)),
        ("crop80", partial(catalog.center_crop, frac=0.8)),
        ("jpeg50", partial(catalog.jpeg_compress, quality=50)),
    ),
    "resize75_blur1_jpeg50": chain(
        ("resize75", partial(catalog.resize_scale, scale=0.75)),
        ("blur1.0", partial(catalog.gaussian_blur, sigma=1.0)),
        ("jpeg50", partial(catalog.jpeg_compress, quality=50)),
    ),
    "crop80_resize50_jpeg50_noise": chain(
        ("crop80", partial(catalog.center_crop, frac=0.8)),
        ("resize50", partial(catalog.resize_scale, scale=0.5)),
        ("jpeg50", partial(catalog.jpeg_compress, quality=50)),
        ("noise0.02", partial(catalog.gaussian_noise, sigma=0.02, seed=0)),
    ),
    "screenshot_jpeg50": chain(
        ("screenshot", partial(screenshot_sim, scale=0.85, border=8, jpeg_quality=85)),
        ("jpeg50", partial(catalog.jpeg_compress, quality=50)),
    ),
}


# ---------------------------------------------------------------------------
# Deterministic random compound chains
# ---------------------------------------------------------------------------

# (family, factory(rng) -> (label, transform)); severities are drawn from realistic ranges.
def _op_jpeg(rng: random.Random):
    q = rng.choice([85, 70, 60, 50, 40])
    return f"jpeg{q}", partial(catalog.jpeg_compress, quality=q)


def _op_resize(rng: random.Random):
    s = rng.choice([0.9, 0.75, 0.6, 0.5])
    return f"resize{int(s * 100)}", partial(catalog.resize_scale, scale=s)


def _op_blur(rng: random.Random):
    s = rng.choice([0.5, 1.0, 1.5, 2.0])
    return f"blur{s}", partial(catalog.gaussian_blur, sigma=s)


def _op_noise(rng: random.Random):
    s = rng.choice([0.01, 0.02, 0.03, 0.05])
    return f"noise{s}", partial(catalog.gaussian_noise, sigma=s, seed=rng.randint(0, 2**31 - 1))


def _op_crop(rng: random.Random):
    f = rng.choice([0.7, 0.8, 0.9])
    seed = rng.randint(0, 2**31 - 1)
    return f"rcrop{int(f * 100)}", partial(catalog.random_crop, frac=f, seed=seed)


def _op_sharpen(rng: random.Random):
    p = rng.choice([120, 150, 200])
    return f"sharp{p}", partial(catalog.unsharp_mask, percent=p)


def _op_color(rng: random.Random):
    fn = rng.choice([catalog.adjust_brightness, catalog.adjust_contrast, catalog.adjust_saturation])
    factor = round(rng.uniform(0.8, 1.2), 3)
    return f"{fn.__name__.replace('adjust_', '')}{factor}", partial(fn, factor=factor)


_OP_POOL = [_op_resize, _op_crop, _op_blur, _op_noise, _op_sharpen, _op_color, _op_jpeg]


def random_compound_chain(seed: int, n_ops: int = 3, force_jpeg_last: bool = True) -> Chain:
    """Build a reproducible compound chain of `n_ops` operations from `seed`.

    force_jpeg_last -- append a JPEG recompression as the final step (the common real case: a
    transformed image is re-encoded on upload), unless one was already sampled last.
    """
    rng = random.Random(seed)
    ops = rng.sample(_OP_POOL, k=min(n_ops, len(_OP_POOL)))
    pairs = [op(rng) for op in ops]
    if force_jpeg_last and not pairs[-1][0].startswith("jpeg"):
        pairs.append(_op_jpeg(rng))
    return chain(*pairs)
