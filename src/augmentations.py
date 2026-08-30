"""Isolated and compound image transforms used for training-time augmentation and for the
robustness evaluation protocol (spec section 5).

Everything here operates on PIL.Image (RGB) and returns PIL.Image (RGB), so the same functions
back both `train.py` (as augmentation) and `eval/robustness.py` (as a fixed perturbation to
score against). This is deliberate: robustness numbers are only meaningful if the eval-time
transform is the same code path the model was trained to be resistant to.

Shortcut-learning safeguard (spec section 6): during training, `sample_symmetric_transform` is
applied identically to real and fake images. If e.g. only fake images were JPEG-compressed, the
model could learn "JPEG artifacts => real" instead of learning real generative artifacts. Never
call any of the per-class dataset code with different transform distributions per class.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance

Transform = Callable[[Image.Image], Image.Image]


def jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def gaussian_blur(img: Image.Image, sigma: float) -> Image.Image:
    import cv2

    arr = np.asarray(img.convert("RGB"))
    # cv2 ksize must be odd and derived from sigma; 0 lets cv2 pick it from sigma.
    blurred = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return Image.fromarray(blurred)


def resize_down_up(img: Image.Image, scale: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BICUBIC)
    return small.resize((w, h), Image.BICUBIC)


def gaussian_noise(img: Image.Image, sigma: float) -> Image.Image:
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    noise = np.random.normal(0.0, sigma, arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))


def color_jitter(img: Image.Image, pct: float = 0.2) -> Image.Image:
    out = img.convert("RGB")
    for enhancer_cls in (ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color):
        factor = 1.0 + random.uniform(-pct, pct)
        out = enhancer_cls(out).enhance(factor)
    return out


def center_crop(img: Image.Image, frac: float) -> Image.Image:
    """Crop the central `frac` fraction of width/height, then resize back to the original
    size so downstream models see a consistent input resolution."""
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    left, top = (w - cw) // 2, (h - ch) // 2
    cropped = img.crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BICUBIC)


def identity(img: Image.Image) -> Image.Image:
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# Isolated transforms (robustness table, spec section 5)
# ---------------------------------------------------------------------------

ISOLATED_TRANSFORMS: dict[str, Transform] = {
    "clean": identity,
    "jpeg_q90": partial(jpeg_compress, quality=90),
    "jpeg_q70": partial(jpeg_compress, quality=70),
    "jpeg_q50": partial(jpeg_compress, quality=50),
    "jpeg_q30": partial(jpeg_compress, quality=30),
    "blur_s0.5": partial(gaussian_blur, sigma=0.5),
    "blur_s1.0": partial(gaussian_blur, sigma=1.0),
    "blur_s2.0": partial(gaussian_blur, sigma=2.0),
    "resize_0.5x": partial(resize_down_up, scale=0.5),
    "resize_0.25x": partial(resize_down_up, scale=0.25),
    "noise_s0.02": partial(gaussian_noise, sigma=0.02),
    "noise_s0.05": partial(gaussian_noise, sigma=0.05),
    "noise_s0.10": partial(gaussian_noise, sigma=0.10),
    "color_jitter_pm20pct": partial(color_jitter, pct=0.2),
    "center_crop_80pct": partial(center_crop, frac=0.8),
}


@dataclass
class Chain:
    """A picklable ordered sequence of transforms (a plain closure isn't picklable, which
    breaks `DataLoader(num_workers>0)` under spawn-based multiprocessing, e.g. on macOS)."""

    fns: tuple[Transform, ...]

    def __call__(self, img: Image.Image) -> Image.Image:
        for fn in self.fns:
            img = fn(img)
        return img


def _chain(*fns: Transform) -> Transform:
    return Chain(fns)


# ---------------------------------------------------------------------------
# Compound chains (differentiator, spec section 5) — applied in the listed order.
# ---------------------------------------------------------------------------

COMPOUND_CHAINS: dict[str, Transform] = {
    "crop80_resize_jpeg50": _chain(partial(center_crop, frac=0.8), partial(jpeg_compress, quality=50)),
    "crop80_resize_jpeg30": _chain(partial(center_crop, frac=0.8), partial(jpeg_compress, quality=30)),
    "blur2_jpeg30": _chain(partial(gaussian_blur, sigma=2.0), partial(jpeg_compress, quality=30)),
}


def all_conditions() -> dict[str, Transform]:
    """All isolated + compound conditions, keyed by condition name — used by both the
    training-time sampler and the robustness eval script so they stay in sync."""
    merged = dict(ISOLATED_TRANSFORMS)
    merged.update(COMPOUND_CHAINS)
    return merged


@dataclass
class SymmetricAugmenter:
    """Samples one condition uniformly (including a compound chain, per spec section 4's Tier 1
    training data) and applies it to an image, regardless of that image's real/fake label.

    Use ONE instance's `__call__` for both classes in a training batch — never construct
    separate augmenters per class — so the transform distribution stays identical across
    real and fake, per the shortcut-learning safeguard in spec section 6.
    """

    include_clean: bool = True
    clean_prob: float = 0.2
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self._rng = self.rng or random.Random()
        self._conditions = list(all_conditions().items())
        self._non_clean = [(n, f) for n, f in self._conditions if n != "clean"]

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.include_clean and self._rng.random() < self.clean_prob:
            return identity(img)
        _, fn = self._rng.choice(self._non_clean)
        return fn(img)


if __name__ == "__main__":
    # Reproducibility demo: apply every condition to a sample image and save thumbnails.
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Preview all isolated + compound transforms on one image.")
    parser.add_argument("image", help="path to a sample image")
    parser.add_argument("--out-dir", default="outputs/figures/augmentation_preview")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    src = Image.open(args.image).convert("RGB")
    for name, fn in all_conditions().items():
        fn(src).save(os.path.join(args.out_dir, f"{name}.png"))
    print(f"Wrote {len(all_conditions())} previews to {args.out_dir}")
