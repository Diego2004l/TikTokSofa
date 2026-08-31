"""Parametrised transform primitives (Feature 1).

Every function takes a PIL.Image and keyword parameters and returns a PIL.Image (RGB). None of
them read randomness from the global RNG except where a `seed` is passed explicitly, so the
whole benchmark is reproducible.

The primitives that already exist in `src/augmentations.py` (JPEG, Gaussian blur, Gaussian
noise, down/up resample, center crop, identity) are imported rather than reimplemented so the
benchmark and the training-time augmenter can never silently diverge.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Re-used verbatim from the training-time augmenter.
from src.augmentations import (  # noqa: F401  (re-exported on purpose)
    center_crop,
    gaussian_blur,
    identity,
    jpeg_compress,
    resize_down_up,
)


def gaussian_noise(img: Image.Image, sigma: float, seed: int | None = None) -> Image.Image:
    """Additive Gaussian noise. Unlike `src.augmentations.gaussian_noise` (which draws from the
    global RNG, fine for training-time augmentation) this is seedable, so the robustness
    benchmark is byte-reproducible. `seed=None` still gives fresh noise each call."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0.0, 1.0)
    return Image.fromarray((noisy * 255).astype(np.uint8))

# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resize_scale(img: Image.Image, scale: float, resample: int = Image.BICUBIC) -> Image.Image:
    """Resize to `scale` of the original width/height and leave it there (the downstream model
    resizes to its own input size). Distinct from `resize_down_up`, which restores the original
    resolution and therefore only tests interpolation loss, not the scale change itself."""
    w, h = img.size
    new = (max(1, round(w * scale)), max(1, round(h * scale)))
    return img.convert("RGB").resize(new, resample)


# ---------------------------------------------------------------------------
# Blur
# ---------------------------------------------------------------------------


def motion_blur(img: Image.Image, kernel_size: int = 9, angle_deg: float = 0.0) -> Image.Image:
    """Linear motion blur via a directional kernel. `angle_deg` is measured counter-clockwise
    from the horizontal. Deterministic given its arguments."""
    import cv2

    kernel_size = max(3, int(kernel_size) | 1)  # force odd, >= 3
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[kernel_size // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((kernel_size / 2 - 0.5, kernel_size / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (kernel_size, kernel_size))
    total = kernel.sum()
    if total > 1e-8:
        kernel /= total
    arr = np.asarray(img.convert("RGB"))
    blurred = cv2.filter2D(arr, -1, kernel)
    return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------


def salt_pepper_noise(img: Image.Image, amount: float = 0.02, seed: int | None = None) -> Image.Image:
    """Replace a fraction `amount` of pixels with pure black or white, split evenly. Seeded so
    the corrupted pixel mask is reproducible across benchmark runs."""
    rng = np.random.default_rng(seed)
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]
    n = int(amount * h * w)
    if n == 0:
        return Image.fromarray(arr)
    ys = rng.integers(0, h, size=n)
    xs = rng.integers(0, w, size=n)
    vals = rng.integers(0, 2, size=n) * 255
    arr[ys, xs, :] = vals[:, None]
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------


def unsharp_mask(img: Image.Image, radius: float = 2.0, percent: int = 150, threshold: int = 3) -> Image.Image:
    """PIL's unsharp mask. `percent` ~150 is a mild sharpen, ~300 is aggressive/haloed."""
    return img.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
    )


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def random_crop(img: Image.Image, frac: float, seed: int | None = None) -> Image.Image:
    """Crop a `frac` x `frac` window at a seeded random position, then resize back to the
    original resolution so the downstream model sees a consistent input size."""
    rng = random.Random(seed)
    w, h = img.size
    cw, ch = max(1, int(w * frac)), max(1, int(h * frac))
    left = rng.randint(0, max(0, w - cw))
    top = rng.randint(0, max(0, h - ch))
    cropped = img.convert("RGB").crop((left, top, left + cw, top + ch))
    return cropped.resize((w, h), Image.BICUBIC)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(img.convert("RGB")).enhance(factor)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Contrast(img.convert("RGB")).enhance(factor)


def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Color(img.convert("RGB")).enhance(factor)


def adjust_hue(img: Image.Image, shift_deg: float) -> Image.Image:
    """Rotate hue by `shift_deg` degrees (-180..180). Kept mild in the registry (<= ~15 deg):
    a large hue rotation is not a realistic distribution shift for uploaded photos."""
    hsv = np.asarray(img.convert("HSV")).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(shift_deg / 360.0 * 255)) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")
