"""Profiler feature extraction (Feature 2).

`profiler_features(img)` returns a single float vector = Tier 2's existing forensic vector
(`src.frequency.features.extract_all_features`) concatenated with a small set of extra
transformation-sensitive indicators that Tier 2 does not already expose as scalars:

  * resolution / aspect info                 -> resize & crop
  * Laplacian variance, high-freq energy     -> blur vs sharpening
  * 8-px blockiness in the pixel domain      -> JPEG compression (complements Tier 2's DCT comb)
  * FFT radial-slope + mid/high band ratio   -> resampling / resize artifacts
  * residual-noise std (already denoised)    -> additive noise
  * colour count / palette flatness / border uniformity -> screenshot-like

At inference time Tier 2's forensic dict is already computed, so pass it in as `forensic=` to
avoid recomputing the expensive part (NL-means denoise + windowed FFT).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from src.frequency.features import extract_all_features

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required for src/transformation/features.py") from exc


EXTRA_FEATURE_NAMES = [
    "log_area", "aspect_ratio", "min_side", "even_dims",
    "laplacian_var", "grad_mean", "grad_p95",
    "highfreq_ratio", "midfreq_ratio", "fft_radial_slope",
    "blockiness_8px", "blockiness_ratio",
    "residual_std", "residual_mad",
    "unique_color_frac", "top_color_frac", "border_uniformity",
    "sat_mean", "sat_std",
]


def _fft_bands(gray: np.ndarray) -> tuple[float, float, float]:
    """Global FFT radial power split into low / mid / high bands + a log-log radial slope
    (steep negative slope = blurred; flattened / bumped high band = sharpened or resampled)."""
    f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    power = np.abs(f) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = r.max() + 1e-8
    rn = r / r_max
    total = power.sum() + 1e-8
    low = power[rn < 0.15].sum() / total
    mid = power[(rn >= 0.15) & (rn < 0.5)].sum() / total
    high = power[rn >= 0.5].sum() / total

    n_bins = 12
    bin_idx = np.clip((rn * n_bins).astype(int), 0, n_bins - 1)
    prof = np.array([power[bin_idx == i].mean() if np.any(bin_idx == i) else 0.0 for i in range(n_bins)])
    prof = np.log1p(prof[1:])  # drop DC
    xs = np.log1p(np.arange(1, n_bins))
    slope = float(np.polyfit(xs, prof, 1)[0]) if np.ptp(prof) > 0 else 0.0
    return float(high), float(mid), slope


def _blockiness(gray: np.ndarray) -> tuple[float, float]:
    """Mean absolute first difference across columns/rows that sit ON an 8-px boundary vs. those
    that do not. A JPEG-compressed image has stronger discontinuities on the grid."""
    dh = np.abs(np.diff(gray, axis=1))
    dv = np.abs(np.diff(gray, axis=0))
    col_on = dh[:, 7::8].mean() if dh.shape[1] > 8 else dh.mean()
    col_off = dh[:, [i for i in range(dh.shape[1]) if i % 8 != 7]].mean() if dh.shape[1] > 8 else dh.mean()
    row_on = dv[7::8, :].mean() if dv.shape[0] > 8 else dv.mean()
    row_off = dv[[i for i in range(dv.shape[0]) if i % 8 != 7], :].mean() if dv.shape[0] > 8 else dv.mean()
    on = 0.5 * (col_on + row_on)
    off = 0.5 * (col_off + row_off) + 1e-8
    return float(on), float(on / off)


def _color_stats(img: Image.Image) -> tuple[float, float, float]:
    small = img.convert("RGB").resize((128, 128))
    arr = np.asarray(small).reshape(-1, 3)
    q = (arr // 8).astype(np.int32)
    keys = q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2]
    _, counts = np.unique(keys, return_counts=True)
    unique_frac = len(counts) / len(keys)
    top_frac = counts.max() / len(keys)
    hsv = np.asarray(small.convert("HSV"))
    return float(unique_frac), float(top_frac), float(hsv[..., 1].mean() / 255.0)


def _border_uniformity(gray: np.ndarray, band: int = 4) -> float:
    if min(gray.shape) <= 2 * band + 2:
        return 0.0
    frame = np.concatenate([
        gray[:band, :].ravel(), gray[-band:, :].ravel(),
        gray[:, :band].ravel(), gray[:, -band:].ravel(),
    ])
    return float(1.0 / (1.0 + frame.std()))


def extra_features(img: Image.Image, forensic: dict | None = None) -> np.ndarray:
    img = img.convert("RGB")
    w, h = img.size
    gray = np.asarray(img.convert("L"), dtype=np.float32)

    lap = cv2.Laplacian(gray, cv2.CV_32F)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    high, mid, slope = _fft_bands(gray)
    block_on, block_ratio = _blockiness(gray)

    if forensic is None:
        forensic = extract_all_features(img)
    residual_std = float(np.sqrt(max(forensic["prnu"][0], 0.0)))
    residual_mad = float(np.mean(np.abs(forensic["autocorr"])))

    uniq, topc, sat_mean = _color_stats(img)
    hsv = np.asarray(img.convert("HSV"))

    feats = [
        np.log1p(w * h), w / (h + 1e-8), float(min(w, h)),
        float((w % 2 == 0) and (h % 2 == 0)),
        float(lap.var()), float(grad.mean()), float(np.percentile(grad, 95)),
        high, mid, slope,
        block_on, block_ratio,
        residual_std, residual_mad,
        uniq, topc, _border_uniformity(gray),
        sat_mean, float(hsv[..., 1].std() / 255.0),
    ]
    return np.asarray(feats, dtype=np.float32)


def profiler_features(img: Image.Image, forensic: dict | None = None) -> np.ndarray:
    """Full profiler feature vector: Tier 2 forensic vector ++ `extra_features`."""
    img = img.convert("RGB")
    if forensic is None:
        forensic = extract_all_features(img)
    vec = np.concatenate([forensic["vector"], extra_features(img, forensic)]).astype(np.float32)
    # Tier 2's kurtosis/skew features are NaN on degenerate (near-constant-residual) images;
    # map any non-finite value to 0 so the profiler's trees see a stable, deterministic vector.
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def feature_names() -> list[str]:
    from src.frequency.features import FEATURE_VECTOR_DIM

    return [f"forensic_{i}" for i in range(FEATURE_VECTOR_DIM)] + EXTRA_FEATURE_NAMES


PROFILER_FEATURE_DIM = len(feature_names())
