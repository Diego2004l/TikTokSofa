"""Tier 2 — hand-built forensic/frequency features (spec section 4, Tier 2).

No pretrained model is used anywhere in this file. Every function below takes a PIL.Image and
returns either a scalar or a small fixed-length numpy vector. `extract_all_features` concatenates
all of them into one feature vector for the RandomForest/SVM classifier, and also exposes a
`degradation_score` — the block-grid + double-JPEG signal strength used by `fusion.py` to
estimate "how transformed is this image" for adaptive reweighting.

Design notes (see README "Related Work" for what's standard vs. novel here):
- FFT and autocorrelation are computed patch-wise with a Hann window rather than on the whole
  image, which makes them far more robust to crops and resizes than a single global spectrum
  (a global FFT's crop-edge leakage and shifted scale are exactly what most simple forensic
  detectors get wrong under the "compound transform" conditions in spec section 5).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.stats import kurtosis, skew

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError("opencv-python is required for src/frequency/features.py") from exc


def _to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.float32)


def _maybe_downscale(img: Image.Image, max_dim: int | None) -> Image.Image:
    """Optional speed knob: cap the longer side at `max_dim` before feature extraction.
    Feature cost is roughly O(pixels) (windowed FFT patch count, NL-means denoise), so going
    from 1024px to ~384px is a large speedup. Trade-off: the 8x8 block-grid / double-JPEG
    signals weaken once the native JPEG grid is resampled away, so leave this unset (None) for
    the final run and only use it for fast iteration. Whatever value is used here MUST also be
    passed to inference (src/infer.py --max-image-dim) so train/serve feature distributions match.
    """
    if max_dim is None:
        return img
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def _radial_profile(power: np.ndarray, n_bins: int = 8) -> np.ndarray:
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = r.max() + 1e-8
    bin_idx = np.clip((r / r_max * n_bins).astype(int), 0, n_bins - 1)
    profile = np.array([power[bin_idx == i].mean() if np.any(bin_idx == i) else 0.0 for i in range(n_bins)])
    return profile


def windowed_fft_features(gray: np.ndarray, patch_sizes=(32, 64, 128), n_bins: int = 8) -> np.ndarray:
    """Tile into patches per scale, apply a Hann window (avoids crop-edge spectral leakage),
    FFT, radially-average the power spectrum, then mean/max-pool across patches."""
    h, w = gray.shape
    feats = []
    for size in patch_sizes:
        if h < size or w < size:
            feats.append(np.zeros(n_bins * 2, dtype=np.float32))
            continue
        window_1d = np.hanning(size)
        window_2d = np.outer(window_1d, window_1d)
        profiles = []
        for y0 in range(0, h - size + 1, size):
            for x0 in range(0, w - size + 1, size):
                patch = gray[y0 : y0 + size, x0 : x0 + size] * window_2d
                spectrum = np.fft.fftshift(np.fft.fft2(patch))
                power = np.log1p(np.abs(spectrum) ** 2)
                profiles.append(_radial_profile(power, n_bins))
        profiles = np.stack(profiles, axis=0)
        feats.append(np.concatenate([profiles.mean(axis=0), profiles.max(axis=0)]))
    return np.concatenate(feats).astype(np.float32)


def _denoise_residual(gray: np.ndarray) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(gray.astype(np.uint8), h=7)
    return gray - denoised.astype(np.float32)


def autocorrelation_features(gray: np.ndarray, n_rings: int = 6) -> np.ndarray:
    """2D autocorrelation of the noise residual. Translation-invariant (unlike raw peak
    detection on the residual itself), so it survives crops much better."""
    residual = _denoise_residual(gray)
    f = np.fft.fft2(residual)
    autocorr = np.fft.fftshift(np.fft.ifft2(f * np.conj(f)).real)
    autocorr /= autocorr.max() + 1e-8

    h, w = autocorr.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    r_max = min(cy, cx)
    ring_edges = np.linspace(0, r_max, n_rings + 1)
    ring_means = []
    for i in range(n_rings):
        mask = (r >= ring_edges[i]) & (r < ring_edges[i + 1])
        ring_means.append(autocorr[mask].mean() if np.any(mask) else 0.0)
    return np.array(ring_means, dtype=np.float32)


def prnu_residual_stats(gray: np.ndarray) -> np.ndarray:
    residual = _denoise_residual(gray)
    flat = residual.ravel()
    return np.array([flat.var(), kurtosis(flat), skew(flat)], dtype=np.float32)


def ela_features(img: Image.Image, quality: int = 90) -> np.ndarray:
    """Error Level Analysis: re-save at a known JPEG quality and diff against the original.
    Regions/images already near that quality's quantization grid show a low, flat diff."""
    import io

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32)
    original = np.asarray(img.convert("RGB"), dtype=np.float32)
    diff = np.abs(original - resaved)
    return np.array([diff.mean(), diff.std(), diff.max(), np.mean(diff > 10)], dtype=np.float32)


def double_jpeg_dct_histogram(gray: np.ndarray, n_bins: int = 32) -> tuple[np.ndarray, float]:
    """Per-8x8-block DCT of the first AC coefficient. A single JPEG encoding gives a smooth
    coefficient histogram; double-quantization (recompress at a different quality, e.g. after a
    crop-then-resave) leaves a periodic comb pattern in that histogram. We measure comb strength
    as the peak-to-mean ratio of the histogram's own FFT magnitude (excluding DC)."""
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    coeffs = []
    for y0 in range(0, h8, 8):
        for x0 in range(0, w8, 8):
            block = gray[y0 : y0 + 8, x0 : x0 + 8]
            dct = cv2.dct(block)
            coeffs.append(dct[0, 1])  # first horizontal AC coefficient
    coeffs = np.array(coeffs)
    hist, _ = np.histogram(coeffs, bins=n_bins)
    hist = hist.astype(np.float32)
    spectrum = np.abs(np.fft.fft(hist))[1 : n_bins // 2]
    comb_strength = float(spectrum.max() / (spectrum.mean() + 1e-8))
    return hist, comb_strength


def block_grid_alignment(gray: np.ndarray) -> float:
    """Sobel edge-discontinuity strength specifically at 8-pixel-grid boundaries vs. interior.
    A real 8x8 JPEG grid produces a small but consistent step at grid lines; a crop or resize
    that shifts the content relative to the grid weakens or misaligns this signal."""
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(sobel_x**2 + sobel_y**2)

    h, w = grad.shape
    grid_mask = np.zeros_like(grad, dtype=bool)
    grid_mask[::8, :] = True
    grid_mask[:, ::8] = True

    grid_strength = grad[grid_mask].mean() if grid_mask.any() else 0.0
    interior_strength = grad[~grid_mask].mean() if (~grid_mask).any() else 1e-8
    return float(grid_strength / (interior_strength + 1e-8))


def extract_all_features(img: Image.Image, max_dim: int | None = None) -> dict:
    img = _maybe_downscale(img, max_dim)
    gray = _to_gray(img)
    fft_feats = windowed_fft_features(gray)
    autocorr_feats = autocorrelation_features(gray)
    prnu_feats = prnu_residual_stats(gray)
    ela_feats = ela_features(img)
    dct_hist, comb_strength = double_jpeg_dct_histogram(gray)
    grid_ratio = block_grid_alignment(gray)

    vector = np.concatenate(
        [fft_feats, autocorr_feats, prnu_feats, ela_feats, dct_hist, [comb_strength, grid_ratio]]
    ).astype(np.float32)

    # Degradation estimate consumed by fusion.py: strong double-JPEG comb + strong block-grid
    # discontinuity both indicate a heavily re-compressed / re-processed image. Normalize each
    # to a rough [0, 1] range via a squashing function so they combine sensibly.
    degradation_score = float(
        0.5 * np.tanh(comb_strength / 10.0) + 0.5 * np.tanh(grid_ratio / 5.0)
    )

    return {
        "vector": vector,
        "fft": fft_feats,
        "autocorr": autocorr_feats,
        "prnu": prnu_feats,
        "ela": ela_feats,
        "dct_hist": dct_hist,
        "double_jpeg_comb_strength": comb_strength,
        "block_grid_ratio": grid_ratio,
        "degradation_score": degradation_score,
    }


FEATURE_VECTOR_DIM = extract_all_features(Image.new("RGB", (128, 128), color=(128, 128, 128)))["vector"].shape[0]
