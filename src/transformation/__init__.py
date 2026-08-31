"""Feature 2 -- explicit transformation profiler.

Given an image, estimate what degradation/transformations it has undergone:

    {
      "jpeg_compression": 0.87,
      "resize_degradation": 0.74,
      "blur": 0.12,
      "noise": 0.21,
      "sharpening": 0.05,
      "crop": 0.63,
      "screenshot_like": 0.71,
      "overall_degradation": 0.78,
    }

Design constraints (see spec Feature 2):
  * NOT another large neural network. The profiler reuses Tier 2's hand-built forensic features
    (double-JPEG DCT stats, block-grid alignment, ELA, noise-residual stats, FFT) plus a handful
    of extra cheap indicators, and feeds them to per-label gradient-boosted trees.
  * Supervised labels are generated automatically: apply known transforms from `src.transforms`
    to clean images, so the label of each synthetic example is known by construction.
  * The per-label outputs are isotonic-**calibrated** on a held-out split, so they may be read as
    calibrated confidence estimates in [0, 1]. `overall_degradation` is a regressor output, not a
    probability -- documented as a degradation score.
"""

from __future__ import annotations

from src.transformation.model import (
    PROFILE_LABELS,
    TransformationProfiler,
    load_profiler,
)

__all__ = ["PROFILE_LABELS", "TransformationProfiler", "load_profiler"]
