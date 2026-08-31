"""Deterministic multi-crop consistency (Feature 4).

Re-score an image under a few fixed crops/resizes and measure how stable the final score is. A
genuine, easy image scores consistently across crops; a borderline or artefact-driven decision
swings. Used as a confidence signal, and run ONLY for hard cases (see `signals.is_hard_case`) so
the cost-aware cascade is preserved.

`multi_crop_eval(score_fn, img)` -- `score_fn(list[PIL.Image]) -> np.ndarray` is supplied by the
caller (infer.py builds it from the loaded tiers + router; the ablation harness from DetectorBank).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Fixed crop windows as (left_frac, top_frac, size_frac): center + 4 quadrants + a mild zoom.
CROP_WINDOWS = [
    (0.0, 0.0, 1.00),
    (0.0, 0.0, 0.75),
    (0.25, 0.0, 0.75),
    (0.0, 0.25, 0.75),
    (0.25, 0.25, 0.75),
    (0.125, 0.125, 0.75),
]


def make_crops(img: Image.Image, n: int = 5) -> list[Image.Image]:
    img = img.convert("RGB")
    w, h = img.size
    out = []
    for lf, tf, sf in CROP_WINDOWS[: max(1, min(n, len(CROP_WINDOWS)))]:
        cw, ch = max(1, int(w * sf)), max(1, int(h * sf))
        left, top = min(int(w * lf), w - cw), min(int(h * tf), h - ch)
        crop = img.crop((left, top, left + cw, top + ch))
        out.append(crop.resize((w, h), Image.BICUBIC))
    return out


def multi_crop_eval(score_fn, img: Image.Image, n: int = 5) -> dict:
    crops = make_crops(img, n)
    scores = np.asarray(score_fn(crops), dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return {"scores": [], "mean": 0.5, "std": 0.0, "consistency": 1.0, "agree_frac": 1.0}
    mean = float(scores.mean())
    std = float(scores.std())
    # consistency in [0,1]: 1 when all crops agree tightly, ->0 as spread approaches 0.5
    consistency = float(np.clip(1.0 - 2.0 * std, 0.0, 1.0))
    agree_frac = float(np.mean((scores >= 0.5) == (mean >= 0.5)))
    return {"scores": scores.round(4).tolist(), "mean": mean, "std": std,
            "consistency": consistency, "agree_frac": agree_frac}
