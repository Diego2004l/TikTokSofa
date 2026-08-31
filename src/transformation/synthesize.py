"""Synthetic transformation-labelled data generator (Feature 2).

Take clean images, apply a known random subset of transform families at known severities, and
emit `(feature_vector, binary_label_vector, severity_vector)` rows. Because the transform is
applied by us, every label is exact -- no manual annotation.

Multi-label by construction: each synthetic example may stack up to `max_ops` families (applied
in a canonical order: geometric -> photometric -> filter -> JPEG-last), so the profiler learns
compound cases, not just isolated ones.

Severity grids are split into `train` and `eval` values so `evaluate.py` can measure
generalisation to **unseen severity levels** (spec Feature 2, "Required evaluation").
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image

from src.transforms import catalog
from src.transforms.screenshot import screenshot_sim
from src.transformation.features import profiler_features

PROFILE_LABELS = [
    "jpeg_compression",
    "resize_degradation",
    "blur",
    "noise",
    "sharpening",
    "crop",
    "screenshot_like",
]

# Per-family severity grids. 'severity' is a normalised 0..1 "how strong" used as the regression
# target; the second element is the concrete transform parameter.
SEVERITY_GRID: dict[str, dict[str, list[tuple[float, float]]]] = {
    "jpeg_compression": {
        "train": [(0.15, 85), (0.40, 65), (0.65, 45), (0.85, 25)],
        "eval": [(0.28, 75), (0.52, 55), (0.75, 35)],
    },
    "resize_degradation": {
        "train": [(0.20, 0.85), (0.45, 0.6), (0.70, 0.4), (0.85, 0.25)],
        "eval": [(0.32, 0.72), (0.58, 0.5), (0.78, 0.3)],
    },
    "blur": {
        "train": [(0.20, 0.6), (0.45, 1.2), (0.70, 2.0), (0.90, 3.0)],
        "eval": [(0.32, 0.9), (0.58, 1.6), (0.80, 2.5)],
    },
    "noise": {
        "train": [(0.20, 0.015), (0.45, 0.035), (0.70, 0.06), (0.90, 0.09)],
        "eval": [(0.32, 0.025), (0.58, 0.045), (0.80, 0.075)],
    },
    "sharpening": {
        "train": [(0.30, 120), (0.55, 180), (0.80, 260), (1.0, 340)],
        "eval": [(0.42, 150), (0.68, 220), (0.90, 300)],
    },
    "crop": {
        "train": [(0.15, 0.9), (0.35, 0.78), (0.60, 0.62), (0.80, 0.48)],
        "eval": [(0.25, 0.84), (0.48, 0.7), (0.70, 0.55)],
    },
    "screenshot_like": {
        "train": [(0.35, 0.9), (0.6, 0.82), (0.85, 0.7)],
        "eval": [(0.48, 0.86), (0.72, 0.76)],
    },
}

# Applied in this order when a synthetic example stacks several families.
CANONICAL_ORDER = ["crop", "resize_degradation", "sharpening", "blur", "noise",
                   "screenshot_like", "jpeg_compression"]


def _apply(family: str, param: float, img: Image.Image, rng: random.Random) -> Image.Image:
    if family == "jpeg_compression":
        return catalog.jpeg_compress(img, quality=int(param))
    if family == "resize_degradation":
        return catalog.resize_scale(img, scale=float(param))
    if family == "blur":
        return catalog.gaussian_blur(img, sigma=float(param))
    if family == "noise":
        return catalog.gaussian_noise(img, sigma=float(param), seed=rng.randint(0, 2**31 - 1))
    if family == "sharpening":
        return catalog.unsharp_mask(img, percent=int(param))
    if family == "crop":
        return catalog.random_crop(img, frac=float(param), seed=rng.randint(0, 2**31 - 1))
    if family == "screenshot_like":
        return screenshot_sim(img, scale=float(param), border=rng.choice([0, 8, 14]),
                              jpeg_quality=rng.choice([70, 80, 90]))
    raise ValueError(family)


@dataclass
class SynthConfig:
    variants_per_image: int = 6
    max_ops: int = 3
    clean_fraction: float = 0.15          # fraction of variants left untransformed
    grid: str = "train"                   # 'train' or 'eval' severity grid
    screenshot_implies_jpeg: bool = True  # screenshot_sim already recompresses -> also flag jpeg
    seed: int = 0


def synth_labels_for_image(
    img: Image.Image, cfg: SynthConfig, rng: random.Random
) -> list[tuple[Image.Image, np.ndarray, np.ndarray]]:
    out = []
    for _ in range(cfg.variants_per_image):
        sev = np.zeros(len(PROFILE_LABELS), dtype=np.float32)
        binary = np.zeros(len(PROFILE_LABELS), dtype=np.float32)

        if rng.random() < cfg.clean_fraction:
            out.append((img.convert("RGB"), binary, sev))
            continue

        k = rng.randint(1, cfg.max_ops)
        families = rng.sample(PROFILE_LABELS, k=k)
        cur = img.convert("RGB")
        for fam in [f for f in CANONICAL_ORDER if f in families]:
            s_norm, param = rng.choice(SEVERITY_GRID[fam][cfg.grid])
            cur = _apply(fam, param, cur, rng)
            i = PROFILE_LABELS.index(fam)
            binary[i] = 1.0
            sev[i] = max(sev[i], s_norm)
            if fam == "screenshot_like" and cfg.screenshot_implies_jpeg:
                j = PROFILE_LABELS.index("jpeg_compression")
                binary[j] = 1.0
                sev[j] = max(sev[j], 0.4)
        out.append((cur, binary, sev))
    return out


def build_synthetic_dataset(
    clean_paths: list[str], cfg: SynthConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X features, Y binary [N, 7], S severity [N, 7])."""
    rng = random.Random(cfg.seed)
    X, Y, S = [], [], []
    for idx, path in enumerate(clean_paths):
        try:
            base = Image.open(path).convert("RGB")
        except Exception:
            continue
        img_rng = random.Random(f"{cfg.seed}:{idx}")
        for variant, binary, sev in synth_labels_for_image(base, cfg, img_rng):
            X.append(profiler_features(variant))
            Y.append(binary)
            S.append(sev)
        if (idx + 1) % 100 == 0:
            print(f"  synth: {idx + 1}/{len(clean_paths)} images -> {len(X)} examples")
    return np.stack(X), np.stack(Y).astype(np.float32), np.stack(S).astype(np.float32)


def overall_severity(sev_row: np.ndarray) -> float:
    """Target for the overall_degradation regressor: soft-OR of per-family severities so that
    several mild transforms still read as clearly degraded."""
    keep = np.clip(sev_row, 0.0, 1.0)
    prod = np.prod(1.0 - keep)
    return float(1.0 - prod)
