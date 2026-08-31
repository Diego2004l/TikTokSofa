"""Router feature assembly (Feature 3).

`build_router_features(evidence, config)` turns one image's evidence bundle into a flat float
vector. Feature groups are toggleable via `RouterConfig` so the ablation study (Feature 5) can
remove one group at a time and measure the contribution.

`Evidence` is a plain dataclass so it can be built from `src/infer.py` at inference time and
from `src/router/train.py` in bulk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.transformation.synthesize import PROFILE_LABELS

NEUTRAL = 0.5

# Order matters and is frozen: feature_names() must match build_vector() exactly.
_DETECTOR_KEYS = ("tier0", "tier1", "tier2", "tier3")
_PROFILE_KEYS = tuple(PROFILE_LABELS) + ("overall_degradation",)


@dataclass
class Evidence:
    tier1: float | None = None
    tier2: float | None = None
    tier3: float | None = None
    tier0: float | None = None          # provenance; None == no usable signal
    degradation: float = 0.0            # Tier 2 block-grid/double-JPEG estimate
    profile: dict | None = None         # Feature 2 output dict, or None
    crop_consistency: float | None = None   # Feature 4: 1 - std of multi-crop scores, or None
    crop_mean: float | None = None          # Feature 4: mean multi-crop score, or None
    min_side: int | None = None             # smallest image dimension in px (feature-quality proxy)

    def available_scores(self) -> list[float]:
        return [s for s in (self.tier1, self.tier2, self.tier3) if s is not None]


@dataclass
class RouterConfig:
    use_disagreement: bool = True
    use_profile: bool = True
    use_uncertainty: bool = False       # Feature 4 multi-crop; off by default
    model: str = "logreg"               # 'logreg' | 'gbdt'
    calibrate: bool = True
    seed: int = 0

    def tag(self) -> str:
        bits = [self.model]
        if self.use_disagreement:
            bits.append("disagree")
        if self.use_profile:
            bits.append("profile")
        if self.use_uncertainty:
            bits.append("uncertainty")
        return "+".join(bits)


def _disagreement(scores: list[float]) -> list[float]:
    if len(scores) < 2:
        return [NEUTRAL, 0.0, NEUTRAL, NEUTRAL, 0.0, 0.0, 0.0, 0.0]
    a = np.array(scores, dtype=float)
    pair = [abs(x - y) for i, x in enumerate(a) for y in a[i + 1:]]
    pair = (pair + [0.0, 0.0, 0.0])[:3]  # up to C(3,2)=3 pairs; pad if fewer detectors
    return [float(a.mean()), float(a.std()), float(a.min()), float(a.max()),
            float(a.max() - a.min()), *[float(p) for p in pair]]


def feature_names(config: RouterConfig) -> list[str]:
    names = list(_DETECTOR_KEYS)
    names += ["tier1_missing", "tier2_missing", "tier3_missing", "provenance_available",
              "n_tiers_available", "degradation", "small_image"]
    if config.use_disagreement:
        names += ["dis_mean", "dis_std", "dis_min", "dis_max", "dis_range",
                  "dis_pair0", "dis_pair1", "dis_pair2"]
    if config.use_profile:
        names += [f"profile_{k}" for k in _PROFILE_KEYS]
    if config.use_uncertainty:
        names += ["crop_consistency", "crop_mean", "crop_available"]
    return names


def build_vector(ev: Evidence, config: RouterConfig) -> np.ndarray:
    t0 = ev.tier0 if ev.tier0 is not None else 0.0
    t1 = ev.tier1 if ev.tier1 is not None else NEUTRAL
    t2 = ev.tier2 if ev.tier2 is not None else NEUTRAL
    t3 = ev.tier3 if ev.tier3 is not None else NEUTRAL

    feats = [t0, t1, t2, t3,
             float(ev.tier1 is None), float(ev.tier2 is None), float(ev.tier3 is None),
             float(ev.tier0 is not None),
             float(len(ev.available_scores())),
             float(ev.degradation),
             float(ev.min_side is not None and ev.min_side < 128)]

    if config.use_disagreement:
        feats += _disagreement(ev.available_scores())

    if config.use_profile:
        prof = ev.profile or {}
        feats += [float(prof.get(k, 0.0)) for k in _PROFILE_KEYS]

    if config.use_uncertainty:
        cc = ev.crop_consistency
        cm = ev.crop_mean
        feats += [float(cc) if cc is not None else 1.0,
                  float(cm) if cm is not None else NEUTRAL,
                  float(cc is not None)]

    return np.asarray(feats, dtype=np.float32)


def build_matrix(evidence: list[Evidence], config: RouterConfig) -> np.ndarray:
    return np.stack([build_vector(e, config) for e in evidence])
