"""Confidence signal extraction (Feature 4).

`confidence_signals(evidence, final_score, multicrop=None)` -> (vector, dict). The vector feeds
`ConfidenceModel`; the dict is kept for the output JSON so a human reviewer sees *why* the system
was (un)confident.
"""

from __future__ import annotations

import numpy as np

from src.router.features import Evidence

SIGNAL_NAMES = [
    "margin",                # 2*|final_score - 0.5|  (0 = on the fence, 1 = extreme)
    "final_score",
    "dis_std", "dis_range", "dis_max_pair",
    "n_tiers_available", "any_tier_missing",
    "degradation", "profile_overall_degradation", "profile_screenshot_like",
    "provenance_available",
    "crop_consistency", "crop_mean_margin", "crop_available",
]


def disagreement_stats(scores: list[float]) -> dict:
    if len(scores) < 2:
        return {"mean": float("nan"), "std": 0.0, "min": float("nan"),
                "max": float("nan"), "range": 0.0, "max_pair": 0.0}
    a = np.asarray(scores, dtype=float)
    pairs = [abs(x - y) for i, x in enumerate(a) for y in a[i + 1:]]
    return {"mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
            "max": float(a.max()), "range": float(a.max() - a.min()),
            "max_pair": float(max(pairs)) if pairs else 0.0}


def confidence_signals(ev: Evidence, final_score: float, multicrop: dict | None = None) -> tuple[np.ndarray, dict]:
    dis = disagreement_stats(ev.available_scores())
    profile = ev.profile or {}

    if multicrop is not None:
        crop_consistency = float(multicrop.get("consistency", 1.0))
        crop_mean_margin = 2 * abs(float(multicrop.get("mean", 0.5)) - 0.5)
        crop_available = 1.0
    else:
        crop_consistency, crop_mean_margin, crop_available = 1.0, 2 * abs(final_score - 0.5), 0.0

    d = {
        "margin": 2 * abs(final_score - 0.5),
        "final_score": float(final_score),
        "dis_std": dis["std"],
        "dis_range": dis["range"],
        "dis_max_pair": dis["max_pair"],
        "n_tiers_available": float(len(ev.available_scores())),
        "any_tier_missing": float(len(ev.available_scores()) < 3),
        "degradation": float(ev.degradation),
        "profile_overall_degradation": float(profile.get("overall_degradation", 0.0)),
        "profile_screenshot_like": float(profile.get("screenshot_like", 0.0)),
        "provenance_available": float(ev.tier0 is not None),
        "crop_consistency": crop_consistency,
        "crop_mean_margin": crop_mean_margin,
        "crop_available": crop_available,
    }
    vec = np.array([d[k] for k in SIGNAL_NAMES], dtype=np.float32)
    return vec, d


def is_hard_case(final_score: float, ev: Evidence, boundary: float = 0.15,
                 disagreement: float = 0.25, severity: float = 0.6) -> bool:
    """Cost-aware trigger for the expensive multi-crop pass (spec Feature 4): only when the
    decision is near the boundary, the detectors disagree, or transformation severity is high."""
    if abs(final_score - 0.5) < boundary:
        return True
    if disagreement_stats(ev.available_scores())["range"] >= disagreement:
        return True
    prof = ev.profile or {}
    return float(prof.get("overall_degradation", ev.degradation)) >= severity
