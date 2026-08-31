"""Inference cost accounting (Feature 5, "Average inference cost" column).

Two modes:
  * relative  -- documented static cost units per component (default), so the research table has
    a hardware-independent cost number.
  * measured  -- wall-clock milliseconds measured on the current machine for a sample of images.

The relative units below are order-of-magnitude estimates (Tier 2's hand-built features and CLIP
dominate; provenance and the linear router/fusion are ~free). Documented, not benchmarked -- the
point is the *ordering* and rough ratios the cascade is designed around.
"""

from __future__ import annotations

# Relative cost units (Tier 2 forensic extraction := 1.0 reference).
COST_UNITS = {
    "tier0_provenance": 0.05,
    "tier1_cnn": 0.6,
    "tier2_forensic": 1.0,
    "tier3_clip": 0.8,
    "transformation_profiler": 0.15,   # reuses Tier 2 features; only the extra indicators + trees
    "fusion": 0.01,
    "router": 0.01,
    "confidence": 0.02,
    "multicrop_per_crop": 1.6,         # re-runs Tier 2 + Tier 3 (+ Tier 1) per crop
}


def method_cost(method: str, *, escalated: bool = True, profiler: bool = False,
                multicrop_crops: int = 0) -> float:
    """Relative cost of one image through `method`.

    `escalated`  -- whether Tier 1 + Tier 3 ran (cascade may skip them).
    `profiler`   -- transformation profiler ran.
    `multicrop_crops` -- number of extra crops scored (Feature 4 hard-case path).
    """
    c = COST_UNITS
    base = {
        "baseline_cnn": c["tier1_cnn"],
        "baseline_forensic": c["tier2_forensic"],
        "baseline_clip": c["tier3_clip"],
        "baseline_static_ensemble": c["tier1_cnn"] + c["tier2_forensic"] + c["tier3_clip"],
        "baseline_existing_fusion": c["tier1_cnn"] + c["tier2_forensic"] + c["tier3_clip"] + c["fusion"],
    }
    if method in base:
        return round(base[method], 3)

    # router / full-system variants
    total = c["tier2_forensic"] + c["tier0_provenance"]
    if escalated:
        total += c["tier1_cnn"] + c["tier3_clip"]
    if profiler:
        total += c["transformation_profiler"]
    total += c["router"] + c["fusion"]
    if "confidence" in method or method in ("full_system", "router_full"):
        total += c["confidence"]
    total += multicrop_crops * c["multicrop_per_crop"]
    return round(total, 3)


def measure_wallclock(bank, paths: list[str], transform=None, warmup: int = 1) -> dict:
    """Measured ms/image for a full DetectorBank pass over `paths`."""
    import time

    for p in paths[:warmup]:
        bank.score_models([p], transform=transform)
    t0 = time.perf_counter()
    bank.score_models(paths, transform=transform)
    dt = (time.perf_counter() - t0) * 1000
    return {"total_ms": round(dt, 1), "ms_per_image": round(dt / max(1, len(paths)), 2), "n": len(paths)}
