"""Comprehensive, reusable image-transformation library for the robustness benchmark
(Feature 1) and the transformation profiler's synthetic training data (Feature 2).

This package is additive: `src/augmentations.py` (used by Tier 1 / Tier 2 training) is left
untouched, and the primitives it already defines are re-used here rather than reimplemented.

Public surface:
    catalog   -- parametrised transform primitives (PIL.Image -> PIL.Image)
    screenshot -- deterministic "social-media screenshot" simulation
    chains    -- Chain composition + deterministic seeded random compound chains
    registry  -- Condition dataclass + build_conditions(): the canonical list of benchmark
                 conditions with family/severity metadata
"""

from __future__ import annotations

from src.transforms.chains import Chain, random_compound_chain
from src.transforms.registry import Condition, build_conditions, iter_conditions
from src.transforms.screenshot import screenshot_sim

__all__ = [
    "Chain",
    "random_compound_chain",
    "Condition",
    "build_conditions",
    "iter_conditions",
    "screenshot_sim",
]
