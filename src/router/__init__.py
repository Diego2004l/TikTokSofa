"""Feature 3 -- adaptive evidence router / fusion.

Upgrades the degradation-aware fusion (`src/fusion.py`, preserved) into an explicitly-defined
adaptive evidence-routing model that learns, from the image condition + every detector's output
+ how much the detectors disagree, *which evidence to trust*.

Kept deliberately lightweight (logistic regression or a small gradient-boosted model -- NOT a
neural network): the point is learned adaptive weighting, not added capacity.

Router inputs (`src/router/features.py:build_router_features`), grouped so ablations can drop a
group at a time:
  * detector outputs      -- tier0/1/2/3 scores (+ neutral fill for missing tiers)
  * reliability signals   -- which tiers ran, provenance availability, Tier 2 degradation estimate
  * transformation profile-- Feature 2's 7 family scores + overall_degradation
  * disagreement features -- mean / std / min / max / range / pairwise abs-diffs of detector scores
  * uncertainty features  -- Feature 4's multi-crop consistency (optional; filled with 0 if absent)

Anti-leakage: the router is trained on a held-out split (the same fusion split, disjoint from
tier-training data AND from the final test benchmark) with out-of-fold predictions for its own
calibration. See `src/router/train.py`.
"""

from __future__ import annotations

from src.router.model import AdaptiveRouter, RouterConfig, load_router

__all__ = ["AdaptiveRouter", "RouterConfig", "load_router"]
