"""Shared detector-scoring layer (Features 1, 3, 4, 5).

`DetectorBank` loads every trained tier once and scores a list of image paths under an optional
transform, returning the raw per-tier scores plus Tier 2's degradation estimate. From those raw
scores it derives every *model* the benchmark compares:

    cnn              -- Tier 1 alone
    forensic         -- Tier 2 alone
    clip             -- Tier 3 alone
    static_ensemble  -- fixed weighted mean, weights in STATIC_ENSEMBLE_WEIGHTS (documented)
    existing_fusion  -- src/fusion.py's degradation-aware logistic-regression meta-model
    adaptive_router  -- Feature 3's learned router, if a callable is registered

Every tier degrades gracefully: a missing checkpoint yields NaN for that tier's raw score and
the derived models fall back to the available tiers (documented per model).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import joblib
import numpy as np
from PIL import Image

from src.frequency.features import extract_all_features
from src.fusion import build_features

# Documented fixed weights for the "simple static weighted ensemble" baseline (spec Feature 3,
# Baseline D). Deliberately near-uniform, with a hair more weight on the CNN.
STATIC_ENSEMBLE_WEIGHTS = {"tier1": 0.34, "tier2": 0.33, "tier3": 0.33}

DEFAULT_PATHS = {
    "tier1": "outputs/tier1_efficientnet_b0.pt",
    "tier2": "outputs/tier2_classifier.joblib",
    "tier3": "outputs/tier3_clip_probe.joblib",
    "fusion": "outputs/fusion_model.joblib",
    "profiler": "outputs/transformation_profiler.joblib",
    "router": "outputs/router_model.joblib",
}

MODEL_NAMES = ("cnn", "forensic", "clip", "static_ensemble", "existing_fusion", "adaptive_router")


@dataclass
class RawScores:
    """Per-image raw signals, all shape (N,). NaN where a tier was unavailable."""

    tier1: np.ndarray
    tier2: np.ndarray
    tier3: np.ndarray
    degradation: np.ndarray
    paths: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tier1)


@dataclass
class DetectorBank:
    tier1_ckpt: str = DEFAULT_PATHS["tier1"]
    tier2_clf: str = DEFAULT_PATHS["tier2"]
    tier3_probe: str = DEFAULT_PATHS["tier3"]
    fusion_model: str = DEFAULT_PATHS["fusion"]
    profiler_path: str | None = None      # Feature 2; loaded only if given and present
    router_path: str | None = None        # Feature 3; loaded only if given and present
    max_image_dim: int | None = None
    device: str = "cpu"
    # Alternative to router_path: register a callable fn(list[Evidence]) -> np.ndarray
    router_fn: Callable | None = None

    def __post_init__(self) -> None:
        self._t1 = self._t2 = self._t3 = self._embedder = self._fusion = None
        self._profiler = self._router = None

        if os.path.exists(self.tier1_ckpt):
            from src.model.model import load_checkpoint

            self._t1 = load_checkpoint(self.tier1_ckpt, self.device)
        if os.path.exists(self.tier2_clf):
            self._t2 = joblib.load(self.tier2_clf)
        if os.path.exists(self.tier3_probe):
            from src.semantic.embed import ClipEmbedder

            self._t3 = joblib.load(self.tier3_probe)
            self._embedder = ClipEmbedder(device=self.device)
        if os.path.exists(self.fusion_model):
            self._fusion = joblib.load(self.fusion_model)
        if self.profiler_path and os.path.exists(self.profiler_path):
            from src.transformation.model import load_profiler

            self._profiler = load_profiler(self.profiler_path)
        if self.router_path and os.path.exists(self.router_path):
            from src.router.model import load_router

            self._router = load_router(self.router_path)

    # ------------------------------------------------------------------
    @property
    def available(self) -> dict[str, bool]:
        return {
            "tier1": self._t1 is not None,
            "tier2": self._t2 is not None,
            "tier3": self._t3 is not None,
            "fusion": self._fusion is not None,
            "profiler": self._profiler is not None,
            "router": self._router is not None or self.router_fn is not None,
        }

    # ------------------------------------------------------------------
    def raw_scores(self, paths: list[str], transform: Callable | None = None,
                   _keep: dict | None = None) -> RawScores:
        n = len(paths)
        t1 = np.full(n, np.nan)
        t2 = np.full(n, np.nan)
        t3 = np.full(n, np.nan)
        deg = np.full(n, np.nan)

        # Load + transform once, reuse the PIL image for tiers 2 and 3.
        images = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            images.append(transform(img) if transform is not None else img)

        if self._t1 is not None:
            from src.model.evaluate import score_images

            # score_images re-opens from disk; give it the transform so tier 1 sees the same pixels.
            t1[:] = score_images(self._t1, paths, transform, self.device)

        forensics: list[dict] = []
        for i, img in enumerate(images):
            feats = extract_all_features(img, max_dim=self.max_image_dim)
            forensics.append(feats)
            deg[i] = feats["degradation_score"]
            if self._t2 is not None:
                t2[i] = float(self._t2.predict_proba(feats["vector"].reshape(1, -1))[0, 1])

        if self._t3 is not None and self._embedder is not None:
            embs = self._embedder.embed_batch(images).numpy()
            t3[:] = self._t3.predict_proba(embs)[:, 1]

        if _keep is not None:
            _keep["images"] = images
            _keep["forensics"] = forensics

        return RawScores(t1, t2, t3, deg, list(paths))

    def raw_scores_from_images(self, images: list, _keep: dict | None = None) -> RawScores:
        """Same as `raw_scores` but takes in-memory PIL images (used by the multi-crop pass,
        which has no file path)."""
        import torch

        n = len(images)
        t1 = np.full(n, np.nan); t2 = np.full(n, np.nan)
        t3 = np.full(n, np.nan); deg = np.full(n, np.nan)
        images = [im.convert("RGB") for im in images]

        if self._t1 is not None:
            from src.model.model import build_transform

            pre = build_transform()
            batch = torch.stack([pre(im) for im in images]).to(self.device)
            with torch.no_grad():
                t1[:] = self._t1.predict_proba(batch).cpu().numpy()

        forensics = []
        for i, img in enumerate(images):
            feats = extract_all_features(img, max_dim=self.max_image_dim)
            forensics.append(feats)
            deg[i] = feats["degradation_score"]
            if self._t2 is not None:
                t2[i] = float(self._t2.predict_proba(feats["vector"].reshape(1, -1))[0, 1])

        if self._t3 is not None and self._embedder is not None:
            t3[:] = self._t3.predict_proba(self._embedder.embed_batch(images).numpy())[:, 1]

        if _keep is not None:
            _keep["images"] = images
            _keep["forensics"] = forensics
        return RawScores(t1, t2, t3, deg, [])

    def build_evidence_from_images(self, images: list, profiler=None):
        from src.router.features import Evidence

        profiler = profiler or self._profiler
        keep: dict = {}
        raw = self.raw_scores_from_images(images, _keep=keep)
        profiles = profiler.predict_batch(keep["images"], keep["forensics"]) if profiler else [None] * len(images)
        ev = [
            Evidence(
                tier1=None if not np.isfinite(raw.tier1[i]) else float(raw.tier1[i]),
                tier2=None if not np.isfinite(raw.tier2[i]) else float(raw.tier2[i]),
                tier3=None if not np.isfinite(raw.tier3[i]) else float(raw.tier3[i]),
                tier0=None, degradation=float(raw.degradation[i]) if np.isfinite(raw.degradation[i]) else 0.0,
                profile=profiles[i], min_side=min(keep["images"][i].size),
            )
            for i in range(len(images))
        ]
        return ev, raw

    def build_evidence(self, paths, transform=None, profiler=None, min_sides=None):
        """Score `paths` and assemble a list of `src.router.features.Evidence`, running the
        transformation profiler (Feature 2) on the SAME transformed pixels / forensic dict."""
        from src.router.features import Evidence

        profiler = profiler or self._profiler
        keep: dict = {}
        raw = self.raw_scores(paths, transform, _keep=keep)
        profiles = [None] * len(paths)
        if profiler is not None:
            profiles = profiler.predict_batch(keep["images"], keep["forensics"])
        ev = []
        for i in range(len(paths)):
            ms = min_sides[i] if min_sides else min(keep["images"][i].size)
            ev.append(Evidence(
                tier1=None if not np.isfinite(raw.tier1[i]) else float(raw.tier1[i]),
                tier2=None if not np.isfinite(raw.tier2[i]) else float(raw.tier2[i]),
                tier3=None if not np.isfinite(raw.tier3[i]) else float(raw.tier3[i]),
                tier0=None,
                degradation=float(raw.degradation[i]) if np.isfinite(raw.degradation[i]) else 0.0,
                profile=profiles[i],
                min_side=int(ms),
            ))
        return ev, raw

    # ------------------------------------------------------------------
    def static_ensemble(self, raw: RawScores) -> np.ndarray:
        cols = {"tier1": raw.tier1, "tier2": raw.tier2, "tier3": raw.tier3}
        out = np.zeros(len(raw))
        wsum = np.zeros(len(raw))
        for k, w in STATIC_ENSEMBLE_WEIGHTS.items():
            v = cols[k]
            m = np.isfinite(v)
            out[m] += w * v[m]
            wsum[m] += w
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(wsum > 0, out / wsum, np.nan)

    def existing_fusion(self, raw: RawScores) -> np.ndarray:
        if self._fusion is None:
            return np.full(len(raw), np.nan)
        feats = np.stack([
            build_features(
                0.0,
                t1 if np.isfinite(t1) else 0.5,
                t2 if np.isfinite(t2) else 0.5,
                t3 if np.isfinite(t3) else 0.5,
                d if np.isfinite(d) else 0.0,
            )
            for t1, t2, t3, d in zip(raw.tier1, raw.tier2, raw.tier3, raw.degradation)
        ])
        return self._fusion.predict_proba(feats)[:, 1]

    def adaptive_router(self, evidence: list | None) -> np.ndarray:
        n = len(evidence) if evidence is not None else 0
        if evidence is None:
            return np.array([])
        if self.router_fn is not None:
            return np.asarray(self.router_fn(evidence), dtype=float)
        if self._router is not None:
            return np.asarray(self._router.predict_proba(evidence), dtype=float)
        return np.full(n, np.nan)

    # ------------------------------------------------------------------
    def all_model_scores(self, raw: RawScores, evidence: list | None = None) -> dict[str, np.ndarray]:
        n = len(raw)
        router = self.adaptive_router(evidence) if evidence is not None else np.full(n, np.nan)
        if router.size == 0:
            router = np.full(n, np.nan)
        return {
            "cnn": raw.tier1,
            "forensic": raw.tier2,
            "clip": raw.tier3,
            "static_ensemble": self.static_ensemble(raw),
            "existing_fusion": self.existing_fusion(raw),
            "adaptive_router": router,
        }

    def score_models(self, paths, transform=None) -> tuple[dict[str, np.ndarray], "RawScores", list]:
        """One call: transform, score every tier, build Evidence (+ profiler + router), and
        return {model_name: scores}, the RawScores, and the Evidence list."""
        need_ev = self._profiler is not None or self._router is not None or self.router_fn is not None
        if need_ev:
            evidence, raw = self.build_evidence(paths, transform=transform)
        else:
            evidence, raw = None, self.raw_scores(paths, transform)
        return self.all_model_scores(raw, evidence), raw, evidence
