"""Assemble the confidence/abstention dataset for a split (Feature 4, shared by tune/evaluate).

For each image: run detectors + profiler + router (via DetectorBank), pick the final score
(router > existing fusion > static ensemble), run the deterministic multi-crop pass for hard
cases only, and compute the confidence signal vector.

A per-image seeded transform condition is applied (like the router's training data) so the
policy is tuned/evaluated across the clean+degraded spread, not just clean images.

ANTI-LEAKAGE: pass a validation split to `tune.py` and a disjoint test split to `evaluate.py`.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.confidence.multicrop import multi_crop_eval
from src.confidence.signals import confidence_signals, is_hard_case
from src.eval.scoring import DetectorBank
from src.model.train import list_images
from src.transforms.registry import build_conditions


@dataclass
class ConfidenceData:
    final_scores: np.ndarray
    labels: np.ndarray
    signals: np.ndarray          # (N, len(SIGNAL_NAMES))
    hard_mask: np.ndarray
    paths: list[str]
    condition_names: list[str]


def _final_score(model_scores: dict, i: int) -> float:
    for key in ("adaptive_router", "existing_fusion", "static_ensemble"):
        v = model_scores[key][i]
        if np.isfinite(v):
            return float(v)
    return 0.5


def _score_fn_factory(bank: DetectorBank):
    def score_fn(imgs):
        ev, raw = bank.build_evidence_from_images(imgs)
        ms = bank.all_model_scores(raw, ev)
        return np.array([_final_score(ms, i) for i in range(len(imgs))])

    return score_fn


def build_confidence_data(
    real_dir: str, fake_dir: str, bank: DetectorBank,
    n_samples: int | None = None, clean_fraction: float = 0.3,
    multicrop_n: int = 5, seed: int = 0, force_all_multicrop: bool = False,
) -> ConfidenceData:
    real = list_images(real_dir)[:n_samples] if n_samples else list_images(real_dir)
    fake = list_images(fake_dir)[:n_samples] if n_samples else list_images(fake_dir)
    paths = real + fake
    labels = np.array([0] * len(real) + [1] * len(fake))
    if len(set(labels.tolist())) < 2:
        raise SystemExit("Need both real and fake images.")

    conds = build_conditions(seed=seed, include_optional=False)
    non_clean = [c for c in conds if c.name != "clean"]
    clean = next(c for c in conds if c.name == "clean")
    rng = random.Random(seed)
    assign = [clean if rng.random() < clean_fraction else rng.choice(non_clean) for _ in paths]

    by_cond: dict[str, list[int]] = {}
    for i, c in enumerate(assign):
        by_cond.setdefault(c.name, []).append(i)
    name_to_cond = {c.name: c for c in conds}
    score_fn = _score_fn_factory(bank)

    N = len(paths)
    final = np.full(N, np.nan)
    sig = np.zeros((N, len(_signal_names())), dtype=np.float32)
    hard = np.zeros(N, dtype=bool)
    cond_names = [c.name for c in assign]

    for k, (cname, idxs) in enumerate(by_cond.items(), 1):
        subset = [paths[i] for i in idxs]
        cond = name_to_cond[cname]
        ev, raw = bank.build_evidence(subset, transform=cond)
        ms = bank.all_model_scores(raw, ev)
        for j, i in enumerate(idxs):
            fs = _final_score(ms, j)
            final[i] = fs
            hard_case = force_all_multicrop or is_hard_case(fs, ev[j])
            hard[i] = hard_case
            mc = None
            if hard_case and multicrop_n > 0:
                from PIL import Image

                img = cond(Image.open(paths[i]).convert("RGB"))
                mc = multi_crop_eval(score_fn, img, n=multicrop_n)
            sig[i] = confidence_signals(ev[j], fs, mc)[0]
        print(f"  [{k}/{len(by_cond)}] {cname}: {len(idxs)} imgs (hard={int(hard[idxs].sum())})")

    return ConfidenceData(final, labels, sig, hard, paths, cond_names)


def _signal_names():
    from src.confidence.signals import SIGNAL_NAMES

    return SIGNAL_NAMES
