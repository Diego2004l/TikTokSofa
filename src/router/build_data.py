"""Assemble router training/eval data (Feature 3, shared with the Feature 5 ablation harness).

Runs every detector + the transformation profiler over a real/fake split, applying a
per-image random transform condition (seeded) so the router sees the full spread of clean and
degraded inputs it must route between. Returns `(evidence, labels, raw, condition_names)`.

ANTI-LEAKAGE: pass a HELD-OUT split here -- the same one used to fit `src/fusion.py`'s
meta-model, which is disjoint from tier-training data and from the final test benchmark. This
module never touches the test set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.eval.scoring import DetectorBank, RawScores
from src.model.train import list_images
from src.transforms.registry import build_conditions


@dataclass
class RouterData:
    evidence: list
    labels: np.ndarray
    raw: object                    # concatenated RawScores-like (arrays)
    condition_names: list[str]
    paths: list[str]


def _assign_conditions(n: int, seed: int, clean_fraction: float):
    conds = build_conditions(seed=seed, include_optional=False)
    non_clean = [c for c in conds if c.name != "clean"]
    clean = next(c for c in conds if c.name == "clean")
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append(clean if rng.random() < clean_fraction else rng.choice(non_clean))
    return out


def build_router_data(
    real_dir: str,
    fake_dir: str,
    bank: DetectorBank,
    profiler=None,
    n_samples: int | None = None,
    clean_fraction: float = 0.25,
    seed: int = 0,
) -> RouterData:
    real = list_images(real_dir)
    fake = list_images(fake_dir)
    if n_samples:
        real, fake = real[:n_samples], fake[:n_samples]
    paths = real + fake
    labels = np.array([0] * len(real) + [1] * len(fake))
    if len(set(labels.tolist())) < 2:
        raise SystemExit("Need both real and fake images.")

    conditions = _assign_conditions(len(paths), seed, clean_fraction)

    # Group by condition so each transform is applied once per batch.
    by_cond: dict[str, list[int]] = {}
    for i, c in enumerate(conditions):
        by_cond.setdefault(c.name, []).append(i)

    all_ev: list = [None] * len(paths)
    t1 = np.full(len(paths), np.nan); t2 = np.full(len(paths), np.nan)
    t3 = np.full(len(paths), np.nan); deg = np.full(len(paths), np.nan)
    cond_by_idx = [c.name for c in conditions]

    name_to_cond = {c.name: c for c in build_conditions(seed=seed, include_optional=False)}
    for k, (cname, idxs) in enumerate(by_cond.items(), 1):
        subset = [paths[i] for i in idxs]
        ev, raw = bank.build_evidence(subset, transform=name_to_cond[cname], profiler=profiler)
        for j, i in enumerate(idxs):
            all_ev[i] = ev[j]
            t1[i], t2[i], t3[i], deg[i] = raw.tier1[j], raw.tier2[j], raw.tier3[j], raw.degradation[j]
        print(f"  [{k}/{len(by_cond)}] {cname}: {len(idxs)} imgs")

    raw_all = RawScores(t1, t2, t3, deg, list(paths))
    return RouterData(all_ev, labels, raw_all, cond_by_idx, paths)
