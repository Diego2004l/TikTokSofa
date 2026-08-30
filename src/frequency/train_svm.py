"""Train Tier 2's classifier (RandomForest or SVM) on the hand-built forensic features.

Trains on AUGMENTED images (isolated + compound transforms applied symmetrically to both
classes), not just clean ones — otherwise Tier 2 would only ever see clean-image feature
distributions and collapse under any real-world compression/resize.
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import joblib
import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from src.augmentations import SymmetricAugmenter
from src.frequency.features import extract_all_features

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def list_images(folder: str) -> list[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(paths)


def _features_for_path(path: str, label: int, n_augments_per_image: int, seed: int, max_image_dim: int | None):
    # Per-image RNG keyed by path so the augmentation draw is deterministic and parallel-safe
    # (a single shared augmenter can't be pickled across joblib workers, and its draw order
    # would depend on scheduling). Still symmetric: label is never an input to the transform.
    augmenter = SymmetricAugmenter(rng=random.Random(f"{seed}:{path}"))
    img = Image.open(path).convert("RGB")
    variants = [img] + [augmenter(img) for _ in range(n_augments_per_image)]
    return [(extract_all_features(v, max_dim=max_image_dim)["vector"], label) for v in variants]


def build_dataset(real_dir: str, fake_dir: str, n_augments_per_image: int, seed: int = 0, max_image_dim: int | None = None, n_jobs: int = -1):
    tasks = [(p, 0) for p in list_images(real_dir)] + [(p, 1) for p in list_images(fake_dir)]
    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_features_for_path)(path, label, n_augments_per_image, seed, max_image_dim) for path, label in tasks
    )
    X, y = [], []
    for group in results:
        for feats, label in group:
            X.append(feats)
            y.append(label)
    return np.stack(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train Tier 2 forensic-feature classifier.")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--classifier", choices=["rf", "svm"], default="rf")
    parser.add_argument("--n-augments-per-image", type=int, default=2)
    parser.add_argument("--out", default="outputs/tier2_classifier.joblib")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-image-dim", type=int, default=None,
                        help="Cap the longer image side (px) before feature extraction — big speedup for iteration. "
                             "Pass the SAME value to src/infer.py. Leave unset for the final run (see features.py).")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Parallel workers for feature extraction (-1 = all cores).")
    args = parser.parse_args()

    X, y = build_dataset(args.real_dir, args.fake_dir, args.n_augments_per_image, args.seed,
                         max_image_dim=args.max_image_dim, n_jobs=args.n_jobs)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    if args.classifier == "rf":
        clf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=args.seed, n_jobs=-1)
    else:
        clf = SVC(kernel="rbf", probability=True, random_state=args.seed)

    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Tier 2 ({args.classifier}) validation AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"Saved classifier to {args.out}")


if __name__ == "__main__":
    main()
