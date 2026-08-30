"""Train a logistic-regression probe on top of frozen CLIP embeddings."""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.model.train import list_images
from src.semantic.embed import ClipEmbedder


def build_dataset(embedder: ClipEmbedder, real_dir: str, fake_dir: str):
    X, y = [], []
    for label, folder in ((0, real_dir), (1, fake_dir)):
        for path in list_images(folder):
            img = Image.open(path).convert("RGB")
            X.append(embedder.embed_image(img).numpy())
            y.append(label)
    return np.stack(X), np.array(y)


def main():
    parser = argparse.ArgumentParser(description="Train Tier 3 CLIP linear probe.")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--out", default="outputs/tier3_clip_probe.joblib")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    embedder = ClipEmbedder()
    X, y = build_dataset(embedder, args.real_dir, args.fake_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=args.seed, stratify=y)

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train, y_train)
    val_probs = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Tier 3 (CLIP linear probe) validation AUC: {auc:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"Saved probe to {args.out}")


if __name__ == "__main__":
    main()
