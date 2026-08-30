"""Driver for the degradation-aware fusion meta-model (README "Reproducing a full run", step 4).

`src/fusion.py` defines `train_fusion()` but nothing collects the per-tier scores it needs.
This script does exactly that: run Tiers 0-3 over a HELD-OUT real/fake split (one the tier
models were NOT trained on), assemble one `{tier0, tier1, tier2, tier3, degradation_score}`
row per image, fit the logistic-regression meta-model, and save it to
`outputs/fusion_model.joblib` so `src/infer.py` and `src/eval/robustness.py` pick it up
automatically.

Why a separate held-out split matters: if the fusion model is fit on the same images Tiers 1-3
trained on, every tier score is over-optimistic and the meta-model learns the wrong weights.
Use a slice of data untouched by `src/model/train.py` / `train_svm.py` / `train_probe.py`
(a fresh split of the training corpus, or SID_Set's eval split). Do NOT use the COCO+DALL-E
validation-only set here — that is reserved for the final benchmark (`data/README.md`).

Every tier degrades gracefully: a missing Tier 1 checkpoint / Tier 3 probe just contributes a
neutral 0.5 for that column (same convention as `src/infer.py`), so this runs end-to-end even
before all tiers are trained -- though the resulting meta-model is only meaningful once the
tiers it weights actually exist.

Tier score computation is the expensive part (CLIP + CNN per image); `--scores-cache` saves the
assembled rows to a .npz so re-fitting with different `--seed` / regularization is instant.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.frequency.features import extract_all_features
from src.fusion import build_feature_matrix, train_fusion
from src.model.train import list_images

TIER_KEYS = ("tier0", "tier1", "tier2", "tier3", "degradation_score")


def _load_tiers(tier1_ckpt: str, tier2_clf: str, tier3_probe: str):
    tier1 = tier2 = tier3 = embedder = None

    if os.path.exists(tier1_ckpt):
        from src.model.model import load_checkpoint

        tier1 = load_checkpoint(tier1_ckpt)
    else:
        print(f"[warn] no Tier 1 checkpoint at {tier1_ckpt} — Tier 1 column defaults to 0.5.")

    if os.path.exists(tier2_clf):
        tier2 = joblib.load(tier2_clf)
    else:
        print(f"[warn] no Tier 2 classifier at {tier2_clf} — Tier 2 column defaults to 0.5.")

    if os.path.exists(tier3_probe):
        from src.semantic.embed import ClipEmbedder

        tier3 = joblib.load(tier3_probe)
        embedder = ClipEmbedder()
    else:
        print(f"[warn] no Tier 3 probe at {tier3_probe} — Tier 3 column defaults to 0.5.")

    return tier1, tier2, tier3, embedder


def build_rows(
    real_dir: str,
    fake_dir: str,
    tier1_ckpt: str,
    tier2_clf: str,
    tier3_probe: str,
    n_samples: int | None,
    max_image_dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (rows, labels): rows is (N, 5) float32 in TIER_KEYS order, labels is (N,) {0,1}."""
    tier1, tier2, tier3, embedder = _load_tiers(tier1_ckpt, tier2_clf, tier3_probe)

    real_paths = list_images(real_dir)
    fake_paths = list_images(fake_dir)
    if n_samples is not None:
        real_paths, fake_paths = real_paths[:n_samples], fake_paths[:n_samples]
    paths = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
    print(f"Scoring {len(real_paths)} real + {len(fake_paths)} fake images through Tiers 0-3...")

    # Tier 1 is cheapest to run in one batched pass over all paths.
    t1_scores = {}
    if tier1 is not None:
        from src.model.evaluate import score_images

        all_paths = [p for p, _ in paths]
        for path, score in zip(all_paths, score_images(tier1, all_paths)):
            t1_scores[path] = score

    rows, labels = [], []
    for i, (path, label) in enumerate(paths):
        img = Image.open(path).convert("RGB")

        forensic = extract_all_features(img, max_dim=max_image_dim)
        t2 = float(tier2.predict_proba(forensic["vector"].reshape(1, -1))[0, 1]) if tier2 is not None else 0.5
        degradation = float(forensic["degradation_score"])

        t1 = t1_scores.get(path, 0.5)

        if tier3 is not None and embedder is not None:
            emb = embedder.embed_image(img).numpy().reshape(1, -1)
            t3 = float(tier3.predict_proba(emb)[0, 1])
        else:
            t3 = 0.5

        # Tier 0 (C2PA) is metadata, not a learnable pixel signal, and is absent from every
        # training corpus — infer.py already passes 0.0 when there is no provenance signal.
        rows.append([0.0, t1, t2, t3, degradation])
        labels.append(label)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(paths)}")

    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Fit the degradation-aware fusion meta-model over a held-out split.")
    parser.add_argument("--real-dir", required=True, help="Held-out real images (NOT used to train Tiers 1-3).")
    parser.add_argument("--fake-dir", required=True, help="Held-out fake images (NOT used to train Tiers 1-3).")
    parser.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    parser.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    parser.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    parser.add_argument("--out", default="outputs/fusion_model.joblib")
    parser.add_argument("--n-samples", type=int, default=None, help="Cap per class (for a fast dry run).")
    parser.add_argument("--max-image-dim", type=int, default=None, help="Match the value used to train Tier 2 (src/frequency/train_svm.py).")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Held-out slice for reporting fusion AUC.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scores-cache", default=None, help="Path to a .npz to save/reuse the assembled tier scores.")
    args = parser.parse_args()

    if args.scores_cache and os.path.exists(args.scores_cache):
        print(f"Loading cached tier scores from {args.scores_cache}")
        cached = np.load(args.scores_cache)
        rows, labels = cached["rows"], cached["labels"]
    else:
        rows, labels = build_rows(
            args.real_dir, args.fake_dir,
            args.tier1_checkpoint, args.tier2_classifier, args.tier3_probe,
            args.n_samples, args.max_image_dim,
        )
        if args.scores_cache:
            os.makedirs(os.path.dirname(args.scores_cache) or ".", exist_ok=True)
            np.savez(args.scores_cache, rows=rows, labels=labels)
            print(f"Cached tier scores to {args.scores_cache}")

    if len(set(labels.tolist())) < 2:
        raise SystemExit("Need both real and fake images to fit the fusion model.")

    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=args.val_frac, random_state=args.seed, stratify=labels)

    row_dicts = [dict(zip(TIER_KEYS, r)) for r in rows]
    clf = train_fusion([row_dicts[i] for i in train_idx], labels[train_idx].tolist(), seed=args.seed)

    val_X = build_feature_matrix([row_dicts[i] for i in val_idx])
    val_probs = clf.predict_proba(val_X)[:, 1]
    auc = roc_auc_score(labels[val_idx], val_probs)
    print(f"\nFusion meta-model validation AUC: {auc:.4f}  (n_train={len(train_idx)}, n_val={len(val_idx)})")
    print("Learned coefficients [tier0, tier1, tier2_reweighted, tier3_reweighted, degradation]:")
    print(f"  {np.round(clf.coef_[0], 4).tolist()}  intercept={clf.intercept_[0]:.4f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(clf, args.out)
    print(f"Saved fusion model to {args.out}")


if __name__ == "__main__":
    main()
