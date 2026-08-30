"""Robustness evaluation protocol (spec section 5): isolated transforms + compound chains,
per-tier AUC, written to outputs/robustness_summary.csv.

Final Score = 0.50 * AUC_clean + 0.50 * AUC_robust, where AUC_robust is the mean AUC across every
non-clean condition (isolated transforms AND compound chains). Reporting per-tier numbers (not
just the fused system) is what makes the eventual error-analysis doc credible — see spec
section 5's "Report metrics per tier" note and docs/error_analysis.md.

Tier 0 is intentionally excluded here: a valid C2PA manifest is metadata, not pixels, and most of
these transforms (recompression, crop, resize) strip or invalidate it by design — that is a
property of provenance signals in general, not something a robustness score over pixel transforms
can meaningfully capture.
"""

from __future__ import annotations

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.augmentations import all_conditions
from src.frequency.features import extract_all_features
from src.fusion import build_features
from src.model.evaluate import score_images
from src.model.model import load_checkpoint
from src.model.train import list_images
from src.semantic.embed import ClipEmbedder


def score_tier2(clf, paths, transform, max_image_dim=None):
    scores, degradations = [], []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if transform is not None:
            img = transform(img)
        feats = extract_all_features(img, max_dim=max_image_dim)
        scores.append(float(clf.predict_proba(feats["vector"].reshape(1, -1))[0, 1]))
        degradations.append(feats["degradation_score"])
    return scores, degradations


def score_tier3(embedder, probe, paths, transform):
    scores = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if transform is not None:
            img = transform(img)
        emb = embedder.embed_image(img).numpy().reshape(1, -1)
        scores.append(float(probe.predict_proba(emb)[0, 1]))
    return scores


def safe_auc(y_true, y_score) -> float:
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def main():
    parser = argparse.ArgumentParser(description="Run the robustness eval protocol across all isolated + compound conditions.")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=100, help="Cap per class per condition, for speed.")
    parser.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    parser.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    parser.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    parser.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    parser.add_argument("--out", default="outputs/robustness_summary.csv")
    parser.add_argument("--max-image-dim", type=int, default=None, help="Match the value used to train Tier 2.")
    args = parser.parse_args()

    real_paths = list_images(args.real_dir)[: args.n_samples]
    fake_paths = list_images(args.fake_dir)[: args.n_samples]
    y_true = [0] * len(real_paths) + [1] * len(fake_paths)

    tier1_model = load_checkpoint(args.tier1_checkpoint) if os.path.exists(args.tier1_checkpoint) else None
    tier2_clf = joblib.load(args.tier2_classifier) if os.path.exists(args.tier2_classifier) else None
    tier3_probe = joblib.load(args.tier3_probe) if os.path.exists(args.tier3_probe) else None
    tier3_embedder = ClipEmbedder() if tier3_probe is not None else None
    fusion_model = joblib.load(args.fusion_model) if os.path.exists(args.fusion_model) else None

    if tier1_model is None:
        print("[warn] no Tier 1 checkpoint — Tier 1 columns will be NaN.")
    if tier2_clf is None:
        print("[warn] no Tier 2 classifier — Tier 2 columns will be NaN.")
    if tier3_probe is None:
        print("[warn] no Tier 3 probe — Tier 3 columns will be NaN.")

    rows = []
    for name, transform in all_conditions().items():
        row = {"condition": name}

        if tier1_model is not None:
            t1_scores = score_images(tier1_model, real_paths + fake_paths, transform)
            row["tier1_auc"] = safe_auc(y_true, t1_scores)
        else:
            t1_scores, row["tier1_auc"] = None, float("nan")

        if tier2_clf is not None:
            t2_scores, degradations = score_tier2(tier2_clf, real_paths + fake_paths, transform, args.max_image_dim)
            row["tier2_auc"] = safe_auc(y_true, t2_scores)
        else:
            t2_scores, degradations, row["tier2_auc"] = None, None, float("nan")

        if tier3_embedder is not None:
            t3_scores = score_tier3(tier3_embedder, tier3_probe, real_paths + fake_paths, transform)
            row["tier3_auc"] = safe_auc(y_true, t3_scores)
        else:
            t3_scores, row["tier3_auc"] = None, float("nan")

        if fusion_model is not None and t1_scores is not None and t2_scores is not None and t3_scores is not None:
            fused = [
                float(fusion_model.predict_proba(build_features(0.0, t1, t2, t3, deg).reshape(1, -1))[0, 1])
                for t1, t2, t3, deg in zip(t1_scores, t2_scores, t3_scores, degradations)
            ]
            row["fused_auc"] = safe_auc(y_true, fused)
        else:
            row["fused_auc"] = float("nan")

        rows.append(row)
        print(row)

    df = pd.DataFrame(rows)
    non_clean = df[df["condition"] != "clean"]
    summary = {"condition": "FINAL_SCORE (0.5*clean + 0.5*mean_robust)"}
    for col in ("tier1_auc", "tier2_auc", "tier3_auc", "fused_auc"):
        clean_val = df.loc[df["condition"] == "clean", col].iloc[0]
        robust_val = non_clean[col].mean()
        summary[col] = 0.5 * clean_val + 0.5 * robust_val
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
