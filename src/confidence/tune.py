"""Fit the confidence model + abstention policy on a VALIDATION split (Feature 4).

    python -m src.confidence.tune \
        --real-dir data/raw/sid_set/val/real --fake-dir data/raw/sid_set/val/fake \
        --profiler outputs/transformation_profiler.joblib --router-model outputs/router_model.joblib \
        --target-fpr 0.05 --min-selective-accuracy 0.92 \
        --out-confidence outputs/confidence_model.joblib --out-policy outputs/abstention_policy.joblib

Thresholds are TUNED here, never hard-coded. The objective explicitly bounds the false-positive
rate (`--target-fpr`) and the selective accuracy, then maximises coverage.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.confidence.build import build_confidence_data
from src.confidence.model import AbstentionPolicy, ConfidenceModel
from src.eval.scoring import DetectorBank


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--out-confidence", default="outputs/confidence_model.joblib")
    ap.add_argument("--out-policy", default="outputs/abstention_policy.joblib")
    ap.add_argument("--profiler", default="outputs/transformation_profiler.joblib")
    ap.add_argument("--router-model", default="outputs/router_model.joblib")
    ap.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    ap.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    ap.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    ap.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    ap.add_argument("--max-image-dim", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--multicrop-n", type=int, default=5)
    ap.add_argument("--all-multicrop", action="store_true", help="Run multi-crop on every image, not just hard cases (for tuning stability).")
    ap.add_argument("--target-fpr", type=float, default=0.1)
    ap.add_argument("--min-selective-accuracy", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = DetectorBank(args.tier1_checkpoint, args.tier2_classifier, args.tier3_probe,
                        args.fusion_model, profiler_path=args.profiler, router_path=args.router_model,
                        max_image_dim=args.max_image_dim)
    print(f"[bank] {bank.available}")

    data = build_confidence_data(args.real_dir, args.fake_dir, bank, n_samples=args.n_samples,
                                 multicrop_n=args.multicrop_n, seed=args.seed,
                                 force_all_multicrop=args.all_multicrop)

    pred = (data.final_scores >= 0.5).astype(int)
    correct = (pred == data.labels).astype(int)
    print(f"\nRaw (no abstention): accuracy={correct.mean():.3f}  n={len(correct)}  hard={int(data.hard_mask.sum())}")

    conf_model = ConfidenceModel(seed=args.seed).fit(data.signals, correct)
    confidence = conf_model.predict_confidence(data.signals)
    from sklearn.metrics import roc_auc_score

    if len(set(correct.tolist())) == 2:
        print(f"Confidence model AUROC (predicting correctness): "
              f"{roc_auc_score(correct, confidence):.3f}")

    policy = AbstentionPolicy().fit(
        data.final_scores, data.labels, confidence,
        target_fpr=args.target_fpr, min_selective_accuracy=args.min_selective_accuracy,
    )
    print(f"\nTuned policy: {policy.params}")
    print(f"Validation fit report: {policy.fit_report_}")

    os.makedirs(os.path.dirname(args.out_confidence) or ".", exist_ok=True)
    import joblib

    joblib.dump(conf_model, args.out_confidence)
    policy.save(args.out_policy)
    print(f"\nSaved confidence model -> {args.out_confidence}")
    print(f"Saved abstention policy -> {args.out_policy} (+ .json)")


if __name__ == "__main__":
    main()
