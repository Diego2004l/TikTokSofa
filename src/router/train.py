"""Train the adaptive evidence router (Feature 3).

    python -m src.router.train \
        --real-dir data/raw/sid_set/eval/real --fake-dir data/raw/sid_set/eval/fake \
        --profiler outputs/transformation_profiler.joblib \
        --out outputs/router_model.joblib --model logreg

Uses the HELD-OUT fusion split (disjoint from tier training AND the final test benchmark). Each
image gets a seeded random transform condition so the router learns to route across clean +
degraded inputs. Reports out-of-fold AUC (the router never scores a sample it trained on) and,
for the linear model, the learned evidence weights.

`--scores-cache` saves the assembled evidence so retraining with a different --model / feature
config is instant.
"""

from __future__ import annotations

import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

from src.eval.metrics import binary_metrics
from src.eval.scoring import DetectorBank
from src.router.build_data import build_router_data
from src.router.model import AdaptiveRouter, RouterConfig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--out", default="outputs/router_model.joblib")
    ap.add_argument("--profiler", default="outputs/transformation_profiler.joblib")
    ap.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    ap.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    ap.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    ap.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    ap.add_argument("--max-image-dim", type=int, default=None)
    ap.add_argument("--model", choices=["logreg", "gbdt"], default="logreg")
    ap.add_argument("--no-disagreement", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--use-uncertainty", action="store_true")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--clean-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scores-cache", default=None)
    args = ap.parse_args()

    profiler = None
    if not args.no_profile and os.path.exists(args.profiler):
        from src.transformation.model import load_profiler

        profiler = load_profiler(args.profiler)
    elif not args.no_profile:
        print(f"[warn] no profiler at {args.profiler} — profile features will be zeros.")

    if args.scores_cache and os.path.exists(args.scores_cache):
        print(f"Loading cached evidence from {args.scores_cache}")
        with open(args.scores_cache, "rb") as f:
            data = pickle.load(f)
    else:
        bank = DetectorBank(args.tier1_checkpoint, args.tier2_classifier, args.tier3_probe,
                            args.fusion_model, max_image_dim=args.max_image_dim)
        print(f"[bank] {bank.available}")
        data = build_router_data(args.real_dir, args.fake_dir, bank, profiler,
                                 n_samples=args.n_samples, clean_fraction=args.clean_fraction,
                                 seed=args.seed)
        if args.scores_cache:
            os.makedirs(os.path.dirname(args.scores_cache) or ".", exist_ok=True)
            with open(args.scores_cache, "wb") as f:
                pickle.dump(data, f)
            print(f"Cached evidence to {args.scores_cache}")

    config = RouterConfig(
        use_disagreement=not args.no_disagreement,
        use_profile=not args.no_profile,
        use_uncertainty=args.use_uncertainty,
        model=args.model,
        seed=args.seed,
    )
    router = AdaptiveRouter(config)
    oof = router.oof_fit_predict(data.evidence, data.labels, n_splits=5)

    m = binary_metrics(data.labels, oof, threshold=0.5)
    print(f"\nRouter [{config.tag()}] out-of-fold: AUC={m['auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"F1={m['f1']:.3f}  FPR={m['fpr']:.3f}  n={m['num_samples']}")

    # Baselines on the SAME evidence, for a quick sanity check (full comparison: src.router.baselines).
    t1, t2, t3 = data.raw.tier1, data.raw.tier2, data.raw.tier3
    for nm, s in [("cnn", t1), ("forensic", t2), ("clip", t3),
                  ("static_ens", np.nanmean(np.vstack([t1, t2, t3]), axis=0))]:
        ok = np.isfinite(s)
        if ok.sum() > 2 and len(set(data.labels[ok].tolist())) == 2:
            print(f"  baseline {nm:11s} AUC={roc_auc_score(data.labels[ok], s[ok]):.4f}")

    coefs = router.coefficients()
    if coefs:
        top = sorted(coefs.items(), key=lambda kv: -abs(kv[1]))[:12]
        print("\nLearned evidence weights (|top 12|):")
        for name, w in top:
            print(f"  {name:24s} {w:+.4f}")

    router.metadata = {
        "config": config.__dict__, "oof_auc": m["auc"], "n_train": int(len(data.labels)),
        "profiler": args.profiler if profiler is not None else None, "seed": args.seed,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    router.save(args.out)
    print(f"\nSaved router to {args.out}")


if __name__ == "__main__":
    main()
