"""Baselines + ablation harness for the adaptive router (Feature 3 "Critical experiment",
reused by Feature 5's ablation study).

Fits every router variant on a TRAIN/VAL split and evaluates *all* methods on exactly the same
held-out TEST samples, under a clean pass and a seeded-degraded pass:

    Baseline A  CNN only
    Baseline B  Forensic only
    Baseline C  CLIP only
    Baseline D  Static weighted ensemble (0.34/0.33/0.33)
    Baseline E  Existing degradation-aware fusion (src/fusion.py)
    Router      adaptive router, base features only
    Router+dis  + disagreement features
    Router+prof + transformation profiler (Feature 2)
    Router+unc  + uncertainty / multi-crop (Feature 4)   [needs --uncertainty-cache]

    python -m src.router.baselines \
        --train-real .../val/real --train-fake .../val/fake \
        --test-real  .../test/real --test-fake .../test/fake \
        --profiler outputs/transformation_profiler.joblib --out-dir outputs/router_ablation
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.eval.metrics import binary_metrics, df_to_markdown, fpr_at_tpr
from src.eval.scoring import STATIC_ENSEMBLE_WEIGHTS, DetectorBank, RawScores
from src.model.train import list_images
from src.router.build_data import build_router_data
from src.router.model import AdaptiveRouter, RouterConfig

ROUTER_VARIANTS = {
    "router_base": RouterConfig(use_disagreement=False, use_profile=False, use_uncertainty=False),
    "router_disagreement": RouterConfig(use_disagreement=True, use_profile=False, use_uncertainty=False),
    "router_profiler": RouterConfig(use_disagreement=True, use_profile=True, use_uncertainty=False),
    "router_full": RouterConfig(use_disagreement=True, use_profile=True, use_uncertainty=True),
}


def _static_ensemble(raw) -> np.ndarray:
    w = STATIC_ENSEMBLE_WEIGHTS
    stack = np.vstack([raw.tier1, raw.tier2, raw.tier3])
    weights = np.array([[w["tier1"]], [w["tier2"]], [w["tier3"]]])
    mask = np.isfinite(stack)
    num = np.nansum(np.where(mask, stack * weights, 0.0), axis=0)
    den = np.nansum(np.where(mask, weights, 0.0), axis=0)
    return np.where(den > 0, num / den, np.nan)


def _method_scores(bank, router_models, ev, raw) -> dict[str, np.ndarray]:
    if not isinstance(raw, RawScores):
        raw = RawScores(raw.tier1, raw.tier2, raw.tier3, raw.degradation, list(getattr(raw, "paths", [])))
    out = {
        "baseline_cnn": np.asarray(raw.tier1, dtype=float),
        "baseline_forensic": np.asarray(raw.tier2, dtype=float),
        "baseline_clip": np.asarray(raw.tier3, dtype=float),
        "baseline_static_ensemble": _static_ensemble(raw),
        "baseline_existing_fusion": bank.existing_fusion(raw),
    }
    for name, router in router_models.items():
        out[name] = router.predict_proba(ev)
    return out


def _evaluate(scores: dict[str, np.ndarray], y, target_tpr: float) -> list[dict]:
    rows = []
    for method, s in scores.items():
        ok = np.isfinite(s)
        if ok.sum() < 2 or len(set(y[ok].tolist())) < 2:
            continue
        m = binary_metrics(y[ok], s[ok], threshold=0.5)
        f_at, thr = fpr_at_tpr(y[ok], s[ok], target_tpr)
        rows.append({"method": method, **{k: m[k] for k in
                     ("auc", "pr_auc", "accuracy", "f1", "fpr", "tpr", "ece")},
                     f"fpr_at_tpr{target_tpr}": f_at, "n": int(ok.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-real", required=True); ap.add_argument("--train-fake", required=True)
    ap.add_argument("--test-real", required=True); ap.add_argument("--test-fake", required=True)
    ap.add_argument("--out-dir", default="outputs/router_ablation")
    ap.add_argument("--profiler", default="outputs/transformation_profiler.joblib")
    ap.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    ap.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    ap.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    ap.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    ap.add_argument("--max-image-dim", type=int, default=None)
    ap.add_argument("--model", choices=["logreg", "gbdt"], default="logreg")
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--target-tpr", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    profiler = None
    if os.path.exists(args.profiler):
        from src.transformation.model import load_profiler

        profiler = load_profiler(args.profiler)

    bank = DetectorBank(args.tier1_checkpoint, args.tier2_classifier, args.tier3_probe,
                        args.fusion_model, profiler_path=args.profiler,
                        max_image_dim=args.max_image_dim)
    print(f"[bank] {bank.available}")

    print("Building TRAIN/VAL router data (degraded mix)...")
    train = build_router_data(args.train_real, args.train_fake, bank, profiler,
                              n_samples=args.n_train, seed=args.seed)

    router_models: dict[str, AdaptiveRouter] = {}
    for name, cfg in ROUTER_VARIANTS.items():
        cfg.model = args.model
        cfg.seed = args.seed
        if cfg.use_uncertainty:
            # No multi-crop cache wired in this harness yet (Feature 4 supplies it); the
            # uncertainty features degrade to their neutral fill, so router_full == router_profiler
            # unless an uncertainty cache is provided. Reported honestly in the table notes.
            pass
        router_models[name] = AdaptiveRouter(cfg).fit(train.evidence, train.labels)
        print(f"  fitted {name} [{cfg.tag()}]")

    # ---- TEST: clean pass + seeded-degraded pass on the SAME images -----------
    test_real = list_images(args.test_real)[: args.n_test] if args.n_test else list_images(args.test_real)
    test_fake = list_images(args.test_fake)[: args.n_test] if args.n_test else list_images(args.test_fake)
    test_paths = test_real + test_fake
    y = np.array([0] * len(test_real) + [1] * len(test_fake))

    from src.transforms.registry import build_conditions

    clean = next(c for c in build_conditions() if c.name == "clean")

    all_rows = []
    for pass_name, transform in [("clean", clean), ("degraded_mix", None)]:
        if pass_name == "degraded_mix":
            # per-image seeded condition
            data = build_router_data(args.test_real, args.test_fake, bank, profiler,
                                     n_samples=args.n_test, seed=args.seed + 777)
            ev, raw, yy = data.evidence, data.raw, data.labels
        else:
            ev, raw = bank.build_evidence(test_paths, transform=transform, profiler=profiler)
            yy = y
        scores = _method_scores(bank, router_models, ev, raw)
        for row in _evaluate(scores, yy, args.target_tpr):
            row["pass"] = pass_name
            all_rows.append(row)
        print(f"  evaluated pass={pass_name}")

    df = pd.DataFrame(all_rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "ablation.csv"), index=False)

    piv = df.pivot_table(index="method", columns="pass", values="auc")
    piv["delta_clean_minus_degraded"] = piv.get("clean") - piv.get("degraded_mix")
    piv.to_csv(os.path.join(args.out_dir, "ablation_auc.csv"))

    lines = ["# Router — Baselines & Ablation", "",
             f"- seed {args.seed} · router model `{args.model}` · target TPR {args.target_tpr}",
             f"- train/val: {len(train.labels)} imgs (degraded mix) · test: {len(y)} imgs",
             "- `router_full` == `router_profiler` unless a Feature 4 multi-crop uncertainty cache is supplied.",
             "", "## ROC-AUC", "", df_to_markdown(piv.round(4)), "",
             "## Full metrics", "", df_to_markdown(df.round(4), index=False), ""]
    with open(os.path.join(args.out_dir, "ablation.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n" + piv.round(4).to_string())
    print(f"\nWrote {args.out_dir}/ablation.csv, ablation_auc.csv, ablation.md")


if __name__ == "__main__":
    main()
