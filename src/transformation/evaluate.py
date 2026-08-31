"""Evaluate a trained transformation profiler (Feature 2, "Required evaluation").

Three regimes, all on a held-out clean image set the profiler was NOT trained on:
  1. isolated       -- one transform family at a time, TRAIN-grid severities
  2. compound       -- 2-3 stacked families, TRAIN-grid severities
  3. unseen_severity -- one family at a time, EVAL-grid severities (never seen in training)

Reports per-transformation precision / recall / F1 and a per-label confusion matrix, plus the
`overall_degradation` MAE. Writes JSON + a Markdown table.

    python -m src.transformation.evaluate --clean-dir data/raw/benchmark/coco_holdout \
        --profiler outputs/transformation_profiler.joblib --out-dir outputs/profiler_eval
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.model.train import list_images
from src.transformation.model import PROFILE_LABELS, load_profiler
from src.transformation.synthesize import SynthConfig, build_synthetic_dataset, overall_severity


def _eval_regime(profiler, paths, cfg, name: str) -> dict:
    X, Y, S = build_synthetic_dataset(paths, cfg)
    cols = profiler._score_matrix(X)
    per_label = {}
    for i, label in enumerate(PROFILE_LABELS):
        y = Y[:, i].astype(int)
        pred = (cols[label] >= 0.5).astype(int)
        if len(set(y.tolist())) < 2:
            continue
        p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        try:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y, cols[label]))
        except ValueError:
            auc = float("nan")
        per_label[label] = {
            "precision": float(p), "recall": float(r), "f1": float(f1), "auc": auc,
            "confusion": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)},
            "support_pos": int(y.sum()), "n": int(len(y)),
        }
    ov_true = np.array([overall_severity(row) for row in S])
    ov_mae = float(np.mean(np.abs(ov_true - cols["overall_degradation"])))
    macro_f1 = float(np.mean([v["f1"] for v in per_label.values()])) if per_label else float("nan")
    return {"regime": name, "n_examples": int(len(X)), "macro_f1": macro_f1,
            "overall_degradation_mae": ov_mae, "per_label": per_label}


def _md(results: list[dict]) -> str:
    lines = ["# Transformation Profiler — Evaluation", ""]
    for res in results:
        lines.append(f"## {res['regime']}  (n={res['n_examples']}, macro-F1={res['macro_f1']:.3f}, "
                     f"overall MAE={res['overall_degradation_mae']:.3f})")
        lines.append("")
        lines.append("| label | precision | recall | F1 | AUC | TP | FP | FN | TN |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for label, v in res["per_label"].items():
            c = v["confusion"]
            lines.append(f"| {label} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
                         f"{v['auc']:.3f} | {c['tp']} | {c['fp']} | {c['fn']} | {c['tn']} |")
        lines.append("")
    lines.append("_Isolated & compound use the training severity grid; `unseen_severity` uses a "
                 "disjoint grid never seen in training. Clean images here are held out from profiler training._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", required=True, action="append")
    ap.add_argument("--profiler", default="outputs/transformation_profiler.joblib")
    ap.add_argument("--out-dir", default="outputs/profiler_eval")
    ap.add_argument("--n-images", type=int, default=None)
    ap.add_argument("--variants-per-image", type=int, default=6)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    profiler = load_profiler(args.profiler)
    paths = sorted(set(p for d in args.clean_dir for p in list_images(d)))
    if args.n_images:
        paths = paths[: args.n_images]
    if not paths:
        raise SystemExit("No clean images found.")

    base = dict(variants_per_image=args.variants_per_image, seed=args.seed)
    regimes = [
        _eval_regime(profiler, paths, SynthConfig(max_ops=1, grid="train", **base), "isolated"),
        _eval_regime(profiler, paths, SynthConfig(max_ops=3, grid="train", **base), "compound"),
        _eval_regime(profiler, paths, SynthConfig(max_ops=1, grid="eval", **base), "unseen_severity"),
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "profiler_eval.json"), "w") as f:
        json.dump({"profiler": args.profiler, "metadata": profiler.metadata, "regimes": regimes}, f, indent=2)
    md_path = os.path.join(args.out_dir, "profiler_eval.md")
    with open(md_path, "w") as f:
        f.write(_md(regimes))
    print(open(md_path).read())
    print(f"\nWrote {md_path}")


if __name__ == "__main__":
    main()
