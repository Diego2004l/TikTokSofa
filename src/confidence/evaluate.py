"""Evaluate the abstention policy on a held-out TEST split (Feature 4).

Reports, at the tuned operating point and across the whole risk-coverage curve:
  coverage · selective accuracy · FPR · TPR · abstention rate

and shows that abstaining on the low-confidence tail improves reliability on the answered set.

    python -m src.confidence.evaluate \
        --real-dir data/raw/benchmark/real --fake-dir data/raw/benchmark/fake \
        --confidence-model outputs/confidence_model.joblib --policy outputs/abstention_policy.joblib \
        --out-dir outputs/abstention_eval
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np

from src.confidence.build import build_confidence_data
from src.confidence.model import load_policy
from src.eval.metrics import binary_metrics
from src.eval.scoring import DetectorBank


def risk_coverage_curve(scores, labels, confidences, n_points: int = 20) -> list[dict]:
    order = np.argsort(-np.asarray(confidences))  # most confident first
    scores, labels = np.asarray(scores)[order], np.asarray(labels)[order]
    n = len(scores)
    out = []
    for frac in np.linspace(1.0, 0.1, n_points):
        k = max(2, int(frac * n))
        s, y = scores[:k], labels[:k]
        if len(set(y.tolist())) < 2:
            continue
        pred = (s >= 0.5).astype(int)
        acc = float((pred == y).mean())
        fp = int(((pred == 1) & (y == 0)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        out.append({"coverage": k / n, "selective_accuracy": acc, "risk": 1 - acc,
                    "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
                    "tpr": tp / (tp + fn) if (tp + fn) else 0.0})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True)
    ap.add_argument("--fake-dir", required=True)
    ap.add_argument("--confidence-model", default="outputs/confidence_model.joblib")
    ap.add_argument("--policy", default="outputs/abstention_policy.joblib")
    ap.add_argument("--out-dir", default="outputs/abstention_eval")
    ap.add_argument("--profiler", default="outputs/transformation_profiler.joblib")
    ap.add_argument("--router-model", default="outputs/router_model.joblib")
    ap.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    ap.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    ap.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    ap.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    ap.add_argument("--max-image-dim", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--multicrop-n", type=int, default=5)
    ap.add_argument("--all-multicrop", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    bank = DetectorBank(args.tier1_checkpoint, args.tier2_classifier, args.tier3_probe,
                        args.fusion_model, profiler_path=args.profiler, router_path=args.router_model,
                        max_image_dim=args.max_image_dim)
    conf_model = joblib.load(args.confidence_model)
    policy = load_policy(args.policy)

    data = build_confidence_data(args.real_dir, args.fake_dir, bank, n_samples=args.n_samples,
                                 multicrop_n=args.multicrop_n, seed=args.seed,
                                 force_all_multicrop=args.all_multicrop)
    confidence = conf_model.predict_confidence(data.signals)
    decisions = policy.decide_batch(data.final_scores, confidence)

    answered = np.array([not d["abstained"] for d in decisions])
    y = data.labels
    pred_all = (data.final_scores >= 0.5).astype(int)

    full = binary_metrics(y, data.final_scores, threshold=0.5)
    sel = binary_metrics(y[answered], data.final_scores[answered], threshold=0.5) if answered.sum() > 2 else {}

    summary = {
        "n": int(len(y)),
        "coverage": float(answered.mean()),
        "abstention_rate": float(1 - answered.mean()),
        "accuracy_no_abstention": full["accuracy"],
        "fpr_no_abstention": full["fpr"],
        "selective_accuracy": sel.get("accuracy"),
        "selective_fpr": sel.get("fpr"),
        "selective_tpr": sel.get("tpr"),
        "unknown_count": int((~answered).sum()),
        "policy": policy.params.__dict__,
    }
    curve = risk_coverage_curve(data.final_scores, y, confidence)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "abstention_eval.json"), "w") as f:
        json.dump({"summary": summary, "risk_coverage": curve,
                   "decisions_sample": decisions[:20]}, f, indent=2)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        cov = [p["coverage"] for p in curve]
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(cov, [p["risk"] for p in curve], marker="o")
        ax[0].set_xlabel("coverage"); ax[0].set_ylabel("risk (1 - selective acc)")
        ax[0].set_title("Risk–coverage"); ax[0].grid(alpha=0.3)
        ax[1].plot(cov, [p["fpr"] for p in curve], marker="o", label="FPR")
        ax[1].plot(cov, [p["tpr"] for p in curve], marker="s", label="TPR")
        ax[1].set_xlabel("coverage"); ax[1].set_title("FPR / TPR vs coverage")
        ax[1].legend(); ax[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "risk_coverage.png"), dpi=120)
        plt.close(fig)
    except Exception as e:  # pragma: no cover
        print(f"[warn] could not draw risk-coverage curve: {e}")

    lines = ["# Abstention Evaluation", "",
             f"- n={summary['n']} · coverage={summary['coverage']:.3f} · "
             f"abstention rate={summary['abstention_rate']:.3f}",
             "",
             "| metric | no abstention | with abstention (answered set) |",
             "|---|---|---|",
             f"| accuracy | {summary['accuracy_no_abstention']:.3f} | "
             f"{_f(summary['selective_accuracy'])} |",
             f"| FPR | {summary['fpr_no_abstention']:.3f} | {_f(summary['selective_fpr'])} |",
             f"| TPR | {full['tpr']:.3f} | {_f(summary['selective_tpr'])} |",
             "",
             "Abstention improves reliability on the answered set when selective accuracy > raw "
             "accuracy and selective FPR < raw FPR.", "",
             "## Risk–coverage", "",
             "| coverage | selective acc | risk | FPR | TPR |", "|---|---|---|---|---|"]
    for p in curve:
        lines.append(f"| {p['coverage']:.2f} | {p['selective_accuracy']:.3f} | {p['risk']:.3f} | "
                     f"{p['fpr']:.3f} | {p['tpr']:.3f} |")
    with open(os.path.join(args.out_dir, "abstention_eval.md"), "w") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, indent=2))
    print(f"\nWrote {args.out_dir}/abstention_eval.{{json,md}} + risk_coverage.png")


def _f(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) and x is not None else "—"


if __name__ == "__main__":
    main()
