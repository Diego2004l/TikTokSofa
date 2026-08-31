"""Feature 1 -- Transformation stress test + robustness benchmark.

Systematically measures every detector *model* (Tier 1 CNN, Tier 2 forensic, Tier 3 CLIP, the
static weighted ensemble, the existing degradation-aware fusion, and -- once Feature 3 lands --
the adaptive router) under every condition in `src.transforms.registry`: isolated transforms at
several severities, named compound chains, and deterministic seeded random compound chains.

Reproducible with a single command:

    python -m src.eval.robustness_bench \
        --real-dir data/raw/benchmark/real --fake-dir data/raw/benchmark/fake \
        --out-dir outputs/robustness --n-samples 300 --seed 0

Outputs (in --out-dir):
    results.csv / results.json  -- one row per (model, condition), full metric set
    report.md                   -- human-readable matrix + per-family degradation summary
    figures/degradation_*.png   -- AUC-vs-severity curves per transform family

ANTI-LEAKAGE: --real-dir / --fake-dir must be a HELD-OUT split never used to train any tier, fit
the fusion meta-model, or tune thresholds. This script only *evaluates*; it never fits anything.
Per-model decision thresholds, if supplied via --thresholds-json, must have been tuned on a
separate validation split (see src/eval/tune_thresholds.py once Feature 4 lands). Absent that,
every model is scored at a raw 0.5 cut and threshold-dependent metrics are clearly caveated.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd

from src.eval.metrics import binary_metrics
from src.eval.scoring import MODEL_NAMES, DetectorBank
from src.model.train import list_images
from src.transforms.registry import build_conditions

CURVE_FAMILIES = ("compression", "resolution", "blur", "noise")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _load_thresholds(path: str | None) -> dict[str, float]:
    if not path or not os.path.exists(path):
        return {m: 0.5 for m in MODEL_NAMES}
    with open(path) as f:
        raw = json.load(f)
    return {m: float(raw.get(m, 0.5)) for m in MODEL_NAMES}


def run_benchmark(args) -> tuple[pd.DataFrame, DetectorBank]:
    real_paths = list_images(args.real_dir)
    fake_paths = list_images(args.fake_dir)
    if args.n_samples:
        real_paths = real_paths[: args.n_samples]
        fake_paths = fake_paths[: args.n_samples]
    paths = real_paths + fake_paths
    y_true = np.array([0] * len(real_paths) + [1] * len(fake_paths))
    if len(set(y_true.tolist())) < 2:
        raise SystemExit("Need both real and fake images.")

    bank = DetectorBank(
        tier1_ckpt=args.tier1_checkpoint,
        tier2_clf=args.tier2_classifier,
        tier3_probe=args.tier3_probe,
        fusion_model=args.fusion_model,
        profiler_path=args.transformation_profiler,
        router_path=args.router_model,
        max_image_dim=args.max_image_dim,
    )
    print(f"[bank] available tiers: {bank.available}")
    thresholds = _load_thresholds(args.thresholds_json)

    conditions = build_conditions(seed=args.seed, include_optional=not args.no_optional)
    print(f"[bench] {len(conditions)} conditions x {len(MODEL_NAMES)} models on "
          f"{len(real_paths)} real + {len(fake_paths)} fake images")

    rows = []
    for ci, cond in enumerate(conditions, 1):
        model_scores, raw, _ = bank.score_models(paths, transform=cond)
        for model, scores in model_scores.items():
            finite = np.isfinite(scores)
            if finite.sum() < 2 or len(set(y_true[finite].tolist())) < 2:
                continue
            m = binary_metrics(y_true[finite], scores[finite], threshold=thresholds[model])
            rows.append({
                "model": model,
                "transformation": cond.name,
                "family": cond.family,
                "severity": round(cond.severity, 4),
                "severity_rank": cond.severity_rank,
                "compound_id": cond.compound_id or "",
                "seed": cond.seed if cond.seed is not None else args.seed,
                **{k: m[k] for k in ("auc", "pr_auc", "accuracy", "precision", "recall", "f1",
                                     "fpr", "tpr", "brier", "ece", "tp", "fp", "fn", "tn")},
                "num_samples": m["num_samples"],
            })
        print(f"  [{ci}/{len(conditions)}] {cond.family}/{cond.name}")

    return pd.DataFrame(rows), bank


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pivot_auc(df: pd.DataFrame) -> pd.DataFrame:
    return df.pivot_table(index="model", columns="transformation", values="auc")


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model aggregate AUCs: clean, robust mean, per-family mean, compound mean."""
    out = []
    for model, g in df.groupby("model"):
        clean = g.loc[g.transformation == "clean", "auc"]
        non_clean = g[g.family != "clean"]
        rec = {
            "model": model,
            "clean_auc": float(clean.iloc[0]) if len(clean) else float("nan"),
            "robust_auc_mean": float(non_clean["auc"].mean()),
            "robust_auc_min": float(non_clean["auc"].min()),
            "compound_auc_mean": float(g.loc[g.family.isin(["compound", "random_compound"]), "auc"].mean()),
        }
        for fam, fg in g.groupby("family"):
            rec[f"{fam}_auc_mean"] = float(fg["auc"].mean())
        rec["final_score"] = 0.5 * rec["clean_auc"] + 0.5 * rec["robust_auc_mean"]
        out.append(rec)
    return pd.DataFrame(out).set_index("model")


def write_degradation_curves(df: pd.DataFrame, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    written = []
    for fam in CURVE_FAMILIES:
        fam_df = df[df.family == fam]
        clean_df = df[df.transformation == "clean"]
        if fam_df.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        for model, g in fam_df.groupby("model"):
            g = g.sort_values("severity_rank")
            xs = [0] + g["severity_rank"].tolist()
            c0 = clean_df.loc[clean_df.model == model, "auc"]
            ys = [float(c0.iloc[0]) if len(c0) else np.nan] + g["auc"].tolist()
            ax.plot(xs, ys, marker="o", label=model)
        ax.set_title(f"Degradation curve — {fam}")
        ax.set_xlabel("severity rank (0 = clean)")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        path = os.path.join(fig_dir, f"degradation_{fam}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path)
    return written


def write_report(df: pd.DataFrame, summary: pd.DataFrame, meta: dict, out_dir: str, figures: list[str]) -> str:
    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and np.isfinite(x) else "—"

    matrix_cols = [
        ("CLEAN", "clean"), ("JPEG50", "jpeg_q50"), ("RESIZE50", "resize_50pct"),
        ("BLUR2", "gauss_blur_s2.0"), ("NOISE.05", "gauss_noise_s0.05"),
        ("CROP50", "center_crop_50"), ("SCREENSHOT", "screenshot_typical"),
    ]
    auc_p = _pivot_auc(df)

    lines = ["# Robustness Benchmark", ""]
    lines.append(f"- git: `{meta['git_sha']}`  ·  generated: {meta['timestamp']}")
    lines.append(f"- split: **{meta['split']}** ({meta['n_real']} real / {meta['n_fake']} fake per condition)")
    lines.append(f"- tiers available: `{meta['tiers_available']}`")
    lines.append(f"- thresholds: `{meta['thresholds_source']}` — threshold-dependent metrics "
                 f"(accuracy/precision/FPR/TPR) are only meaningful if tuned on validation, not here.")
    lines.append("")
    lines.append("## ROC-AUC matrix (isolated conditions)")
    lines.append("")
    header = "| Model | " + " | ".join(h for h, _ in matrix_cols) + " | COMPOUND | ROBUST-MEAN |"
    lines.append(header)
    lines.append("|" + "---|" * (len(matrix_cols) + 3))
    for model in summary.index:
        cells = []
        for _, cond_name in matrix_cols:
            v = auc_p.loc[model, cond_name] if (model in auc_p.index and cond_name in auc_p.columns) else np.nan
            cells.append(fmt(float(v)))
        cells.append(fmt(summary.loc[model, "compound_auc_mean"]))
        cells.append(fmt(summary.loc[model, "robust_auc_mean"]))
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-model aggregate")
    lines.append("")
    agg_cols = ["clean_auc", "robust_auc_mean", "robust_auc_min", "compound_auc_mean", "final_score"]
    lines.append("| Model | " + " | ".join(agg_cols) + " |")
    lines.append("|" + "---|" * (len(agg_cols) + 1))
    for model in summary.index:
        lines.append(f"| {model} | " + " | ".join(fmt(summary.loc[model, c]) for c in agg_cols) + " |")
    lines.append("")

    lines.append("## Per-family AUC (mean over that family's conditions)")
    lines.append("")
    fam_cols = sorted(c for c in summary.columns if c.endswith("_auc_mean") and not c.startswith("robust")
                      and not c.startswith("compound"))
    lines.append("| Model | " + " | ".join(c.replace("_auc_mean", "") for c in fam_cols) + " |")
    lines.append("|" + "---|" * (len(fam_cols) + 1))
    for model in summary.index:
        lines.append(f"| {model} | " + " | ".join(fmt(summary.loc[model, c]) for c in fam_cols) + " |")
    lines.append("")

    if figures:
        lines.append("## Degradation curves")
        lines.append("")
        for p in figures:
            rel = os.path.relpath(p, out_dir)
            lines.append(f"![{os.path.basename(p)}]({rel})")
        lines.append("")

    lines.append("## Worst conditions per model (lowest AUC)")
    lines.append("")
    for model, g in df.groupby("model"):
        worst = g.nsmallest(3, "auc")[["transformation", "auc"]]
        pairs = ", ".join(f"{r.transformation} ({r.auc:.3f})" for r in worst.itertuples())
        lines.append(f"- **{model}**: {pairs}")
    lines.append("")

    report = "\n".join(lines)
    path = os.path.join(out_dir, "report.md")
    with open(path, "w") as f:
        f.write(report)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-dir", required=True, help="HELD-OUT real images (never used for training/tuning).")
    ap.add_argument("--fake-dir", required=True, help="HELD-OUT fake images (never used for training/tuning).")
    ap.add_argument("--out-dir", default="outputs/robustness")
    ap.add_argument("--split", default="test", help="Label for the split, written into results metadata.")
    ap.add_argument("--n-samples", type=int, default=300, help="Cap per class per condition.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-optional", action="store_true", help="Skip motion blur / salt-and-pepper.")
    ap.add_argument("--thresholds-json", default=None, help="Per-model decision thresholds from a VALIDATION tune.")
    ap.add_argument("--tier1-checkpoint", default="outputs/tier1_efficientnet_b0.pt")
    ap.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    ap.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    ap.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    ap.add_argument("--transformation-profiler", default="outputs/transformation_profiler.joblib")
    ap.add_argument("--router-model", default="outputs/router_model.joblib")
    ap.add_argument("--max-image-dim", type=int, default=None, help="Match the value Tier 2 was trained with.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()
    df, bank = run_benchmark(args)
    if df.empty:
        raise SystemExit("No results — are any tier artifacts present?")

    summary = build_summary(df)
    figures = write_degradation_curves(df, args.out_dir)

    real_n = len(list_images(args.real_dir)[: args.n_samples] if args.n_samples else list_images(args.real_dir))
    fake_n = len(list_images(args.fake_dir)[: args.n_samples] if args.n_samples else list_images(args.fake_dir))
    meta = {
        "git_sha": _git_sha(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "split": args.split,
        "n_real": real_n,
        "n_fake": fake_n,
        "n_samples_cap": args.n_samples,
        "seed": args.seed,
        "tiers_available": bank.available,
        "thresholds_source": args.thresholds_json or "raw 0.5 (untuned)",
        "elapsed_sec": round(time.time() - t0, 1),
    }

    csv_path = os.path.join(args.out_dir, "results.csv")
    json_path = os.path.join(args.out_dir, "results.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "summary": summary.reset_index().to_dict(orient="records"),
                   "results": df.to_dict(orient="records")}, f, indent=2)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"))
    report_path = write_report(df, summary, meta, args.out_dir, figures)

    print(f"\nWrote:\n  {csv_path}\n  {json_path}\n  {report_path}")
    for p in figures:
        print(f"  {p}")
    print(f"\nElapsed {meta['elapsed_sec']}s")


if __name__ == "__main__":
    main()
