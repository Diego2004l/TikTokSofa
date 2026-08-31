"""Assemble the final research table (Feature 5).

Merges the artifacts the other feature evaluations already wrote into one table:

    Method | Clean AUC | Robust AUC | Unseen-generator AUC | FPR @ target TPR | Abstention rate | Avg inference cost

Inputs (all optional -- missing cells render as "—"):
  --robustness   outputs/robustness/summary.csv            (Feature 1)  -> clean / robust AUC
  --ablation     outputs/router_ablation/ablation.csv      (Feature 3)  -> FPR @ target TPR
  --cross-gen    outputs/cross_generator/known_vs_unseen.csv(Feature 5) -> unseen-generator AUC
  --abstention   outputs/abstention_eval/abstention_eval.json(Feature 4)-> abstention rate

    python -m src.eval.research_table --out-dir outputs/research_table
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

from src.eval.cost_model import method_cost
from src.eval.metrics import df_to_markdown

# canonical method -> (robustness_bench name, ablation name, cross_gen name)
METHODS = {
    "CNN only":            ("cnn", "baseline_cnn", "cnn"),
    "Forensic only":       ("forensic", "baseline_forensic", "forensic"),
    "CLIP only":           ("clip", "baseline_clip", "clip"),
    "Static ensemble":     ("static_ensemble", "baseline_static_ensemble", "static_ensemble"),
    "Existing fusion":     ("existing_fusion", "baseline_existing_fusion", "existing_fusion"),
    "Adaptive router":     ("adaptive_router", "router_profiler", "adaptive_router"),
    "Full system (+abstention)": ("adaptive_router", "router_full", "adaptive_router"),
}

COST_KEY = {
    "CNN only": "baseline_cnn", "Forensic only": "baseline_forensic", "CLIP only": "baseline_clip",
    "Static ensemble": "baseline_static_ensemble", "Existing fusion": "baseline_existing_fusion",
    "Adaptive router": "router_profiler", "Full system (+abstention)": "router_full",
}


def _load_csv(path):
    return pd.read_csv(path) if path and os.path.exists(path) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robustness", default="outputs/robustness/summary.csv")
    ap.add_argument("--ablation", default="outputs/router_ablation/ablation.csv")
    ap.add_argument("--cross-gen", default="outputs/cross_generator/known_vs_unseen.csv")
    ap.add_argument("--abstention", default="outputs/abstention_eval/abstention_eval.json")
    ap.add_argument("--target-tpr", type=float, default=0.9)
    ap.add_argument("--out-dir", default="outputs/research_table")
    args = ap.parse_args()

    rob = _load_csv(args.robustness)
    if rob is not None:
        rob = rob.set_index(rob.columns[0])
    abl = _load_csv(args.ablation)
    xg = _load_csv(args.cross_gen)
    if xg is not None:
        xg = xg.set_index(xg.columns[0])

    abst_rate = None
    if os.path.exists(args.abstention):
        with open(args.abstention) as f:
            abst_rate = json.load(f).get("summary", {}).get("abstention_rate")

    fpr_col = f"fpr_at_tpr{args.target_tpr}"
    rows = []
    for name, (rb, ab, cg) in METHODS.items():
        clean = robust = unseen = fpr = abst = None
        if rob is not None and rb in rob.index:
            clean = float(rob.loc[rb, "clean_auc"]) if "clean_auc" in rob.columns else None
            robust = float(rob.loc[rb, "robust_auc_mean"]) if "robust_auc_mean" in rob.columns else None
        if abl is not None and fpr_col in abl.columns:
            sub = abl[(abl["method"] == ab) & (abl["pass"] == "degraded_mix")]
            if len(sub):
                fpr = float(sub[fpr_col].iloc[0])
        if xg is not None and ab.replace("baseline_", "").replace("router_profiler", "adaptive_router") in xg.index:
            key = cg
            if key in xg.index and "unseen_generator" in xg.columns:
                unseen = float(xg.loc[key, "unseen_generator"])
        if name.startswith("Full system"):
            abst = abst_rate
        rows.append({
            "Method": name,
            "Clean AUC": clean,
            "Robust AUC": robust,
            "Unseen-generator AUC": unseen,
            f"FPR @ TPR {args.target_tpr}": fpr,
            "Abstention rate": abst if abst is not None else (0.0 if not name.startswith("Full") else None),
            "Avg inference cost (rel.)": method_cost(
                COST_KEY[name], escalated=True,
                profiler=name in ("Adaptive router", "Full system (+abstention)"),
                multicrop_crops=5 if name.startswith("Full system") else 0),
        })

    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "research_table.csv"), index=False)
    md = ["# Final Research Table", "",
          "Merged from the Feature 1/3/4/5 evaluation artifacts. Missing inputs render as `—` — "
          "run the corresponding eval first (see each column's source in the module docstring).",
          "", df_to_markdown(df.round(4), index=False), "",
          "- **Clean / Robust AUC**: `src.eval.robustness_bench` summary.",
          "- **Unseen-generator AUC**: `src.eval.cross_generator_full` known-vs-unseen (mean over folds).",
          f"- **FPR @ TPR {args.target_tpr}**: `src.router.baselines` degraded-mix pass.",
          "- **Abstention rate**: `src.confidence.evaluate` (applies to the full system row).",
          "- **Avg inference cost**: `src.eval.cost_model` relative units (Tier 2 forensic = 1.0)."]
    with open(os.path.join(args.out_dir, "research_table.md"), "w") as f:
        f.write("\n".join(md))
    print("\n".join(md))
    print(f"\nWrote {args.out_dir}/research_table.{{csv,md}}")


if __name__ == "__main__":
    main()
