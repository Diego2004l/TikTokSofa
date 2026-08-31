"""Full leave-one-generator-family-out rotation + known-vs-unseen summary (Feature 5).

For each generator family F in --data-root:
  * (unless --skip-training) train Tier 1/2/3, fit fusion, train the transformation profiler,
    train the router, and tune the abstention policy -- ALL on the other families only. F is
    never used for tier training, fusion training, threshold tuning, profiler tuning, or model
    selection. It is a genuine unseen-generator test.
  * evaluate every method (CNN / forensic / CLIP / static ensemble / existing fusion / adaptive
    router) on F (clean + seeded-degraded), and on a held-out slice of the *training* families
    ("known generators") for the side-by-side.

Outputs a per-family table and the known-vs-unseen summary the write-up needs.

    # reuse already-trained artifacts (fast; only sound if they were trained without leakage):
    python -m src.eval.cross_generator_full --data-root data/raw/wildfake --skip-training \
        --out-dir outputs/cross_generator

    # full rotation (heavy; Kaggle/Colab):
    python -m src.eval.cross_generator_full --data-root data/raw/wildfake \
        --epochs 4 --out-dir outputs/cross_generator
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from src.eval.cross_generator import (
    discover_families,
    evaluate_all_methods_on_holdout,
    merge_families,
)
from src.eval.metrics import df_to_markdown
from src.model.train import list_images

ARTIFACTS = dict(
    tier1="outputs/_xgen/tier1_{f}.pt",
    tier2="outputs/_xgen/tier2_{f}.joblib",
    tier3="outputs/_xgen/tier3_{f}.joblib",
    fusion="outputs/_xgen/fusion_{f}.joblib",
    profiler="outputs/_xgen/profiler_{f}.joblib",
    router="outputs/_xgen/router_{f}.joblib",
)


def _run(cmd: list[str]) -> None:
    print("  $", " ".join(cmd))
    subprocess.check_call([sys.executable, "-m", *cmd])


def _split_known_holdout(merged_root: str, out_root: str, val_frac: float, seed: int):
    """Carve a held-out 'known generators' eval slice out of the merged training families."""
    rng = np.random.default_rng(seed)
    for cls in ("real", "fake"):
        src = os.path.join(merged_root, cls)
        paths = list_images(src)
        rng.shuffle(paths)
        n_val = int(len(paths) * val_frac)
        for name, subset in (("train", paths[n_val:]), ("known_eval", paths[:n_val])):
            d = os.path.join(out_root, name, cls)
            os.makedirs(d, exist_ok=True)
            for p in subset:
                link = os.path.join(d, os.path.basename(p))
                if not os.path.exists(link):
                    os.symlink(os.path.abspath(p), link)


def train_fold(data_root: str, family: str, args) -> dict:
    a = {k: v.format(f=family) for k, v in ARTIFACTS.items()}
    os.makedirs("outputs/_xgen", exist_ok=True)
    merged = f"outputs/_xgen/merged_{family}"
    merge_families(data_root, family, merged)
    split_root = f"outputs/_xgen/split_{family}"
    if os.path.exists(split_root):
        shutil.rmtree(split_root)
    _split_known_holdout(merged, split_root, args.val_frac, args.seed)

    tr = (os.path.join(split_root, "train", "real"), os.path.join(split_root, "train", "fake"))
    ke = (os.path.join(split_root, "known_eval", "real"), os.path.join(split_root, "known_eval", "fake"))

    dim = ["--max-image-dim", str(args.max_image_dim)] if args.max_image_dim else []
    _run(["src.frequency.train_svm", "--real-dir", tr[0], "--fake-dir", tr[1], *dim, "--out", a["tier2"]])
    _run(["src.semantic.train_probe", "--real-dir", tr[0], "--fake-dir", tr[1], "--out", a["tier3"]])
    _run(["src.model.train", "--real-dir", tr[0], "--fake-dir", tr[1],
          "--epochs", str(args.epochs), "--out", a["tier1"]])
    _run(["src.train_fusion", "--real-dir", ke[0], "--fake-dir", ke[1], *dim,
          "--tier1-checkpoint", a["tier1"], "--tier2-classifier", a["tier2"],
          "--tier3-probe", a["tier3"], "--out", a["fusion"]])
    _run(["src.transformation.train", "--clean-dir", ke[0], "--out", a["profiler"],
          "--variants-per-image", "5"])
    _run(["src.router.train", "--real-dir", ke[0], "--fake-dir", ke[1], *dim,
          "--profiler", a["profiler"], "--tier1-checkpoint", a["tier1"],
          "--tier2-classifier", a["tier2"], "--tier3-probe", a["tier3"],
          "--fusion-model", a["fusion"], "--out", a["router"]])
    return {**a, "known_eval": ke}


def default_artifacts() -> dict:
    return {"tier1": "outputs/tier1_efficientnet_b0.pt", "tier2": "outputs/tier2_classifier.joblib",
            "tier3": "outputs/tier3_clip_probe.joblib", "fusion": "outputs/fusion_model.joblib",
            "profiler": "outputs/transformation_profiler.joblib", "router": "outputs/router_model.joblib"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out-dir", default="outputs/cross_generator")
    ap.add_argument("--families", nargs="*", default=None, help="Subset of families to rotate (default: all).")
    ap.add_argument("--skip-training", action="store_true", help="Reuse default outputs/ artifacts (no per-fold retrain).")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--max-image-dim", type=int, default=None)
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--target-tpr", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    families = args.families or discover_families(args.data_root)
    if not families:
        raise SystemExit(f"No <family>/{{real,fake}} subdirs under {args.data_root}")
    print(f"Rotating over families: {families}")

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []
    for fam in families:
        print(f"\n=== holdout: {fam} ===")
        art = default_artifacts() if args.skip_training else train_fold(args.data_root, fam, args)

        unseen = evaluate_all_methods_on_holdout(
            args.data_root, fam, tier1_ckpt=art["tier1"], tier2_clf=art["tier2"],
            tier3_probe=art["tier3"], fusion_model=art["fusion"], profiler_path=art["profiler"],
            router_path=art["router"], max_image_dim=args.max_image_dim,
            n_samples=args.n_samples, seed=args.seed, target_tpr=args.target_tpr,
        )
        for r in unseen["rows"]:
            r["regime"] = "unseen_generator"
            all_rows.append(r)

        if not args.skip_training:
            ke = art["known_eval"]
            from src.eval.cross_generator import evaluate_all_methods_on_holdout as _e  # same fn, known slice
            # Point the "holdout" at a temp family dir built from the known-eval slice.
            known_dir = f"outputs/_xgen/knownfam_{fam}"
            for cls, src in (("real", ke[0]), ("fake", ke[1])):
                d = os.path.join(known_dir, "known", cls)
                os.makedirs(d, exist_ok=True)
                for p in list_images(src):
                    link = os.path.join(d, os.path.basename(p))
                    if not os.path.exists(link):
                        os.symlink(os.path.abspath(p), link)
            known = _e(known_dir, "known", tier1_ckpt=art["tier1"], tier2_clf=art["tier2"],
                       tier3_probe=art["tier3"], fusion_model=art["fusion"],
                       profiler_path=art["profiler"], router_path=art["router"],
                       max_image_dim=args.max_image_dim, n_samples=args.n_samples,
                       seed=args.seed, target_tpr=args.target_tpr)
            for r in known["rows"]:
                r["holdout_family"] = fam
                r["regime"] = "known_generators"
                all_rows.append(r)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(args.out_dir, "cross_generator_full.csv"), index=False)

    # Known vs unseen summary (mean AUC over families, clean pass).
    clean = df[df["pass"] == "clean"]
    summary = clean.pivot_table(index="method", columns="regime", values="auc", aggfunc="mean")
    if "known_generators" in summary and "unseen_generator" in summary:
        summary["generalisation_gap"] = summary["known_generators"] - summary["unseen_generator"]
    summary.to_csv(os.path.join(args.out_dir, "known_vs_unseen.csv"))

    per_family = clean[clean.regime == "unseen_generator"].pivot_table(
        index="method", columns="holdout_family", values="auc")

    lines = ["# Cross-Generator Holdout (leave-one-family-out)", "",
             f"- families: {families}", f"- seed {args.seed} · target TPR {args.target_tpr}"
             + ("" if not args.skip_training else " · **--skip-training: reused default artifacts**"),
             "", "## Known vs unseen generators (mean ROC-AUC, clean pass)", "",
             df_to_markdown(summary.round(4)), "",
             "## Unseen-generator AUC per held-out family (clean pass)", "",
             df_to_markdown(per_family.round(4)), "",
             "## All rows", "", df_to_markdown(df.round(4), index=False), ""]
    with open(os.path.join(args.out_dir, "cross_generator_full.md"), "w") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.out_dir, "cross_generator_full.json"), "w") as f:
        json.dump({"families": families, "rows": all_rows,
                   "known_vs_unseen": summary.reset_index().to_dict(orient="records")}, f, indent=2)

    print("\n" + df_to_markdown(summary.round(4)))
    print(f"\nWrote {args.out_dir}/cross_generator_full.{{csv,md,json}} + known_vs_unseen.csv")


if __name__ == "__main__":
    main()
