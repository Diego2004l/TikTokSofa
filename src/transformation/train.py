"""Train the transformation profiler on auto-labelled synthetic data (Feature 2).

    python -m src.transformation.train --clean-dir data/raw/benchmark/real \
        --out outputs/transformation_profiler.joblib --variants-per-image 6

`--clean-dir` should hold images that are as close to un-transformed as you have (the benchmark
real set, or COCO). They are NEVER used as a detector training signal here -- only as a substrate
for known synthetic transforms. Held out from this: nothing detector-related; the profiler is a
separate model. But keep `--clean-dir` disjoint from the robustness benchmark's real split if you
want the profiler's own eval (evaluate.py) to be clean.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src.model.train import list_images
from src.transformation.model import PROFILE_LABELS, TransformationProfiler
from src.transformation.synthesize import SynthConfig, build_synthetic_dataset, overall_severity


def _report(profiler: TransformationProfiler, X, Y, S, tag: str) -> dict:
    cols = profiler._score_matrix(X)
    print(f"\n[{tag}] per-label (n={len(X)}):")
    print(f"  {'label':<20s} {'pos':>5s} {'AUC':>7s} {'AP':>7s} {'F1@0.5':>7s}")
    out = {}
    for i, label in enumerate(PROFILE_LABELS):
        y = Y[:, i].astype(int)
        p = cols[label]
        if len(set(y.tolist())) < 2:
            print(f"  {label:<20s} {int(y.sum()):>5d}   (single-class)")
            continue
        auc = roc_auc_score(y, p)
        ap = average_precision_score(y, p)
        f1 = f1_score(y, (p >= 0.5).astype(int))
        out[label] = {"auc": auc, "ap": ap, "f1": f1, "pos": int(y.sum())}
        print(f"  {label:<20s} {int(y.sum()):>5d} {auc:>7.3f} {ap:>7.3f} {f1:>7.3f}")
    ov_true = np.array([overall_severity(r) for r in S])
    ov_pred = cols["overall_degradation"]
    mae = float(np.mean(np.abs(ov_true - ov_pred)))
    print(f"  overall_degradation MAE: {mae:.3f}")
    out["overall_degradation"] = {"mae": mae}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clean-dir", required=True, action="append",
                    help="Dir of ~clean images. Repeatable.")
    ap.add_argument("--out", default="outputs/transformation_profiler.joblib")
    ap.add_argument("--variants-per-image", type=int, default=6)
    ap.add_argument("--max-ops", type=int, default=3)
    ap.add_argument("--n-images", type=int, default=None, help="Cap clean images (speed).")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    paths = []
    for d in args.clean_dir:
        paths.extend(list_images(d))
    paths = sorted(set(paths))
    if args.n_images:
        paths = paths[: args.n_images]
    if not paths:
        raise SystemExit("No clean images found.")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(paths))
    n_val = int(len(paths) * args.val_frac)
    val_paths = [paths[i] for i in perm[:n_val]]
    train_paths = [paths[i] for i in perm[n_val:]]
    print(f"{len(train_paths)} train / {len(val_paths)} val clean images")

    train_cfg = SynthConfig(variants_per_image=args.variants_per_image, max_ops=args.max_ops,
                            grid="train", seed=args.seed)
    val_cfg = SynthConfig(variants_per_image=args.variants_per_image, max_ops=args.max_ops,
                          grid="train", seed=args.seed + 1)

    print("Building synthetic TRAIN set...")
    Xtr, Ytr, Str = build_synthetic_dataset(train_paths, train_cfg)
    print("Building synthetic VAL set (same severity grid)...")
    Xva, Yva, Sva = build_synthetic_dataset(val_paths, val_cfg)

    profiler = TransformationProfiler(calibrate=not args.no_calibrate, random_state=args.seed)
    profiler.fit(Xtr, Ytr, Str)
    profiler.metadata = {
        "n_clean_train": len(train_paths), "n_clean_val": len(val_paths),
        "n_synth_train": len(Xtr), "variants_per_image": args.variants_per_image,
        "max_ops": args.max_ops, "seed": args.seed, "calibrated": not args.no_calibrate,
    }

    _report(profiler, Xtr, Ytr, Str, "train")
    profiler.metadata["val_metrics"] = _report(profiler, Xva, Yva, Sva, "val")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    profiler.save(args.out)
    print(f"\nSaved profiler to {args.out}")
    print("Run `python -m src.transformation.evaluate` for isolated / compound / unseen-severity metrics.")


if __name__ == "__main__":
    main()
