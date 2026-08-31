"""Cross-generator holdout test (spec section 5 + section 6 shortcut-learning safeguard).

Train on every generator family in `--data-root` EXCEPT `--holdout-family`, then evaluate purely
on the held-out family. This is the real generalization test: a model that only memorized
artifacts specific to the generators it trained on will show a large AUC drop here relative to
its in-distribution validation AUC, which is exactly the gap the write-up in
docs/shortcut_learning_check.md needs to report honestly.

Expects `--data-root` laid out as `<family>/{real,fake}/...` (see data/README.md's WildFake
section).

Feature 5 additions:
  * `--all-methods` -- after training the held-out Tier 1, evaluate EVERY method (CNN, forensic,
    CLIP, static ensemble, existing fusion, adaptive router) on the held-out family, clean +
    seeded-degraded, with AUC / PR-AUC / FPR / TPR. Writes JSON + Markdown.
  * `src/eval/cross_generator_full.py` runs the full leave-one-family-out rotation and the
    known-vs-unseen summary table in one command.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from src.model.train import list_images


def merge_families(data_root: str, exclude: str, dest_root: str) -> None:
    if os.path.exists(dest_root):
        shutil.rmtree(dest_root)
    for cls in ("real", "fake"):
        os.makedirs(os.path.join(dest_root, cls), exist_ok=True)

    for family in sorted(os.listdir(data_root)):
        if family == exclude:
            continue
        family_dir = os.path.join(data_root, family)
        if not os.path.isdir(family_dir):
            continue
        for cls in ("real", "fake"):
            src_dir = os.path.join(family_dir, cls)
            if not os.path.isdir(src_dir):
                continue
            for path in list_images(src_dir):
                link = os.path.join(dest_root, cls, f"{family}_{os.path.basename(path)}")
                if not os.path.exists(link):
                    os.symlink(os.path.abspath(path), link)


def discover_families(data_root: str) -> list[str]:
    return sorted(
        d for d in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, d, "real"))
        and os.path.isdir(os.path.join(data_root, d, "fake"))
    )


def evaluate_all_methods_on_holdout(
    data_root: str, holdout_family: str, *, tier1_ckpt: str, tier2_clf: str, tier3_probe: str,
    fusion_model: str, profiler_path: str | None, router_path: str | None,
    max_image_dim: int | None = None, n_samples: int | None = None, seed: int = 0,
    target_tpr: float = 0.9,
) -> dict:
    """Every method on the held-out family. Tiers/fusion/profiler/router MUST have been fit
    without this family (that is the caller's responsibility -- see cross_generator_full.py)."""
    from src.eval.metrics import binary_metrics, fpr_at_tpr
    from src.eval.scoring import DetectorBank
    from src.transforms.registry import build_conditions

    bank = DetectorBank(tier1_ckpt, tier2_clf, tier3_probe, fusion_model,
                        profiler_path=profiler_path, router_path=router_path,
                        max_image_dim=max_image_dim)

    real = list_images(os.path.join(data_root, holdout_family, "real"))
    fake = list_images(os.path.join(data_root, holdout_family, "fake"))
    if n_samples:
        real, fake = real[:n_samples], fake[:n_samples]
    paths = real + fake
    y = np.array([0] * len(real) + [1] * len(fake))

    clean = next(c for c in build_conditions() if c.name == "clean")
    rng = np.random.default_rng(seed)
    non_clean = [c for c in build_conditions(seed=seed, include_optional=False) if c.name != "clean"]

    rows = []
    for pass_name in ("clean", "degraded_mix"):
        if pass_name == "clean":
            model_scores, _, _ = bank.score_models(paths, transform=clean)
        else:
            # per-image seeded condition, grouped
            assign = [non_clean[rng.integers(len(non_clean))] for _ in paths]
            by_cond: dict = {}
            for i, c in enumerate(assign):
                by_cond.setdefault(c.name, []).append(i)
            name_to_cond = {c.name: c for c in non_clean}
            acc: dict[str, np.ndarray] = {}
            for cname, idxs in by_cond.items():
                ms, _, _ = bank.score_models([paths[i] for i in idxs], transform=name_to_cond[cname])
                for model, s in ms.items():
                    acc.setdefault(model, np.full(len(paths), np.nan))
                    acc[model][idxs] = s
            model_scores = acc
        for model, s in model_scores.items():
            s = np.asarray(s, dtype=float)
            ok = np.isfinite(s)
            if ok.sum() < 2 or len(set(y[ok].tolist())) < 2:
                continue
            m = binary_metrics(y[ok], s[ok], threshold=0.5)
            f_at, _ = fpr_at_tpr(y[ok], s[ok], target_tpr)
            rows.append({"holdout_family": holdout_family, "pass": pass_name, "method": model,
                         **{k: m[k] for k in ("auc", "pr_auc", "accuracy", "f1", "fpr", "tpr")},
                         f"fpr_at_tpr{target_tpr}": f_at, "n": int(ok.sum())})
    return {"holdout_family": holdout_family, "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Cross-generator holdout generalization test.")
    parser.add_argument("--data-root", required=True, help="Root dir with one subfolder per generator family, each containing real/ and fake/.")
    parser.add_argument("--holdout-family", required=True)
    parser.add_argument("--tmp-root", default="outputs/_cross_generator_train")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--all-methods", action="store_true",
                        help="Also evaluate every method (not just Tier 1) on the held-out family.")
    parser.add_argument("--skip-tier1-train", action="store_true",
                        help="Reuse --tier1-checkpoint instead of training a held-out Tier 1.")
    parser.add_argument("--tier1-checkpoint", default=None)
    parser.add_argument("--tier2-classifier", default="outputs/tier2_classifier.joblib")
    parser.add_argument("--tier3-probe", default="outputs/tier3_clip_probe.joblib")
    parser.add_argument("--fusion-model", default="outputs/fusion_model.joblib")
    parser.add_argument("--profiler", default="outputs/transformation_profiler.joblib")
    parser.add_argument("--router-model", default="outputs/router_model.joblib")
    parser.add_argument("--max-image-dim", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/cross_generator")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    holdout_real = os.path.join(args.data_root, args.holdout_family, "real")
    holdout_fake = os.path.join(args.data_root, args.holdout_family, "fake")
    ckpt = args.tier1_checkpoint or f"outputs/tier1_cross_gen_holdout_{args.holdout_family}.pt"

    if not args.skip_tier1_train:
        merge_families(args.data_root, args.holdout_family, args.tmp_root)
        train_real, train_fake = os.path.join(args.tmp_root, "real"), os.path.join(args.tmp_root, "fake")
        print(f"Training on all families except '{args.holdout_family}'...")
        print(f"  train real: {len(list_images(train_real))} images, train fake: {len(list_images(train_fake))} images")
        print(f"  holdout '{args.holdout_family}' real: {len(list_images(holdout_real))}, fake: {len(list_images(holdout_fake))}")

        import sys

        sys.argv = ["train.py", "--real-dir", train_real, "--fake-dir", train_fake,
                    "--epochs", str(args.epochs), "--out", ckpt]
        from src.model.train import main as train_main

        train_main()

    from src.model.evaluate import evaluate_auc
    from src.model.model import load_checkpoint

    model = load_checkpoint(ckpt)
    holdout_auc = evaluate_auc(model, holdout_real, holdout_fake)
    print(f"\nTier 1 AUC on held-out family '{args.holdout_family}': {holdout_auc:.4f}")
    print("Compare this against the in-distribution validation AUC printed during training above —")
    print("a large gap indicates the model leaned on generator-specific shortcuts rather than")
    print("generalizable AIGC artifacts. Record both numbers in docs/shortcut_learning_check.md.")

    if args.all_methods:
        os.makedirs(args.out_dir, exist_ok=True)
        res = evaluate_all_methods_on_holdout(
            args.data_root, args.holdout_family,
            tier1_ckpt=ckpt, tier2_clf=args.tier2_classifier, tier3_probe=args.tier3_probe,
            fusion_model=args.fusion_model, profiler_path=args.profiler, router_path=args.router_model,
            max_image_dim=args.max_image_dim, n_samples=args.n_samples, seed=args.seed,
        )
        out = os.path.join(args.out_dir, f"holdout_{args.holdout_family}.json")
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nAll-methods held-out results -> {out}")
        for r in res["rows"]:
            print(f"  {r['pass']:13s} {r['method']:20s} AUC={r['auc']:.4f} FPR={r['fpr']:.3f} TPR={r['tpr']:.3f}")


if __name__ == "__main__":
    main()
