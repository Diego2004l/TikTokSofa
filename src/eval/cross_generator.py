"""Cross-generator holdout test (spec section 5 + section 6 shortcut-learning safeguard).

Train on every generator family in `--data-root` EXCEPT `--holdout-family`, then evaluate purely
on the held-out family. This is the real generalization test: a model that only memorized
artifacts specific to the generators it trained on will show a large AUC drop here relative to
its in-distribution validation AUC, which is exactly the gap the write-up in
docs/shortcut_learning_check.md needs to report honestly.

Expects `--data-root` laid out as `<family>/{real,fake}/...` (see data/README.md's WildFake
section).
"""

from __future__ import annotations

import argparse
import os
import shutil

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


def main():
    parser = argparse.ArgumentParser(description="Cross-generator holdout generalization test.")
    parser.add_argument("--data-root", required=True, help="Root dir with one subfolder per generator family, each containing real/ and fake/.")
    parser.add_argument("--holdout-family", required=True)
    parser.add_argument("--tmp-root", default="outputs/_cross_generator_train")
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    merge_families(args.data_root, args.holdout_family, args.tmp_root)
    train_real, train_fake = os.path.join(args.tmp_root, "real"), os.path.join(args.tmp_root, "fake")
    holdout_real = os.path.join(args.data_root, args.holdout_family, "real")
    holdout_fake = os.path.join(args.data_root, args.holdout_family, "fake")

    print(f"Training on all families except '{args.holdout_family}'...")
    print(f"  train real: {len(list_images(train_real))} images, train fake: {len(list_images(train_fake))} images")
    print(f"  holdout '{args.holdout_family}' real: {len(list_images(holdout_real))}, fake: {len(list_images(holdout_fake))}")

    import sys

    sys.argv = [
        "train.py", "--real-dir", train_real, "--fake-dir", train_fake,
        "--epochs", str(args.epochs), "--out", f"outputs/tier1_cross_gen_holdout_{args.holdout_family}.pt",
    ]
    from src.model.train import main as train_main

    train_main()

    from src.model.evaluate import evaluate_auc
    from src.model.model import load_checkpoint

    model = load_checkpoint(f"outputs/tier1_cross_gen_holdout_{args.holdout_family}.pt")
    holdout_auc = evaluate_auc(model, holdout_real, holdout_fake)
    print(f"\nTier 1 AUC on held-out family '{args.holdout_family}': {holdout_auc:.4f}")
    print("Compare this against the in-distribution validation AUC printed during training above —")
    print("a large gap indicates the model leaned on generator-specific shortcuts rather than")
    print("generalizable AIGC artifacts. Record both numbers in docs/shortcut_learning_check.md.")


if __name__ == "__main__":
    main()
