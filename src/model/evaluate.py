"""Score a Tier 1 checkpoint against a real/fake directory pair, optionally under one named
transform condition from `src.augmentations` — this is what `src/eval/robustness.py` calls
per-condition to fill in Tier 1's row of the robustness table.
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from src.augmentations import Transform, all_conditions
from src.model.model import Tier1CNN, build_transform, load_checkpoint
from src.model.train import list_images


@torch.no_grad()
def score_images(model: Tier1CNN, paths: list[str], transform: Transform | None = None, device: str = "cpu") -> list[float]:
    preprocess = build_transform()
    scores = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        if transform is not None:
            img = transform(img)
        tensor = preprocess(img).unsqueeze(0).to(device)
        scores.append(model.predict_proba(tensor).item())
    return scores


def evaluate_auc(model: Tier1CNN, real_dir: str, fake_dir: str, transform: Transform | None = None, device: str = "cpu") -> float:
    real_scores = score_images(model, list_images(real_dir), transform, device)
    fake_scores = score_images(model, list_images(fake_dir), transform, device)
    y_true = [0] * len(real_scores) + [1] * len(fake_scores)
    y_score = real_scores + fake_scores
    return roc_auc_score(y_true, y_score)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Tier 1 checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--condition", default="clean", choices=list(all_conditions().keys()))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = load_checkpoint(args.checkpoint, args.device)
    transform = all_conditions()[args.condition]
    auc = evaluate_auc(model, args.real_dir, args.fake_dir, transform, args.device)
    print(f"Tier 1 AUC [{args.condition}]: {auc:.4f}")


if __name__ == "__main__":
    main()
