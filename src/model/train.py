"""Fine-tune Tier 1 (EfficientNet-B0) with compound-augmented, class-symmetric training data.

Symmetric augmentation (spec section 6): the same `SymmetricAugmenter` instance samples a
transform independently for every image regardless of its label, so the model can't learn
"has JPEG artifacts => real" or similar shortcuts from an accidental correlation between
augmentation and class.
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from src.augmentations import SymmetricAugmenter
from src.model.model import Tier1CNN, build_transform

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def list_images(folder: str) -> list[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"**/*{ext}"), recursive=True))
    return sorted(paths)


class RealFakeDataset(Dataset):
    def __init__(self, real_dir: str, fake_dir: str, train: bool, seed: int = 0):
        self.paths = [(p, 0) for p in list_images(real_dir)] + [(p, 1) for p in list_images(fake_dir)]
        self.train = train
        self.augmenter = SymmetricAugmenter(rng=random.Random(seed)) if train else None
        self.transform = build_transform()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path, label = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.train:
            img = self.augmenter(img)
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32)


def split_train_val(real_dir: str, fake_dir: str, val_frac: float, seed: int):
    rng = random.Random(seed)
    real_paths, fake_paths = list_images(real_dir), list_images(fake_dir)
    rng.shuffle(real_paths)
    rng.shuffle(fake_paths)
    n_real_val, n_fake_val = int(len(real_paths) * val_frac), int(len(fake_paths) * val_frac)
    return (
        real_paths[n_real_val:], fake_paths[n_fake_val:],
        real_paths[:n_real_val], fake_paths[:n_fake_val],
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Tier 1 EfficientNet-B0.")
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--fake-dir", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="outputs/tier1_efficientnet_b0.pt")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    tmp_dir = os.path.dirname(args.out) or "."

    train_real, train_fake, val_real, val_fake = split_train_val(args.real_dir, args.fake_dir, args.val_frac, args.seed)

    # RealFakeDataset expects directories, so materialize the split as a symlink layout under
    # outputs/_splits rather than duplicating image bytes.
    split_root = os.path.join(tmp_dir, "_splits")
    for name, real_list, fake_list in (("train", train_real, train_fake), ("val", val_real, val_fake)):
        for cls, paths in (("real", real_list), ("fake", fake_list)):
            d = os.path.join(split_root, name, cls)
            os.makedirs(d, exist_ok=True)
            for p in paths:
                link = os.path.join(d, os.path.basename(p))
                if not os.path.exists(link):
                    os.symlink(os.path.abspath(p), link)

    train_ds = RealFakeDataset(os.path.join(split_root, "train", "real"), os.path.join(split_root, "train", "fake"), train=True, seed=args.seed)
    val_ds = RealFakeDataset(os.path.join(split_root, "val", "real"), os.path.join(split_root, "val", "fake"), train=False, seed=args.seed)
    pin = args.device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=pin)

    model = Tier1CNN(pretrained=True).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * x.size(0)

        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                probs = model.predict_proba(x.to(args.device)).cpu()
                val_probs.extend(probs.tolist())
                val_labels.extend(y.tolist())
        auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else float("nan")
        print(f"epoch {epoch + 1}/{args.epochs} | train_loss={total_loss / len(train_ds):.4f} | val_auc={auc:.4f}")

        if auc >= best_auc:
            best_auc = auc
            torch.save(model.state_dict(), args.out)

    print(f"Best val AUC: {best_auc:.4f}. Saved best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
