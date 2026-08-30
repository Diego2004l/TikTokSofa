"""Materialize a SID_Set subset as real/fake image directories the training scripts can consume.

SID_Set (HuggingFace `saberzl/SID_Set`, public, no token) ships as ~500 MB parquet shards with
columns: img_id, image (PIL), mask (PIL), width, height, label. Label convention:
  0 = real, 1 = fully synthetic (AI-generated), 2 = tampered (locally edited/inpainted).

This script downloads a chosen number of shards and writes decoded images to
  <out>/real/<img_id>.png   (label 0)
  <out>/fake/<img_id>.png   (label 1, and label 2 unless --exclude-tampered)
which is exactly the `--real-dir` / `--fake-dir` layout every src/ training script expects.

The full dataset is 140 GB; a few shards (default 4 train + 1 validation ~= a few thousand
images per class) is plenty to train a first real version of every tier. Bump --train-shards
for the final run. Run on Colab/Kaggle where the HF download is fast — unauthenticated local
downloads are heavily rate-limited (set HF_TOKEN in .env to speed them up).

Usage:
  python -m scripts.prep_sid_set --out data/raw/sid_set --train-shards 4 --val-shards 1
"""

from __future__ import annotations

import argparse
import io
import os

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image
from tqdm import tqdm

REPO = "saberzl/SID_Set"
N_TRAIN_SHARDS_TOTAL = 249
N_VAL_SHARDS_TOTAL = 34


def shard_name(split: str, i: int, total: int) -> str:
    return f"data/{split}-{i:05d}-of-{total:05d}.parquet"


def label_to_class(label: int, exclude_tampered: bool) -> str | None:
    if label == 0:
        return "real"
    if label == 1:
        return "fake"
    if label == 2:
        return None if exclude_tampered else "fake"
    return None


def extract_shard(path: str, out_dir: str, exclude_tampered: bool, max_per_shard: int | None) -> dict[str, int]:
    counts = {"real": 0, "fake": 0}
    pf = pq.ParquetFile(path)
    n = 0
    for batch in pf.iter_batches(batch_size=64, columns=["img_id", "image", "label"]):
        rows = batch.to_pylist()
        for row in rows:
            cls = label_to_class(row["label"], exclude_tampered)
            if cls is None:
                continue
            img_bytes = row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            dest = os.path.join(out_dir, cls, f"{row['img_id']}.png")
            img.save(dest)
            counts[cls] += 1
            n += 1
        if max_per_shard is not None and n >= max_per_shard:
            break
    return counts


def main():
    parser = argparse.ArgumentParser(description="Extract a SID_Set subset into real/ and fake/ image dirs.")
    parser.add_argument("--out", default="data/raw/sid_set")
    parser.add_argument("--train-shards", type=int, default=4)
    parser.add_argument("--val-shards", type=int, default=1)
    parser.add_argument("--exclude-tampered", action="store_true", help="Drop label==2 (locally edited) images instead of treating them as fake.")
    parser.add_argument("--max-per-shard", type=int, default=None, help="Stop after this many images per shard (fast dry run).")
    args = parser.parse_args()

    for split_dir in ("train", "val"):
        for cls in ("real", "fake"):
            os.makedirs(os.path.join(args.out, split_dir, cls), exist_ok=True)

    plan = [("train", i) for i in range(min(args.train_shards, N_TRAIN_SHARDS_TOTAL))]
    plan += [("validation", i) for i in range(min(args.val_shards, N_VAL_SHARDS_TOTAL))]

    grand = {"real": 0, "fake": 0}
    for split, i in tqdm(plan, desc="shards"):
        total = N_TRAIN_SHARDS_TOTAL if split == "train" else N_VAL_SHARDS_TOTAL
        fname = shard_name(split, i, total)
        local = hf_hub_download(REPO, fname, repo_type="dataset")
        out_split = "train" if split == "train" else "val"
        counts = extract_shard(local, os.path.join(args.out, out_split), args.exclude_tampered, args.max_per_shard)
        for k, v in counts.items():
            grand[k] += v
        os.remove(local)  # reclaim disk — the decoded PNGs are what we keep
        tqdm.write(f"  {fname}: +{counts}")

    print(f"\nDone. Total extracted: {grand}")
    print(f"  train: {args.out}/train/{{real,fake}}")
    print(f"  held-out (for fusion + eval): {args.out}/val/{{real,fake}}")


if __name__ == "__main__":
    main()
