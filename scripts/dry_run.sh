#!/usr/bin/env bash
# End-to-end dry run on a TINY subset — shakes out runtime bugs in every training/eval script
# before committing to a full (hours-long, GPU) training run.
#
# Usage:
#   scripts/dry_run.sh <real_dir> <fake_dir> [n_per_class]
#
# Example (after fetching CIFAKE per data/README.md):
#   scripts/dry_run.sh data/raw/cifake/train/REAL data/raw/cifake/train/FAKE 40
#
# What it proves: the four training scripts + the fusion driver + infer + robustness all run to
# completion on real image files and produce their output artifacts. It proves NOTHING about
# accuracy — 40 images and 1 epoch is far too little to learn anything.

set -euo pipefail

REAL_DIR="${1:?usage: dry_run.sh <real_dir> <fake_dir> [n_per_class]}"
FAKE_DIR="${2:?usage: dry_run.sh <real_dir> <fake_dir> [n_per_class]}"
N="${3:-40}"

PY="${PYTHON:-.venv/bin/python}"
WORK="outputs/_dry_run"
SUB_REAL="$WORK/real"
SUB_FAKE="$WORK/fake"

echo "==> Building tiny subset ($N per class) under $WORK"
rm -rf "$WORK"
mkdir -p "$SUB_REAL" "$SUB_FAKE"
# Symlink the first N images of each class (no byte copies).
find "$REAL_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | sort | head -n "$N" | while read -r f; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$SUB_REAL/"; done
find "$FAKE_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.webp' \) | sort | head -n "$N" | while read -r f; do ln -sf "$(cd "$(dirname "$f")" && pwd)/$(basename "$f")" "$SUB_FAKE/"; done
echo "    real: $(ls "$SUB_REAL" | wc -l | tr -d ' ')  fake: $(ls "$SUB_FAKE" | wc -l | tr -d ' ')"

echo "==> [1/6] Tier 2 — forensic-feature classifier"
$PY -m src.frequency.train_svm --real-dir "$SUB_REAL" --fake-dir "$SUB_FAKE" \
    --n-augments-per-image 1 --out outputs/tier2_classifier.joblib

echo "==> [2/6] Tier 3 — CLIP linear probe (downloads ViT-B/32 weights on first run)"
$PY -m src.semantic.train_probe --real-dir "$SUB_REAL" --fake-dir "$SUB_FAKE" \
    --out outputs/tier3_clip_probe.joblib

echo "==> [3/6] Tier 1 — EfficientNet-B0, 1 epoch (downloads ImageNet weights on first run)"
$PY -m src.model.train --real-dir "$SUB_REAL" --fake-dir "$SUB_FAKE" \
    --epochs 1 --batch-size 8 --val-frac 0.25 --out outputs/tier1_efficientnet_b0.pt

echo "==> [4/6] Fusion meta-model"
$PY -m src.train_fusion --real-dir "$SUB_REAL" --fake-dir "$SUB_FAKE" \
    --n-samples "$N" --scores-cache outputs/_dry_run_scores.npz

echo "==> [5/6] Cascade inference"
$PY -m src.infer "$SUB_FAKE" --out outputs/_dry_run_predictions.json --always-escalate
echo "    wrote outputs/_dry_run_predictions.json"

echo "==> [6/6] Robustness eval (small sample)"
$PY -m src.eval.robustness --real-dir "$SUB_REAL" --fake-dir "$SUB_FAKE" \
    --n-samples 15 --out outputs/_dry_run_robustness.csv

echo
echo "==> Dry run complete. Every stage ran. Artifacts:"
ls -la outputs/tier1_efficientnet_b0.pt outputs/tier2_classifier.joblib outputs/tier3_clip_probe.joblib outputs/fusion_model.joblib outputs/_dry_run_predictions.json outputs/_dry_run_robustness.csv
echo
echo "These are TOY models trained on $N images — delete them before the real run:"
echo "  rm outputs/tier*.joblib outputs/tier*.pt outputs/fusion_model.joblib outputs/_dry_run*"
