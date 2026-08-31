#!/usr/bin/env bash
# Full pipeline locally: fetch SID_Set subset -> train Tiers 1/2/3 -> fusion -> eval.
# No GPU required (Tier 1 just trains slower on CPU). Run from the repo root with the venv active,
# or set PYTHON=/path/to/python.
#
#   scripts/run_all.sh                 # fast first pass (defaults below)
#   TRAIN_SHARDS=20 MAX_PER_SHARD=0 MAX_IMG_DIM=0 EPOCHS=6 scripts/run_all.sh   # full run
#
# 0 for MAX_PER_SHARD / MAX_IMG_DIM means "no cap".

set -euo pipefail

PY="${PYTHON:-.venv/bin/python}"
TRAIN_SHARDS="${TRAIN_SHARDS:-4}"
VAL_SHARDS="${VAL_SHARDS:-1}"
MAX_PER_SHARD="${MAX_PER_SHARD:-250}"
MAX_IMG_DIM="${MAX_IMG_DIM:-384}"
EPOCHS="${EPOCHS:-4}"
N_AUGMENTS_T2="${N_AUGMENTS_T2:-1}"

cap_flag=""; [ "$MAX_PER_SHARD" != "0" ] && cap_flag="--max-per-shard $MAX_PER_SHARD"
dim_flag="";  [ "$MAX_IMG_DIM"   != "0" ] && dim_flag="--max-image-dim $MAX_IMG_DIM"

R_TRAIN=data/raw/sid_set/train/real; F_TRAIN=data/raw/sid_set/train/fake
R_VAL=data/raw/sid_set/val/real;     F_VAL=data/raw/sid_set/val/fake

echo "==> [1/6] Fetch SID_Set subset ($TRAIN_SHARDS train + $VAL_SHARDS val shards)"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" $PY scripts/prep_sid_set.py --out data/raw/sid_set \
    --train-shards "$TRAIN_SHARDS" --val-shards "$VAL_SHARDS" $cap_flag
echo "    train real=$(ls $R_TRAIN | wc -l | tr -d ' ') fake=$(ls $F_TRAIN | wc -l | tr -d ' ')  |  val real=$(ls $R_VAL | wc -l | tr -d ' ') fake=$(ls $F_VAL | wc -l | tr -d ' ')"

echo "==> [2/6] Tier 2 — forensic features (parallel)"
$PY -m src.frequency.train_svm --real-dir $R_TRAIN --fake-dir $F_TRAIN --classifier rf \
    --n-augments-per-image "$N_AUGMENTS_T2" $dim_flag --out outputs/tier2_classifier.joblib

echo "==> [3/6] Tier 3 — CLIP linear probe"
$PY -m src.semantic.train_probe --real-dir $R_TRAIN --fake-dir $F_TRAIN --out outputs/tier3_clip_probe.joblib

echo "==> [4/6] Tier 1 — EfficientNet-B0 ($EPOCHS epochs)"
$PY -m src.model.train --real-dir $R_TRAIN --fake-dir $F_TRAIN --epochs "$EPOCHS" --batch-size 32 \
    --out outputs/tier1_efficientnet_b0.pt

echo "==> [5/6] Fusion meta-model (held-out val split)"
$PY -m src.train_fusion --real-dir $R_VAL --fake-dir $F_VAL $dim_flag --scores-cache outputs/_fusion_scores.npz

# Adaptive evidence system (Features 2-4). Opt in with RUN_ADAPTIVE=1 (adds ~a few min on CPU).
if [ "${RUN_ADAPTIVE:-0}" != "0" ]; then
    echo "==> [extra] Transformation profiler (F2) + adaptive router (F3) + abstention policy (F4)"
    $PY -m src.transformation.train --clean-dir $R_VAL --out outputs/transformation_profiler.joblib --variants-per-image 5
    $PY -m src.router.train --real-dir $R_VAL --fake-dir $F_VAL $dim_flag \
        --profiler outputs/transformation_profiler.joblib --out outputs/router_model.joblib
    $PY -m src.confidence.tune --real-dir $R_VAL --fake-dir $F_VAL $dim_flag \
        --target-fpr "${TARGET_FPR:-0.1}" --min-selective-accuracy "${MIN_SEL_ACC:-0.85}" --all-multicrop
fi

echo "==> [6/6] Eval — robustness table + cascade inference"
$PY -m src.eval.robustness --real-dir $R_VAL --fake-dir $F_VAL --n-samples 120 $dim_flag --out outputs/robustness_summary.csv
$PY -m src.infer $F_VAL --out outputs/predictions.json --always-escalate $dim_flag

# Full robustness benchmark (Feature 1). Much heavier (50+ conditions x 6 models); opt in with
# RUN_FULL_BENCH=1. Ideally point --real-dir/--fake-dir at the held-out COCO+DALL-E benchmark set.
if [ "${RUN_FULL_BENCH:-0}" != "0" ]; then
    echo "==> [extra] Full robustness benchmark (src.eval.robustness_bench)"
    $PY -m src.eval.robustness_bench --real-dir $R_VAL --fake-dir $F_VAL \
        --n-samples "${BENCH_SAMPLES:-150}" --seed 0 $dim_flag --out-dir outputs/robustness
fi

echo
echo "==> Done. Commit outputs/robustness_summary.csv + outputs/predictions.json and fill in docs/error_analysis.md."
echo "    (outputs/*.pt and *.joblib are gitignored — don't commit the weights.)"
