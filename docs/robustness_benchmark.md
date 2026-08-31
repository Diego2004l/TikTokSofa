# Robustness Benchmark (Feature 1)

A reproducible, quantitative measure of how every detector *model* behaves under realistic image
transformations. Not a demo script — one command produces machine-readable results, a
human-readable report, and degradation curves.

```bash
python -m src.eval.robustness_bench \
    --real-dir data/raw/benchmark/real --fake-dir data/raw/benchmark/fake \
    --out-dir outputs/robustness --n-samples 300 --seed 0
```

## What it evaluates

**Models** (`src/eval/scoring.py:DetectorBank`, all scored on the *same* transformed images):

| Model | Definition |
|---|---|
| `cnn` | Tier 1 EfficientNet-B0 score alone |
| `forensic` | Tier 2 hand-built forensic/frequency classifier alone |
| `clip` | Tier 3 CLIP linear-probe score alone |
| `static_ensemble` | fixed weighted mean, weights `{tier1: 0.34, tier2: 0.33, tier3: 0.33}` (`STATIC_ENSEMBLE_WEIGHTS`) |
| `existing_fusion` | `src/fusion.py` degradation-aware logistic-regression meta-model |
| `adaptive_router` | Feature 3's learned router (NaN / skipped until that lands) |

**Conditions** (`src/transforms/registry.py:build_conditions`, ~54 by default):

- **Compression** — JPEG q95/85/70/50/30
- **Resolution** — resize to 90/75/50/25 %, plus downsample→upsample
- **Blur** — Gaussian σ 0.5/1.0/2.0/3.0; motion blur (optional)
- **Noise** — Gaussian σ 0.02/0.05/0.10 (seeded); salt-and-pepper (optional)
- **Sharpening** — unsharp mask 150 % (mild) / 300 % (aggressive)
- **Cropping** — center + seeded-random crop at 90/75/50 %
- **Colour** — brightness / contrast / saturation ±, mild hue shift (≤15°)
- **Screenshot simulation** — light / typical / heavy. **This is a simulation** of the transform
  stack a re-uploaded screenshot accumulates (viewport rescale → RGB → optional UI border → JPEG
  recompression → sub-pixel resample); it does not claim to reproduce a real platform screenshot.
- **Compound chains** — 7 named chains (`resize→jpeg`, `crop→jpeg`, `blur→jpeg`,
  `resize→crop→jpeg`, `resize→blur→jpeg`, `crop→resize→jpeg→noise`, `screenshot→jpeg`)
- **Random compound** — 8 deterministic seeded chains (seeds 0–7, 2–4 ops each, JPEG forced last)

Every condition is deterministic given `--seed`: `test_conditions_are_deterministic` asserts
byte-identical output on repeat calls.

## Metrics

Per (model, condition), `src/eval/metrics.py:binary_metrics` reports **all** of: ROC-AUC, PR-AUC,
accuracy, precision, recall, F1, FPR, TPR, the 2×2 confusion matrix (`tp/fp/fn/tn`), and
calibration (Brier score, expected calibration error). Never accuracy alone.

`final_score` in the summary is the project's existing blend: `0.5·AUC_clean + 0.5·mean(AUC_robust)`.

## Outputs (`--out-dir`)

| File | Contents |
|---|---|
| `results.csv` | one row per (model, condition): `model, transformation, family, severity, severity_rank, compound_id, seed, auc, pr_auc, accuracy, precision, recall, f1, fpr, tpr, brier, ece, tp, fp, fn, tn, num_samples` |
| `results.json` | same rows + a `meta` block (git SHA, timestamp, split, sample counts, tier availability, threshold source) + the per-model `summary` |
| `summary.csv` | per-model aggregates: clean AUC, robust mean/min AUC, compound mean, per-family means, final score |
| `report.md` | the human-readable AUC matrix, per-family table, worst-conditions list, embedded curves |
| `figures/degradation_*.png` | AUC-vs-severity curves for compression / resolution / blur / noise |

## Anti-leakage rules

1. `--real-dir` / `--fake-dir` **must** be a held-out split never used to train any tier, fit the
   fusion meta-model, or tune thresholds. The script only evaluates; it fits nothing.
2. Threshold-dependent metrics (accuracy, precision, FPR, TPR, confusion) default to a raw 0.5 cut
   and are flagged as such in `report.md`. Supply `--thresholds-json` only with per-model
   thresholds tuned on a **separate validation** split (Feature 4's `tune_thresholds`).
3. Headline numbers should come from the COCO val2017 + DALL-E benchmark set (`data/README.md`),
   not the SID_Set split used for iteration.

## Reproducing / testing

- `pytest tests/test_transforms.py tests/test_metrics.py tests/test_robustness_bench.py`
  (or run each module standalone: `python -m tests.test_transforms`).
- `python -m src.transforms.registry` prints the full condition list with metadata.
