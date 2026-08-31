# Adaptive Evidence System (Features 2–4)

Three components layered on top of the 4-tier cascade. All are additive — `src/infer.py`,
`src/fusion.py` and the original CLIs still run unchanged when the new artifacts are absent.

```
tiers ──► transformation profiler ──► adaptive router ──► confidence + abstention ──► {AI, REAL, UNKNOWN}
 (F2 estimates degradation)   (F3 learns which evidence to trust)   (F4 decides whether to answer)
```

---

## Feature 2 — Transformation Profiler  (`src/transformation/`)

Estimates what degradation an image has undergone:

```json
{"jpeg_compression": 0.87, "resize_degradation": 0.74, "blur": 0.12, "noise": 0.21,
 "sharpening": 0.05, "crop": 0.63, "screenshot_like": 0.71, "overall_degradation": 0.78}
```

- **No new large network.** Reuses Tier 2's forensic vector (double-JPEG DCT, block-grid, ELA,
  noise-residual stats, FFT) + ~19 extra cheap indicators (`features.py:EXTRA_FEATURE_NAMES`:
  resolution/aspect, Laplacian variance, high-freq energy, pixel-domain blockiness, FFT radial
  slope, colour-count / border uniformity for screenshots). Per-label `HistGradientBoosting`
  classifiers, isotonic-calibrated → the 7 family scores are **calibrated confidence** in [0,1].
  `overall_degradation` is a regressor output — a degradation **score**, not a probability.
- **Auto-labelled training data** (`synthesize.py`): apply known transforms from `src/transforms`
  to clean images; the label is exact by construction. Multi-label (stacks up to 3 families in a
  canonical order). Train and eval use **disjoint severity grids** so unseen-severity
  generalisation is measurable.

```bash
python -m src.transformation.train --clean-dir data/raw/benchmark/real --out outputs/transformation_profiler.joblib
python -m src.transformation.evaluate --clean-dir data/raw/benchmark/coco_holdout --profiler outputs/transformation_profiler.joblib
```

`evaluate.py` reports per-transformation precision / recall / F1 + confusion matrix for three
regimes: **isolated**, **compound**, **unseen_severity**.

---

## Feature 3 — Adaptive Evidence Router  (`src/router/`)

Learns, from the image condition + every detector output + how much they disagree, which
evidence to trust. Lightweight by design: `logreg` (a calibrated linear model whose coefficients
*are* the learned evidence weights — `AdaptiveRouter.coefficients()`) or a small `gbdt`.

Feature groups (toggleable for ablation — `RouterConfig`):

| group | features |
|---|---|
| detector outputs | tier0/1/2/3 (+ neutral fill for missing tiers) |
| reliability | which tiers ran, provenance available, Tier 2 degradation estimate, small-image flag |
| disagreement | mean / std / min / max / range + pairwise abs-diffs of detector scores |
| transformation profile | Feature 2's 7 family scores + `overall_degradation` |
| uncertainty | Feature 4 multi-crop consistency (optional) |

**Anti-leakage:** trained on the held-out fusion split (disjoint from tier training *and* the
final test benchmark), with **out-of-fold** predictions for its own calibration
(`AdaptiveRouter.oof_fit_predict`). Each training image gets a seeded random transform so the
router sees the full clean→degraded spread.

```bash
python -m src.router.train --real-dir .../val/real --fake-dir .../val/fake \
    --profiler outputs/transformation_profiler.joblib --out outputs/router_model.joblib --model logreg

# Baselines A–E + progressive ablation, all on the SAME held-out test samples:
python -m src.router.baselines --train-real .../val/real --train-fake .../val/fake \
    --test-real .../test/real --test-fake .../test/fake --out-dir outputs/router_ablation
```

`baselines.py` variants: `router_base` → `+disagreement` → `+profiler` → `+uncertainty`, vs
Baseline A (CNN), B (forensic), C (CLIP), D (static ensemble `0.34/0.33/0.33`), E (existing
fusion). In `src/infer.py` the router is the **primary combiner**; existing fusion is the
fallback, then a plain mean.

---

## Feature 4 — Confidence, Abstention & Disagreement  (`src/confidence/`)

The system may answer **UNKNOWN** ("evidence insufficient or contradictory — human review
recommended") instead of being forced to guess.

```json
{"label": "UNKNOWN", "score": 0.57, "confidence": "LOW", "abstained": true}
```

- **Confidence ≠ raw probability.** `ConfidenceModel` is a calibrated linear model predicting
  P(the prediction is correct) from `signals.SIGNAL_NAMES`: decision margin, detector
  disagreement (std / range / max-pair), transformation severity, missing-tier count, provenance
  strength, and multi-crop consistency.
- **Multi-crop consistency** (`multicrop.py`): 5 deterministic crops re-scored end-to-end,
  consistency = `clip(1 − 2·std, 0, 1)`. Run **only for hard cases** (`signals.is_hard_case`:
  near the boundary, high disagreement, or high transformation severity) — the cost-aware
  cascade is preserved.
- **Abstention policy** (`AbstentionPolicy`): `AI` if `score ≥ hi` & `confidence ≥ conf_min`;
  `REAL` if `score ≤ lo` & `confidence ≥ conf_min`; else `UNKNOWN`. Thresholds `(lo, hi,
  conf_min)` are **grid-searched on validation** to minimise abstention subject to a target FPR
  and a minimum selective accuracy — never hard-coded.

```bash
python -m src.confidence.tune --real-dir .../val/real --fake-dir .../val/fake \
    --target-fpr 0.05 --min-selective-accuracy 0.92 \
    --out-confidence outputs/confidence_model.joblib --out-policy outputs/abstention_policy.joblib

python -m src.confidence.evaluate --real-dir .../test/real --fake-dir .../test/fake \
    --out-dir outputs/abstention_eval
```

`evaluate.py` reports coverage, selective accuracy, FPR/TPR on the answered set, abstention rate,
and the full risk–coverage curve (`risk_coverage.png`) — showing abstention improves reliability
on the ambiguous tail.
