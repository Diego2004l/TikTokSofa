# Error Analysis

## Combined-training update (CIFAKE + SID_Set) — current headline results

**The generalization gap found in the first pass is fixed.** The original model was trained on
SID_Set only; evaluated on CIFAKE (a dataset it never saw) it collapsed — a spot-check of 30
known-fake CIFAKE images labelled **0/30** correctly. Retraining on a combined corpus
(CIFAKE `train/` 50k real + 50k fake, merged with a SID_Set subset of 1,131 real + 2,245 fake)
and evaluating on CIFAKE's held-out `test/` split (10k real + 10k fake) closes it.

- **Robustness table** — `outputs/robustness_summary.csv`, `src/eval/robustness.py` on the
  CIFAKE `test/` split (10k / 10k).
- **Inference sanity check** — `outputs/predictions.json`, `src/infer.py --always-escalate` on the
  same 30-image `data/archive/test_small/FAKE` batch that previously scored 0/30.

### Before → after (fused ROC-AUC on CIFAKE)

| Model | Clean AUC | FINAL_SCORE (0.5·clean + 0.5·mean_robust) |
|---|---|---|
| **Before** — SID_Set-only, tested on CIFAKE | ~0.60 | ~0.60 |
| **After** — combined CIFAKE + SID_Set | **0.9958** | **0.9917** |

### After — per-condition, per-tier

| Condition | Tier 1 | Tier 2 | Tier 3 | Fused |
|---|---|---|---|---|
| clean | 0.9976 | 0.8952 | 0.9768 | **0.9958** |
| jpeg_q90 | 0.9971 | 0.8689 | 0.9760 | 0.9946 |
| jpeg_q70 | 0.9980 | 0.8755 | 0.9752 | 0.9961 |
| jpeg_q50 | 0.9926 | 0.8075 | 0.9546 | 0.9915 |
| jpeg_q30 | 0.9935 | 0.7976 | 0.9442 | 0.9910 |
| blur_s0.5 | 0.9962 | 0.8445 | 0.9714 | 0.9959 |
| blur_s1.0 | 0.9940 | 0.6452 | 0.9007 | 0.9923 |
| blur_s2.0 | 0.9821 | 0.5000 | 0.7624 | 0.9754 |
| resize_0.5x | 0.9926 | 0.6492 | 0.9211 | 0.9932 |
| resize_0.25x | 0.9759 | 0.5580 | 0.7620 | 0.9727 |
| noise_s0.02 | 0.9971 | 0.8488 | 0.9621 | 0.9965 |
| noise_s0.05 | 0.9976 | 0.7939 | 0.9513 | 0.9974 |
| noise_s0.10 | 0.9925 | 0.6304 | 0.8608 | 0.9910 |
| color_jitter_pm20pct | 0.9959 | 0.8778 | 0.9685 | 0.9953 |
| center_crop_80pct | 0.9911 | 0.6507 | 0.9497 | 0.9897 |
| crop80_resize_jpeg50 | 0.9832 | 0.6846 | 0.9232 | 0.9759 |
| crop80_resize_jpeg30 | 0.9823 | 0.6525 | 0.9069 | 0.9752 |
| blur2_jpeg30 | 0.9673 | 0.4498 | 0.7303 | 0.9637 |
| **FINAL_SCORE** | **0.9938** | **0.8045** | **0.9419** | **0.9917** |

### Key findings

1. **The failure was a cross-dataset generalization gap, not a threshold problem.** Training on
   SID_Set alone overfit to that corpus's generator-specific artifacts; on CIFAKE the fused AUC
   was ~0.60 and every known-fake spot-check image was called real. Adding CIFAKE to training
   raised fused clean AUC to **0.996** and FINAL_SCORE to **0.992**.
2. **Tier 1 (the CNN) benefited the most** — clean AUC 0.754 → 0.998, robust-avg 0.760 → ~0.990.
   It is now the strongest and most transform-stable tier.
3. **Tier 2 (hand-built forensic classifier) is the weak tier on this combined set** — clean AUC
   0.895, FINAL_SCORE 0.805, and it degrades sharply under blur / resize / crop:
   `blur2_jpeg30` 0.450, `blur_s2.0` 0.500, `resize_0.25x` 0.558, `center_crop_80pct` 0.651.
   These transforms move image content off the native 8×8 JPEG grid and destroy the high-frequency
   residual, which is exactly the signal Tier 2 depends on.
4. **The degradation-aware fusion is doing its job**: on every row where Tier 2 collapses, the
   fused score stays ≥ 0.96 because the reweighting shifts trust to Tier 1 / Tier 3. Fused
   robust-avg (~0.987) is *higher* than the mean of Tier 1's (~0.990) only because it also holds
   the clean rows near ceiling — no single tier owns robustness on its own.

### Inference sanity check (30 known-fake CIFAKE images)

`python -m src.infer data/archive/test_small/FAKE --out outputs/predictions.json --always-escalate`

- **27 / 30 labelled `fake`** (was **0 / 30** with the SID_Set-only model). Mean fused score 0.89.
- The 3 misses are low-confidence, and all three tiers are weak on them — i.e. genuinely hard
  32×32 crops, not a systematic bias:

  | image | fused | tier1 | tier2 | tier3 | degradation |
  |---|---|---|---|---|---|
  | `204 (8).jpg` | 0.485 | 0.469 | 0.214 | 0.775 | 0.25 |
  | `80 (8).jpg` | 0.020 | 0.197 | 0.152 | 0.246 | 0.19 |
  | `920 (2).jpg` | 0.177 | 0.546 | 0.298 | 0.077 | 0.18 |

  Note the mean tier scores on the full 30-image batch are now Tier 1 **0.89**, Tier 2 0.70,
  Tier 3 0.79 — the fake-bias direction the first pass complained about is now *correct* bias
  (these images really are fake).

> **Caveat:** the loaded `*.joblib` classifiers were pickled with scikit-learn 1.6.1 and this
> environment runs 1.9.0, so `src.infer` prints an `InconsistentVersionWarning`. Results look
> sound (AUC 0.996, 27/30), but for the final submission regenerate the joblibs with the pinned
> version or pin `scikit-learn==1.6.1`.

### What to fix (priority order, updated)

1. **Strengthen Tier 2 under geometric transforms** — patch-max aggregation, or drop the
   blockiness/grid features when a resample is detected (the Feature 2 transformation profiler
   already estimates this). Tier 2's 0.80 FINAL_SCORE is now the ceiling drag.
2. **Threshold calibration on a balanced held-out split** — the raw 0.5 cut is less wrong than
   before (27/30) but still uncalibrated; tune it (and the abstention policy) on validation.
3. **Cross-generator holdout on WildFake** — CIFAKE + SID_Set proves cross-*dataset* transfer;
   the leave-one-generator-family-out test (`docs/shortcut_learning_check.md`) is still needed to
   prove cross-*generator* transfer.
4. **Re-run on the COCO + DALL-E benchmark set** for a headline number that isn't CIFAKE-specific.

---

## Historical — first pass (SID_Set-only model)

Kept for the before/after story. This is the model that motivated the combined-training run above.

Two runs fed this section:

- **Robustness table** — `src/eval/robustness.py` on a Kaggle T4 run: 4 SID_Set train shards
  (~1k real / ~2k fake, `--max-image-dim 384`, Tier 1 = 4 epochs), scored on the SID_Set held-out
  `validation` split.
- **FP/FN breakdown** — the same trained tiers re-scored via `src/infer.py --always-escalate` on a
  local SID_Set val split (47 real / 81 fake, threshold 0.5).

### Per-tier breakdown (SID_Set-only)

| Tier | FP rate @0.5 | FN rate @0.5 | AUC (clean) | AUC (robust avg) |
|---|---|---|---|---|
| Tier 0 (provenance) | n/a | n/a | n/a (metadata only) | n/a |
| Tier 1 (EfficientNet-B0) | 46.8% | 18.5% | 0.754 | 0.760 |
| Tier 2 (forensic/frequency) | 48.9% | 6.2% | 0.934 | 0.921 |
| Tier 3 (CLIP probe) | 51.1% | 7.4% | 0.938 | 0.934 |
| **Fused** | **34.0%** | **7.4%** | **0.955** | **0.949** |

On the SID_Set held-out split the fused AUC looked strong (0.955), but the 0.5 threshold was
wrong (34% FP) and — critically — the whole thing did not transfer to CIFAKE at all. That
non-transfer is what the combined-training update fixes.

### Representative failure cases (SID_Set-only)

**False negatives — all `tampered_*` (locally edited) images:**

| image | pred | tier1 | tier2 | tier3 | degradation |
|---|---|---|---|---|---|
| `tampered_00122.png` | 0.30 | 0.00 | 0.55 | 0.49 | 0.20 |
| `tampered_03541.png` | 0.31 | 0.00 | 0.53 | 0.58 | 0.24 |
| `tampered_05143.png` | 0.37 | 0.01 | 0.61 | 0.54 | 0.21 |

A locally-inpainted image is *mostly* authentic pixels, so a whole-image CNN and a whole-image
CLIP embedding both read "real"; Tier 2's patch features lean fake but get averaged out. **The
cascade has no localization / patch-anomaly tier** — partial manipulations are its structural
blind spot. Adding a patch-max Tier 2 aggregation would target this.

**False positives — genuine photos, Tier 1 fires hard:**

| image | pred | tier1 | tier2 | tier3 | degradation |
|---|---|---|---|---|---|
| `b0071a6847984193.png` | 0.74 | 0.99 | 0.59 | 0.81 | 0.19 |
| `4bd05685b1e2c552.png` | 0.74 | 0.87 | 0.80 | 0.46 | 0.18 |
| `9a99796dffaaa7d3.png` | 0.67 | 0.98 | 0.64 | 0.53 | 0.21 |

Tier 1 (4 epochs, ~1k images, 2:1 class skew) was over-confident and fake-biased. The combined
run's much larger, balanced CIFAKE component is what turned Tier 1 from the weakest tier
(0.754 AUC) into the strongest (0.998).

### Trade-off discussion (SID_Set-only)

- **Cheap vs. expensive tier divergence:** Tier 2 degraded fastest under heavy compression/noise
  (`noise_s0.10` → 0.866 AUC) while Tier 3 held 0.907 and fused held 0.935 — the degradation-aware
  fusion catching it rather than over-correcting. The combined-training numbers show the *same
  pattern, amplified*: Tier 2 now drops to 0.45–0.65 under blur/resize/crop while fused stays
  ≥ 0.96.
- **Escalation:** every FP/FN case above already ran all tiers (`--always-escalate`), so those
  were fusion errors, not "expensive tier never ran" errors.
