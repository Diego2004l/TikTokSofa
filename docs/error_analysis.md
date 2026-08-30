# Error Analysis

First-pass results. Two runs feed this doc:

- **Robustness table** — `outputs/robustness_summary.csv`, produced by `src/eval/robustness.py` on a
  Kaggle T4 run: 4 SID_Set train shards (~1k real / ~2k fake, `--max-image-dim 384`, Tier 1 = 4
  epochs), scored on the SID_Set held-out `validation` split.
- **FP/FN breakdown** — `outputs/predictions.json`, the same trained tiers re-scored via
  `src/infer.py --always-escalate` on a local SID_Set val split (47 real / 81 fake, threshold 0.5).

Both are deliberately small (fast-iteration config). The final submission run should use the
COCO val2017 + DALL-E Advanced benchmark set (`data/README.md`), more train shards, and
`--max-image-dim` unset — see "What to fix" below.

## Per-tier breakdown

| Tier | FP rate @0.5 | FN rate @0.5 | AUC (clean) | AUC (robust avg) |
|---|---|---|---|---|
| Tier 0 (provenance) | n/a | n/a | n/a (metadata only) | n/a |
| Tier 1 (EfficientNet-B0) | 46.8% | 18.5% | 0.754 | 0.760 |
| Tier 2 (forensic/frequency) | 48.9% | 6.2% | 0.934 | 0.921 |
| Tier 3 (CLIP probe) | 51.1% | 7.4% | 0.938 | 0.934 |
| **Fused** | **34.0%** | **7.4%** | **0.955** | **0.949** |

Two things jump out:

1. **AUC is high but the 0.5 threshold is wrong.** Tier 2/3 separate the classes well (AUC ~0.93)
   yet mis-flag ~half of real images at a raw 0.5 cut — their probe outputs are uncalibrated and
   sit above 0.5 for most inputs. The fusion LR partially corrects this (FP 51% -> 34%) but not
   enough, because the training split is ~2:1 fake-heavy and nothing re-balances it.
2. **Fusion still beats every individual tier on AUC** (0.955 clean, 0.949 robust-avg) and cuts the
   FP rate roughly in half vs. any single tier — the degradation-aware reweighting is pulling its
   weight even at this data scale.

## Representative failure cases

**False negatives — all are `tampered_*` (locally edited) images:**

| image | pred | tier1 | tier2 | tier3 | degradation |
|---|---|---|---|---|---|
| `tampered_00122.png` | 0.30 | 0.00 | 0.55 | 0.49 | 0.20 |
| `tampered_03541.png` | 0.31 | 0.00 | 0.53 | 0.58 | 0.24 |
| `tampered_05143.png` | 0.37 | 0.01 | 0.61 | 0.54 | 0.21 |

Likely cause: a locally-inpainted image is *mostly* authentic pixels, so a whole-image CNN (Tier 1)
and a whole-image CLIP embedding (Tier 3) both read "real" with high confidence. Tier 2's
patch-based features lean fake but only weakly, because the forensic anomaly is confined to a small
region and gets averaged out by the mean/max pooling across patches. **The cascade has no
localization / patch-anomaly tier** — every tier produces one image-level score — so partial
manipulations are its structural blind spot. Treating SID_Set label 2 as a separate "tampered"
class, or adding a patch-max aggregation to Tier 2, would target this directly.

**False positives — genuine photos, Tier 1 fires hard:**

| image | pred | tier1 | tier2 | tier3 | degradation |
|---|---|---|---|---|---|
| `b0071a6847984193.png` | 0.74 | 0.99 | 0.59 | 0.81 | 0.19 |
| `4bd05685b1e2c552.png` | 0.74 | 0.87 | 0.80 | 0.46 | 0.18 |
| `9a99796dffaaa7d3.png` | 0.67 | 0.98 | 0.64 | 0.53 | 0.21 |

Mean tier score on all FP cases: Tier 1 **0.91**, Tier 2 0.60, Tier 3 0.54. Tier 1 is
over-confident and fake-biased — expected from 4 epochs on ~1k images with a 2:1 class skew. It
contributes a fusion coefficient of 1.19, enough to drag borderline real images over the line even
when Tier 3 disagrees.

## Trade-off discussion

- **Where the cheap tier (2) diverges from the expensive tiers:** Tier 2 degrades fastest under
  heavy compression/noise — `noise_s0.10` drops it to 0.866 AUC while Tier 3 holds 0.907 and the
  fused score holds 0.935. This is exactly the case the degradation-aware fusion is designed for,
  and the robust-avg numbers (Tier 2 0.921 vs Fused 0.949) show the reweighting is catching it
  rather than over-correcting.
- **Compound transforms:** `crop80_resize_jpeg30` is the worst condition for Tier 2 (0.883) and
  the best for Tier 1 (0.808 — its only above-0.8 row). Tier 3 barely moves (0.924). Fused stays
  at 0.939. No single tier owns robustness; the fusion is what stays flat.
- **Escalation:** every FP/FN case above already ran all tiers (`--always-escalate` for this
  analysis), so the errors are fusion errors, not "the expensive tier never ran" errors. With the
  default `--escalation-margin 0.15`, Tier 2's confident-but-wrong scores on the FN `tampered_*`
  cases (0.53-0.61) *would* trigger escalation — but escalation doesn't help when Tier 1/3 are also
  wrong on partial manipulations.
- **Cross-generator gap:** not yet measured — needs the WildFake split (see
  `docs/shortcut_learning_check.md`). That number belongs here once it exists.

## What to fix (priority order)

1. **Class balance + threshold calibration.** Train on a 1:1 real/fake split (or class-weight the
   losses) and calibrate the fusion output on a held-out split. Biggest single lever on the 34% FP.
2. **More Tier 1 training** — more shards, more epochs, or freeze fewer blocks. Its 0.76 AUC is the
   ceiling drag on the fused score.
3. **Handle partial manipulation** — separate "tampered" handling or a patch-max Tier 2 aggregation.
4. **Re-run on the COCO + DALL-E benchmark set** for the headline numbers, per spec section 2.
