# Shortcut-Learning Check

Two cheap, high-payoff safeguards against the model learning a spurious correlation instead of
real generative artifacts (spec section 6).

## 1. Symmetric augmentation (implemented, structural — not a one-time test)

`src/augmentations.py`'s `SymmetricAugmenter` is the single call site every training script
(`src/model/train.py`, `src/frequency/train_svm.py`) uses to augment BOTH classes. There is no
code path in this repo that applies a different transform distribution to real vs. fake images.

Why this matters concretely: if only fake images in training were ever JPEG-recompressed (e.g.
because a scraped "AIGC" dataset happens to be more compressed than a "real photos" dataset), a
classifier can hit high clean-data accuracy by learning "has JPEG artifacts => real" — a shortcut
that says nothing about actual generative artifacts and collapses the moment a real photo gets
uploaded through TikTok's own recompression pipeline.

**Verification**: `SymmetricAugmenter.__call__` is invoked once per image, and the image's
real/fake label is never an input to the transform draw. In `RealFakeDataset`
(`src/model/train.py`) a single augmenter instance is shared across both classes. In
`build_dataset` (`src/frequency/train_svm.py`) feature extraction is parallelized across
processes, so each image gets its own augmenter seeded by `f"{seed}:{path}"` — the seed value
is label-independent, so the sampled transform distribution is identical for real and fake.
There is no per-class branch anywhere in the augmentation path in either script.

## 2. Cross-dataset generalization — the shortcut we actually caught (and fixed)

The first-pass model was trained on **SID_Set only**. When evaluated on **CIFAKE** — a different
AIGC dataset, different generators, different real-image source — it did not transfer:

| Model | CIFAKE clean fused AUC | 30-image known-fake spot-check |
|---|---|---|
| SID_Set-only | **~0.60** | **0 / 30** labelled fake |
| Combined CIFAKE `train/` (50k+50k) + SID_Set subset (1,131 + 2,245) | **0.9958** | **27 / 30** labelled fake |

This is a textbook shortcut: on SID_Set's own held-out split the model looked strong (fused AUC
0.955, see `docs/error_analysis.md`), but that number was measuring *"can it re-recognise
SID_Set's generator artifacts"*, not *"can it detect AIGC"*. Mixing a second dataset into training
forced it to learn features common to both, and cross-dataset AUC jumped from ~0.60 to ~0.996.
Tier 1 (the CNN) was the tier that had memorised SID_Set the hardest — its clean AUC went
0.754 → 0.998 once CIFAKE was added.

Evaluation split for the "after" numbers: CIFAKE held-out `test/` (10k real + 10k fake), never
seen in training. Full per-condition table in `outputs/robustness_summary.csv` /
`docs/error_analysis.md`.

## 3. Cross-*generator* holdout test (leave-one-generator-family-out)

Cross-dataset transfer (section 2) is necessary but not sufficient — CIFAKE and SID_Set still
share *types* of generators. The stronger test: train on every generator family in the WildFake
split except one, evaluate purely on the held-out family.

```bash
# per-family, all methods:
python -m src.eval.cross_generator --data-root data/raw/wildfake --holdout-family <family_name> --all-methods
# full rotation + known-vs-unseen summary (Feature 5):
python -m src.eval.cross_generator_full --data-root data/raw/wildfake --out-dir outputs/cross_generator
```

### Results

**Still pending** — needs the WildFake split (`data/README.md`); the combined training set
(CIFAKE + SID_Set) is not organised by generator family. The rotation infrastructure now exists
(`src/eval/cross_generator_full.py`, `docs/cross_generator_holdout.md`) and enforces that the
held-out family is absent from tier training, fusion fitting, profiler/router training, and
threshold tuning. Fill this once WildFake is fetched:

| Holdout family | Known-generator AUC | Held-out-family AUC | Gap |
|---|---|---|---|
| _(e.g. diffusion)_ | | | |

**How to read the gap**: a gap under ~0.05 AUC suggests generalizable artifacts; a large gap
(>0.15) is a signal to inspect Tier 2's feature importances and Tier 1's saliency on
held-out-family false negatives before claiming robustness.
