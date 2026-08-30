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

## 2. Cross-generator holdout test (run via `src/eval/cross_generator.py`)

Methodology: train on every generator family in the WildFake split except one, then evaluate
purely on the held-out family (`data/README.md` documents the expected `<family>/{real,fake}`
layout). This is the real generalization test the spec calls out — a model overfit to
generator-specific quirks (a particular diffusion model's exact noise signature, say) will show a
large AUC drop on a family it never saw, even if its in-distribution validation AUC looks great.

```bash
python -m src.eval.cross_generator --data-root data/raw/wildfake --holdout-family <family_name>
```

### Results

**Still pending** — the first-pass training run used SID_Set, which is not split by generator
family, so the cross-generator holdout could not be run yet. It needs the WildFake split
(`data/README.md`); run the command above for at least one held-out family before the final
submission and fill this table:

| Holdout family | In-distribution val AUC | Held-out family AUC | Gap |
|---|---|---|---|
| _(e.g. diffusion)_ | | | |

Interim note from the SID_Set run: the false negatives in `docs/error_analysis.md` are all
locally-tampered images, not a generator-family artifact — so the largest known error mode so far
is a *manipulation-type* blind spot, not an obvious generator-specific shortcut. The
cross-generator number is still needed to rule out the latter.

**How to read the gap**: a gap under ~0.05 AUC suggests the model is learning generalizable
artifacts; a large gap (>0.15) is a signal to inspect Tier 2's feature importances and Tier 1's
saliency on held-out-family false negatives before claiming robustness in the write-up.
