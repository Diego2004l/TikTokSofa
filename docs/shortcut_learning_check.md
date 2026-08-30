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

**Verification**: `SymmetricAugmenter.__call__` is invoked once per image inside a loop that
iterates over both `real_dir` and `fake_dir` with the same augmenter instance and the same RNG
sequence (see `build_dataset` in `src/frequency/train_svm.py` and `RealFakeDataset` in
`src/model/train.py`) — there is no per-class branch anywhere in the augmentation path.

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

*Pending a training run with real WildFake data — this repo was built without GPU/dataset access
in this environment. Fill in this table after running the command above for at least one held-out
family:*

| Holdout family | In-distribution val AUC | Held-out family AUC | Gap |
|---|---|---|---|
| _(e.g. diffusion)_ | | | |

**How to read the gap**: a gap under ~0.05 AUC suggests the model is learning generalizable
artifacts; a large gap (>0.15) is a signal to inspect Tier 2's feature importances and Tier 1's
saliency on held-out-family false negatives before claiming robustness in the write-up.
