# Error Analysis

*Template — fill in after running `src/eval/robustness.py` and `src/infer.py` on a labeled
validation set (COCO val2017 + DALL-E Advanced, per `data/README.md`; never train on this split).
This repo was built without GPU/dataset access, so no real predictions exist yet to analyze.*

## Per-tier breakdown

For each tier, report on the clean validation set:
- False positive rate (real images scored as fake) and 3-5 representative examples
- False negative rate (fake images scored as real) and 3-5 representative examples
- Which forensic/frequency sub-feature (Tier 2) or which visual property (Tier 1/3) plausibly
  explains each representative error

| Tier | FP rate | FN rate | AUC (clean) | AUC (robust avg) |
|---|---|---|---|---|
| Tier 0 (provenance) | | | n/a (metadata only) | n/a |
| Tier 1 (EfficientNet-B0) | | | | |
| Tier 2 (forensic/frequency) | | | | |
| Tier 3 (CLIP probe) | | | | |
| Fused | | | | |

## Representative failure cases

Fill in once `outputs/predictions.json` exists:

- **FP example**: `<image_path>` — predicted fake, actually real. Tier scores: ... Likely cause: ...
- **FN example**: `<image_path>` — predicted real, actually fake. Tier scores: ... Likely cause: ...

## Trade-off discussion

- Where does Tier 2 (cheap, CPU-only) diverge most from Tier 1/3 (expensive)? Does the
  degradation-aware fusion actually catch those divergences, or does it under/over-correct?
- Under the compound-transform conditions (`outputs/robustness_summary.csv`), which tier degrades
  fastest, and does the escalation logic in `src/infer.py` (`--escalation-margin`) trigger the
  expensive tiers often enough to compensate?
- Reference `docs/shortcut_learning_check.md`'s cross-generator gap here if it's large — that gap
  IS an error-analysis finding, not a separate concern.
