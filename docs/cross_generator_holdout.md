# Cross-Generator Holdout + Research Table (Feature 5)

Proves the detector learns general AIGC characteristics, not per-generator fingerprints.

## Leave-one-generator-family-out rotation

`src/eval/cross_generator_full.py` rotates over every family F in `--data-root`
(`<family>/{real,fake}/`, e.g. WildFake):

1. Train Tier 1/2/3, fit fusion, train the transformation profiler, train the router — **all on
   the other families only**. F is never used for tier training, fusion training, threshold
   tuning, profiler tuning, or model selection.
2. Evaluate every method on F (**unseen generator**) and on a held-out slice of the training
   families (**known generators**), clean + seeded-degraded passes.

```bash
# full rotation (heavy — Kaggle/Colab):
python -m src.eval.cross_generator_full --data-root data/raw/wildfake --epochs 4 --out-dir outputs/cross_generator

# reuse already-trained artifacts (only sound if they were trained leakage-free):
python -m src.eval.cross_generator_full --data-root data/raw/wildfake --skip-training --out-dir outputs/cross_generator

# single family, Tier-1 focus (original script, still works):
python -m src.eval.cross_generator --data-root data/raw/wildfake --holdout-family diffusion --all-methods
```

Outputs: `cross_generator_full.csv/md/json` (per-family, per-method AUC / PR-AUC / FPR / TPR /
FPR@TPR) and `known_vs_unseen.csv` — the summary table:

```
                 Known generators   Unseen generator   Gap
CNN                    X                  X             X
Forensic               X                  X             X
CLIP                   X                  X             X
Static ensemble        X                  X             X
Existing fusion        X                  X             X
Adaptive router        X                  X             X
```

A gap < ~0.05 AUC ⇒ generalisable artifacts; a large gap ⇒ inspect Tier 2 feature importances
and Tier 1 saliency on held-out-family false negatives before claiming robustness.

## Ablation study

`src/router/baselines.py` (Feature 3) is the ablation harness — progressive add:
`router_base → +disagreement → +profiler → +uncertainty`, each vs. Baselines A–E, on identical
held-out test samples. Answers "why is this better than a normal ensemble?".

## Final research table

`src/eval/research_table.py` merges the artifacts the other evaluations wrote:

```bash
python -m src.eval.research_table --out-dir outputs/research_table
```

| Method | Clean AUC | Robust AUC | Unseen-generator AUC | FPR @ target TPR | Abstention rate | Avg inference cost |
|---|---|---|---|---|---|---|
| … from `outputs/robustness/summary.csv` | | from `known_vs_unseen.csv` | from `router_ablation/ablation.csv` | from `abstention_eval.json` | from `src/eval/cost_model.py` |

Inference cost is in relative units (Tier 2 forensic extraction := 1.0); `cost_model.measure_wallclock`
gives measured ms/image on the current machine.

## Anti-leakage checklist

- Held-out generator family: absent from tier training, fusion fit, profiler training, router
  training, threshold tuning, model selection. ✔ enforced by `cross_generator_full.train_fold`
  (merges all families *except* F, carves its own known-eval slice for fusion/router/profiler).
- Robustness benchmark + abstention eval: run on held-out `test` split, never the fusion/val
  split used for tuning.
- Router: out-of-fold predictions for its own calibration.
