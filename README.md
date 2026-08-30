# Robust AIGC Image Detector

A 4-tier cascading AI-generated-image detector built for the TikTok TechJam hackathon: provenance
check -> CNN -> hand-built forensic/frequency features -> semantic embeddings -> degradation-aware
adaptive fusion. See the full design spec this repo implements for rationale on every choice.

## Why a cascade, not one classifier

Most AIGC detectors are a single CNN reported at clean-data accuracy. The brief's own framing —
transforms destroy different tiers' signals differently, and generalizing to *new* generators is
the real test — is what this architecture is built around:

1. **Tier 0 — Provenance** (`src/provenance/`): parses embedded C2PA manifests
   (`c2pa_check.py`) for an instant, near-zero-cost early exit when a trusted signal names a known
   AI generator. An optional, clearly-labeled OpenAI-assisted heuristic (`openai_check.py`) can
   supplement this behind `--use-provenance-api` — it is NOT a verified credential the way a valid
   C2PA manifest is.
2. **Tier 1 — CNN** (`src/model/`): EfficientNet-B0 (`timm`, ImageNet-pretrained), fine-tuned on
   compound-augmented data — crop -> resize -> JPEG chains, not just isolated transforms.
3. **Tier 2 — Forensic/frequency features** (`src/frequency/`): fully hand-built, no pretrained
   model — windowed/patch-based FFT (Hann-windowed to avoid crop-edge spectral leakage),
   autocorrelation of the denoised noise residual, PRNU-style residual stats, ELA, double-JPEG DCT
   histogram, and 8x8 block-grid alignment — feeding a RandomForest/SVM.
4. **Tier 3 — Semantic** (`src/semantic/`): frozen CLIP ViT-B/32 (`open_clip`) embeddings with a
   logistic-regression probe. Semantic signal survives transforms that destroy pixel-level
   forensic signal.
5. **Fusion** (`src/fusion.py`): NOT a static stacked ensemble. Tier 2's block-grid/double-JPEG
   signal strength estimates how transformed an image is, and that estimate reweights Tier 2 vs.
   Tier 3 *before* a logistic-regression meta-model combines everything into one calibrated score.

## Deployment story

Tier 0 + Tier 2 run on every upload — both are cheap and CPU-only. Tier 1 + Tier 3 only run when
Tier 0/2 are inconclusive or disagree (`src/infer.py`'s `--escalation-margin`), mirroring the cost
constraints of screening at platform scale. Every tier's score is retained in the output JSON, so
a human moderation reviewer sees evidence ("Tier 2 flagged double-JPEG recompression; Tier 3's
CLIP probe still scored it 0.92 fake") rather than a single opaque number.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in your OWN tokens if you use gated features; never commit .env
```

`c2patool` (Tier 0) is a separate Rust binary, not pip-installable — install it from
https://github.com/contentauth/c2pa-rs/releases and put it on your PATH. Tier 0 degrades to
`detected: False` gracefully if it's missing, so nothing else breaks without it.

Datasets are NOT bundled — see `data/README.md` for exact fetch commands for SID_Set, CIFAKE,
WildFake, and the COCO/DALL-E validation-only split. All fetching requires *your own* free
HuggingFace/Kaggle accounts; there is no shared token, and no token belongs in this repo or in
chat with an assistant.

## Reproducing a full run

```bash
# 1. Tier 1 — CNN
python -m src.model.train --real-dir data/raw/cifake/train/REAL --fake-dir data/raw/cifake/train/FAKE --out outputs/tier1_efficientnet_b0.pt

# 2. Tier 2 — forensic features
python -m src.frequency.train_svm --real-dir data/raw/cifake/train/REAL --fake-dir data/raw/cifake/train/FAKE --out outputs/tier2_classifier.joblib

# 3. Tier 3 — CLIP linear probe
python -m src.semantic.train_probe --real-dir data/raw/cifake/train/REAL --fake-dir data/raw/cifake/train/FAKE --out outputs/tier3_clip_probe.joblib

# 4. Fusion meta-model — runs Tiers 0-3 over a HELD-OUT split (not used to train tiers 1-3),
#    then fits the degradation-aware logistic-regression meta-model.
python -m src.train_fusion --real-dir data/raw/sid_set/eval/real --fake-dir data/raw/sid_set/eval/fake --scores-cache outputs/_fusion_scores.npz

# 5. Run the cascade end-to-end
python -m src.infer path/to/image_dir --out outputs/predictions.json

# 6. Robustness table (isolated + compound transforms, per tier)
python -m src.eval.robustness --real-dir data/raw/val_coco/val2017 --fake-dir data/raw/val_dalle --out outputs/robustness_summary.csv

# 7. Cross-generator holdout (the real generalization test)
python -m src.eval.cross_generator --data-root data/raw/wildfake --holdout-family <family_name>
```

`src/infer.py` is designed to degrade gracefully: any tier whose trained artifact isn't present
yet is skipped with a `[warn]` and a note in the output, rather than crashing — run it any time
after step 1 to confirm the JSON schema end-to-end even before every tier is trained.

## Limitations (current state of this repo)

**First-pass results are in** (`outputs/robustness_summary.csv`, `docs/error_analysis.md`): a
small Kaggle-T4 run on 4 SID_Set shards — fused AUC 0.955 clean / 0.949 robust-avg, beating every
individual tier. `notebooks/colab_train.ipynb` (Colab **and** Kaggle) or `scripts/run_all.sh`
(local, no GPU needed) reproduce it end to end.

Known gaps before the final submission, in priority order:
- Trained on a ~2:1 fake-heavy split with no threshold calibration -> ~34% false-positive rate at
  a raw 0.5 cut despite the high AUC (`docs/error_analysis.md` has the fix).
- Tier 1 is only lightly trained (4 epochs, ~1k images) — its 0.76 AUC drags the fused ceiling.
- Cross-generator holdout (`docs/shortcut_learning_check.md`) not yet run — needs the WildFake split.
- Headline numbers should come from the COCO + DALL-E benchmark set (spec section 2), not the
  SID_Set held-out split used for iteration.
- `*.joblib` / `*.pt` weights are gitignored — rerun the notebook/script to regenerate them.

## Related work — what's standard vs. what's this project's contribution

Frequency/noise forensics for AIGC detection (FFT peaks, PRNU-style residual stats, ELA) and
CLIP-embedding linear probes are both standard, widely-published techniques — we don't claim to
have invented them. An existing open-source project combines CLIP ViT-L/14 + DINOv2 ViT-L/14 +
global forensic features into a static LR/MLP stacking ensemble; to stay clearly differentiated
from that specific combination:

| Their approach | This project |
|---|---|
| CLIP ViT-L/14 + DINOv2 ViT-L/14 together | CLIP ViT-B/32 alone |
| No provenance tier | Tier 0 (C2PA) |
| No trained CNN tier | Tier 1 (fine-tuned EfficientNet-B0) |
| Global FFT/PRNU features | Patch-based, Hann-windowed FFT + autocorrelation + ELA + double-JPEG histogram + block-grid alignment |
| Static stacked ensemble | Degradation-aware adaptive fusion (Tier 2's own transform-strength estimate reweights Tier 2 vs. Tier 3) |

The project's own contributions are specifically: the patch/windowed computation of the frequency
features (crop-robust in a way a single global FFT isn't), the provenance-first cascade with
cost-aware escalation, and the degradation-aware adaptive fusion.

## Repo structure

See `src/` for the four tiers + fusion + `infer.py`, `src/eval/` for the robustness and
cross-generator scripts, `docs/` for the shortcut-learning check and error analysis, and
`notebooks/` for exploratory work.

## License

MIT — see `LICENSE`. All pretrained backbones used (EfficientNet-B0 via `timm`, CLIP ViT-B/32 via
`open_clip`) are public and separately licensed (Apache 2.0 / MIT respectively); see
`docs/` and the Bill of Materials in the original design spec for the full list.
