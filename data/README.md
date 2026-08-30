# Datasets

All datasets are public/licensed. Nothing is bundled in this repo — run the fetch steps below
into `data/raw/<dataset_name>/` (gitignored). You need **your own** free accounts/tokens for
HuggingFace and Kaggle — there is no shared or "open-source" token that works for everyone, and
tokens should never be committed or pasted into chat. Put them in a local `.env` (copied from
`.env.example`, gitignored) or in the tool's own config file as noted below.

## SID_Set (train/eval)

HuggingFace dataset, public, no token required to download.

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='saberzl/SID_Set', repo_type='dataset', local_dir='data/raw/sid_set')
"
```

If HuggingFace ever rate-limits anonymous downloads, create a free account at huggingface.co,
generate a token under Settings -> Access Tokens (read-only is enough), and put it in `.env` as
`HF_TOKEN=...`.

## CIFAKE (train/eval)

Kaggle dataset: `birdy654/cifake-real-and-ai-generated-synthetic-images`.

1. Create a free Kaggle account, then go to Account -> Create New API Token. This downloads
   `kaggle.json` containing *your* username + key.
2. Either place that file at `~/.kaggle/kaggle.json` (`chmod 600`), or copy the two values into
   `.env` as `KAGGLE_USERNAME` / `KAGGLE_KEY`.
3. Fetch:

```bash
pip install kaggle
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/raw/cifake --unzip
```

## WildFake (cross-generator holdout)

ModelScope dataset: `hy2628982280/WildFake`. Used specifically to hold out one full generator
family for the cross-generator generalization test (see `docs/shortcut_learning_check.md`).

ModelScope's UI requires a one-time manual step before the download links resolve: open
https://modelscope.cn/datasets/hy2628982280/WildFake in a browser, use the site's built-in
page-translate control (if the page loads in Chinese) to view it in English, then accept the
dataset's terms on the page. This click-through cannot be scripted — do it once per account.
After that:

```bash
pip install modelscope
python -c "
from modelscope.msdatasets import MsDataset
ds = MsDataset.load('hy2628982280/WildFake', download_dir='data/raw/wildfake')
"
```

## Validation-only set (demo/benchmark — never train on this)

- COCO val2017 (4998 non-AIGC images): https://cocodataset.org/#download
- DALL·E Advanced subset (8843 AIGC images) — from the same benchmark source as your demo.

```bash
mkdir -p data/raw/val_coco data/raw/val_dalle
curl -L http://images.cocodataset.org/zips/val2017.zip -o /tmp/val2017.zip
unzip /tmp/val2017.zip -d data/raw/val_coco
```

Place the DALL·E Advanced images at `data/raw/val_dalle/`. This split is for the demo + final
robustness/benchmark numbers only — never used for training or hyperparameter selection.

## Expected layout after fetching

```
data/raw/
├── sid_set/{real,fake}/...
├── cifake/{train,test}/{REAL,FAKE}/...
├── wildfake/<generator_family>/{real,fake}/...
├── val_coco/val2017/*.jpg
└── val_dalle/*.png
```

Training/eval scripts in `src/` expect a `real/` and `fake/` (or equivalent, mapped in each
script's `--real-dir`/`--fake-dir` args) split — see each script's `--help`.
