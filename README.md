# mw-fMRI analysis

Code for nested cross-validation of demographic, structural MRI, resting-state
fMRI, and movie-watching fMRI features. Data are not included.

## Requirements

- Python 3.10 or later
- MATLAB with SPM12

Install the Python package and create the local configuration:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e .
cp config.example.json config.json
export MWFMRI_CONFIG="$PWD/config.json"
```

## Data layout

Place the protected inputs under `data/` as follows:

```text
data/
├── classification/
│   ├── tabular/baseline_features.npz
│   └── whole_brain/baseline_features.npz
├── forecast/
│   ├── tabular/baseline_features.npz
│   └── whole_brain/baseline_features.npz
├── contrasts/
│   └── <participant_id>_con_0015.nii
└── resources/
    ├── gray_matter_mask.nii.gz
    ├── AAL3v1_1mm.nii
    ├── AAL3v1_1mm_Labels.mat
    └── post_movie_scores.csv
```

The tabular and whole-brain archives for a task must use the same participant
ordering. If local names or locations differ, update `config.json`. This file
and the complete `data/` directory are ignored by Git.

## Reproduce the analysis

Create the fixed outer and inner cross-validation splits:

```bash
python prepare_masks.py --task classification
python prepare_masks.py --task forecast
```

In MATLAB, add SPM12 and `shared/` to the path and generate the training-fold
group masks:

```matlab
addpath('/path/to/spm12')
addpath('shared')

make_fold_masks('outputs/fold_masks/classification/mask_manifest.json', ...
    'outputs/fold_masks/classification', 'data/contrasts', ...
    '%s_con_0015.nii', 0.01, 16)

make_fold_masks('outputs/fold_masks/forecast/mask_manifest.json', ...
    'outputs/fold_masks/forecast', 'data/contrasts', ...
    '%s_con_0015.nii', 0.01, 16)
```

Run all feature sets for both tasks and summarize the 100 outer folds:

```bash
python run_models.py \
  --classification-mask-root outputs/fold_masks/classification \
  --forecast-mask-root outputs/fold_masks/forecast \
  --pearson-threshold 0.01 \
  --chunk-size 10 \
  --max-workers 4

python summarize_models.py
```

The model summary is written to `outputs/model_summary.csv`. To generate the
spatial selection-frequency maps and AAL3 regional summaries, run:

```bash
python localization/build_localization.py
python localization/plot_localization.py
```

## Validation

```bash
python -m unittest discover -s tests -v
```
