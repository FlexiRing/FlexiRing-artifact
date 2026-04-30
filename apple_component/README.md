# Apple Component

This module contains the intermediate pipeline used in our IMU gesture workflow:

- base feature extractor training (`ModelTrain.py`)
- Delta-Encoder resources (`encoder_training/`)
- per-user add-head adaptation (`add_head/train.py`)
- fixed-manifest baseline runner (`run_fixed_manifest_baselines.py`)

It is packaged as a runnable smoke-test release with bundled example data.

## Directory Layout

```text
apple_component/
  ModelTrain.py
  model.py
  run_fixed_manifest_baselines.py
  best_model.pth
  preset_gesture.npz
  encoder_training/
    gesture_augmentation.py
    delta_encoder_best.pth
    delta_vectors_bank.npy
  add_head/
    config.py
    pipeline.py
    train.py
  example_data/
    preset_gestures/all/*.csv
    custom_gestures/<user>/<class_id>/{support,test}/*.csv
```

## Install

From the release root:

```bash
pip install -r apple_component/requirements.txt
```

## Quick Smoke Tests

Run from the release root.

### 1) Base model training

```bash
python apple_component/ModelTrain.py --epochs 1 --batch_size 8 --num_workers 0 --skip_tsne --output_dir apple_component/outputs/modeltrain_smoke
```

Expected outputs include:

- `apple_component/outputs/modeltrain_smoke/best_model.pth`
- `evaluation_metrics.csv`
- confusion matrices and training curves

### 2) Add-head training (per user)

```bash
python apple_component/add_head/train.py --users P16_23_male --classes 0 1 2 3 --n_support 5 --max_epochs 1 --num_workers 0
```

Expected outputs include:

- `apple_component/outputs/add_head_training/P16_23_male/add_head_best.pth`
- `training_log.csv`, `eval_metrics.csv`, `summary.csv`, `summary.json`

### 3) Fixed-manifest baselines

```bash
python apple_component/run_fixed_manifest_baselines.py --shots 1 --max-epochs 1 --preset-limit 64 --artifact-root apple_component/outputs/fixed_manifest_smoke
```

Expected outputs include:

- `apple_component/outputs/fixed_manifest_smoke/new_only_head/...`
- `apple_component/outputs/fixed_manifest_smoke/keep_old_head/...`

## Data and Resource Defaults

Default paths are configured in `apple_component/add_head/config.py`:

- custom data: `apple_component/example_data/custom_gestures/`
- base model: `apple_component/best_model.pth`
- Delta-Encoder: `apple_component/encoder_training/delta_encoder_best.pth`
- Delta-vector bank: `apple_component/encoder_training/delta_vectors_bank.npy`
- preset features: `apple_component/preset_gesture.npz`

For custom datasets, override paths via CLI flags (recommended) or update config values.

## Notes

- This release is designed for smoke testing and interface validation, not full paper-level reproduction.
- `model.py`, `encoder_training/gesture_augmentation.py`, and files under `add_head/` are library modules and are invoked through the entry scripts above.
