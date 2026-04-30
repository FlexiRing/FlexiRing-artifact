# Open-Source Test Guide

This document lists the commands that can be used to verify the current
release package. Run the commands from the repository root:

```bash
cd gesture_open_source_release
```

The bundled example data is intentionally small. These commands are smoke
tests for code paths, data formats, checkpoints, and entry points. They are not
paper-level reproduction experiments.

## 1. Environment

Install dependencies:

```bash
pip install -r main_method/requirements.txt
pip install -r apple_component/requirements.txt
```

Syntax/import check for all Python files:

```bash
python -m py_compile main_method/preprocess_data.py main_method/fewshot_demo.py apple_component/ModelTrain.py apple_component/model.py apple_component/encoder_training/gesture_augmentation.py apple_component/add_head/config.py apple_component/add_head/pipeline.py apple_component/add_head/train.py apple_component/run_fixed_manifest_baselines.py dtw_core/dtw_core.py dtw_core/example.py
```

Purpose:

- Confirms every source file is syntactically valid in the current Python
  environment.
- This command may create `__pycache__/` folders. They are generated cache
  files and are ignored by `.gitignore`.

## 2. Main Method

### 2.1 `main_method/preprocess_data.py`

Purpose:

- Converts gesture CSV files into compact `.npz` arrays.
- Writes a `manifest.csv` describing the processed source files.
- This is a data-format and preprocessing smoke test for the main-method
  package.

Default data source:

- Preset data: `main_method/example_data/preset_gestures/all/`
- Custom data: `main_method/example_data/custom_gestures/`
- Config: `main_method/configs/preprocess_config.json`

Quick test:

```bash
python main_method/preprocess_data.py --preset-only
```

Expected outputs:

- `main_method/processed/preset_gestures.npz`
- `main_method/processed/manifest.csv`

All bundled example data:

```bash
python main_method/preprocess_data.py
```

Useful parameters:

- `--config`: path to a JSON preprocessing config.
- `--preset-csv-dir`: override the preset CSV directory.
- `--custom-csv-dir`: override the custom gesture root directory.
- `--output-dir`: output folder for generated `.npz` and manifest files.
- `--target-length`: resample each sequence to this length. Default is from
  config, currently 180.
- `--preset-only`: process only preset gesture CSV files.
- `--custom-only`: process only custom gesture CSV files.
- `--max-files`: optional per-dataset file limit for faster local checks.

### 2.2 `main_method/fewshot_demo.py`

Purpose:

- Runs a simple few-shot nearest-centroid baseline on custom gestures.
- Uses support samples as class prototypes and evaluates on the matching test
  split.
- This is a lightweight example-data test, not the full main-method training
  pipeline.

Default data source:

- `main_method/example_data/custom_gestures/`
- Current bundled user: `P16_23_male`
- Current bundled classes: `0 1 2 3`

Quick test:

```bash
python main_method/fewshot_demo.py --shots 1
```

Useful parameters:

- `--custom-root`: custom gesture root directory.
- `--user`: user folder to evaluate. If omitted, the first user folder is used.
- `--classes`: class IDs to evaluate. Default: `0 1 2 3`.
- `--shots`: number of support samples per class. Default: 1.
- `--target-length`: resample each CSV sequence to this length. Default: 180.

## 3. Apple Component

### 3.1 `apple_component/ModelTrain.py`

Purpose:

- Trains the Apple-component base gesture feature extractor/classifier.
- Keeps the original model architecture and training flow.
- Saves only the best validation checkpoint as `best_model.pth` inside the
  chosen output directory.
- With full data, it uses the original user-level split. With the bundled small
  example data, it falls back to a stratified sample-level split so the smoke
  test can run.

Default data source:

- `apple_component/example_data/preset_gestures/all/`
- This folder is a small 12-class subset copied from the original preset data
  source (path omitted in this release to avoid machine-specific references).

Quick CPU smoke test:

```bash
python apple_component/ModelTrain.py --epochs 1 --batch_size 8 --num_workers 0 --skip_tsne --output_dir apple_component/outputs/modeltrain_smoke
```

Expected outputs:

- `apple_component/outputs/modeltrain_smoke/best_model.pth`
- `evaluation_metrics.csv`
- `confusion_matrix_raw.png`
- `confusion_matrix_norm.png`
- `training_curves.png`
- `training_*.log`

Useful parameters:

- `--data_dir`: flat directory of preset gesture CSV files.
- `--output_dir`: where logs, figures, metrics, and best checkpoint are saved.
- `--num_classes`: number of output classes. Default: 12.
- `--batch_size`: training batch size. Default: 8 for the bundled smoke test.
- `--epochs`: number of training epochs. Default: 1 for the bundled smoke test.
- `--lr`: Adam learning rate.
- `--lr_step`: StepLR decay interval.
- `--lr_gamma`: StepLR decay multiplier.
- `--num_workers`: DataLoader worker count. Default is 0 for the safest Windows
  smoke test.
- `--seed`: random seed.
- `--train_users`: number of users for the original user-level train split.
- `--val_users`: number of users for the original user-level validation split.
- `--skip_tsne`: skip t-SNE generation for faster checks. This is enabled by
  default.
- `--run_tsne`: generate the t-SNE figure.

Notes:

- The smoke test does not need `--data_dir`; by default the script uses the
  bundled example CSV files in `apple_component/example_data/preset_gestures/all/`.
- Use `--data_dir` only when replacing the bundled example data with an
  external full preset CSV directory. Do not copy placeholder paths literally.

Bundled example data with more epochs:

```bash
python apple_component/ModelTrain.py --epochs 20 --batch_size 8 --num_workers 0 --skip_tsne --output_dir apple_component/outputs/modeltrain_example_20ep
```

### 3.2 `apple_component/add_head/train.py`

Purpose:

- Trains a per-user add-head for custom gestures.
- Uses the frozen base feature extractor, delta encoder, delta-vector bank, and
  preset gesture features.
- This tests the Apple add-head adaptation pipeline on bundled custom example
  data.
- Defaults are smoke-test settings: bundled example data, classes `0 1 2 3`,
  `n_support=5`, `max_epochs=1`, and `num_workers=0`.

Default data source and resources:

- Custom gestures: `apple_component/example_data/custom_gestures/`
- Base feature extractor: `apple_component/best_model.pth`
- Delta encoder: `apple_component/encoder_training/delta_encoder_best.pth`
- Delta-vector bank: `apple_component/encoder_training/delta_vectors_bank.npy`
- Preset features: `apple_component/preset_gesture.npz`
- These paths are configured in `apple_component/add_head/config.py`.

Quick smoke test:

```bash
python apple_component/add_head/train.py --users P16_23_male --classes 0 1 2 3 --n_support 5 --max_epochs 1 --num_workers 0
```

Known compatibility note:

- On some newer PyTorch builds, `ReduceLROnPlateau` may reject the `verbose`
  argument and raise:
  `TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'`.
- If this occurs, remove `verbose=False` from
  `apple_component/add_head/train.py` (scheduler initialization), or use a
  PyTorch build where this argument is supported.

Expected outputs:

- `apple_component/outputs/add_head_training/P16_23_male/add_head_best.pth`
- `training_log.csv`
- `training_curves.png`
- `confusion_matrix.png`
- `eval_metrics.csv`
- `eval_report.txt`
- `class_map.json`
- `apple_component/outputs/add_head_training/summary.csv`
- `apple_component/outputs/add_head_training/summary.json`

Useful parameters:

- `--data_root`: override custom gesture root. Expected layout:
  `user/class_id/support/*.csv` and `user/class_id/test/*.csv`.
- `--users`: one or more user folder names to train.
- `--classes`: class IDs to include.
- `--n_support`: number of support examples per class. Default: 5.
- `--max_epochs`: override the configured epoch count for quick checks. Default: 1.
- `--num_workers`: DataLoader worker count. Default is 0 for the safest Windows
  smoke test.

### 3.3 `apple_component/run_fixed_manifest_baselines.py`

Purpose:

- Runs fixed-manifest Apple-component baseline evaluations.
- Compares two add-head baseline settings:
  `new_only_head` and `keep_old_head`.
- If no manifest is provided, it builds a one-round manifest from the bundled
  custom example data.
- Defaults are smoke-test settings: `shots=[1]`, `max_epochs=1`,
  `preset_limit=64`, and output under
  `apple_component/outputs/fixed_manifest_smoke`.

Default data source and resources:

- Custom gestures: `apple_component/example_data/custom_gestures/`
- Base feature extractor: `apple_component/best_model.pth`
- Delta encoder: `apple_component/encoder_training/delta_encoder_best.pth`
- Delta-vector bank: `apple_component/encoder_training/delta_vectors_bank.npy`
- Preset features: `apple_component/preset_gesture.npz`

Quick smoke test:

```bash
python apple_component/run_fixed_manifest_baselines.py --shots 1 --max-epochs 1 --preset-limit 64 --artifact-root apple_component/outputs/fixed_manifest_smoke
```

Expected outputs:

- `apple_component/outputs/fixed_manifest_smoke/new_only_head/...`
- `apple_component/outputs/fixed_manifest_smoke/keep_old_head/...`
- Per-setting summaries, predictions, logs, confusion matrices, and `best.pth`
  files under the output folder.

Useful parameters:

- `--custom-data-root`: override custom gesture root.
- `--manifest-path`: fixed split manifest JSON. If omitted, an example manifest
  is generated from bundled custom data.
- `--artifact-root`: output folder for all baseline artifacts.
- `--shots`: one or more support-shot counts, for example `--shots 1 3 5`.
- `--max-epochs`: training epochs per baseline run.
- `--preset-limit`: limit preset features for faster smoke tests.
- `--device`: force device, for example `cpu` or `cuda`.

## 4. DTW Core

### `dtw_core/example.py`

Purpose:

- Demonstrates the standalone Dynamic Time Warping reference implementation.
- Does not require gesture CSV data.
- Useful as a minimal sanity check for the DTW module.

Command:

```bash
python dtw_core/example.py
```

Library file:

- `dtw_core/dtw_core.py` contains reusable DTW functions/classes.
- It is tested by running `dtw_core/example.py` and by the `py_compile`
  command in Section 1.

## 5. Non-CLI Library Files

These files are not meant to be run directly as scripts:

- `apple_component/model.py`: delta encoder/decoder model definitions.
- `apple_component/encoder_training/gesture_augmentation.py`: gesture
  augmentation functions.
- `apple_component/add_head/config.py`: path and training configuration.
- `apple_component/add_head/pipeline.py`: shared add-head data and feature
  pipeline utilities.
- `dtw_core/dtw_core.py`: reusable DTW implementation.

Use the `python -m py_compile ...` command in Section 1 to check them, and use
the runnable scripts above to exercise them through the intended entry points.

## 6. Generated Files

The test commands generate output folders such as:

- `main_method/processed/`
- `apple_component/outputs/`
- `results/`
- `artifacts/`
- `__pycache__/`

These are generated test artifacts and are ignored by `.gitignore`.
