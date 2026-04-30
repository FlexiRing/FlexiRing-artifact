# Gesture Recognition Pre-Acceptance Release

This package is a runnable pre-acceptance release. It provides example gesture
data, data preprocessing utilities, a few-shot evaluation script, resource
files, an Apple component package, and a DTW reference implementation. The
complete implementation and paper-aligned experiment runners will be released
after paper acceptance.

## Structure

```text
gesture_open_source_release/
  main_method/
    example_data/
    configs/
    best_backbone_CE_SupCon_Triplet.pth
    preprocess_data.py
    fewshot_demo.py
    README.md
  apple_component/
    example_data/
    add_head/
    encoder_training/
    best_model.pth
    preset_gesture.npz
    ModelTrain.py
    model.py
    run_fixed_manifest_baselines.py
    README.md
  dtw_core/
    dtw_core.py
    example.py
    README.md
```

## Components

- `main_method/`: main-method release package. It contains example preset and
  custom CSV data, the standalone preprocessing script, the simple few-shot
  demo, and a backbone resource file.
- `apple_component/`: Apple-component release package. It contains Apple
  resource files, bundled custom CSV data, public data augmentation utilities,
  and training entry points for the base feature extractor and per-user
  add-head.
- `dtw_core/`: lightweight Dynamic Time Warping reference implementation for
  template-based gesture recognition. It is self-contained and documented in
  its own README.

## Quick Checks

For detailed test commands, parameter meanings, data sources, and module
purposes, see [TESTING.md](TESTING.md). For a shorter quick-run guide,
see [QUICK_TEST.md](QUICK_TEST.md).

Known compatibility note: on some newer PyTorch builds, the add-head smoke
test may require removing `verbose=False` from
`apple_component/add_head/train.py` (scheduler initialization). See
`TESTING.md` for details.

```bash
cd gesture_open_source_release

pip install -r main_method/requirements.txt
pip install -r apple_component/requirements.txt
python main_method/preprocess_data.py --preset-only
python main_method/fewshot_demo.py --shots 1
python apple_component/ModelTrain.py --epochs 1 --batch_size 8 --num_workers 0 --skip_tsne --output_dir apple_component/outputs/modeltrain_smoke
python apple_component/add_head/train.py --users P16_23_male --classes 0 1 2 3 --n_support 5 --max_epochs 1 --num_workers 0
python apple_component/run_fixed_manifest_baselines.py --shots 1 --max-epochs 1 --preset-limit 64 --artifact-root apple_component/outputs/fixed_manifest_smoke
python dtw_core/example.py
```

The bundled data is intentionally small so that users can verify file formats
and command-line entry points before replacing it with their own CSV data.

## Release Scope

- Included now: example preset/custom CSV data, preprocessing, a simple
  main-method few-shot baseline, the final main-method backbone checkpoint
  (`main_method/best_backbone_CE_SupCon_Triplet.pth`), Apple resource files,
  Apple training/source utilities, and DTW reference code.
- Not included now: full implementation source code and paper reproduction
  runners.
- Planned after acceptance: complete source code for reproducible paper-level
  experiments.
