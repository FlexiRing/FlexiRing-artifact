# Quick Test Guide

This is a short quick-run guide for the current release package.
For the full test matrix and parameter details, see `TESTING.md`.

Run from the repository root:

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

Known compatibility note:

- On some newer PyTorch builds, `apple_component/add_head/train.py` may fail if
  `ReduceLROnPlateau` does not accept the `verbose` argument.
- If needed, remove `verbose=False` from scheduler initialization in
  `apple_component/add_head/train.py`.
