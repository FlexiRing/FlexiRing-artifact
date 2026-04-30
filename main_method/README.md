# Main Method

This folder contains the pre-acceptance main-method release: bundled example
CSV data, a standalone preprocessing script, and a simple few-shot evaluation
script. The complete implementation will be released after paper acceptance.

## Contents

- `example_data/preset_gestures/all/`: small preset-gesture CSV subset.
- `example_data/custom_gestures/`: bundled custom-gesture support/test data.
- `configs/preprocess_config.json`: default paths for the bundled example data.
- `best_backbone_CE_SupCon_Triplet.pth`: backbone resource file.
- `preprocess_data.py`: converts raw CSV files into compact `.npz` arrays.
- `fewshot_demo.py`: nearest-centroid few-shot baseline for custom gestures.

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

From the release root:

```bash
python main_method/preprocess_data.py --preset-only
python main_method/fewshot_demo.py --shots 1
python main_method/fewshot_demo.py
```

Expected output for the default 5-shot run:

```text
User: P16_23_male
Classes: [0, 1, 2, 3]
Shots per class: 5
Support samples: 20
Test samples: 160
Accuracy: 1.0000
Macro F1: 1.0000
```

The bundled few-shot example is selected for stable verification: 1-shot reaches
about `0.9938` accuracy, and 3-shot or more reaches `1.0000` on the included
test split.

## Use Your Own Data

Preset gesture CSV files should be stored in one flat folder:

```text
your_preset_data/
  GestureName_P01_user_00001_0.csv
  GestureName_P01_user_00002_0.csv
```

Custom few-shot data should follow this structure:

```text
your_custom_data/
  UserName/
    0/
      support/*.csv
      test/*.csv
    1/
      support/*.csv
      test/*.csv
```

Run with explicit paths:

```bash
python main_method/preprocess_data.py --preset-csv-dir your_preset_data --custom-csv-dir your_custom_data
python main_method/fewshot_demo.py --custom-root your_custom_data --user UserName --classes 0 1 --shots 5
```

Preprocessing writes:

```text
processed/
  preset_gestures.npz
  custom_gestures.npz
  manifest.csv
```
