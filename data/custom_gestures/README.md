# Custom Gesture Dataset

## Overview

This folder contains the **raw CSV data only** for the custom gesture set used in our experiments.

- Participants: **20**
- Gesture classes per participant: **12** (class IDs `0..11`)
- Samples per class: **50** (`support=10`, `test=40`)
- Total files: **12,000 CSV**
- Demographics: **14 male, 6 female**; age **20-24** (mean **22.5**)

No images, processing scripts, or session metadata are required to use this release.

## Directory Layout

```text
custom_gestures/
├── P01_<AGE>_<GENDER>/
│   ├── 0/
│   │   ├── support/*.csv
│   │   └── test/*.csv
│   ├── 1/
│   │   ├── support/*.csv
│   │   └── test/*.csv
│   └── ...
├── P02_<AGE>_<GENDER>/
│   └── ...
└── ...
```

Participant folder naming rule:

```text
P[ID]_[Age]_[Gender]
```

## File Format

Each CSV file contains one gesture sample:

- Rows: **181** (`1` header row + `180` data rows)
- Columns: **6**
- Header:

```text
ax, ay, az, gx, gy, gz
```

Column meaning:

- `ax`, `ay`, `az`: accelerometer (X/Y/Z)
- `gx`, `gy`, `gz`: gyroscope (X/Y/Z)

## Notes for Reproduction

- The `support`/`test` split is already prepared in this release.
- File names include class IDs and can be parsed directly in training pipelines.
- This dataset is intended for custom-gesture few-shot training and evaluation.
