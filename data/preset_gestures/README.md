# Predefined Gesture Dataset

## Overview

This folder contains the **raw CSV data only** for the predefined gesture set.

- Participants: **34** (`P01` to `P34`)
- Gesture classes: **12** (`0..11`)
- Samples per participant per class: **100**
- Total files: **40,800 CSV**
- Demographics: **26 male, 8 female**; age **19-24** (mean **21.5**)

## Directory Structure

```text
preset_gestures/
└── all/
    └── <GestureName>_<ParticipantID>_<Age>-<Gender>_<SampleIndex>_<ClassLabel>.csv
```

Example:

```text
Check_P01_<AGE>-<GENDER>_00001_10.csv
```

## Filename Fields

| Field | Description |
|---|---|
| `GestureName` | Gesture name |
| `ParticipantID` | Anonymous participant ID (e.g., `P01`) |
| `Age-Gender` | Participant age and gender |
| `SampleIndex` | 5-digit sample index |
| `ClassLabel` | Numeric class label (`0..11`) |

## Data Format

Each CSV file stores one sample:

- Rows: **181** (`1` header row + `180` data rows)
- Columns: **6**
- Header:

```text
ax, ay, az, gx, gy, gz
```

Column meaning:

- `ax`, `ay`, `az`: accelerometer (X/Y/Z)
- `gx`, `gy`, `gz`: gyroscope (X/Y/Z)

## Class Mapping

| Label | Gesture |
|---|---|
| 0 | UpDown |
| 1 | DownUp |
| 2 | LeftRight |
| 3 | RightLeft |
| 4 | CircleCW |
| 5 | CircleCCW |
| 6 | ZigZag |
| 7 | Triangle |
| 8 | Square |
| 9 | Infinity |
| 10 | Check |
| 11 | Cross |
