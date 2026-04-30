# DTW Core

## Overview

This directory contains a lightweight Dynamic Time Warping (DTW) implementation for template-based gesture recognition.

It is intended to be reusable in other applications without depending on any particular experiment protocol.

The implementation includes:

- sequence loading from csv
- preprocessing for accelerometer-based gesture signals
- DTW distance computation
- template construction
- nearest-template classification

---

## Files

- `dtw_core.py`
  - core implementation
- `example.py`
  - minimal usage example that runs on bundled example CSV files

---

## Input Format

The default input format is:

- rows: time steps
- columns: sensor channels

By default, only the first `3` channels are used as accelerometer signals.

---

## Preprocessing Pipeline

The current preprocessing pipeline is:

1. keep accelerometer channels
2. temporal compression
3. quantization

### Temporal Compression

- window: `50 ms`
- stride: `30 ms`

### Quantization

Continuous values are mapped into integer levels in `[-16, 16]`.

---

## DTW Formulation

For two sequences `Q=(q_1,...,q_n)` and `T=(t_1,...,t_m)`:

Point-wise distance:

```math
d(i,j)=\|q_i-t_j\|_2
```

Dynamic programming recursion:

```math
D(i,j)=d(i,j)+\min\{D(i-1,j),D(i,j-1),D(i-1,j-1)\}
```

Final DTW distance:

```math
DTW(Q,T)=D(n,m)
```

---

## Template Classification

For a query sequence, compute DTW distance to all class templates and assign the label of the nearest template:

```math
\hat{y}=\arg\min_c DTW(Q,T_c)
```

---

## Usage

### Example

```bash
python dtw_core/example.py
```

Expected output predicts class `2` for the bundled `Beta_003_2_2.csv`
query and prints the best DTW distance.

### Python API

Available functions in `dtw_core.py`:

- `load_csv_sequence()`
- `keep_accel_channels()`
- `compress_sequence()`
- `quantize_sequence()`
- `preprocess_sequence()`
- `dtw_distance()`
- `build_templates()`
- `classify_by_templates()`

---

## Configuration

The main parameters are managed through `DTWConfig`:

- `sample_hz`
- `window_ms`
- `stride_ms`
- `quant_levels`
- `accel_channels`

Example:

```python
config = DTWConfig(
    sample_hz=60,
    window_ms=50,
    stride_ms=30,
    quant_levels=16,
)
```

---

## Notes

- This implementation is designed for template-based recognition.
- It is suitable for few-shot or nearest-template style applications.
- If needed, the preprocessing steps and quantization strategy can be modified independently from the DTW core.
