from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class DTWConfig:
    sample_hz: int = 60
    window_ms: int = 50
    stride_ms: int = 30
    quant_levels: int = 16
    accel_channels: int = 3


def load_csv_sequence(path: str | Path) -> np.ndarray:
    """Load one gesture csv as shape (T, C)."""
    df = pd.read_csv(path)
    return df.values.astype(np.float32)


def keep_accel_channels(sequence: np.ndarray, accel_channels: int = 3) -> np.ndarray:
    """Keep the first accelerometer channels.

    Input shape: (T, C)
    Output shape: (T, accel_channels)
    """
    return sequence[:, :accel_channels]


def compress_sequence(sequence: np.ndarray, sample_hz: int = 60, window_ms: int = 50, stride_ms: int = 30) -> np.ndarray:
    """Temporal compression by sliding-window mean.

    Input shape: (T, C)
    Output shape: (T', C)
    """
    window = max(1, int(round(window_ms * sample_hz / 1000.0)))
    stride = max(1, int(round(stride_ms * sample_hz / 1000.0)))
    compressed = []
    for start in range(0, sequence.shape[0] - window + 1, stride):
        compressed.append(sequence[start:start + window].mean(axis=0))
    if not compressed:
        compressed.append(sequence.mean(axis=0))
    return np.stack(compressed, axis=0)


def quantize_sequence(sequence: np.ndarray, quant_levels: int = 16, clip_range: float = 4.0) -> np.ndarray:
    """Quantize continuous values to integer levels in [-quant_levels, quant_levels].

    This implementation uses logarithmic compression before rounding.
    """
    clipped = np.clip(sequence, -clip_range, clip_range)
    scaled = np.sign(clipped) * np.log1p(np.abs(clipped)) / np.log1p(clip_range)
    return np.clip(np.rint(scaled * quant_levels), -quant_levels, quant_levels).astype(np.int16)


def preprocess_sequence(sequence: np.ndarray, config: DTWConfig | None = None) -> np.ndarray:
    """Full DTW preprocessing pipeline.

    Steps:
    1. keep accelerometer channels
    2. temporal compression
    3. quantization
    """
    if config is None:
        config = DTWConfig()
    seq = keep_accel_channels(sequence, config.accel_channels)
    seq = compress_sequence(seq, config.sample_hz, config.window_ms, config.stride_ms)
    seq = quantize_sequence(seq, config.quant_levels)
    return seq


def dtw_distance(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Compute DTW distance between two preprocessed sequences.

    seq1, seq2 shape: (T, C)
    """
    n, m = len(seq1), len(seq2)
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        ai = seq1[i - 1]
        for j in range(1, m + 1):
            cost = np.linalg.norm(ai - seq2[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m])


def build_templates(sequences: list[np.ndarray], labels: list[int], templates_per_class: int = 1, config: DTWConfig | None = None):
    """Build template set from raw sequences.

    Returns a list of (label, template_sequence).
    """
    if config is None:
        config = DTWConfig()
    grouped: dict[int, list[np.ndarray]] = {}
    for seq, label in zip(sequences, labels):
        grouped.setdefault(int(label), []).append(seq)
    templates = []
    for label in sorted(grouped):
        for seq in grouped[label][:templates_per_class]:
            templates.append((label, preprocess_sequence(seq, config)))
    return templates


def classify_by_templates(query_sequence: np.ndarray, templates, config: DTWConfig | None = None) -> tuple[int, float]:
    """Classify one raw query sequence by nearest template.

    Returns:
    - predicted label
    - best DTW distance
    """
    if config is None:
        config = DTWConfig()
    query = preprocess_sequence(query_sequence, config)
    best_label = None
    best_distance = float('inf')
    for label, template in templates:
        dist = dtw_distance(query, template)
        if dist < best_distance:
            best_distance = dist
            best_label = int(label)
    return best_label, best_distance
