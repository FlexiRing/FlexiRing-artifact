"""Preprocess gesture CSV files into compact NPZ files.

The generated arrays are convenient for lightweight baselines, data inspection,
and downstream experiments that do not need to stream individual CSV files.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # Keep preprocessing runnable in minimal environments.
    # Run the tqdm helper.
    def tqdm(iterable, **_kwargs):
        return iterable


CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PRESET_CSV_DIR = (SCRIPT_DIR / "example_data/preset_gestures/all").resolve()
DEFAULT_CUSTOM_CSV_DIR = (SCRIPT_DIR / "example_data/custom_gestures").resolve()
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "processed"
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs/preprocess_config.json"


@dataclass(frozen=True)
class SampleRecord:
    """Metadata for one source CSV file before array packing."""

    path: Path
    dataset: str
    label: int
    user: str
    split: str


# Run the load config helper.
def load_config(path: Path | None) -> dict:
    """Load preprocessing options from a JSON file when one is available."""

    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# Run the read imu csv helper.
def read_imu_csv(path: Path, target_length: int) -> np.ndarray:
    """Read one IMU CSV file and return a channel-first fixed-length array."""

    frame = pd.read_csv(path)
    missing = [name for name in CHANNELS if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    data = frame.loc[:, CHANNELS].to_numpy(dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != len(CHANNELS):
        raise ValueError(f"{path} has invalid shape: {data.shape}")

    if data.shape[0] != target_length:
        data = resample_time_axis(data, target_length)

    return data.T.astype(np.float32, copy=False)


# Run the resample time axis helper.
def resample_time_axis(data: np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample the time axis to a fixed sequence length."""

    if data.shape[0] < 2:
        raise ValueError("Cannot resample a sequence with fewer than two rows")

    old_x = np.linspace(0.0, 1.0, num=data.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    out = np.empty((target_length, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        out[:, channel] = np.interp(new_x, old_x, data[:, channel])
    return out


# Run the collect preset samples helper.
def collect_preset_samples(csv_dir: Path) -> list[SampleRecord]:
    """Collect preset gesture CSV records from a flat directory."""

    records: list[SampleRecord] = []
    for path in sorted(csv_dir.glob("*.csv")):
        stem = path.stem
        parts = stem.split("_")
        if len(parts) < 5:
            raise ValueError(f"Unexpected preset filename: {path.name}")
        user = parts[1]
        label = int(parts[-1])
        records.append(SampleRecord(path, "preset", label, user, "all"))
    return records


# Run the collect custom samples helper.
def collect_custom_samples(root: Path) -> list[SampleRecord]:
    """Collect custom gesture CSV records from user/class/split folders."""

    records: list[SampleRecord] = []
    for user_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in user_dir.iterdir() if p.is_dir()):
            if not re.fullmatch(r"\d+", class_dir.name):
                continue
            label = int(class_dir.name)
            for split in ("support", "test"):
                split_dir = class_dir / split
                if not split_dir.is_dir():
                    continue
                for path in sorted(split_dir.glob("*.csv")):
                    records.append(SampleRecord(path, "custom", label, user_dir.name, split))
    return records


# Run the pack records helper.
def pack_records(records: list[SampleRecord], target_length: int) -> dict[str, np.ndarray]:
    """Pack CSV records into arrays ready for NPZ serialization."""

    x = np.empty((len(records), len(CHANNELS), target_length), dtype=np.float32)
    y = np.empty((len(records),), dtype=np.int64)
    users: list[str] = []
    splits: list[str] = []
    files: list[str] = []

    for idx, record in enumerate(tqdm(records, desc="Reading CSV")):
        x[idx] = read_imu_csv(record.path, target_length)
        y[idx] = record.label
        users.append(record.user)
        splits.append(record.split)
        files.append(str(record.path))

    return {
        "x": x,
        "y": y,
        "users": np.array(users, dtype=object),
        "splits": np.array(splits, dtype=object),
        "files": np.array(files, dtype=object),
        "channels": np.array(CHANNELS, dtype=object),
    }


# Run the write npz helper.
def write_npz(records: list[SampleRecord], output_path: Path, target_length: int) -> None:
    """Write packed gesture arrays to a compressed NPZ file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = pack_records(records, target_length)
    np.savez_compressed(output_path, **arrays)
    print(f"Wrote {output_path} ({len(records)} samples)")


# Run the write manifest helper.
def write_manifest(records: Iterable[SampleRecord], output_path: Path) -> None:
    """Write a CSV manifest describing the processed source files."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "label", "user", "split", "file"])
        for record in records:
            writer.writerow([record.dataset, record.label, record.user, record.split, record.path])
    print(f"Wrote {output_path}")


# Parse command-line arguments.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--preset-csv-dir", type=Path)
    parser.add_argument("--custom-csv-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--target-length", type=int)
    parser.add_argument("--preset-only", action="store_true")
    parser.add_argument("--custom-only", action="store_true")
    parser.add_argument("--max-files", type=int, default=None, help="Optional per-dataset file limit for fast checks.")
    return parser.parse_args()


# Run the command-line entry point.
def main() -> None:
    """Run preprocessing for preset and/or custom gesture datasets."""

    args = parse_args()
    config = load_config(args.config if args.config.exists() else None)

    preset_csv_dir = args.preset_csv_dir or Path(config.get("preset_csv_dir", DEFAULT_PRESET_CSV_DIR))
    custom_csv_dir = args.custom_csv_dir or Path(config.get("custom_csv_dir", DEFAULT_CUSTOM_CSV_DIR))
    output_dir = args.output_dir or Path(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    if not preset_csv_dir.is_absolute():
        preset_csv_dir = (SCRIPT_DIR / preset_csv_dir).resolve()
    if not custom_csv_dir.is_absolute():
        custom_csv_dir = (SCRIPT_DIR / custom_csv_dir).resolve()
    if args.output_dir is None and not output_dir.is_absolute():
        output_dir = (SCRIPT_DIR / output_dir).resolve()
    target_length = args.target_length or int(config.get("target_length", 180))

    include_preset = bool(config.get("include_preset", True))
    include_custom = bool(config.get("include_custom", True))
    if args.preset_only:
        include_preset, include_custom = True, False
    if args.custom_only:
        include_preset, include_custom = False, True

    all_records: list[SampleRecord] = []

    if include_preset:
        preset_records = collect_preset_samples(preset_csv_dir)
        if args.max_files is not None:
            preset_records = preset_records[: args.max_files]
        write_npz(preset_records, output_dir / "preset_gestures.npz", target_length)
        all_records.extend(preset_records)

    if include_custom:
        custom_records = collect_custom_samples(custom_csv_dir)
        if args.max_files is not None:
            custom_records = custom_records[: args.max_files]
        write_npz(custom_records, output_dir / "custom_gestures.npz", target_length)
        all_records.extend(custom_records)

    write_manifest(all_records, output_dir / "manifest.csv")


if __name__ == "__main__":
    main()
