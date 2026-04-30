"""Simple few-shot baseline for custom gesture data.

The demo uses support samples as prototypes and evaluates nearest-centroid
classification on the matching test split. It does not require the backbone
`src` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler


CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CUSTOM_ROOT = (SCRIPT_DIR / "example_data/custom_gestures").resolve()


# Run the read flattened csv helper.
def read_flattened_csv(path: Path, target_length: int = 180) -> np.ndarray:
    frame = pd.read_csv(path)
    data = frame.loc[:, CHANNELS].to_numpy(dtype=np.float32)
    if data.shape[0] != target_length:
        data = resample(data, target_length)
    return data.T.reshape(-1)


# Run the resample helper.
def resample(data: np.ndarray, target_length: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, num=data.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
    out = np.empty((target_length, data.shape[1]), dtype=np.float32)
    for channel in range(data.shape[1]):
        out[:, channel] = np.interp(new_x, old_x, data[:, channel])
    return out


# Run the choose default user helper.
def choose_default_user(custom_root: Path) -> str:
    if not custom_root.is_dir():
        raise FileNotFoundError(
            f"Custom gesture root does not exist: {custom_root}. "
            "Pass --custom-root to point at data/data/custom_gestures."
        )
    users = sorted(p.name for p in custom_root.iterdir() if p.is_dir())
    if not users:
        raise FileNotFoundError(f"No user folders found under {custom_root}")
    return users[0]


# Run the load split helper.
def load_split(
    custom_root: Path,
    user: str,
    labels: list[int],
    split: str,
    shots: int | None,
    target_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    targets: list[int] = []

    for label in labels:
        split_dir = custom_root / user / str(label) / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        files = sorted(split_dir.glob("*.csv"))
        if split == "support" and shots is not None:
            files = files[:shots]
        if not files:
            raise FileNotFoundError(f"No CSV files found in {split_dir}")
        for path in files:
            features.append(read_flattened_csv(path, target_length))
            targets.append(label)

    return np.stack(features).astype(np.float32), np.array(targets, dtype=np.int64)


# Run the nearest centroid predict helper.
def nearest_centroid_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    labels = np.array(sorted(set(train_y.tolist())), dtype=np.int64)
    centroids = np.stack([train_x[train_y == label].mean(axis=0) for label in labels])
    distances = ((test_x[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return labels[np.argmin(distances, axis=1)]


# Parse command-line arguments.
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--custom-root", type=Path, default=DEFAULT_CUSTOM_ROOT)
    parser.add_argument("--user", type=str, default=None)
    parser.add_argument("--classes", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--shots", type=int, default=1)
    parser.add_argument("--target-length", type=int, default=180)
    return parser.parse_args()


# Run the command-line entry point.
def main() -> None:
    args = parse_args()
    user = args.user or choose_default_user(args.custom_root)

    support_x, support_y = load_split(
        args.custom_root, user, args.classes, "support", args.shots, args.target_length
    )
    test_x, test_y = load_split(
        args.custom_root, user, args.classes, "test", None, args.target_length
    )

    scaler = StandardScaler()
    support_x = scaler.fit_transform(support_x)
    test_x = scaler.transform(test_x)

    pred_y = nearest_centroid_predict(support_x, support_y, test_x)
    acc = accuracy_score(test_y, pred_y)
    macro_f1 = f1_score(test_y, pred_y, average="macro", zero_division=0)

    print(f"User: {user}")
    print(f"Classes: {args.classes}")
    print(f"Shots per class: {args.shots}")
    print(f"Support samples: {len(support_y)}")
    print(f"Test samples: {len(test_y)}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print()
    print(classification_report(test_y, pred_y, zero_division=0))


if __name__ == "__main__":
    main()
