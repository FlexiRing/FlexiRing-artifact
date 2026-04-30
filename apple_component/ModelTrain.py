#!/usr/bin/env python3
"""
Gesture Recognition Model with SE-enhanced MBConv blocks (~106K params).

Architecture (from diagram):
  Part A - Feature Embedding Extraction:
    6 IMU channels → bandpass filter (4 sub-bands each)
    → per-channel MBConv1+SE → MBConv6+SE
    → Concatenate → SeparableConv → MaxPool → Flatten(120)
  Part B - Inference:
    120 → [Dense+BN+ReLU+Drop]×4 → Output(num_classes)

Training:
  - CrossEntropy loss, Adam optimizer
  - 200 epochs, StepLR decay ×0.1 every 50 epochs

Features:
  - Real-time training log with tqdm progress bars
  - Confusion matrices (normalized + unnormalized)
  - t-SNE visualization of 120-dim embeddings
  - Accuracy, F1-score, Recall metrics
"""

import argparse
import os
import glob
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend, resample

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import (confusion_matrix, classification_report,
                             accuracy_score, f1_score, recall_score)
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

try:
    from tqdm import tqdm
except ImportError:
    class _TqdmFallback:
        def __init__(self, iterable, **_kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **_kwargs):
            pass

    def tqdm(iterable, **kwargs):
        return _TqdmFallback(iterable, **kwargs)
import logging


SCRIPT_DIR = Path(__file__).resolve().parent


# ============================================================
# 0. Logging Setup
# ============================================================

def setup_logger(log_dir="results"):
    """Setup dual logging: console (real-time) + file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger("GestureModel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f"Log file: {log_file}")
    return logger, log_dir


# ============================================================
# 1. Signal Preprocessing
# ============================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 0.001)
    high = min(highcut / nyq, 0.999)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_filters(signal_1d, fs=100.0):
    raw = signal_1d.copy()
    padlen = min(12, len(raw) - 1)

    b, a = butter_bandpass(0.22, 8.0, fs, order=4)
    low_pass = filtfilt(b, a, raw, padlen=padlen)

    b, a = butter_bandpass(8.0, 32.0, fs, order=4)
    mid_pass = filtfilt(b, a, raw, padlen=padlen)

    b, a = butter_bandpass(32.0, 49.0, fs, order=4)
    high_pass = filtfilt(b, a, raw, padlen=padlen)

    return np.stack([raw, low_pass, mid_pass, high_pass], axis=-1)


def preprocess_sample(data_180x6):
    data = detrend(data_180x6, axis=0)
    data_100x6 = resample(data, 100, axis=0)
    channels = []
    for ch in range(6):
        filtered = apply_filters(data_100x6[:, ch], fs=100.0)
        channels.append(filtered)
    return np.stack(channels, axis=0).astype(np.float32)


# ============================================================
# 2. Dataset
# ============================================================

class GestureDataset(Dataset):
    def __init__(self, data_dir, cache=True):
        self.cache = cache
        self.cached_data = {}

        self.file_list = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
        if not self.file_list:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")

        self.labels = []
        self.user_ids = []
        self.class_names = {}
        for f in self.file_list:
            parts = os.path.splitext(os.path.basename(f))[0].split("_")
            label = int(parts[-1])
            self.labels.append(label)
            self.user_ids.append(parts[1])  # e.g. P01
            if label not in self.class_names:
                self.class_names[label] = parts[0]

        self.num_classes = len(set(self.labels))
        self.sorted_class_names = [self.class_names.get(i, str(i))
                                   for i in range(self.num_classes)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if self.cache and idx in self.cached_data:
            return self.cached_data[idx]

        df = pd.read_csv(self.file_list[idx], header=0)
        data = df.values.astype(np.float32)
        x = torch.from_numpy(preprocess_sample(data))
        y = self.labels[idx]

        if self.cache:
            self.cached_data[idx] = (x, y)
        return x, y


# ============================================================
# 3. Model Components
# ============================================================

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduced_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, reduced_channels, bias=False)
        self.fc2 = nn.Linear(reduced_channels, channels, bias=False)

    def forward(self, x):
        b, c, _ = x.shape
        s = self.pool(x).view(b, c)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(-1)


class MBConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, expansion, kernel_size=10,
                 stride=1, se_reduction=2):
        super().__init__()
        mid_ch = in_ch * expansion
        self.use_residual = (stride == 1 and in_ch == out_ch)
        pad = (kernel_size - 1) // 2

        layers = []
        if expansion != 1:
            layers += [
                nn.Conv1d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm1d(mid_ch),
                nn.ReLU6(inplace=True),
            ]
        layers += [
            nn.Conv1d(mid_ch, mid_ch, kernel_size, stride=stride,
                      padding=pad, groups=mid_ch, bias=False),
            nn.BatchNorm1d(mid_ch),
            nn.ReLU6(inplace=True),
        ]
        self.pre_se = nn.Sequential(*layers)

        se_reduced = max(1, mid_ch // se_reduction)
        self.se = SqueezeExcitation(mid_ch, se_reduced)

        self.projection = nn.Sequential(
            nn.Conv1d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

    def forward(self, x):
        out = self.pre_se(x)
        out = self.se(out)
        out = self.projection(out)
        if self.use_residual:
            out = out + x
        return out


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size, padding=0,
                      groups=in_ch, bias=False),
            nn.Conv1d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ChannelBranch(nn.Module):
    def __init__(self, se_reduction=2):
        super().__init__()
        self.mbconv1 = MBConvBlock(4, 16, expansion=1, kernel_size=10,
                                    stride=1, se_reduction=se_reduction)
        self.mbconv6 = MBConvBlock(16, 24, expansion=6, kernel_size=10,
                                    stride=2, se_reduction=se_reduction)

    def forward(self, x):
        return self.mbconv6(self.mbconv1(x))


# ============================================================
# 4. Full Model
# ============================================================

class GestureModel(nn.Module):
    def __init__(self, num_classes=12, se_reduction=2):
        super().__init__()

        self.branches = nn.ModuleList([ChannelBranch(se_reduction) for _ in range(6)])
        self.sep_conv = DepthwiseSeparableConv1d(144, 24, kernel_size=10)
        self.maxpool = nn.MaxPool1d(kernel_size=8)
        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(120, 80),  nn.BatchNorm1d(80),  nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(80, 40),   nn.BatchNorm1d(40),  nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(40, 20),   nn.BatchNorm1d(20),  nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(20, 10),   nn.BatchNorm1d(10),  nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(10, num_classes),
        )

    def forward(self, x):
        outs = []
        for i in range(6):
            ch = x[:, i, :, :].permute(0, 2, 1)
            outs.append(self.branches[i](ch))
        out = torch.cat(outs, dim=1)
        out = self.sep_conv(out)
        out = self.maxpool(out)
        out = self.flatten(out)
        return self.classifier(out)

    def get_embedding(self, x):
        outs = []
        for i in range(6):
            ch = x[:, i, :, :].permute(0, 2, 1)
            outs.append(self.branches[i](ch))
        out = torch.cat(outs, dim=1)
        out = self.sep_conv(out)
        out = self.maxpool(out)
        return self.flatten(out)


# ============================================================
# 5. Training & Evaluation
# ============================================================

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, criterion, optimizer, device, epoch, logger):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"  Train Epoch {epoch:>3d}", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += batch_size

        # Real-time update on progress bar
        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{100.*correct/total:.2f}%")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, desc="  Val"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc=desc, leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        correct += outputs.argmax(1).eq(targets).sum().item()
        total += batch_size

        pbar.set_postfix(loss=f"{total_loss/total:.4f}",
                         acc=f"{100.*correct/total:.2f}%")

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def collect_predictions_and_embeddings(model, loader, device):
    """Collect all predictions, true labels, and 120-dim embeddings."""
    model.eval()
    all_preds, all_labels, all_embeds = [], [], []

    pbar = tqdm(loader, desc="  Collecting", leave=False,
                bar_format="{l_bar}{bar:30}{r_bar}")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = torch.as_tensor(targets, dtype=torch.long, device=device)

        logits = model(inputs)
        embeddings = model.get_embedding(inputs)

        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(targets.cpu().numpy())
        all_embeds.append(embeddings.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_embeds = np.concatenate(all_embeds)

    return all_preds, all_labels, all_embeds


# ============================================================
# 6. Visualization & Metrics
# ============================================================

def plot_confusion_matrices(y_true, y_pred, class_names, save_dir):
    """Save both unnormalized and normalized confusion matrices."""
    label_ids = list(range(len(class_names)))
    cm_raw = confusion_matrix(y_true, y_pred, labels=label_ids)
    cm_norm = confusion_matrix(y_true, y_pred, labels=label_ids, normalize='true')
    cm_norm = np.nan_to_num(cm_norm)

    for cm_data, title_suffix, fname, fmt, vmax in [
        (cm_raw,  "(Unnormalized)", "confusion_matrix_raw.png",    "d",   None),
        (cm_norm, "(Normalized)",   "confusion_matrix_norm.png",  ".2f",  1.0),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm_data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names,
                    ax=ax, vmin=0, vmax=vmax,
                    linewidths=0.5, linecolor='gray')
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title(f"Confusion Matrix {title_suffix}", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()

        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def plot_tsne(embeddings, labels, class_names, save_dir, perplexity=30, seed=42):
    """Compute t-SNE on 120-dim embeddings and save scatter plot."""
    if len(embeddings) < 3:
        print("  Skipping t-SNE: need at least 3 test embeddings.")
        return
    perplexity = min(perplexity, max(2, (len(embeddings) - 1) // 3))
    print("  Computing t-SNE (this may take a moment)...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                n_iter=1000, init='pca', learning_rate='auto')
    coords = tsne.fit_transform(embeddings)

    num_classes = len(class_names)
    cmap = cm.get_cmap('tab20' if num_classes > 10 else 'tab10', num_classes)

    fig, ax = plt.subplots(figsize=(12, 10))
    for cls_idx in range(num_classes):
        mask = labels == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(cls_idx)], label=class_names[cls_idx],
                   s=8, alpha=0.6, edgecolors='none')

    ax.legend(loc='best', fontsize=8, markerscale=3, framealpha=0.8)
    ax.set_title("t-SNE Visualization of 120-dim Embeddings", fontsize=14)
    ax.set_xlabel("t-SNE Dim 1", fontsize=11)
    ax.set_ylabel("t-SNE Dim 2", fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(save_dir, "tsne_embeddings.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_training_curves(history, save_dir):
    """Save training/validation loss and accuracy curves."""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=1.2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=1.2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=1.2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=1.2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training & Validation Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def compute_and_log_metrics(y_true, y_pred, class_names, logger):
    """Compute and log accuracy, macro/weighted F1, macro/weighted recall."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)

    logger.info("=" * 60)
    logger.info("FINAL EVALUATION METRICS")
    logger.info("=" * 60)
    logger.info(f"  Accuracy:          {acc:.4f}  ({acc*100:.2f}%)")
    logger.info(f"  F1 (macro):        {f1_macro:.4f}")
    logger.info(f"  F1 (weighted):     {f1_weighted:.4f}")
    logger.info(f"  Recall (macro):    {recall_macro:.4f}")
    logger.info(f"  Recall (weighted): {recall_weighted:.4f}")
    logger.info("-" * 60)

    # Per-class report
    report = classification_report(y_true, y_pred,
                                   labels=list(range(len(class_names))),
                                   target_names=class_names,
                                   digits=4, zero_division=0)
    logger.info("Per-class Classification Report:\n" + report)

    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
    }


def save_metrics_csv(metrics_dict, y_true, y_pred, class_names, save_dir):
    """Save per-class and overall metrics to CSV."""
    # Per-class metrics
    label_ids = list(range(len(class_names)))
    f1_per = f1_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    recall_per = recall_score(y_true, y_pred, labels=label_ids, average=None, zero_division=0)
    cm_raw = confusion_matrix(y_true, y_pred, labels=label_ids)
    support = cm_raw.sum(axis=1)

    rows = []
    for i, name in enumerate(class_names):
        correct_i = cm_raw[i, i]
        total_i = support[i]
        rows.append({
            'class': name,
            'class_id': i,
            'accuracy': correct_i / total_i if total_i > 0 else 0,
            'f1_score': f1_per[i],
            'recall': recall_per[i],
            'support': int(total_i),
        })

    # Overall row
    rows.append({
        'class': 'OVERALL',
        'class_id': -1,
        'accuracy': metrics_dict['accuracy'],
        'f1_score': metrics_dict['f1_macro'],
        'recall': metrics_dict['recall_macro'],
        'support': int(len(y_true)),
    })

    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, "evaluation_metrics.csv")
    df.to_csv(path, index=False, float_format='%.4f')
    print(f"  Saved: {path}")


# ============================================================
# 7. Main
# ============================================================

def parse_args():
    default_data_dir = SCRIPT_DIR / "example_data" / "preset_gestures" / "all"
    default_output_dir = SCRIPT_DIR / "outputs" / "modeltrain_smoke"
    parser = argparse.ArgumentParser(
        description="Train the preset GestureModel on CSV IMU gesture data."
    )
    parser.add_argument("--data_dir", type=str, default=str(default_data_dir),
                        help="Directory containing preset gesture CSV files.")
    parser.add_argument("--output_dir", type=str, default=str(default_output_dir),
                        help="Directory for logs, checkpoints, and figures.")
    parser.add_argument("--num_classes", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step", type=int, default=50)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_users", type=int, default=17)
    parser.add_argument("--val_users", type=int, default=3)
    parser.add_argument("--skip_tsne", action="store_true", default=True,
                        help="Skip t-SNE figure generation.")
    parser.add_argument("--run_tsne", action="store_false", dest="skip_tsne",
                        help="Generate the t-SNE figure.")
    return parser.parse_args()


def build_split_indices(dataset, train_users_n, val_users_n, seed, logger):
    unique_users = sorted(set(dataset.user_ids))
    required_users = train_users_n + val_users_n + 1

    if len(unique_users) >= required_users:
        import random as _rnd
        split_users = unique_users[:]
        _rnd.Random(seed).shuffle(split_users)
        train_users = set(split_users[:train_users_n])
        val_users = set(split_users[train_users_n:train_users_n + val_users_n])
        test_users = set(split_users[train_users_n + val_users_n:])
        logger.info("Using user-level split.")
        logger.info(f"Train users ({len(train_users)}): {sorted(train_users)}")
        logger.info(f"Val users   ({len(val_users)}):   {sorted(val_users)}")
        logger.info(f"Test users  ({len(test_users)}):  {sorted(test_users)}")

        train_idx = [i for i, u in enumerate(dataset.user_ids) if u in train_users]
        val_idx = [i for i, u in enumerate(dataset.user_ids) if u in val_users]
        test_idx = [i for i, u in enumerate(dataset.user_ids) if u in test_users]
        return train_idx, val_idx, test_idx

    logger.info(
        "Using stratified sample-level split because the bundled example data "
        f"has {len(unique_users)} user(s), fewer than {required_users}."
    )
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for label in sorted(set(dataset.labels)):
        cls_idx = np.array([i for i, y in enumerate(dataset.labels) if y == label])
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        if n >= 3:
            n_test = max(1, int(round(n * 0.15)))
            n_val = max(1, int(round(n * 0.15)))
            n_train = max(1, n - n_val - n_test)
            train_idx.extend(cls_idx[:n_train].tolist())
            val_idx.extend(cls_idx[n_train:n_train + n_val].tolist())
            test_idx.extend(cls_idx[n_train + n_val:].tolist())
        elif n == 2:
            train_idx.append(int(cls_idx[0]))
            test_idx.append(int(cls_idx[1]))
        else:
            train_idx.append(int(cls_idx[0]))

    if not val_idx:
        val_idx = test_idx[:]
    if not test_idx:
        test_idx = val_idx[:]

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def main():
    # ---- Configuration ----
    args = parse_args()
    DATA_DIR = args.data_dir
    NUM_CLS = args.num_classes
    BATCH = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    LR_STEP = args.lr_step
    LR_GAMMA = args.lr_gamma
    N_TRAIN_USERS = args.train_users
    N_VAL_USERS = args.val_users
    SEED = args.seed

    # ---- Setup ----
    logger, save_dir = setup_logger(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Results directory: {save_dir}")
    logger.info(f"Data directory: {DATA_DIR}")

    # ---- Data ----
    logger.info("Loading dataset...")
    full_dataset = GestureDataset(DATA_DIR, cache=True)
    logger.info(f"Full dataset: {len(full_dataset)} samples, {full_dataset.num_classes} classes")
    if full_dataset.num_classes > NUM_CLS:
        raise ValueError(
            f"Dataset has {full_dataset.num_classes} labels but --num_classes={NUM_CLS}."
        )

    class_names = full_dataset.sorted_class_names[:NUM_CLS]
    logger.info(f"Dataset: {len(full_dataset)} samples, {NUM_CLS} classes")
    logger.info(f"Classes: {class_names}")

    # ---- Split ----
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    train_idx, val_idx, test_idx = build_split_indices(
        full_dataset, N_TRAIN_USERS, N_VAL_USERS, SEED, logger
    )

    train_set = torch.utils.data.Subset(full_dataset, train_idx)
    val_set   = torch.utils.data.Subset(full_dataset, val_idx)
    test_set  = torch.utils.data.Subset(full_dataset, test_idx)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(val_set,   batch_size=BATCH, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(test_set,  batch_size=BATCH, shuffle=False,
                              num_workers=args.num_workers, pin_memory=pin_memory)
    logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # ---- Model ----
    model = GestureModel(num_classes=NUM_CLS, se_reduction=2).to(device)
    n_params = count_parameters(model)
    logger.info(f"Parameters: {n_params:,} ({n_params/1000:.1f}K)")

    # Verify forward pass
    dummy = torch.randn(2, 6, 100, 4).to(device)
    with torch.no_grad():
        out = model(dummy)
    logger.info(f"Output shape check: {out.shape} (expected [2, {NUM_CLS}])")

    # ---- Optimizer & Scheduler ----
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=LR_STEP, gamma=LR_GAMMA)

    # ---- Training Loop ----
    best_acc = -1.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    logger.info("")
    logger.info(f"{'='*80}")
    logger.info(f"{'Epoch':>6} | {'TrainLoss':>9} | {'TrainAcc':>8} | "
                f"{'ValLoss':>9} | {'ValAcc':>8} | {'LR':>10} | {'Best':>6}")
    logger.info(f"{'='*80}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        t_loss, t_acc = train_one_epoch(model, train_loader, criterion,
                                         optimizer, device, epoch, logger)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)
        lr_now = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        # Record history
        history['train_loss'].append(t_loss)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_loss)
        history['val_acc'].append(v_acc)

        # Track best
        is_best = v_acc > best_acc
        if is_best:
            best_acc = v_acc
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'val_acc': v_acc}, os.path.join(save_dir, 'best_model.pth'))

        # Log every epoch
        best_marker = " *" if is_best else ""
        logger.info(f"{epoch:>6} | {t_loss:>9.4f} | {t_acc:>7.2f}% | "
                     f"{v_loss:>9.4f} | {v_acc:>7.2f}% | {lr_now:>10.1e} | "
                     f"{best_acc:>5.2f}%{best_marker}")

        scheduler.step()

    logger.info(f"{'='*80}")
    logger.info(f"Training complete. Best Val Acc: {best_acc:.2f}%")

    # ---- Load Best Model ----
    logger.info("\nLoading best model for final evaluation...")
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded best model from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.2f}%)")

    # ---- Collect Predictions & Embeddings ----
    logger.info("Collecting predictions and embeddings on test set...")
    y_pred, y_true, embeddings = collect_predictions_and_embeddings(
        model, test_loader, device)

    # ---- Metrics ----
    metrics = compute_and_log_metrics(y_true, y_pred, class_names, logger)

    # ---- Save Metrics CSV ----
    save_metrics_csv(metrics, y_true, y_pred, class_names, save_dir)

    # ---- Confusion Matrices ----
    logger.info("\nGenerating confusion matrices...")
    plot_confusion_matrices(y_true, y_pred, class_names, save_dir)

    # ---- Training Curves ----
    logger.info("Generating training curves...")
    plot_training_curves(history, save_dir)

    # ---- t-SNE ----
    if args.skip_tsne:
        logger.info("Skipping t-SNE visualization.")
    else:
        logger.info("Generating t-SNE visualization...")
        plot_tsne(embeddings, y_true, class_names, save_dir, perplexity=30, seed=SEED)

    # ---- Summary ----
    logger.info("")
    logger.info("=" * 60)
    logger.info("ALL RESULTS SAVED")
    logger.info("=" * 60)
    logger.info(f"  Directory:              {save_dir}/")
    logger.info(f"  Training log:           training_*.log")
    logger.info(f"  Metrics CSV:            evaluation_metrics.csv")
    logger.info(f"  Confusion (raw):        confusion_matrix_raw.png")
    logger.info(f"  Confusion (normalized): confusion_matrix_norm.png")
    logger.info(f"  Training curves:        training_curves.png")
    logger.info(f"  t-SNE:                  tsne_embeddings.png")
    logger.info(f"  Best model:             best_model.pth")
    logger.info(f"  Best Val Accuracy:      {best_acc:.2f}%")
    logger.info(f"  Parameters:             {n_params:,}")


if __name__ == "__main__":
    main()
