#!/usr/bin/env python3
"""
Add-head data pipeline (per-user).

For each user:
  1) Load raw IMU signals from user_dir/{cls}/{support|test}/*.csv
  2) Select NUM_SELECT_CLS classes and sample N_SUPPORT support examples
  3) Split support seeds into train/val (4:1) before augmentation
  4) Apply signal-level augmentations and extract frozen 120-d embeddings
  5) Synthesize additional embeddings via delta-encoder
  6) Merge preset gesture embeddings as negative class
  7) Use test/ samples as independent test set
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import torch

# Module imports
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "encoder_training"))

from ModelTrain import GestureModel, preprocess_sample
from model import DeltaEncoderModel
from gesture_augmentation import (
    POS_AUG_COMBOS, POS_AUG_FNS,
    NEG_AUG_FNS, NEG_AUG_NAMES,
)
from config import (
    FEATURE_EXTRACTOR_PATH, DELTA_ENCODER_PATH, DELTA_VECTORS_PATH,
    PRESET_FEATURES_PATH, CUSTOM_DATA_ROOT,
    NUM_SELECT_CLS, SELECTED_CLASSES, N_SUPPORT, TOTAL_CLASSES,
    NEG_LABEL, SEED, SYNTH_MULT,
    PRESET_TRAIN_N, PRESET_VAL_N,
)

DELTA_EMB_DIM   = 120
DELTA_DELTA_DIM = 5
DELTA_HIDDEN    = 4096
DELTA_ALPHA     = 0.3
FEAT_BATCH      = 256


# ══════════════════════════════════════════════════════════════════
# 1) Load frozen models
# ══════════════════════════════════════════════════════════════════

def load_feature_extractor(device: torch.device) -> GestureModel:
    """Load frozen feature extractor."""
    model = GestureModel(num_classes=12, se_reduction=2).to(device)
    ckpt  = torch.load(FEATURE_EXTRACTOR_PATH, map_location=device,
                        weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_delta_components(device: torch.device):
    """Load frozen delta-encoder and delta-vector bank."""
    delta_model = DeltaEncoderModel(DELTA_EMB_DIM, DELTA_DELTA_DIM,
                                    DELTA_HIDDEN, DELTA_ALPHA).to(device)
    ckpt = torch.load(DELTA_ENCODER_PATH, map_location=device,
                       weights_only=False)
    delta_model.load_state_dict(ckpt['model_state_dict'])
    delta_model.eval()
    for p in delta_model.parameters():
        p.requires_grad_(False)
    delta_bank = np.load(DELTA_VECTORS_PATH).astype(np.float32)
    return delta_model, delta_bank


def load_preset_features():
    """Load preset gesture embeddings (40800, 120)."""
    preset = np.load(PRESET_FEATURES_PATH, allow_pickle=True)
    return preset['features'].astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# 2) Signal-level utilities
# ══════════════════════════════════════════════════════════════════

def extract_embeddings(signals: list, feat_model: GestureModel,
                       device: torch.device) -> np.ndarray:
    """signals: list of (6,100,4) → (N, 120)"""
    if len(signals) == 0:
        return np.empty((0, 120), dtype=np.float32)
    all_feats = []
    with torch.no_grad():
        for i in range(0, len(signals), FEAT_BATCH):
            batch = torch.from_numpy(
                np.stack(signals[i:i + FEAT_BATCH])).to(device)
            feats = feat_model.get_embedding(batch)
            all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0).astype(np.float32)


def synthesize(embeddings: np.ndarray, delta_model, delta_bank,
               device: torch.device, n_synth: int = SYNTH_MULT,
               rng=None) -> np.ndarray:
    """Synthesize embeddings via delta-encoder, repeated (n_synth+1)x including original."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    parts = [embeddings]
    M     = len(embeddings)
    if M == 0:
        return embeddings
    with torch.no_grad():
        for _ in range(n_synth):
            idx    = rng.choice(len(delta_bank), M, replace=True)
            deltas = torch.from_numpy(delta_bank[idx]).to(device)
            refs   = torch.from_numpy(embeddings).to(device)
            synth  = delta_model.decode(deltas, refs)
            parts.append(synth.cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def _apply_pos_combo(sig, combo):
    out = sig
    for name in combo:
        out = POS_AUG_FNS[name](out)
    return out


def augment_pos(seeds: list, feat_model, device) -> np.ndarray:
    """Apply positive augmentations and extract embeddings. seeds: list of (6,100,4)."""
    augmented = []
    for sig in seeds:
        augmented.append(sig)
        for combo in POS_AUG_COMBOS:
            augmented.append(_apply_pos_combo(sig, combo))
    return extract_embeddings(augmented, feat_model, device)


def augment_neg(seeds: list, feat_model, device) -> np.ndarray:
    """Apply negative augmentations (+positive combos) and extract embeddings."""
    augmented = []
    for sig in seeds:
        for neg_name in NEG_AUG_NAMES:
            neg_sig = NEG_AUG_FNS[neg_name](sig)
            augmented.append(neg_sig)
            for combo in POS_AUG_COMBOS:
                augmented.append(_apply_pos_combo(neg_sig, combo))
    return extract_embeddings(augmented, feat_model, device)


# ══════════════════════════════════════════════════════════════════
# 3) User discovery and data loading
# ══════════════════════════════════════════════════════════════════

def discover_users(root_dir: str = None) -> list:
    """
    Discover user directories.
    Returns [(user_name, user_dir_path), ...]
    """
    if root_dir is None:
        root_dir = CUSTOM_DATA_ROOT
    users = []
    for name in sorted(os.listdir(root_dir)):
        d = os.path.join(root_dir, name)
        if os.path.isdir(d):
            users.append((name, d))
    return users


def select_classes(rng, total: int = TOTAL_CLASSES,
                   n: int = NUM_SELECT_CLS) -> list:
    """Randomly select n classes and return a sorted list."""
    if SELECTED_CLASSES is not None:
        return sorted(SELECTED_CLASSES)
    return sorted(rng.choice(total, n, replace=False).tolist())


def load_signals_from_dir(directory: str) -> list:
    """Read all CSV files in a directory and return a list of (6,100,4) signals."""
    csvs = sorted(glob.glob(os.path.join(directory, "*.csv")))
    signals = []
    for fp in csvs:
        df  = pd.read_csv(fp, header=0)
        sig = preprocess_sample(df.values.astype(np.float32))
        signals.append(sig)
    return signals


# ══════════════════════════════════════════════════════════════════
# 4) Build datasets for one user
# ══════════════════════════════════════════════════════════════════

def build_user_datasets(user_name: str, user_dir: str,
                        selected_classes: list,
                        feat_model, delta_model, delta_bank,
                        preset_feats: np.ndarray,
                        device: torch.device,
                        n_support: int = N_SUPPORT):
    """
    Build train/val/test datasets for one user.

    Returns:
      train_feats, train_labels, val_feats, val_labels,
      test_feats, test_labels, class_map
    """
    rng = np.random.default_rng(SEED)
    num_cls = len(selected_classes)

    # class_map: original class id -> model label id (0..num_cls-1)
    class_map = {orig: idx for idx, orig in enumerate(selected_classes)}
    print(f"\n  [{user_name}] selected classes: {selected_classes}  map: {class_map}")

    # 1) Load support and test signals
    train_pos_feats_all, train_pos_labels_all = [], []
    train_neg_feats_all, train_neg_labels_all = [], []
    val_feats_all,       val_labels_all       = [], []
    test_feats_all,      test_labels_all      = [], []

    for orig_cls in selected_classes:
        mapped_label = class_map[orig_cls]
        support_dir  = os.path.join(user_dir, str(orig_cls), "support")
        test_dir     = os.path.join(user_dir, str(orig_cls), "test")

        # Load support signals
        support_signals = load_signals_from_dir(support_dir)
        n_available     = len(support_signals)
        n_use           = min(n_support, n_available)
        assert n_available > 0, f"class {orig_cls} has empty support set: {support_dir}"

        # Randomly choose n_use samples
        chosen_idx = rng.choice(n_available, n_use, replace=False)
        chosen     = [support_signals[i] for i in chosen_idx]

        # Split seeds into train/val using 4:1 ratio
        val_n   = max(0, n_use // 5)
        train_n = n_use - val_n

        perm = rng.permutation(n_use)
        train_sigs = [chosen[i] for i in perm[:train_n]]
        val_sigs   = [chosen[i] for i in perm[train_n:train_n + val_n]]

        # Train seeds: positive augmentations -> embeddings -> delta synthesis
        if len(train_sigs) > 0:
            pos_embs = augment_pos(train_sigs, feat_model, device)
            pos_embs = synthesize(pos_embs, delta_model, delta_bank, device,
                                  rng=rng)
            train_pos_feats_all.append(pos_embs)
            train_pos_labels_all.append(
                np.full(len(pos_embs), mapped_label, dtype=np.int64))

            # Train seeds: negative augmentations -> embeddings -> delta synthesis
            neg_embs = augment_neg(train_sigs, feat_model, device)
            neg_embs = synthesize(neg_embs, delta_model, delta_bank, device,
                                  rng=rng)
            train_neg_feats_all.append(neg_embs)
            train_neg_labels_all.append(
                np.full(len(neg_embs), NEG_LABEL, dtype=np.int64))

        # Validation seeds: no augmentation, embeddings only
        if len(val_sigs) > 0:
            v_embs = extract_embeddings(val_sigs, feat_model, device)
            val_feats_all.append(v_embs)
            val_labels_all.append(
                np.full(len(v_embs), mapped_label, dtype=np.int64))

        # Test set: load all files under test/
        test_signals = load_signals_from_dir(test_dir)
        if len(test_signals) > 0:
            t_embs = extract_embeddings(test_signals, feat_model, device)
            test_feats_all.append(t_embs)
            test_labels_all.append(
                np.full(len(t_embs), mapped_label, dtype=np.int64))

        print(f"    class {orig_cls} -> label {mapped_label}: "
              f"support={n_available}(use={n_use}, train={train_n}/val={val_n})  "
              f"test={len(test_signals)}")

    # 2) Preset negatives
    preset_idx       = rng.permutation(len(preset_feats))
    train_preset     = preset_feats[preset_idx[:PRESET_TRAIN_N]]
    val_preset       = preset_feats[preset_idx[PRESET_TRAIN_N:
                                               PRESET_TRAIN_N + PRESET_VAL_N]]

    # 3) Merge training set
    parts_f = train_pos_feats_all + train_neg_feats_all + [train_preset]
    parts_l = (train_pos_labels_all + train_neg_labels_all
               + [np.full(len(train_preset), NEG_LABEL, dtype=np.int64)])
    train_feats  = np.concatenate(parts_f, axis=0)
    train_labels = np.concatenate(parts_l, axis=0)

    # 4) Merge validation set
    val_feats_all.append(val_preset)
    val_labels_all.append(np.full(len(val_preset), NEG_LABEL, dtype=np.int64))
    val_feats  = np.concatenate(val_feats_all, axis=0)
    val_labels = np.concatenate(val_labels_all, axis=0)

    # 5) Merge test set (positives + full preset negatives)
    test_feats_all.append(preset_feats)   # all 40800 used as test negatives
    test_labels_all.append(
        np.full(len(preset_feats), NEG_LABEL, dtype=np.int64))
    test_feats  = np.concatenate(test_feats_all, axis=0)
    test_labels = np.concatenate(test_labels_all, axis=0)

    # 6) Stats
    print(f"    train: {len(train_feats)}  (positive={int((train_labels < NEG_LABEL).sum())}, "
          f"negative={int((train_labels == NEG_LABEL).sum())})")
    print(f"    val: {len(val_feats)}  test: {len(test_feats)}")

    return (train_feats, train_labels.astype(np.int64),
            val_feats,   val_labels.astype(np.int64),
            test_feats,  test_labels.astype(np.int64),
            class_map)
