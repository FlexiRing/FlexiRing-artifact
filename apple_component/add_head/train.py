#!/usr/bin/env python3
"""
Add-head training entry point (per-user).

Each user gets an independent add-head:
  architecture: 120 -> Dense(1024, ReLU) -> Dense(5)
  loss: CE_clean + lambda * CE_adv (lambda=0.2, epsilon=0.2)
  sampling: WeightedRandomSampler for class balance in mini-batches

Examples:
  python add_head/train.py
  python add_head/train.py --data_root /path/to/custom_gestures
  python add_head/train.py --users P01_A P03_B --classes 1 3 6 9 --n_support 5
"""

import os
import sys
import json
import argparse
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import (f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PROJECT_DIR)

from config import (
    SAVE_ROOT, SEED,
    NUM_SELECT_CLS, NUM_TOTAL_CLS, NEG_LABEL,
    EMB_DIM, HIDDEN_DIM,
    BATCH_SIZE, LR, MAX_EPOCHS, EARLY_STOP,
    LR_PATIENCE, LR_FACTOR,
    LAMBDA_ADV, EPSILON_ADV,
    LOG_EVERY, NUM_WORKERS,
    N_SUPPORT, CUSTOM_DATA_ROOT, SELECTED_CLASSES,
)
from pipeline import (
    load_feature_extractor, load_delta_components, load_preset_features,
    discover_users, select_classes, build_user_datasets,
)


RUNTIME_MAX_EPOCHS = MAX_EPOCHS
RUNTIME_NUM_WORKERS = NUM_WORKERS


# ══════════════════════════════════════════════════════════════════
# 1) Model
# ══════════════════════════════════════════════════════════════════

class AddHead(nn.Module):
    """
    Add-head classifier.
    Fixed architecture used in the release:
      Input(120) -> Dense(1024, ReLU) -> Dense(NUM_SELECT_CLS+1)
    """
    def __init__(self, emb_dim: int = EMB_DIM,
                 hidden: int = HIDDEN_DIM,
                 num_cls: int = NUM_TOTAL_CLS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_cls),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════
# 2) Dataset & Sampler
# ══════════════════════════════════════════════════════════════════

class EmbeddingDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        self.feats  = torch.from_numpy(feats).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


def make_weighted_sampler(labels: np.ndarray,
                          num_cls: int = NUM_TOTAL_CLS) -> WeightedRandomSampler:
    class_counts  = np.bincount(labels, minlength=num_cls).astype(float)
    class_counts  = np.maximum(class_counts, 1)
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights     = torch.from_numpy(sample_weights).float(),
        num_samples = len(labels),
        replacement = True,
    )


# ══════════════════════════════════════════════════════════════════
# 3) Evaluation
# ══════════════════════════════════════════════════════════════════

def make_class_names(class_map: dict) -> list:
    """Build display names per output label from class_map."""
    inv = {v: k for k, v in class_map.items()}
    names = [f"Cls_{inv.get(i, '?')}" for i in range(NUM_SELECT_CLS)]
    names.append("Negative")
    return names


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0

    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())

    y_true   = np.concatenate(all_labels)
    y_pred   = np.concatenate(all_preds)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_TOTAL_CLS)), zero_division=0)

    neg_mask = (y_true == NEG_LABEL)
    fpr = float((y_pred[neg_mask] != NEG_LABEL).mean()) if neg_mask.any() else 0.0

    return {
        'loss': total_loss / max(n, 1), 'macro_f1': macro_f1,
        'prec': prec, 'rec': rec, 'f1': f1, 'sup': sup,
        'y_true': y_true, 'y_pred': y_pred, 'fpr': fpr,
    }


def save_final_report(metrics: dict, save_dir: str, class_names: list,
                      class_map: dict, n_support: int):
    y_true, y_pred = metrics['y_true'], metrics['y_pred']

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_TOTAL_CLS)))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (mat, title, fmt) in zip(axes, [
        (cm,  'Confusion Matrix (Raw)',  'd'),
        (cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9),
         'Confusion Matrix (Norm)', '.2f'),
    ]):
        sns.heatmap(mat, annot=True, fmt=fmt, ax=ax,
                    xticklabels=class_names, yticklabels=class_names,
                    cmap='Blues')
        ax.set_title(title); ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # Per-class metrics
    rows = []
    for i, name in enumerate(class_names):
        rows.append({'class': name, 'precision': metrics['prec'][i],
                     'recall': metrics['rec'][i], 'f1': metrics['f1'][i],
                     'support': int(metrics['sup'][i])})
    rows.append({'class': 'macro_avg',
                 'precision': float(np.mean(metrics['prec'])),
                 'recall':    float(np.mean(metrics['rec'])),
                 'f1':        metrics['macro_f1'],
                 'support':   int(sum(metrics['sup']))})
    pd.DataFrame(rows).to_csv(os.path.join(save_dir, "eval_metrics.csv"),
                               index=False, float_format='%.4f')

    # Text report
    rpt_path = os.path.join(save_dir, "eval_report.txt")
    with open(rpt_path, 'w') as f:
        f.write(f"N_SUPPORT: {n_support}\n")
        f.write(f"class_map: {class_map}\n\n")
        f.write(classification_report(
            y_true, y_pred, target_names=class_names, zero_division=0))
        f.write(f"\nMacro-F1         : {metrics['macro_f1']:.4f}\n")
        f.write(f"FPR (neg→pos)    : {metrics['fpr']:.4f}\n")


def save_loss_curve(log_df: pd.DataFrame, save_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
    ax1.plot(log_df['epoch'], log_df['train_loss'], label='Train')
    if 'val_loss' in log_df:
        ax1.plot(log_df['epoch'], log_df['val_loss'], label='Val')
    ax1.set_title('Loss'); ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=0.3)

    if 'val_macro_f1' in log_df:
        ax2.plot(log_df['epoch'], log_df['val_macro_f1'], color='green')
        best_ep = log_df.loc[log_df['val_macro_f1'].idxmax(), 'epoch']
        ax2.axvline(best_ep, color='red', ls='--', lw=1, label=f'Best ep={best_ep}')
    ax2.set_title('Val Macro-F1'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════
# 4) Single-user training
# ══════════════════════════════════════════════════════════════════

def train_one_user(user_name, train_feats, train_labels,
                   val_feats, val_labels,
                   test_feats, test_labels,
                   class_map, save_dir, device, n_support):
    """
    Train the add-head for one user and evaluate on that user's test split.
    """
    os.makedirs(save_dir, exist_ok=True)
    class_names = make_class_names(class_map)

    train_ds = EmbeddingDataset(train_feats, train_labels)
    val_ds   = EmbeddingDataset(val_feats,   val_labels)
    test_ds  = EmbeddingDataset(test_feats,  test_labels)

    sampler      = make_weighted_sampler(train_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=RUNTIME_NUM_WORKERS,
                              pin_memory=device.type == 'cuda')
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=RUNTIME_NUM_WORKERS,
                              pin_memory=device.type == 'cuda')
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=RUNTIME_NUM_WORKERS,
                              pin_memory=device.type == 'cuda')

    # Check whether validation contains any positive classes
    has_pos_val = int((val_labels < NEG_LABEL).sum()) > 0

    model     = AddHead().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR,
                                  patience=LR_PATIENCE, verbose=False)

    best_f1, best_ep, no_improve = -1.0, 0, 0
    ckpt_path = os.path.join(save_dir, "add_head_best.pth")
    history   = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_macro_f1': []}

    for epoch in range(1, RUNTIME_MAX_EPOCHS + 1):
        # Train
        model.train()
        ep_loss, ep_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x.requires_grad_(True)
            logits_clean = model(x)
            loss_clean   = criterion(logits_clean, y)

            grad_x = torch.autograd.grad(loss_clean, x, retain_graph=True)[0]
            x_adv  = (x.detach() + EPSILON_ADV * grad_x.sign()).detach()
            loss_adv = criterion(model(x_adv), y)
            loss = loss_clean + LAMBDA_ADV * loss_adv
            loss.backward()
            optimizer.step()
            ep_loss += loss_clean.item() * len(y)
            ep_n    += len(y)

        train_loss = ep_loss / max(ep_n, 1)

        # Validate
        if has_pos_val:
            val_m    = evaluate(model, val_loader, criterion, device)
            val_loss = val_m['loss']
            macro_f1 = val_m['macro_f1']
            scheduler.step(macro_f1)
        else:
            val_loss = train_loss
            macro_f1 = 0.0

        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_macro_f1'].append(macro_f1)

        is_best = (macro_f1 > best_f1) or (not has_pos_val and epoch == RUNTIME_MAX_EPOCHS)
        if is_best:
            best_f1, best_ep, no_improve = macro_f1, epoch, 0
            torch.save({
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_macro_f1': macro_f1, 'class_map': class_map,
                'n_support': n_support,
            }, ckpt_path)
        else:
            no_improve += 1

        if epoch % LOG_EVERY == 0 or epoch == 1 or epoch == RUNTIME_MAX_EPOCHS:
            lr_now = optimizer.param_groups[0]['lr']
            mark   = " ★" if is_best else ""
            if has_pos_val:
                print(f"    Ep{epoch:3d} | t_loss={train_loss:.5f} "
                      f"v_loss={val_loss:.5f} F1={macro_f1:.4f} "
                      f"FPR={val_m['fpr']:.4f} lr={lr_now:.2e}{mark}")
            else:
                print(f"    Ep{epoch:3d} | t_loss={train_loss:.5f} "
                      f"(no positive validation samples) lr={lr_now:.2e}")

        if has_pos_val and no_improve >= EARLY_STOP:
            print(f"    [Early stop] no improvement for {EARLY_STOP} epochs")
            break

    # Load best checkpoint and evaluate on test set
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    test_m = evaluate(model, test_loader, criterion, device)

    # Save artifacts
    log_df = pd.DataFrame(history)
    log_df.to_csv(os.path.join(save_dir, "training_log.csv"),
                  index=False, float_format='%.8f')
    save_loss_curve(log_df, save_dir)
    save_final_report(test_m, save_dir, class_names, class_map, n_support)

    # Save class mapping
    with open(os.path.join(save_dir, "class_map.json"), 'w') as f:
        json.dump({str(k): int(v) for k, v in class_map.items()}, f, indent=2)

    return {
        'user': user_name, 'best_epoch': best_ep,
        'val_macro_f1': best_f1, 'test_macro_f1': test_m['macro_f1'],
        'test_fpr': test_m['fpr'],
        'per_class_f1': {class_names[i]: float(test_m['f1'][i])
                         for i in range(NUM_TOTAL_CLS)},
    }


# ══════════════════════════════════════════════════════════════════
# 5) Main entry
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Per-user add-head training")
    parser.add_argument("--data_root", type=str, default=None,
                        help="Custom gesture root directory (overrides CUSTOM_DATA_ROOT in config.py)")
    parser.add_argument("--users", nargs='+', default=None,
                        help="Train only the specified users (folder names). Default: all users")
    parser.add_argument("--classes", nargs='+', type=int, default=None,
                        help="Manually select classes, for example: 1 3 6 9")
    parser.add_argument("--n_support", type=int, default=None,
                        help="Number of support samples per class (overrides N_SUPPORT)")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Override MAX_EPOCHS in config.py (useful for smoke tests)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Override NUM_WORKERS in config.py")
    args = parser.parse_args()

    global RUNTIME_MAX_EPOCHS, RUNTIME_NUM_WORKERS
    if args.max_epochs is not None:
        RUNTIME_MAX_EPOCHS = args.max_epochs
    if args.num_workers is not None:
        RUNTIME_NUM_WORKERS = args.num_workers

    t0 = time.time()
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    data_root  = args.data_root or CUSTOM_DATA_ROOT
    n_support  = args.n_support or N_SUPPORT
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(SAVE_ROOT, exist_ok=True)

    print("=" * 65)
    print("  Add-head training (per-user)")
    print("=" * 65)
    print(f"  Device     : {device}")
    print(f"  Data root  : {data_root}")
    print(f"  N_SUPPORT  : {n_support}")
    print(f"  Save root  : {SAVE_ROOT}")

    # Load shared frozen models once
    print("\n[Global] Loading frozen models...")
    feat_model              = load_feature_extractor(device)
    delta_model, delta_bank = load_delta_components(device)
    preset_feats            = load_preset_features()
    print(f"  Feature extractor OK | delta-encoder OK | preset ({preset_feats.shape[0]}) OK")

    # Discover users
    all_users = discover_users(data_root)
    if args.users:
        all_users = [(n, d) for n, d in all_users if n in args.users]
    print(f"\n[Users] total: {len(all_users)}")

    # Choose class subset (global fixed or per-user random)
    rng = np.random.default_rng(SEED)
    if args.classes:
        global_classes = sorted(args.classes)
        print(f"  Manually selected classes: {global_classes}")
    elif SELECTED_CLASSES is not None:
        global_classes = sorted(SELECTED_CLASSES)
        print(f"  Config-selected classes: {global_classes}")
    else:
        global_classes = None

    # Train per user
    all_results = []
    for idx, (user_name, user_dir) in enumerate(all_users):
        print(f"\n{'─'*65}")
        print(f"  [{idx+1}/{len(all_users)}] User: {user_name}")
        print(f"{'─'*65}")

        # Allow per-user random class sampling if no global class list is provided
        if global_classes is not None:
            sel_classes = global_classes
        else:
            user_rng    = np.random.default_rng(SEED + idx)
            sel_classes = select_classes(user_rng)

        # Build datasets
        (train_f, train_l, val_f, val_l,
         test_f, test_l, cmap) = build_user_datasets(
            user_name, user_dir, sel_classes,
            feat_model, delta_model, delta_bank,
            preset_feats, device, n_support)

        # Train
        user_save = os.path.join(SAVE_ROOT, user_name)
        res = train_one_user(
            user_name, train_f, train_l, val_f, val_l,
            test_f, test_l, cmap, user_save, device, n_support)
        all_results.append(res)

        print(f"  → test Macro-F1={res['test_macro_f1']:.4f}  "
              f"FPR={res['test_fpr']:.4f}")

    # Summary
    print(f"\n{'═'*65}")
    print("  Summary")
    print(f"{'═'*65}")
    summary_rows = []
    for r in all_results:
        print(f"  {r['user']:15s}  test_F1={r['test_macro_f1']:.4f}  "
              f"FPR={r['test_fpr']:.4f}  best_ep={r['best_epoch']}")
        summary_rows.append(r)

    avg_f1  = np.mean([r['test_macro_f1'] for r in all_results])
    avg_fpr = np.mean([r['test_fpr']      for r in all_results])
    print(f"\n  Mean test Macro-F1 : {avg_f1:.4f}")
    print(f"  Mean test FPR      : {avg_fpr:.4f}")
    print(f"  Total time         : {(time.time()-t0)/60:.1f} min")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(SAVE_ROOT, "summary.csv")
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    print(f"  Summary file: {summary_path}")

    # Save summary JSON (including per-class F1)
    with open(os.path.join(SAVE_ROOT, "summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
