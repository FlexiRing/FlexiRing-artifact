from __future__ import annotations

import json
import os
import random
import sys
import time
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


APPLE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APPLE_DIR.parent
INTER_DIR = APPLE_DIR
sys.path.insert(0, str(INTER_DIR / 'add_head'))
sys.path.insert(0, str(INTER_DIR))

from add_head.config import (
    EMB_DIM, HIDDEN_DIM,
    BATCH_SIZE, LR, MAX_EPOCHS, EARLY_STOP, LR_PATIENCE, LR_FACTOR,
    LAMBDA_ADV, EPSILON_ADV, LOG_EVERY, NUM_WORKERS,
)
from add_head.pipeline import (
    extract_embeddings, synthesize, augment_pos, augment_neg,
)
from model import DeltaEncoderModel


def load_delta_components(device: torch.device):
    delta_model = DeltaEncoderModel(120, 5, 4096, 0.3).to(device)
    ckpt = torch.load(CONFIG['delta_encoder_path'], map_location=device, weights_only=False)
    delta_model.load_state_dict(ckpt['model_state_dict'])
    delta_model.eval()
    for p in delta_model.parameters():
        p.requires_grad_(False)
    delta_bank = np.load(CONFIG['delta_vectors_path']).astype(np.float32)
    return delta_model, delta_bank


def load_feature_extractor(device: torch.device):
    from ModelTrain import GestureModel
    model = GestureModel(num_classes=12, se_reduction=2).to(device)
    ckpt = torch.load(CONFIG['feature_extractor_path'], map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_preset_features():
    preset = np.load(CONFIG['preset_features_path'], allow_pickle=True)
    return preset['features'].astype(np.float32)


CONFIG = {
    'custom_data_root': APPLE_DIR / 'example_data' / 'custom_gestures',
    'manifest_path': PROJECT_ROOT / 'artifacts' / 'evaluation' / 'baseline' / 'split_manifest.json',
    'feature_extractor_path': APPLE_DIR / 'best_model.pth',
    'delta_encoder_path': APPLE_DIR / 'encoder_training' / 'delta_encoder_best.pth',
    'delta_vectors_path': APPLE_DIR / 'encoder_training' / 'delta_vectors_bank.npy',
    'preset_features_path': APPLE_DIR / 'preset_gesture.npz',
    'shots': [1],
    'artifact_root': APPLE_DIR / 'outputs' / 'fixed_manifest_smoke',
    'seed': 20260417,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'max_epochs': 1,
    'preset_limit': 64,
}


class AddHeadKeepOld(nn.Module):
    def __init__(self, num_new_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN_DIM, 12 + num_new_classes + 1),
        )

    def forward(self, x):
        return self.net(x)


class AddHeadNewOnly(nn.Module):
    def __init__(self, num_new_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMB_DIM, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(HIDDEN_DIM, num_new_classes + 1),
        )

    def forward(self, x):
        return self.net(x)


class EmbeddingDataset(Dataset):
    def __init__(self, feats: np.ndarray, labels: np.ndarray):
        self.feats = torch.from_numpy(feats).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.feats[idx], self.labels[idx]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def append_jsonl(path: Path, records: list[dict]):
    if not records:
        return
    with path.open('a', encoding='utf-8') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def load_manifest(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def build_example_manifest(custom_root: Path, shots: list[int]):
    """Build a one-round manifest from bundled example custom gesture data."""
    subjects = [p.name for p in discover_users(custom_root)]
    if not subjects:
        raise FileNotFoundError(f'No example users found in {custom_root}')
    first_user = custom_root / subjects[0]
    classes = sorted(int(p.name) for p in first_user.iterdir() if p.is_dir() and p.name.isdigit())
    if not classes:
        raise FileNotFoundError(f'No class folders found in {first_user}')

    round_subjects = {}
    for user in subjects:
        user_dir = custom_root / user
        support_by_shot = {str(shot): [] for shot in shots}
        query_paths = []
        for cls in classes:
            support_files = sorted((user_dir / str(cls) / 'support').glob('*.csv'))
            test_files = sorted((user_dir / str(cls) / 'test').glob('*.csv'))
            if not support_files or not test_files:
                raise FileNotFoundError(f'Missing support/test CSVs for {user} class {cls}')
            for shot in shots:
                support_by_shot[str(shot)].extend(
                    str(path.relative_to(custom_root)) for path in support_files[:shot]
                )
            query_paths.extend(str(path.relative_to(custom_root)) for path in test_files)
        round_subjects[user] = {
            'support_paths_by_shot': support_by_shot,
            'query_paths': query_paths,
        }

    return {
        'subjects': subjects,
        'rounds': [
            {
                'round_id': 0,
                'classes': classes,
                'subjects': round_subjects,
            }
        ],
    }


def discover_users(root_dir: Path):
    return sorted([p for p in root_dir.iterdir() if p.is_dir()])


def load_signals_from_paths(paths: list[str]) -> list:
    from ModelTrain import preprocess_sample
    signals = []
    for fp in paths:
        df = pd.read_csv(fp, header=0)
        sig = preprocess_sample(df.values.astype(np.float32))
        signals.append(sig)
    return signals


def make_weighted_sampler(labels: np.ndarray, num_cls: int):
    class_counts = np.bincount(labels, minlength=num_cls).astype(float)
    class_counts = np.maximum(class_counts, 1)
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(torch.from_numpy(sample_weights).float(), num_samples=len(labels), replacement=True)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        n += len(y)
        all_preds.append(logits.argmax(1).cpu().numpy())
        all_labels.append(y.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=sorted(set(y_true.tolist())), zero_division=0)
    return {
        'loss': total_loss / max(n, 1),
        'macro_f1': macro_f1,
        'prec': prec,
        'rec': rec,
        'f1': f1,
        'sup': sup,
        'y_true': y_true,
        'y_pred': y_pred,
    }


def save_confusion(y_true, y_pred, labels, names, output_path: Path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (mat, title, fmt) in zip(axes, [
        (cm, 'Confusion Matrix (Raw)', 'd'),
        (cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9), 'Confusion Matrix (Norm)', '.2f'),
    ]):
        sns.heatmap(mat, annot=True, fmt=fmt, ax=ax, xticklabels=names, yticklabels=names, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def build_class_map(round_classes: list[int], mode: str):
    if mode == 'keep_old_head':
        return {orig: 12 + idx for idx, orig in enumerate(round_classes)}
    if mode == 'new_only_head':
        return {orig: idx for idx, orig in enumerate(round_classes)}
    raise ValueError(mode)


def train_one_run(mode: str, user_name: str, round_id: int, shot: int, support_paths: list[str], query_paths: list[str], round_classes: list[int], feat_model, delta_model, delta_bank, preset_feats, device, save_dir: Path):
    rng = np.random.default_rng(CONFIG['seed'] + round_id)
    class_map = build_class_map(round_classes, mode)

    train_pos_feats_all, train_pos_labels_all = [], []
    train_neg_feats_all, train_neg_labels_all = [], []
    val_feats_all, val_labels_all = [], []
    test_feats_all, test_labels_all = [], []
    detailed_records = []

    support_by_class = defaultdict(list)
    for path in support_paths:
        cls = int(Path(path).parts[-3])
        support_by_class[cls].append(path)

    query_by_class = defaultdict(list)
    for path in query_paths:
        cls = int(Path(path).parts[-3])
        query_by_class[cls].append(path)

    for orig_cls in round_classes:
        mapped_label = class_map[orig_cls]
        support_signals = load_signals_from_paths(support_by_class[orig_cls])
        n_use = min(shot, len(support_signals))
        chosen_idx = rng.choice(len(support_signals), n_use, replace=False)
        chosen = [support_signals[i] for i in chosen_idx]
        val_n = max(0, n_use // 5)
        train_n = n_use - val_n
        perm = rng.permutation(n_use)
        train_sigs = [chosen[i] for i in perm[:train_n]]
        val_sigs = [chosen[i] for i in perm[train_n:train_n + val_n]]

        if len(train_sigs) > 0:
            pos_embs = augment_pos(train_sigs, feat_model, device)
            pos_embs = synthesize(pos_embs, delta_model, delta_bank, device, rng=rng)
            train_pos_feats_all.append(pos_embs)
            train_pos_labels_all.append(np.full(len(pos_embs), mapped_label, dtype=np.int64))

            neg_embs = augment_neg(train_sigs, feat_model, device)
            neg_embs = synthesize(neg_embs, delta_model, delta_bank, device, rng=rng)
            neg_label = 12 + len(round_classes) if mode == 'keep_old_head' else len(round_classes)
            train_neg_feats_all.append(neg_embs)
            train_neg_labels_all.append(np.full(len(neg_embs), neg_label, dtype=np.int64))

        if len(val_sigs) > 0:
            v_embs = extract_embeddings(val_sigs, feat_model, device)
            val_feats_all.append(v_embs)
            val_labels_all.append(np.full(len(v_embs), mapped_label, dtype=np.int64))

        test_signals = load_signals_from_paths(query_by_class[orig_cls])
        if len(test_signals) > 0:
            t_embs = extract_embeddings(test_signals, feat_model, device)
            test_feats_all.append(t_embs)
            test_labels_all.append(np.full(len(t_embs), mapped_label, dtype=np.int64))

    neg_label = 12 + len(round_classes) if mode == 'keep_old_head' else len(round_classes)
    preset_idx = rng.permutation(len(preset_feats))
    train_preset_n = min(32640, max(1, len(preset_feats) * 4 // 5))
    val_preset_n = min(8160, max(1, len(preset_feats) - train_preset_n))
    train_preset = preset_feats[preset_idx[:train_preset_n]]
    val_preset = preset_feats[preset_idx[train_preset_n:train_preset_n + val_preset_n]]

    train_feats = np.concatenate(train_pos_feats_all + train_neg_feats_all + ([train_preset] if mode == 'new_only_head' else []), axis=0)
    train_labels = np.concatenate(train_pos_labels_all + train_neg_labels_all + ([np.full(len(train_preset), neg_label, dtype=np.int64)] if mode == 'new_only_head' else []), axis=0)
    val_feats = np.concatenate(val_feats_all + [val_preset], axis=0)
    val_labels = np.concatenate(val_labels_all + [np.full(len(val_preset), neg_label, dtype=np.int64)], axis=0)
    test_feats = np.concatenate(test_feats_all + ([preset_feats] if mode == 'new_only_head' else []), axis=0)
    test_labels = np.concatenate(test_labels_all + ([np.full(len(preset_feats), neg_label, dtype=np.int64)] if mode == 'new_only_head' else []), axis=0)

    num_cls = (12 + len(round_classes) + 1) if mode == 'keep_old_head' else (len(round_classes) + 1)
    model = AddHeadKeepOld(len(round_classes)).to(device) if mode == 'keep_old_head' else AddHeadNewOnly(len(round_classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=LR_FACTOR, patience=LR_PATIENCE)

    train_ds = EmbeddingDataset(train_feats, train_labels)
    val_ds = EmbeddingDataset(val_feats, val_labels)
    test_ds = EmbeddingDataset(test_feats, test_labels)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=make_weighted_sampler(train_labels, num_cls), num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_f1, best_ep, no_improve = -1.0, 0, 0
    ckpt_path = save_dir / 'best.pth'
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_macro_f1': []}

    max_epochs = int(CONFIG.get('max_epochs', MAX_EPOCHS))
    for epoch in range(1, max_epochs + 1):
        model.train()
        ep_loss, ep_n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            x.requires_grad_(True)
            logits_clean = model(x)
            loss_clean = criterion(logits_clean, y)
            grad_x = torch.autograd.grad(loss_clean, x, retain_graph=True)[0]
            x_adv = (x.detach() + EPSILON_ADV * grad_x.sign()).detach()
            loss_adv = criterion(model(x_adv), y)
            loss = loss_clean + LAMBDA_ADV * loss_adv
            loss.backward()
            optimizer.step()
            ep_loss += loss_clean.item() * len(y)
            ep_n += len(y)
        train_loss = ep_loss / max(ep_n, 1)
        val_m = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_m['macro_f1'])
        history['epoch'].append(epoch)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_m['loss'])
        history['val_macro_f1'].append(val_m['macro_f1'])
        if val_m['macro_f1'] > best_f1:
            best_f1, best_ep, no_improve = val_m['macro_f1'], epoch, 0
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, ckpt_path)
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP:
            break

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_m = evaluate(model, test_loader, criterion, device)

    # detailed records
    for idx, (t, p) in enumerate(zip(test_m['y_true'].tolist(), test_m['y_pred'].tolist())):
        detailed_records.append({
            'user': user_name,
            'round': round_id,
            'shot': shot,
            'mode': mode,
            'true_label': int(t),
            'pred_label': int(p),
            'correct': bool(int(t) == int(p)),
            'class_map': {str(k): int(v) for k, v in class_map.items()},
        })

    label_names = []
    if mode == 'keep_old_head':
        label_names = [f'Old_{i}' for i in range(12)] + [f'New_{c}' for c in round_classes] + ['Negative']
    else:
        label_names = [f'New_{c}' for c in round_classes] + ['Negative']
    save_confusion(test_m['y_true'], test_m['y_pred'], list(range(num_cls)), label_names, save_dir / 'confusion_matrix.png')
    pd.DataFrame(history).to_csv(save_dir / 'training_log.csv', index=False)

    return {
        'best_epoch': best_ep,
        'val_macro_f1': best_f1,
        'test_macro_f1': float(test_m['macro_f1']),
        'detailed_records': detailed_records,
    }


def main():
    parser = argparse.ArgumentParser(description='Run manifest-based Apple-component baselines.')
    parser.add_argument('--custom-data-root', type=Path, default=CONFIG['custom_data_root'])
    parser.add_argument('--manifest-path', type=Path, default=None)
    parser.add_argument('--artifact-root', type=Path, default=CONFIG['artifact_root'])
    parser.add_argument('--shots', nargs='+', type=int, default=CONFIG['shots'])
    parser.add_argument('--max-epochs', type=int, default=CONFIG['max_epochs'])
    parser.add_argument('--preset-limit', type=int, default=CONFIG['preset_limit'])
    parser.add_argument('--device', type=str, default=CONFIG['device'])
    args = parser.parse_args()

    CONFIG['custom_data_root'] = args.custom_data_root
    CONFIG['artifact_root'] = args.artifact_root
    CONFIG['shots'] = args.shots
    CONFIG['max_epochs'] = args.max_epochs
    CONFIG['preset_limit'] = args.preset_limit
    CONFIG['device'] = args.device
    if args.manifest_path is not None:
        CONFIG['manifest_path'] = args.manifest_path

    out = ensure_dir(CONFIG['artifact_root'])
    print("=" * 72)
    print("Running Apple-component fixed-manifest baselines")
    print("=" * 72)
    print(f"Output root      : {out}")
    print(f"Custom data root : {CONFIG['custom_data_root']}")
    print(f"Shots            : {CONFIG['shots']}")
    print(f"Max epochs       : {CONFIG['max_epochs']}")
    print(f"Preset limit     : {CONFIG['preset_limit']}")
    print(f"Device           : {CONFIG['device']}")
    if args.manifest_path is not None:
        manifest = load_manifest(CONFIG['manifest_path'])
        print(f"Manifest         : {CONFIG['manifest_path']}")
    elif CONFIG['manifest_path'].exists():
        manifest = load_manifest(CONFIG['manifest_path'])
        print(f"Manifest         : {CONFIG['manifest_path']}")
    else:
        manifest = build_example_manifest(CONFIG['custom_data_root'], CONFIG['shots'])
        print("Manifest         : built from bundled example data")
    subjects = discover_users(CONFIG['custom_data_root'])
    print(f"Subjects         : {[p.name for p in subjects]}")
    feat_model = load_feature_extractor(torch.device(CONFIG['device']))
    delta_model, delta_bank = load_delta_components(torch.device(CONFIG['device']))
    preset_feats = load_preset_features()
    if CONFIG.get('preset_limit') is not None:
        preset_feats = preset_feats[:int(CONFIG['preset_limit'])]

    save_json(out / 'config.json', json_safe({**CONFIG, 'custom_data_root': str(CONFIG['custom_data_root']), 'manifest_path': str(CONFIG['manifest_path'])}))

    for mode in ['keep_old_head', 'new_only_head']:
        print("-" * 72)
        print(f"Mode: {mode}")
        mode_dir = ensure_dir(out / mode)
        results_summary = {}
        subjectwise = {}
        roundwise = {}
        detailed_path = mode_dir / 'detailed_predictions.jsonl'
        if detailed_path.exists():
            detailed_path.unlink()

        for shot in CONFIG['shots']:
            shot_key = f'{shot}-shot'
            print(f"  Shot setting: {shot_key}")
            subjectwise[shot_key] = {}
            roundwise[shot_key] = {}
            round_scores = []
            for round_entry in manifest['rounds']:
                round_key = f"round_{round_entry['round_id']:02d}"
                round_subject_scores = {}
                for user_path in subjects:
                    user_name = user_path.name
                    support_paths = [str(CONFIG['custom_data_root'] / p.replace('\\', '/')) for p in round_entry['subjects'][user_name]['support_paths_by_shot'][str(shot)]]
                    query_paths = [str(CONFIG['custom_data_root'] / p.replace('\\', '/')) for p in round_entry['subjects'][user_name]['query_paths']]
                    save_dir = ensure_dir(mode_dir / shot_key / round_key / user_name)
                    result = train_one_run(mode, user_name, round_entry['round_id'], shot, support_paths, query_paths, round_entry['classes'], feat_model, delta_model, delta_bank, preset_feats, torch.device(CONFIG['device']), save_dir)
                    round_subject_scores[user_name] = {'test_macro_f1': result['test_macro_f1'], 'best_epoch': result['best_epoch']}
                    print(f"    {round_key} {user_name}: macro_f1={result['test_macro_f1']:.4f}, best_epoch={result['best_epoch']}")
                    append_jsonl(detailed_path, result['detailed_records'])
                round_mean = float(np.mean([v['test_macro_f1'] for v in round_subject_scores.values()]))
                round_scores.append(round_mean)
                roundwise[shot_key][round_key] = {'classes': round_entry['classes'], 'mean': round_mean, 'subject_scores': round_subject_scores}
            subjectwise[shot_key] = {u: float(np.mean([roundwise[shot_key][rk]['subject_scores'][u]['test_macro_f1'] for rk in roundwise[shot_key]])) for u in manifest['subjects']}
            results_summary[shot_key] = {'mean': float(np.mean(round_scores)), 'std_round': float(np.std(round_scores)), 'subjects': len(subjects), 'rounds': len(round_scores)}

        save_json(mode_dir / 'results_summary.json', results_summary)
        save_json(mode_dir / 'subjectwise_results.json', subjectwise)
        save_json(mode_dir / 'roundwise_results.json', roundwise)
        (mode_dir / 'summary.txt').write_text('\n'.join([f"- {k}: mean={v['mean']:.6f}, std_round={v['std_round']:.6f}" for k, v in results_summary.items()]) + '\n', encoding='utf-8')
        for shot_key, row in results_summary.items():
            print(f"  Summary {mode} {shot_key}: mean={row['mean']:.4f}, std_round={row['std_round']:.4f}")

    print("=" * 72)
    print(f"Done. Results saved to: {out}")
    print("=" * 72)


if __name__ == '__main__':
    main()
