#!/usr/bin/env python3
"""
Gesture augmentation module (gesture_augmentation.py).

Positive augmentations (3 types -> 7 non-empty combinations):
  - time_scale
  - amp_scale
  - time_warp

Negative augmentations (3 types):
  - truncate
  - invert
  - shuffle

Data expansion per original sample:
  - positives: 1 original + 7 positive combos = 8
  - negatives: 3 negative transforms x (1 base + 7 positive combos) = 24
  - total: 32 records per original sample
"""

import numpy as np
from itertools import combinations
from scipy.signal import resample
from scipy.interpolate import interp1d


# Positive augmentation names and all non-empty combinations
POS_AUG_NAMES = ('time_scale', 'amp_scale', 'time_warp')

POS_AUG_COMBOS = []
for _r in range(1, len(POS_AUG_NAMES) + 1):
    for _combo in combinations(POS_AUG_NAMES, _r):
        POS_AUG_COMBOS.append(_combo)
# 7 combos in total

NEG_AUG_NAMES = ('truncate', 'invert', 'shuffle')


# Positive augmentation functions

def augment_time_scale(x: np.ndarray) -> np.ndarray:
    """
    Time scaling to simulate different gesture speeds.
    scale ~ U(0.9, 1.0): signal is compressed to scale*T, tail is zero-padded.
    x : (6, T, 4) float32
    """
    T = x.shape[1]
    scale = np.random.uniform(0.9, 1.0)
    new_T = max(1, int(round(T * scale)))
    result = np.zeros_like(x)
    for ch in range(x.shape[0]):
        for fb in range(x.shape[2]):
            result[ch, :new_T, fb] = resample(x[ch, :, fb], new_T)
    return result


def augment_amp_scale(x: np.ndarray) -> np.ndarray:
    """
    Amplitude scaling to simulate different gesture strengths.
    s ~ N(1, 0.2²), clipped to [0, 2].
    x : (6, T, 4) float32
    """
    s = float(np.clip(np.random.normal(1.0, 0.2), 0.0, 2.0))
    return (x * s).astype(np.float32)


def augment_time_warp(x: np.ndarray, num_knots: int = 2, sigma: float = 0.05) -> np.ndarray:
    """
    Time warping to simulate temporal variability.
    Uses two interpolation knots, w ~ N(1, 0.05²), clipped to [0, 2].
    x : (6, T, 4) float32
    """
    T = x.shape[1]
    t = np.arange(T, dtype=float)

    # Evenly-spaced knots including endpoints: [0, t1, t2, T-1]
    knot_locs = np.linspace(0.0, T - 1.0, num_knots + 2)

    # Warp factors for the first num_knots interior segments
    w = np.clip(np.random.normal(1.0, sigma, num_knots), 0.0, 2.0)

    # Original segment lengths
    seg_lens = np.diff(knot_locs)           # shape: (num_knots+1,)
    warped_lens = seg_lens.copy()
    for i in range(num_knots):
        warped_lens[i] *= w[i]

    # Normalise so warped total == T-1
    total = warped_lens.sum()
    if total > 0:
        warped_lens = warped_lens / total * (T - 1.0)

    # Warped knot positions in the new (output) time axis
    warped_knots = np.concatenate([[0.0], np.cumsum(warped_lens)])

    # Mapping: new uniform time → original time (for interpolation)
    map_fn = interp1d(warped_knots, knot_locs, kind='linear',
                      bounds_error=False,
                      fill_value=(knot_locs[0], knot_locs[-1]))
    sample_locs = np.clip(map_fn(t), 0.0, T - 1.0)

    result = np.zeros_like(x)
    for ch in range(x.shape[0]):
        for fb in range(x.shape[2]):
            sig_fn = interp1d(t, x[ch, :, fb], kind='linear',
                              bounds_error=False, fill_value='extrapolate')
            result[ch, :, fb] = sig_fn(sample_locs)
    return result.astype(np.float32)


# Negative augmentation functions

def negative_truncate(x: np.ndarray) -> np.ndarray:
    """
    Truncate: randomly zero out a contiguous 0.5s (50-step) window.
    x : (6, T, 4) float32
    """
    T = x.shape[1]
    window = T // 2          # 50 samples @ 100 Hz = 0.5 s
    start = np.random.randint(0, T - window + 1)
    result = x.copy()
    result[:, start:start + window, :] = 0.0
    return result


def negative_invert(x: np.ndarray) -> np.ndarray:
    """
    Invert: flip the signal along the temporal axis.
    x : (6, T, 4) float32
    """
    return x[:, ::-1, :].copy()


def negative_shuffle(x: np.ndarray) -> np.ndarray:
    """
    Shuffle: split into 0.1s (10-step) segments and randomly reorder.
    x : (6, T, 4) float32
    """
    T = x.shape[1]
    seg_len = 10              # 0.1 s @ 100 Hz
    n_segs = T // seg_len
    segs = [x[:, i * seg_len:(i + 1) * seg_len, :] for i in range(n_segs)]
    perm = np.random.permutation(n_segs)
    shuffled = np.concatenate([segs[p] for p in perm], axis=1)
    remainder = x[:, n_segs * seg_len:, :]      # handle T not divisible by 10
    if remainder.shape[1] > 0:
        shuffled = np.concatenate([shuffled, remainder], axis=1)
    return shuffled


# Dispatch dictionaries
POS_AUG_FNS = {
    'time_scale': augment_time_scale,
    'amp_scale':  augment_amp_scale,
    'time_warp':  augment_time_warp,
}

NEG_AUG_FNS = {
    'truncate': negative_truncate,
    'invert':   negative_invert,
    'shuffle':  negative_shuffle,
}


# Core augmentation flow

def _apply_pos_combo(x: np.ndarray, combo: tuple) -> np.ndarray:
    """Apply a sequence of positive augmentations in order."""
    out = x
    for name in combo:
        out = POS_AUG_FNS[name](out)
    return out


def augment_sample(x: np.ndarray, label: int,
                   class_name: str, user_id: str) -> list:
    """
    Apply the full augmentation flow to one raw sample and return records.

    Each record is a dict containing:
      signal      : (6, T, 4) float32
      label       : int
      class_name  : str
      user_id     : str
      is_negative : bool      - True means negative sample
      neg_aug     : str       - 'none' or 'truncate'/'invert'/'shuffle'
      pos_augs    : str       - 'none' or combinations like 'time_scale+amp_scale'
      sample_type : str       - 'original'/'pos_aug'/'neg_base'/'neg_pos_aug'

    Count: 1 + 7 + 3 + 21 = 32 records per raw sample.
    """
    records = []

    def _rec(sig, is_neg, neg_name, combo, stype):
        return {
            'signal':      sig.astype(np.float32),
            'label':       int(label),
            'class_name':  class_name,
            'user_id':     user_id,
            'is_negative': bool(is_neg),
            'neg_aug':     neg_name,
            'pos_augs':    '+'.join(combo) if combo else 'none',
            'sample_type': stype,
        }

    # Step 1: original + 7 positive combinations -> positives (8)
    records.append(_rec(x, False, 'none', (), 'original'))
    for combo in POS_AUG_COMBOS:
        records.append(_rec(_apply_pos_combo(x, combo), False, 'none', combo, 'pos_aug'))

    # Step 2: 3 negative transforms x (1 base + 7 positive combos) -> negatives (24)
    for neg_name in NEG_AUG_NAMES:
        neg_x = NEG_AUG_FNS[neg_name](x)
        records.append(_rec(neg_x, True, neg_name, (), 'neg_base'))
        for combo in POS_AUG_COMBOS:
            records.append(_rec(_apply_pos_combo(neg_x, combo), True, neg_name, combo, 'neg_pos_aug'))

    return records   # 1 + 7 + 3*(1+7) = 32 records


def augment_dataset(samples: list, labels: list,
                    class_names: list, user_ids: list) -> list:
    """
    Apply full augmentation to a batch of raw samples and return all records.

    Parameters
    ----------
    samples     : list of (6, T, 4) float32 arrays
    labels      : list of int
    class_names : list of str
    user_ids    : list of str
    """
    all_records = []
    for x, lbl, cn, uid in zip(samples, labels, class_names, user_ids):
        all_records.extend(augment_sample(x, lbl, cn, uid))
    return all_records
