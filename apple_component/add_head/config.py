#!/usr/bin/env python3
"""Configuration for per-user add-head training."""

from __future__ import annotations

import os


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

# Paths are relative to this apple_component package.
FEATURE_EXTRACTOR_PATH = os.path.join(PROJECT_DIR, "best_model.pth")
DELTA_ENCODER_PATH = os.path.join(
    PROJECT_DIR, "encoder_training", "delta_encoder_best.pth"
)
DELTA_VECTORS_PATH = os.path.join(
    PROJECT_DIR, "encoder_training", "delta_vectors_bank.npy"
)
PRESET_FEATURES_PATH = os.path.join(PROJECT_DIR, "preset_gesture.npz")

# Bundled custom gesture examples copied from main_method/example_data.
CUSTOM_DATA_ROOT = os.path.join(PROJECT_DIR, "example_data", "custom_gestures")
SAVE_ROOT = os.path.join(PROJECT_DIR, "outputs", "add_head_training")

SEED = 42
NUM_SELECT_CLS = 4
SELECTED_CLASSES = [0, 1, 2, 3]
N_SUPPORT = 5
TOTAL_CLASSES = 12

NUM_TOTAL_CLS = NUM_SELECT_CLS + 1
NEG_LABEL = NUM_SELECT_CLS

PRESET_TOTAL = 40800
PRESET_TRAIN_N = 32640
PRESET_VAL_N = 8160

SYNTH_MULT = 9

EMB_DIM = 120
HIDDEN_DIM = 1024

BATCH_SIZE = 128
LR = 1e-3
MAX_EPOCHS = 1
EARLY_STOP = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5
LAMBDA_ADV = 0.2
EPSILON_ADV = 0.2
LOG_EVERY = 5
NUM_WORKERS = 0
