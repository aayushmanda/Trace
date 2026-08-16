"""Hyperparameters and global settings for the phase transition experiment."""

import torch

# --- Optimization ---
batch_size = 256
block_size = 64
steps_per_run = 6000
learning_rate = 3e-4
min_learning_rate = 1e-5
warmup_steps = 600
weight_decay = 0.01
dropout = 0.05
label_smoothing = 0.1
max_grad_norm = 1.0
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Baseline Reference for Fixed-Token Sweep ---
REFERENCE_BATCH_SIZE = 256
TOTAL_TOKEN_BUDGET = REFERENCE_BATCH_SIZE * steps_per_run  # 1,536,000 tokens

# --- Model ---
n_embd = 128
n_head = 4
n_layer = 4

# --- Data ---
# Scaled to equal TOTAL_TOKEN_BUDGET so no samples are recycled across runs
DATASET_SIZE = 1536000
DATASET_SEED = 100
VAL_SEED = 999999              
CACHE_DIR = "./dataset_cache"

# --- Experiment sweep ---
RATIOS_TO_TEST = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0]
SEEDS_TO_TEST = [2200, 1337, 2026, 2003, 10]
BATCH_SIZES_TO_TEST = [32, 64, 128, 256, 512]
VAL_WORDS = 1000
PLOT_FILENAME = "phase_transition_low_variance.png"