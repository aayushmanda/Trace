# config.py

task_name = "word_index"

rho_values = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
batch_sizes = [32, 64, 128, 256]
model_seeds = [2000, 2100, 2200]

# Data
train_size = 50000
train_seed = 10

val_size = 1000
val_seed = 101
val_batch_size = 256

# Training
steps = 6000

learning_rate = 3e-3
min_learning_rate = 1e-5
warmup_steps = 600
weight_decay = 0.01
max_grad_norm = 1.0

# Model
n_embd = 128
n_head = 4
n_layer = 8
dropout = 0.05

# DataLoader
batch_seed = 12345
num_workers = 4

# Runtime
USE_COMPILE = True
SAVE_MODELS = False
REGENERATE_VAL = False

# W&B
USE_WANDB = True
WANDB_PROJECT = "trace"
WANDB_LOG_EVERY = 20