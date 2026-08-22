# config.py

task_name = "word_index"

rho_values = [0.0,  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# [
#     0.0,
#     0.04,
#     0.08,
#     0.12,
#     0.15,
#     0.20,
#     0.25,
#     0.30,
#     0.35, 
#     0.4, 
#     0.45,


# ] 
batch_sizes = [128]
model_seeds = [2001, 2002, 2003]

# Data
train_size = 500000
train_seed = 501

val_size = 1000
val_seed = 101
val_batch_size = 128

# Training
steps = 8000

learning_rate = 3e-4
min_learning_rate = 3e-4
warmup_steps = 0
weight_decay = 0.0
max_grad_norm = 1.0

# Model
n_embd = 128
n_head = 4
n_layer = 8
dropout = 0.0

# DataLoader
batch_seed = 12345
num_workers = 4

# Runtime
USE_COMPILE = True
SAVE_MODELS = False
REGENERATE_VAL = False

# W&B
USE_WANDB = False
WANDB_PROJECT = "trace_batch_128"
WANDB_LOG_EVERY = 20