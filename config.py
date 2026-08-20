# config.py

task_name = "word_index"

rho_values = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
model_seeds = [2000, 2100, 2200]

batch_size = 128
steps = 6000

learning_rate = 3e-3
min_learning_rate = 1e-5
warmup_steps = 600
weight_decay = 0.01
max_grad_norm = 1.0

n_embd = 128
n_head = 4
n_layer = 8
dropout = 0.05

val_size = 1000
val_batch_size = 256

batch_seed = 12345
val_seed = 101

num_workers = 4

USE_COMPILE = True
SAVE_MODELS = False
REGENERATE_VAL = False