"""Dataset generation, ratio mixing, and batching without disk serialization."""

import random
import string
import torch
from config import DATASET_SEED, DATASET_SIZE, VAL_SEED, block_size, device
from tokenizer import PAD_ID, encode


def sample_word(length):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def make_example(word, mode="correct_think"):
    gold = random.randrange(len(word))
    c = word[gold]
    prompt = f"{word};{c}"

    if mode == "correct_think":
        trace = " ".join([f"{i}{ch}" for i, ch in enumerate(word)])
        answer = f" {trace} : {gold}"
    elif mode == "wrong_think":
        fake_indices = list(range(len(word)))
        random.shuffle(fake_indices)
        trace = " ".join(
            [
                f"{fake_idx}{random.choice(string.ascii_lowercase)}"
                for fake_idx in fake_indices
            ]
        )
        answer = f" {trace} : {gold}"
    else:
        answer = f" {gold}"

    return prompt, answer


def _encode_example(prompt, answer):
    p_ids = encode(prompt)
    t_ids = encode(answer + "\n")
    full = (p_ids + t_ids)[:block_size]
    Lp = min(len(p_ids), len(full))

    x, y = full[:-1], full[1:]
    mask = [1.0 if (i + 1) >= Lp else 0.0 for i in range(len(full) - 1)]

    pad_len = (block_size - 1) - len(x)
    x += [PAD_ID] * pad_len
    y += [PAD_ID] * pad_len
    mask += [0.0] * pad_len

    return x, y, mask


def get_or_create_master_pools(
    n_samples=DATASET_SIZE, data_seed=DATASET_SEED, cache_dir=None
):
    """Generates master pools in RAM without saving gigabytes to disk."""
    print(f"\n[RAM] Generating Master Dataset Pools in RAM ({n_samples:,} samples)...")
    rng_state_py = random.getstate()
    random.seed(data_seed)

    correct_xs, correct_ys, correct_masks = [], [], []
    wrong_xs, wrong_ys, wrong_masks = [], [], []

    for _ in range(n_samples):
        L = random.randint(4, 10)
        w = sample_word(L)

        x_c, y_c, mask_c = _encode_example(*make_example(w, mode="correct_think"))
        correct_xs.append(x_c)
        correct_ys.append(y_c)
        correct_masks.append(mask_c)

        x_w, y_w, mask_w = _encode_example(*make_example(w, mode="wrong_think"))
        wrong_xs.append(x_w)
        wrong_ys.append(y_w)
        wrong_masks.append(mask_w)

    random.setstate(rng_state_py)

    master_correct = (
        torch.tensor(correct_xs, dtype=torch.long, device=device),
        torch.tensor(correct_ys, dtype=torch.long, device=device),
        torch.tensor(correct_masks, dtype=torch.float, device=device),
    )
    master_wrong = (
        torch.tensor(wrong_xs, dtype=torch.long, device=device),
        torch.tensor(wrong_ys, dtype=torch.long, device=device),
        torch.tensor(wrong_masks, dtype=torch.float, device=device),
    )

    print("[RAM] Dataset loaded successfully into GPU/RAM memory!\n")
    return master_correct, master_wrong


def get_mixed_dataset_for_ratio(master_correct, master_wrong, ratio):
    n_samples = master_correct[0].shape[0]
    n_correct = int(n_samples * ratio)

    c_x, c_y, c_m = (
        master_correct[0][:n_correct],
        master_correct[1][:n_correct],
        master_correct[2][:n_correct],
    )
    w_x, w_y, w_m = (
        master_wrong[0][n_correct:],
        master_wrong[1][n_correct:],
        master_wrong[2][n_correct:],
    )

    mix_x = torch.cat([c_x, w_x], dim=0)
    mix_y = torch.cat([c_y, w_y], dim=0)
    mix_m = torch.cat([c_m, w_m], dim=0)

    return mix_x, mix_y, mix_m


def get_batch_from_fixed_dataset(dataset_tensors, batch_size):
    xs_all, ys_all, masks_all = dataset_tensors
    n_samples = xs_all.shape[0]
    ix = torch.randint(0, n_samples, (batch_size,), device=device)
    return xs_all[ix], ys_all[ix], masks_all[ix]


def build_validation_dataset(n_words=500, correct_ratio=1.0, data_seed=VAL_SEED):
    rng_state_py = random.getstate()
    random.seed(data_seed)

    examples = []
    seen = set()
    while len(seen) < n_words:
        L = random.randint(4, 10)
        w = sample_word(L)
        if w in seen:
            continue
        seen.add(w)

        mode = "correct_think" if random.random() < correct_ratio else "wrong_think"
        prompt_part, answer_part = make_example(w, mode=mode)
        examples.append((prompt_part, answer_part + "\n"))

    random.setstate(rng_state_py)
    return examples