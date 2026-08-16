"""Batch-size sweep: re-run the ratio sweep at several batch sizes.

Uses your existing train_model(train_dataset_tensors, seed, n_steps, batch_size)
signature directly -- no changes to train.py needed.

Per the theory, steps_per_run is held FIXED across all batch sizes (not epochs),
so this is a true test of rho_c(B) ~ B^{-1/2} rather than a fixed-epoch protocol.
"""

from config import (
    CACHE_DIR,
    DATASET_SEED,
    DATASET_SIZE,
    RATIOS_TO_TEST,
    SEEDS_TO_TEST,
    VAL_WORDS,
    steps_per_run,
    BATCH_SIZES_TO_TEST
)
from data import (
    build_validation_dataset,
    get_mixed_dataset_for_ratio,
    get_or_create_master_pools,
)
from evaluate import evaluate_accuracy
from train import train_model

# --- Batch sizes to sweep ---
BATCH_SIZES_TO_TEST = [32, 64, 128, 256, 512]


def summarize(results, batch_size):
    print(f"\n{'Correct %':<12} | {'Mean Acc %':<12} | {'Std Dev %':<12}")
    print("-" * 45)
    for ratio, accs in results.items():
        mean_acc = sum(accs) / len(accs)
        std_acc = (
            (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5
            if len(accs) > 1
            else 0.0
        )
        print(
            f"{ratio * 100:>10.1f}% | {mean_acc * 100:>10.2f}% | "
            f"± {std_acc * 100:>8.2f}%"
        )


def run_batch_size_experiment(
    batch_sizes=BATCH_SIZES_TO_TEST,
    ratios_to_test=RATIOS_TO_TEST,
    seeds_to_test=SEEDS_TO_TEST,
):
    val_clean = build_validation_dataset(
        n_words=VAL_WORDS, correct_ratio=1.0, data_seed=DATASET_SEED
    )
    master_correct, master_wrong = get_or_create_master_pools(
        n_samples=DATASET_SIZE, data_seed=DATASET_SEED, cache_dir=CACHE_DIR
    )

    all_results = {}

    for B in batch_sizes:
        print("\n" + "=" * 70)
        print(f"   BATCH SIZE = {B}   (steps_per_run held fixed at {steps_per_run})")
        print("=" * 70)

        results = {}
        for ratio in ratios_to_test:
            train_dataset_tensors = get_mixed_dataset_for_ratio(
                master_correct, master_wrong, ratio
            )
            accs = []
            for seed in seeds_to_test[:3]:
                model = train_model(
                    train_dataset_tensors,
                    seed,
                    n_steps=steps_per_run,
                    batch_size=B,
                )
                acc = evaluate_accuracy(model, val_clean)
                accs.append(acc)
                print(
                    f"  [B={B}] ratio={ratio * 100:5.1f}%  seed={seed}  "
                    f"acc={acc * 100:6.2f}%"
                )
            results[ratio] = accs

        summarize(results, B)
        all_results[B] = results

    return all_results


if __name__ == "__main__":
    run_batch_size_experiment()