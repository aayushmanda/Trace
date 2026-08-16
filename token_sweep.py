"""Fixed-Token Budget Sweep: re-run the ratio sweep at several batch sizes while

holding the total sample/token exposure constant across all runs.

Per the pure SNR theory, n_steps is scaled inversely with batch size:
    n_steps(B) = TOTAL_TOKEN_BUDGET / B

This isolates mini-batch gradient variance from cumulative compute effects
to validate the theoretical scaling exponent rho_c(B) ~ B^{-1/2}.
"""

from config import (
    CACHE_DIR,
    DATASET_SEED,
    DATASET_SIZE,
    RATIOS_TO_TEST,
    SEEDS_TO_TEST,
    VAL_WORDS,
    steps_per_run,
    BATCH_SIZES_TO_TEST,
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

# --- Fix overall token exposure ---
# Uses B=256 at default steps_per_run as the reference token baseline
REFERENCE_BATCH_SIZE = 256
TOTAL_TOKEN_BUDGET = REFERENCE_BATCH_SIZE * steps_per_run


def summarize(results, batch_size, steps):
    print(
        f"\n{'Correct %':<12} | {'Mean Acc %':<12} | {'Std Dev %':<12} "
        f"(B={batch_size}, steps={steps})"
    )
    print("-" * 55)
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


def run_fixed_token_experiment(
    batch_sizes=BATCH_SIZES_TO_TEST,
    ratios_to_test=RATIOS_TO_TEST,
    seeds_to_test=SEEDS_TO_TEST,
    total_token_budget=TOTAL_TOKEN_BUDGET,
):
    val_clean = build_validation_dataset(
        n_words=VAL_WORDS, correct_ratio=1.0, data_seed=DATASET_SEED
    )
    master_correct, master_wrong = get_or_create_master_pools(
        n_samples=DATASET_SIZE, data_seed=DATASET_SEED, cache_dir=CACHE_DIR
    )

    all_results = {}

    for B in batch_sizes:
        # Scale steps inversely with batch size to maintain fixed total token exposure
        scaled_steps = int(total_token_budget / B)

        print("\n" + "=" * 70)
        print(
            f"   BATCH SIZE = {B:3d} | SCALED STEPS = {scaled_steps:5d} "
            f"(Total Token Budget = {total_token_budget:,})"
        )
        print("=" * 70)

        results = {}
        for ratio in ratios_to_test:
            train_dataset_tensors = get_mixed_dataset_for_ratio(
                master_correct, master_wrong, ratio
            )
            accs = []
            for seed in seeds_to_test[:1]:
                model = train_model(
                    train_dataset_tensors,
                    seed,
                    n_steps=scaled_steps,
                    batch_size=B,
                )
                acc = evaluate_accuracy(model, val_clean)
                accs.append(acc)
                print(
                    f"  [B={B:3d} | steps={scaled_steps:5d}] ratio={ratio * 100:5.1f}%  "
                    f"seed={seed}  acc={acc * 100:6.2f}%"
                )
            results[ratio] = accs

        summarize(results, B, scaled_steps)
        all_results[B] = results

    return all_results


if __name__ == "__main__":
    run_fixed_token_experiment()