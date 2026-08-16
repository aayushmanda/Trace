"""Phase transition sweep: accuracy vs. fraction of correct think traces."""

from config import (
    CACHE_DIR,
    DATASET_SEED,
    DATASET_SIZE,
    PLOT_FILENAME,
    RATIOS_TO_TEST,
    SEEDS_TO_TEST,
    VAL_WORDS,
    batch_size,
    device,
    dropout,
    weight_decay,
    batch_size,
)
from data import (
    build_validation_dataset,
    get_mixed_dataset_for_ratio,
    get_or_create_master_pools,
)
from evaluate import evaluate_accuracy
from model import GPTModel
from plotting import save_phase_transition_plot
from train import train_model


def summarize(results):
    print("\n" + "=" * 70)
    print("        STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"{'Correct %':<12} | {'Wrong %':<12} | {'Mean Acc %':<15} | {'Std Dev %':<12}")
    print("-" * 70)

    for ratio, accs in results.items():
        mean_acc = sum(accs) / len(accs)
        std_acc = (
            (sum((x - mean_acc) ** 2 for x in accs) / len(accs)) ** 0.5
            if len(accs) > 1
            else 0.0
        )
        print(
            f"{ratio * 100:>10.1f}% | {(1 - ratio) * 100:>10.1f}% | "
            f"{mean_acc * 100:>13.2f}% | ± {std_acc * 100:>8.2f}%"
        )

    print("=" * 70)


def run_clean_phase_experiment(
    ratios_to_test=RATIOS_TO_TEST,
    seeds_to_test=SEEDS_TO_TEST,
    plot_filename=PLOT_FILENAME,
):
    print(
        f"Model Parameters: "
        f"{sum(p.numel() for p in GPTModel().parameters() if p.requires_grad):,}"
    )

    results = {}

    val_clean = build_validation_dataset(
        n_words=VAL_WORDS, correct_ratio=1.0, data_seed=DATASET_SEED
    )

    # 1. GET OR CREATE MASTER DATASET POOLS FROM DISK
    master_correct, master_wrong = get_or_create_master_pools(
        n_samples=DATASET_SIZE, data_seed=DATASET_SEED, cache_dir=CACHE_DIR
    )

    print("=" * 70)
    print("   PHASE TRANSITION EXPERIMENT (LABEL SMOOTHED NOISE TRAINING)")
    print(
        f"Device: {device} | Batch Size: {batch_size} | "
        f"Weight Decay: {weight_decay} | Dropout: {dropout}"
    )
    print("=" * 70)

    for ratio in ratios_to_test:
        results[ratio] = []
        print(
            f"\n[Testing Ratio: {ratio * 100:.1f}% Correct | "
            f"{(1 - ratio) * 100:.1f}% Wrong]"
        )

        # 2. SLICE DIRECTLY FROM MASTER POOLS
        train_dataset_tensors = get_mixed_dataset_for_ratio(
            master_correct, master_wrong, ratio
        )

        for seed in seeds_to_test:
            model = train_model(train_dataset_tensors, seed)
            acc = evaluate_accuracy(model, val_clean)
            results[ratio].append(acc)
            print(f"Seed {seed} Accuracy: {acc * 100:.2f}%")

    summarize(results)
    save_phase_transition_plot(results, seeds_to_test, filename=plot_filename)

    return results
