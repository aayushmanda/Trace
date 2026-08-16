"""Phase transition plot."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from config import PLOT_FILENAME  # noqa: E402


def save_phase_transition_plot(results, seeds, filename=PLOT_FILENAME):
    ratios_pct = [r * 100 for r in results.keys()]

    plt.figure(figsize=(10, 6), dpi=300)

    colors = ["#7f7f7f", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for i, seed in enumerate(seeds):
        seed_accs = [
            results[r][i] * 100 if i < len(results[r]) else None for r in results.keys()
        ]
        plt.scatter(
            ratios_pct,
            seed_accs,
            color=colors[i % len(colors)],
            alpha=0.7,
            s=25,
            zorder=3,
            label=f"Seed {seed}",
        )

    means = [(sum(accs) / len(accs)) * 100 for accs in results.values()]
    stds = [
        (
            (sum(((x * 100) - m) ** 2 for x in accs) / len(accs)) ** 0.5
            if len(accs) > 1
            else 0.0
        )
        for m, accs in zip(means, results.values())
    ]

    lower = [max(0.0, m - s) for m, s in zip(means, stds)]
    upper = [min(100.0, m + s) for m, s in zip(means, stds)]
    plt.fill_between(
        ratios_pct,
        lower,
        upper,
        color="#1a73e8",
        alpha=0.15,
        label="±1 Std Dev Range",
    )

    plt.plot(
        ratios_pct,
        means,
        color="#1a73e8",
        marker="o",
        linewidth=2.5,
        markersize=6,
        zorder=4,
        label="Mean Accuracy",
    )

    plt.title(
        "Accuracy vs. Think Trace Ratio (Prompt Masking Only)",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    plt.xlabel(
        "Correct Think Trace Ratio (%)", fontsize=11, fontweight="bold", labelpad=8
    )
    plt.ylabel("Accuracy (%)", fontsize=11, fontweight="bold", labelpad=8)

    plt.xticks(ratios_pct, [f"{int(r)}%" for r in ratios_pct], rotation=45, fontsize=9)
    plt.ylim(-2, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()

    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"\n[INFO] Plot saved successfully as '{filename}'")
