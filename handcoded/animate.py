"""Training-dynamics animation and a catalog of every evaluated circuit."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from handcoded.eval import circuit_answer_matrix, gold_answer_matrix, make_circuit_prompts
from src.plot_style import apply_style

OUTCOME, PROCESS, FIXED, HIGHLIGHT = "#c56b08", "#087eaa", "#7653ad", "#e84393"


def animate_training_dynamics(history, fixed_circuit, fixed_metrics, interval_ms=140,
                              *, dpi=140, figsize=(18, 14), selected_start=None):
    """FuncAnimation from real checkpoints (not interpolated tokens)."""
    apply_style({"legend.fontsize": 13, "legend.title_fontsize": 14})
    frames = {mode: history[history["mode"] == mode].sort_values("step").reset_index(drop=True)
              for mode in ("outcome", "process")}
    steps = frames["outcome"]["step"].tolist()
    size = fixed_circuit["answer_matrix"].shape[0]
    state_width = max(1, int(np.ceil(np.log2(size))))
    if selected_start is None:
        selected_start = min(8, size - 1)   # 8 at K=16, else the nearest valid state
    if not 0 <= selected_start < size:
        raise ValueError(
            f"selected_start={selected_start} is not a state: this run has {size} states (0..{size - 1})"
        )

    colors = {"outcome": OUTCOME, "process": PROCESS}
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    outer = fig.add_gridspec(8, 1, height_ratios=[0.72, 0.38, 2.5, 0.55, 3.1, 0.55, 2.5, 0.85],
                            left=0.07, right=0.95, top=0.97, bottom=0.03, hspace=0.32)
    header = fig.add_subplot(outer[0]); header.axis("off")
    header.text(0, 0.82, "Learning a Boolean circuit", fontsize=28, weight="bold", color="#172b4d")
    circuit_text = " → ".join(fixed_circuit["gates"])
    header.text(0, 0.18, f"Circuit: {circuit_text}   |   Fixed outcome: {len(fixed_circuit['layers'])} blocks; learned models: 1 block",
                fontsize=16, color="#475569")
    progress = header.text(1, 0.82, "", ha="right", fontsize=16, color="#172b4d")

    legend_ax = fig.add_subplot(outer[1]); legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color=OUTCOME, lw=3, label="Outcome-trained"),
            Line2D([0], [0], color=PROCESS, lw=3, label="Process-trained"),
            Line2D([0], [0], color=OUTCOME, lw=3, ls="--", label="Outcome · train sample"),
            Line2D([0], [0], color=PROCESS, lw=3, ls="--", label="Process · train sample"),
            Line2D([0], [0], color=FIXED, lw=3, ls=":", label="Fixed outcome reference"),
            Line2D([0], [0], color=HIGHLIGHT, lw=3, label="Selected start state"),
        ],
        loc="center", ncol=3, fontsize=14, title="Legend", title_fontsize=15, frameon=True,
    )

    curve_grid = outer[2].subgridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(curve_grid[0, i]) for i in range(3)]
    specs = [("train_loss", 0, "-"), ("train_answer_accuracy_sample", 1, "--"),
             ("test_answer_accuracy", 1, "-"), ("test_exact_continuation", 2, "-")]
    lines, cursors = {}, []
    for metric, panel, style in specs:
        for mode, color in colors.items():
            lines[metric, mode], = axes[panel].plot([], [], style, color=color, linewidth=2.6,
                                                  marker="o", markersize=4, markevery=[-1])
    for ax, title, metric in zip(axes,
        ["1  Training loss ↓", "2  Generated answer accuracy ↑", "3  Exact test continuation ↑"],
        ["train_loss", "test_answer_accuracy", "test_exact_continuation"]):
        ax.set_title(title, loc="left", fontsize=16, weight="bold", pad=12)
        ax.axhline(fixed_metrics[metric], color=FIXED, linestyle=":", linewidth=2.4)
        ax.set_xlim(0, max(1, steps[-1]))
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("Optimization step (linear to 100, then logarithmic)", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        cursors.append(ax.axvline(0, color="#64748b", alpha=0.4, linewidth=1.2))
    axes[0].set_ylabel("Next-token cross-entropy", fontsize=14)
    axes[0].set_ylim(0, max(0.1, history.train_loss.max(), fixed_metrics["train_loss"]) * 1.12)
    for ax in axes[1:]:
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_ylim(-0.025, 1.065)
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    def text_band(slot, title, explanation):
        ax = fig.add_subplot(slot); ax.axis("off")
        ax.text(0, 0.80, title, fontsize=17, weight="bold", color="#172b4d")
        ax.text(0, 0.13, explanation, fontsize=14, color="#475569")

    text_band(outer[3], "4  Whole-circuit answer probabilities",
              "Rows = starting states; columns = answer states. Pink row follows the selected input.")

    def draw_matrix(ax, matrix, title, show_all_ticks=True):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=15, pad=10)
        ticks = np.arange(size) if show_all_ticks else np.arange(0, size, 4)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.tick_params(labelsize=11)
        ax.set_xlabel("Resulting state (integer)", fontsize=13)
        ax.set_ylabel("Starting state (integer)", fontsize=13)
        ax.axhline(selected_start, color=HIGHLIGHT, linewidth=1.6, alpha=0.9)
        return image

    matrix_grid = outer[4].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.26)
    matrix_axes = [fig.add_subplot(matrix_grid[0, i]) for i in range(3)]
    fixed_image = draw_matrix(matrix_axes[0], fixed_circuit["answer_matrix"], "Hand-coded outcome • fixed")
    images = {mode: draw_matrix(ax, frames[mode].iloc[0].circuit_matrix, f"{mode.title()}-trained")
              for ax, mode in zip(matrix_axes[1:], ["outcome", "process"])}
    colorbar = fig.colorbar(fixed_image, cax=fig.add_subplot(matrix_grid[0, 3]))
    colorbar.set_label("Answer-token probability", fontsize=13)
    colorbar.ax.tick_params(labelsize=11)

    text_band(outer[5], "5  Inside the hand-coded outcome Transformer • fixed throughout training",
              "Each matrix is a captured hidden state after all gates up to this block.")
    layer_count = len(fixed_circuit["layers"])
    layer_grid = outer[6].subgridspec(1, layer_count + 1, width_ratios=[1] * layer_count + [0.05], wspace=0.30)
    trajectory = []
    for index, (gate, matrix) in enumerate(zip(fixed_circuit["gates"], fixed_circuit["layers"])):
        ax = fig.add_subplot(layer_grid[0, index])
        layer_image = draw_matrix(ax, matrix, f"Block {index + 1} · apply {gate}", False)
        state = int(matrix[selected_start].argmax())
        trajectory.append(state)
        ax.scatter([state], [selected_start], s=120, marker="s", facecolors="none",
                   edgecolors=HIGHLIGHT, linewidths=2.2)
    layer_bar = fig.colorbar(layer_image, cax=fig.add_subplot(layer_grid[0, -1]))
    layer_bar.set_label("Hidden feature activation", fontsize=13)
    layer_bar.ax.tick_params(labelsize=11)

    footer = fig.add_subplot(outer[7]); footer.axis("off")
    path = f"S{selected_start:0{state_width}b}" + "".join(
        f"  —{gate}→ S{state:0{state_width}b}"
        for gate, state in zip(fixed_circuit["gates"], trajectory))
    footer.text(0, 0.95, "Selected input inside the fixed model: " + path, fontsize=15, color="#172b4d")
    live_summary = footer.text(0, 0.45, "", fontsize=14, color="#475569")
    footer.text(0, 0, "Frames are measured training checkpoints. Fixed panels never train.",
                fontsize=13, color="#64748b")

    def update_frame(index):
        progress.set_text(f"Step {steps[index]:,} / {steps[-1]:,}  •  Frame {index + 1} / {len(steps)}")
        for (metric, mode), line in lines.items():
            data = frames[mode].iloc[:index + 1]
            line.set_data(data.step, data[metric])
        summaries = []
        for ax, mode in zip(matrix_axes[1:], ["outcome", "process"]):
            row = frames[mode].iloc[index]
            images[mode].set_data(row.circuit_matrix)
            ax.set_title(f"{mode.title()}-trained • test answer {row.test_answer_accuracy:.1%}", fontsize=15)
            summaries.append(f"{mode}: train sample {row.train_answer_accuracy_sample:.1%}, test {row.test_answer_accuracy:.1%}")
        for cursor in cursors:
            cursor.set_xdata([steps[index], steps[index]])
        live_summary.set_text("Generated-answer accuracy at this checkpoint — " + "  |  ".join(summaries))
        return [progress, live_summary, *lines.values(), *images.values(), *cursors]

    animation = FuncAnimation(fig, update_frame, frames=len(steps), init_func=lambda: update_frame(0),
                              interval=interval_ms, repeat=False, blit=False)
    plt.close(fig)
    return animation


def animate_architecture_dynamics(history, interval_ms=140, *, dpi=140,
                                 figsize=(16, 9), architectures=None):
    """FuncAnimation for the 2x2 run from `run_architecture_experiment`.

    `animate_training_dynamics` filters on `mode` alone, so it silently merges the
    two architectures when handed a 2x2 history. This one keys on the
    (architecture, mode) pair: colour is the supervision format, line style is the
    architecture, and each pair gets its own evolving answer matrix.
    """
    apply_style({"legend.fontsize": 12, "legend.title_fontsize": 13})
    if architectures is None:
        architectures = list(dict.fromkeys(history["architecture"]))
    modes = list(dict.fromkeys(history["mode"]))
    pairs = [(a, m) for a in architectures for m in modes]

    frames = {}
    for pair in pairs:
        rows = history[(history["architecture"] == pair[0]) & (history["mode"] == pair[1])]
        if rows.empty:
            raise ValueError(f"no rows for architecture={pair[0]!r} mode={pair[1]!r}")
        frames[pair] = rows.sort_values("step").reset_index(drop=True)
    steps = frames[pairs[0]]["step"].tolist()
    for pair, rows in frames.items():
        if rows["step"].tolist() != steps:
            raise ValueError(f"{pair} has a different checkpoint grid; cannot share frames")

    colour = {m: (OUTCOME if m == "outcome" else PROCESS) for m in modes}
    dashes = {a: style for a, style in zip(architectures, ["-", "--", ":", "-."])}

    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[0.12, 1, 1], hspace=0.42,
                             left=0.06, right=0.97, top=0.94, bottom=0.07)
    header = fig.add_subplot(outer[0]); header.axis("off")
    header.text(0, 0.7, "Supervision x architecture", fontsize=24, weight="bold", color="#172b4d")
    header.text(0, 0.05, "Colour = where the loss is applied.  Line style = which architecture.",
                fontsize=14, color="#475569")
    progress = header.text(1, 0.7, "", ha="right", fontsize=15, color="#172b4d")

    curve_grid = outer[1].subgridspec(1, 2, wspace=0.22)
    loss_ax = fig.add_subplot(curve_grid[0, 0])
    acc_ax = fig.add_subplot(curve_grid[0, 1])
    lines, cursors = {}, []
    for ax, metric, title, ylabel in (
        (loss_ax, "train_loss", "Training loss", "cross-entropy"),
        (acc_ax, "test_answer_accuracy", "Held-out answer accuracy", "accuracy"),
    ):
        for pair in pairs:
            lines[(metric, pair)], = ax.plot(
                [], [], color=colour[pair[1]], linestyle=dashes[pair[0]], linewidth=2.0,
                label=f"{pair[0]} / {pair[1]}")
        ax.set_title(title, loc="left", fontsize=15, weight="bold", pad=10)
        ax.set_xlabel("Optimization step (linear to 100, then logarithmic)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, max(steps) if max(steps) else 1)
        ax.grid(alpha=0.18)
        cursors.append(ax.axvline(steps[0], color="#94a3b8", linewidth=1.2, alpha=0.8))
    finite = history["train_loss"].replace([np.inf, -np.inf], np.nan).dropna()
    loss_ax.set_yscale("log")
    if len(finite):
        loss_ax.set_ylim(max(finite.min() * 0.6, 1e-6), finite.max() * 1.6)
    acc_ax.set_ylim(-0.03, 1.03)
    acc_ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    acc_ax.legend(loc="upper left", ncol=1, frameon=True)

    matrix_grid = outer[2].subgridspec(1, len(pairs) + 1,
                                       width_ratios=[1] * len(pairs) + [0.05], wspace=0.28)
    images, matrix_axes = {}, {}
    for index, pair in enumerate(pairs):
        ax = fig.add_subplot(matrix_grid[0, index])
        images[pair] = ax.imshow(frames[pair].iloc[0].circuit_matrix, vmin=0, vmax=1,
                                 cmap="viridis", interpolation="nearest")
        ax.set_xlabel("answer state", fontsize=11)
        ax.set_ylabel("start state", fontsize=11)
        ax.tick_params(labelsize=9)
        matrix_axes[pair] = ax
    bar = fig.colorbar(images[pairs[0]], cax=fig.add_subplot(matrix_grid[0, -1]))
    bar.set_label("answer-token probability", fontsize=11)
    bar.ax.tick_params(labelsize=9)

    def update_frame(index):
        progress.set_text(f"Step {steps[index]:,} / {steps[-1]:,}  \u2022  "
                          f"Frame {index + 1} / {len(steps)}")
        for (metric, pair), line in lines.items():
            data = frames[pair].iloc[:index + 1]
            line.set_data(data.step, data[metric])
        for pair in pairs:
            row = frames[pair].iloc[index]
            images[pair].set_data(row.circuit_matrix)
            matrix_axes[pair].set_title(
                f"{pair[0]} / {pair[1]}\n test answer {row.test_answer_accuracy:.1%}", fontsize=12)
        for cursor in cursors:
            cursor.set_xdata([steps[index], steps[index]])
        return [progress, *lines.values(), *images.values(), *cursors]

    animation = FuncAnimation(fig, update_frame, frames=len(steps),
                              init_func=lambda: update_frame(0),
                              interval=interval_ms, repeat=False, blit=False)
    plt.close(fig)
    return animation


def _circuit_catalog_cache(circuits, models, tokenizer, device):
    from tqdm.auto import tqdm
    seen, unique = set(), []
    for circuit in circuits:
        key = tuple(circuit.gates)
        if key in seen:
            continue
        seen.add(key)
        unique.append(circuit)
    n_bits = int(np.log2(tokenizer.n_states))
    cache = {}
    for circuit in tqdm(unique, desc="circuit matrices"):
        key = tuple(circuit.gates)
        prompts = make_circuit_prompts(circuit.gates, tokenizer, device)
        cache[key] = {
            "gold": gold_answer_matrix(circuit.gates, n_bits=n_bits),
            "outcome": circuit_answer_matrix(models["outcome"], prompts, tokenizer, "outcome"),
            "process": circuit_answer_matrix(models["process"], prompts, tokenizer, "process"),
        }
    return cache


def animate_all_circuits(circuits, models, tokenizer, device, interval_ms=280,
                         *, dpi=120, figsize=(18, 8)):
    """One frame per circuit: gold φ vs learned outcome vs learned process answer matrices."""
    apply_style({"legend.fontsize": 14, "legend.title_fontsize": 15})
    cache = _circuit_catalog_cache(circuits, models, tokenizer, device)
    size = tokenizer.n_states
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    outer = fig.add_gridspec(3, 1, height_ratios=[0.7, 0.4, 3.4],
                             left=0.06, right=0.92, top=0.96, bottom=0.08, hspace=0.18)
    header = fig.add_subplot(outer[0]); header.axis("off")
    header.text(0, 0.78, "Every evaluated circuit", fontsize=28, weight="bold", color="#172b4d")
    subtitle = header.text(0, 0.12, "", fontsize=16, color="#475569")
    progress = header.text(1, 0.78, "", ha="right", fontsize=16, color="#172b4d")

    legend_ax = fig.add_subplot(outer[1]); legend_ax.axis("off")
    legend_ax.legend(
        handles=[
            Line2D([0], [0], color=FIXED, lw=4, label="Gold φ (exact permutation)"),
            Line2D([0], [0], color=OUTCOME, lw=4, label="Learned outcome"),
            Line2D([0], [0], color=PROCESS, lw=4, label="Learned process"),
            Line2D([0], [0], color=HIGHLIGHT, lw=4, label="This circuit's start state"),
        ],
        loc="center", ncol=4, fontsize=15, title="Legend", title_fontsize=16, frameon=True,
    )

    matrix_grid = outer[2].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.22)
    titles = ("Gold φ", "Learned outcome", "Learned process")
    keys = ("gold", "outcome", "process")
    title_by_key = dict(zip(keys, titles))
    first = cache[tuple(circuits[0].gates)]
    images, axes = {}, []
    for index, (title, key) in enumerate(zip(titles, keys)):
        ax = fig.add_subplot(matrix_grid[0, index])
        image = ax.imshow(first[key], vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=18, pad=10)
        ax.set_xticks(np.arange(size)); ax.set_yticks(np.arange(size))
        ax.tick_params(labelsize=12)
        ax.set_xlabel("Answer state", fontsize=15)
        ax.set_ylabel("Start state", fontsize=15)
        highlight = ax.axhline(circuits[0].start, color=HIGHLIGHT, linewidth=2.2)
        images[key] = image
        axes.append((ax, highlight))
    colorbar = fig.colorbar(images["gold"], cax=fig.add_subplot(matrix_grid[0, 3]))
    colorbar.set_label("Answer-token probability", fontsize=14)
    colorbar.ax.tick_params(labelsize=12)

    def update_frame(index):
        circuit = circuits[index]
        panels = cache[tuple(circuit.gates)]
        gate_text = " → ".join(circuit.gates)
        subtitle.set_text(
            f"Start S{circuit.start:04b}   |   {gate_text}   |   gold answer S{circuit.answer:04b}"
        )
        progress.set_text(f"Circuit {index + 1} / {len(circuits)}")
        for key, (ax, highlight) in zip(keys, axes):
            images[key].set_data(panels[key])
            highlight.set_ydata([circuit.start, circuit.start])
            predicted = int(np.asarray(panels[key])[circuit.start].argmax())
            mass = float(np.asarray(panels[key])[circuit.start].sum())
            ax.set_title(
                f"{title_by_key[key]}  •  P(S{predicted:04b}) row-sum {mass:.2f}",
                fontsize=16,
            )
        return [subtitle, progress, *images.values()]

    animation = FuncAnimation(fig, update_frame, frames=len(circuits), init_func=lambda: update_frame(0),
                              interval=interval_ms, repeat=False, blit=False)
    plt.close(fig)
    return animation


def export_training_animation(animation, save_path="training_dynamics.html", embed_limit_mb=160):
    import re
    import warnings
    from IPython.display import HTML
    with plt.rc_context({"animation.embed_limit": embed_limit_mb}), warnings.catch_warnings(record=True) as caught:
        html = animation.to_jshtml(default_mode="once")
    expected = len(list(animation.new_frame_seq()))
    embedded = len(re.findall(r'frames\[\d+\] = "data:image', html))
    if embedded != expected:
        raise RuntimeError(
            f"Only {embedded}/{expected} frames embedded. Reduce dpi or increase embed_limit_mb."
        )
    for warning in caught:
        warnings.warn(warning.message, warning.category)
    page = """<!doctype html><html><head><meta charset="utf-8">
    <title>Transformer circuit learning</title></head><body>
    <h1>Transformer circuit learning</h1>
    """ + html + "</body></html>"
    Path(save_path).write_text(page, encoding="utf-8")
    print(f"Exported {embedded} frames to {save_path} ({len(page.encode()) / 1024**2:.1f} MiB)")
    return HTML(page)


def save_training_mp4(animation, filename="training_dynamics.mp4", *, fps=7, dpi=180):
    from matplotlib.animation import FFMpegWriter, writers
    ffmpeg_path = plt.rcParams["animation.ffmpeg_path"]
    if not writers.is_available("ffmpeg"):
        try:
            import imageio_ffmpeg
        except ImportError as error:
            raise RuntimeError("MP4 export needs FFmpeg or `imageio-ffmpeg`.") from error
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    with plt.rc_context({"animation.ffmpeg_path": ffmpeg_path}):
        if not writers.is_available("ffmpeg"):
            raise RuntimeError(f"FFmpeg is unavailable at {ffmpeg_path}")
        writer = FFMpegWriter(
            fps=fps, codec="libx264",
            extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        animation.save(filename, writer=writer, dpi=dpi)
    print(f"Saved MP4: {filename}")
    return Path(filename)
