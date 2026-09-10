"""Training-dynamics animation: answer matrices + fixed-outcome residual snapshots."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import PercentFormatter

from handcoded.plotting import apply_style


def animate_training_dynamics(history, fixed_circuit, fixed_metrics, interval_ms=140,
                              *, dpi=140, figsize=(18, 14), selected_start=8):
    """FuncAnimation from real checkpoints (not interpolated tokens)."""
    apply_style()
    if interval_ms <= 0 or dpi <= 0:
        raise ValueError("Playback interval and dpi must be positive")
    frames = {mode: history[history["mode"] == mode].sort_values("step").reset_index(drop=True)
              for mode in ("outcome", "process")}
    steps = frames["outcome"]["step"].tolist()
    if not steps or steps != frames["process"]["step"].tolist():
        raise ValueError("Both models need matching, nonempty checkpoint steps")
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Checkpoint steps must be unique and increasing")
    size = fixed_circuit["answer_matrix"].shape[0]
    if not isinstance(selected_start, int) or not 0 <= selected_start < size:
        raise ValueError("selected_start must be a valid integer state")
    if len(fixed_circuit["layers"]) != len(fixed_circuit["gates"]) or not fixed_circuit["layers"]:
        raise ValueError("Each circuit gate needs its captured fixed-layer matrix")
    for mode in frames:
        if any(np.asarray(matrix).shape != (size, size) for matrix in frames[mode].circuit_matrix):
            raise ValueError("All circuit matrices must match the fixed matrix shape")

    colors = {"outcome": "#c56b08", "process": "#087eaa"}
    fixed_color, highlight = "#7653ad", "#e84393"
    fig = plt.figure(figsize=figsize, dpi=dpi, facecolor="white")
    outer = fig.add_gridspec(7, 1, height_ratios=[0.65, 2.5, 0.55, 3.1, 0.55, 2.5, 0.75],
                            left=0.06, right=0.95, top=0.98, bottom=0.025, hspace=0.30)
    header = fig.add_subplot(outer[0]); header.axis("off")
    header.text(0, 0.85, "Learning a Boolean circuit", fontsize=23, weight="bold", color="#172b4d")
    circuit_text = " → ".join(fixed_circuit["gates"])
    header.text(0, 0.25, f"Circuit: {circuit_text}   |   Fixed outcome: {len(fixed_circuit['layers'])} blocks; learned models: 1 block",
                fontsize=12, color="#475569")
    progress = header.text(1, 0.85, "", ha="right", fontsize=12, color="#172b4d")

    curve_grid = outer[1].subgridspec(1, 3, wspace=0.28)
    axes = [fig.add_subplot(curve_grid[0, i]) for i in range(3)]
    specs = [("train_loss", 0, "-"), ("train_answer_accuracy_sample", 1, "--"),
             ("test_answer_accuracy", 1, "-"), ("test_exact_continuation", 2, "-")]
    lines, cursors = {}, []
    for metric, panel, style in specs:
        for mode, color in colors.items():
            split = "training sample" if metric == "train_answer_accuracy_sample" else "test"
            label = f"{mode}: {split}" if panel == 1 else mode
            lines[metric, mode], = axes[panel].plot([], [], style, color=color, linewidth=2.2,
                                                  marker="o", markersize=3, markevery=[-1], label=label)
    for ax, title, metric in zip(axes,
        ["1  Training loss ↓", "2  Generated answer accuracy ↑", "3  Exact test continuation ↑"],
        ["train_loss", "test_answer_accuracy", "test_exact_continuation"]):
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=12)
        ax.axhline(fixed_metrics[metric], color=fixed_color, linestyle=":", linewidth=2,
                   label="Fixed outcome reference")
        ax.set_xlim(0, max(1, steps[-1]))
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlabel("Optimization step (linear to 100, then logarithmic)", fontsize=8)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=7, loc="best")
        cursors.append(ax.axvline(0, color="#64748b", alpha=0.4, linewidth=1))
    axes[0].set(ylabel="Next-token cross-entropy",
                ylim=(0, max(0.1, history.train_loss.max(), fixed_metrics["train_loss"]) * 1.12))
    for ax in axes[1:]:
        ax.set(ylabel="Accuracy", ylim=(-0.025, 1.065))
        ax.yaxis.set_major_formatter(PercentFormatter(1))

    def text_band(slot, title, explanation):
        ax = fig.add_subplot(slot); ax.axis("off")
        ax.text(0, 0.80, title, fontsize=13, weight="bold", color="#172b4d")
        ax.text(0, 0.13, explanation, fontsize=10, color="#475569")

    text_band(outer[2], "4  Whole-circuit answer probabilities",
              "Rows = starting states; columns = answer states. Pink row follows the selected input.")

    def draw_matrix(ax, matrix, title, show_all_ticks=True):
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
        ax.set_title(title, fontsize=11, pad=10)
        ticks = np.arange(size) if show_all_ticks else np.arange(0, size, 4)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Resulting state (integer)", fontsize=8)
        ax.set_ylabel("Starting state (integer)", fontsize=8)
        ax.axhline(selected_start, color=highlight, linewidth=1.1, alpha=0.8)
        return image

    matrix_grid = outer[3].subgridspec(1, 4, width_ratios=[1, 1, 1, 0.035], wspace=0.26)
    matrix_axes = [fig.add_subplot(matrix_grid[0, i]) for i in range(3)]
    fixed_image = draw_matrix(matrix_axes[0], fixed_circuit["answer_matrix"], "Hand-coded outcome • fixed")
    images = {mode: draw_matrix(ax, frames[mode].iloc[0].circuit_matrix, f"{mode.title()}-trained")
              for ax, mode in zip(matrix_axes[1:], ["outcome", "process"])}
    fig.colorbar(fixed_image, cax=fig.add_subplot(matrix_grid[0, 3]), label="Answer-token probability")

    text_band(outer[4], "5  Inside the hand-coded outcome Transformer • fixed throughout training",
              "Each matrix is a captured hidden state after all gates up to this block.")
    layer_count = len(fixed_circuit["layers"])
    layer_grid = outer[5].subgridspec(1, layer_count + 1, width_ratios=[1] * layer_count + [0.045], wspace=0.30)
    trajectory = []
    for index, (gate, matrix) in enumerate(zip(fixed_circuit["gates"], fixed_circuit["layers"])):
        ax = fig.add_subplot(layer_grid[0, index])
        layer_image = draw_matrix(ax, matrix, f"Block {index + 1} · apply {gate}", False)
        state = int(matrix[selected_start].argmax())
        trajectory.append(state)
        ax.scatter([state], [selected_start], s=95, marker="s", facecolors="none",
                   edgecolors=highlight, linewidths=1.8)
    fig.colorbar(layer_image, cax=fig.add_subplot(layer_grid[0, -1]), label="Hidden feature activation")

    footer = fig.add_subplot(outer[6]); footer.axis("off")
    path = f"S{selected_start:04b}" + "".join(
        f"  —{gate}→ S{state:04b}" for gate, state in zip(fixed_circuit["gates"], trajectory))
    footer.text(0, 0.95, "Selected input inside the fixed model: " + path, fontsize=11, color="#172b4d")
    live_summary = footer.text(0, 0.45, "", fontsize=10, color="#475569")
    footer.text(0, 0, "Frames are measured training checkpoints. Fixed panels never train.",
                fontsize=9, color="#64748b")

    def update_frame(index):
        progress.set_text(f"Step {steps[index]:,} / {steps[-1]:,}  •  Frame {index + 1} / {len(steps)}")
        for (metric, mode), line in lines.items():
            data = frames[mode].iloc[:index + 1]
            line.set_data(data.step, data[metric])
        summaries = []
        for ax, mode in zip(matrix_axes[1:], ["outcome", "process"]):
            row = frames[mode].iloc[index]
            images[mode].set_data(row.circuit_matrix)
            ax.set_title(f"{mode.title()}-trained • test answer {row.test_answer_accuracy:.1%}", fontsize=11)
            summaries.append(f"{mode}: train sample {row.train_answer_accuracy_sample:.1%}, test {row.test_answer_accuracy:.1%}")
        for cursor in cursors:
            cursor.set_xdata([steps[index], steps[index]])
        live_summary.set_text("Generated-answer accuracy at this checkpoint — " + "  |  ".join(summaries))
        return [progress, live_summary, *lines.values(), *images.values(), *cursors]

    animation = FuncAnimation(fig, update_frame, frames=len(steps), init_func=lambda: update_frame(0),
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
    if Path(filename).suffix.lower() != ".mp4":
        raise ValueError("Use an .mp4 filename")
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
