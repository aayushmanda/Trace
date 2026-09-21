"""Compose documented crops of existing video frames; no training or digitization.

Run: python -m src.experiments.plotting.plot_optimization_dynamics
The fixed crops are specific to the archived 3240 x 2520 training video.
Curves and matrix colors are retained as rendered; no numeric history is
recovered from pixels. Original hidden-state panels are deliberately omitted.
"""
from pathlib import Path
import hashlib
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
VIDEO = ROOT / "docs/assets/training_dynamics.mp4"
SOURCE_SHA256 = "3bfdbb0ee341fa088543af2be406d29158fb0174cf91516167a48093f274f172"
# Zero-based frame indices; labels are visible in each exported frame.
CHECKPOINTS = [(0, 0), (28, 1583), (129, 10000)]
# PIL boxes: left, upper, right, lower, in unscaled source pixels.
CURVES = [(180, 399, 1075, 875), (1140, 399, 2100, 875), (2160, 399, 3140, 875)]
MAPS = {"Outcome": (1300, 1060, 1813, 1575), "Process": (2238, 1060, 2751, 1575)}


def main():
    if hashlib.sha256(VIDEO.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise ValueError("Video changed: recheck frame indices and crop boxes before rebuilding")
    with tempfile.TemporaryDirectory(prefix="trace-illustrative-") as temporary:
        target = Path(temporary) / "frame-%02d.png"
        selection = "+".join(f"eq(n\\,{frame})" for frame, _ in CHECKPOINTS)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(VIDEO), "-vf", f"select={selection}",
                        "-vsync", "0", "-frames:v", "3", str(target)], check=True)
        frames = [Image.open(Path(temporary) / f"frame-{i:02d}.png").convert("RGB") for i in (1, 2, 3)]
        if any(frame.size != (3240, 2520) for frame in frames):
            raise ValueError("Unexpected video geometry")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
        fig = plt.figure(figsize=(8, 5.05), facecolor="white")
        fig.text(.035, 1.01, "(a) Measured training trajectories", weight="bold", va="top", fontsize=11)
        for left, title, box in zip([.045, .365, .685],
                                    ["Training loss", "Generated-answer accuracy", "Exact test continuation"], CURVES):
            ax = fig.add_axes([left, .745, .275, .19])
            ax.imshow(frames[-1].crop(box))
            ax.axis("off")
            ax.set_title(title, fontsize=9, pad=2)
        fig.text(.5, .721, "Optimization step (linear to 100, then logarithmic)", ha="center", fontsize=8)
        fig.legend(handles=[Line2D([0], [0], color="#c56b08", lw=2, label="Outcome"),
                            Line2D([0], [0], color="#087eaa", lw=2, label="Process"),
                            Line2D([0], [0], color="#7653ad", lw=2, ls=":", label="Fixed reference")],
                   loc="center", bbox_to_anchor=(.5, .68), ncol=3, frameon=False, fontsize=8)
        for row, (mode, bottom, heading) in enumerate([
            ("Outcome", .385, .645), ("Process", .055, .325)
        ]):
            fig.text(.035, heading, f"({'bc'[row]}) {mode}: whole-circuit answer maps", weight="bold", fontsize=11)
            for index, (frame, (_, step)) in enumerate(zip(frames, CHECKPOINTS)):
                ax = fig.add_axes([.115 + index * .30, bottom, .14, .222])
                ax.imshow(frame.crop(MAPS[mode]), extent=(-.5, 15.5, 15.5, -.5), interpolation="nearest")
                ax.set_title(f"Step {step:,}", fontsize=9, pad=3)
                ax.set_xticks([0, 15]); ax.set_yticks([0, 15]); ax.tick_params(labelsize=8, length=2, pad=1)
                if index == 0:
                    ax.set_ylabel("Start state", fontsize=8, labelpad=1)
                if row == 1:
                    ax.set_xlabel("Answer state", fontsize=8, labelpad=1)
        # Same colormap and [0,1] range used by the source animation, not inferred values.
        bar = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"),
                           cax=fig.add_axes([.94, .10, .015, .47]), ticks=[0, .5, 1])
        bar.set_label("Answer-token probability", fontsize=8)
        bar.ax.tick_params(labelsize=8, length=2)
        destination = ROOT / "Paper/figures/optimization_dynamics"
        destination.parent.mkdir(exist_ok=True, parents=True)
        for suffix in ("pdf", "png"):
            fig.savefig(destination.with_suffix("." + suffix), dpi=240, bbox_inches="tight", pad_inches=.04)
        plt.close(fig)
        print("Wrote", destination.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
