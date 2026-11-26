# Trace — Silent Routing

Code, paper, and explainer animations for **Silent Routing: How Step Supervision Trains Transformers Where Outcome Supervision Stalls** (ICML 2026).

<p align="center">
  <img src="animations/previews/fig1_overview.png" alt="Figure 1: Step vs. outcome supervision and silent lag routing" width="100%" />
</p>

## Artifacts

- **Paper:** [`papers/silent_routing/main.pdf`](papers/silent_routing/main.pdf) ([LaTeX sources](papers/silent_routing/))
- **Full Explainer Video (56s, 1080p60):** [`animations/00_Silent_Routing_Full_Explainer.mp4`](animations/00_Silent_Routing_Full_Explainer.mp4) ([Manim source](animations/silent_routing_scenes.py))

<table>
  <tr>
    <td align="center" width="33%">
      <a href="animations/01_Outcome_Plateau.mp4"><img src="animations/previews/01_outcome_plateau.png" alt="Outcome Plateau" width="100%" /></a><br />
      <a href="animations/01_Outcome_Plateau.mp4"><strong>1. Outcome Plateau</strong></a>
    </td>
    <td align="center" width="33%">
      <a href="animations/02_Silent_Routing.mp4"><img src="animations/previews/02_silent_routing.png" alt="Silent Routing" width="100%" /></a><br />
      <a href="animations/02_Silent_Routing.mp4"><strong>2. Silent Routing</strong></a>
    </td>
    <td align="center" width="33%">
      <a href="animations/03_Stateless_Block_RMS.mp4"><img src="animations/previews/03_stateless_block_rms.png" alt="Stateless Block-RMS" width="100%" /></a><br />
      <a href="animations/03_Stateless_Block_RMS.mp4"><strong>3. Stateless Block-RMS</strong></a>
    </td>
  </tr>
</table>

## Quickstart

```bash
uv sync
uv run python handcoded/lettertrace.py task=count word_len=6 mod=2 steps=8000  # Python smoke test
make -C cpp all check                                                          # C++20 suite + grad check
cd papers/silent_routing && tectonic -X compile main.tex                       # Build PDF
```

## Layout

| Path | Contents |
|---|---|
| [`papers/silent_routing/`](papers/silent_routing/README.md) | ICML 2026 LaTeX source, figures, and `main.pdf` |
| [`animations/`](animations/README.md) | Manim source, 4 rendered 1080p60 MP4s, and README previews |
| [`cpp/`](cpp/README.md) | C++20 27-block causal interventions, Block-RMS SGD, and `(Step-ODE)` solver |
| [`handcoded/`](handcoded/README.md) | Two-block Transformer (`lettertrace.py`), exact executors, plateau checks, demo notebook |
| `src/`, `configs/`, `data/` | GPT-2L and LoRA transfer experiments (Boolean-8, register machine, BBH) |

Full reproduction commands are in [RUN.md](RUN.md).

## Citation & License

```bibtex
@inproceedings{silentrouting2026,
  title     = {Silent Routing: How Step Supervision Trains Transformers Where Outcome Supervision Stalls},
  author    = {Anonymous Authors},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

Released under the [MIT License](LICENSE).
