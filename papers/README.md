# Papers

Self-contained LaTeX source for the paper, compiling cleanly with [tectonic](https://tectonic-typesetting.github.io/):

```bash
cd papers/silent_routing && tectonic -X compile main.tex
```

| Folder | Paper | Venue / Style | Compiled PDF |
|---|---|---|---|
| `silent_routing/` | *Silent Routing: How Step Supervision Trains Transformers Where Outcome Supervision Stalls* | ICML 2026 (`icml2026`) | [`silent_routing/main.pdf`](silent_routing/main.pdf) |

Experiment code and reproduction commands live at the repository root in [`cpp/`](../cpp/README.md),
[`handcoded/`](../handcoded/), and [`src/`](../src/) (see [`RUN.md`](../RUN.md)). Explainer animations live in
[`animations/`](../animations/README.md).
