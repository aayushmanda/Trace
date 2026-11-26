# `animations/` — Explainer Videos

1080p60 Manim animations for *Silent Routing*:

- [`00_full_explainer.mp4`](00_full_explainer.mp4) — full 56s walkthrough
- [`01_outcome_plateau.mp4`](01_outcome_plateau.mp4) — `Scene1_OutcomePlateau` (Thm. 1)
- [`02_silent_routing.mp4`](02_silent_routing.mp4) — `Scene2_SilentRouting` (Thm. 2)
- [`03_stateless_block_rms.mp4`](03_stateless_block_rms.mp4) — `Scene3_StatelessBlockRMS` (Thm. 3)
- `previews/` — PNG thumbnails embedded in the root README

Re-render from `scenes.py` (requires Manim, LaTeX, `ffmpeg`):

```bash
manim -pqh scenes.py Scene1_OutcomePlateau Scene2_SilentRouting Scene3_StatelessBlockRMS
ffmpeg -y -f concat -safe 0 -i concat_list.txt -c copy 00_full_explainer.mp4
```
