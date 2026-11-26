# `animations/` — Explainer Videos

1080p60 Manim animations for *Silent Routing*:

- [`00_Silent_Routing_Full_Explainer.mp4`](00_Silent_Routing_Full_Explainer.mp4) — full 56s walkthrough
- [`01_Outcome_Plateau.mp4`](01_Outcome_Plateau.mp4) — `Scene1_OutcomePlateau` (Thm. 1)
- [`02_Silent_Routing.mp4`](02_Silent_Routing.mp4) — `Scene2_SilentRouting` (Thm. 2)
- [`03_Stateless_Block_RMS.mp4`](03_Stateless_Block_RMS.mp4) — `Scene3_StatelessBlockRMS` (Thm. 3)
- `previews/` — PNG thumbnails embedded in the root README

Re-render from `silent_routing_scenes.py` (requires Manim, LaTeX, `ffmpeg`):

```bash
manim -pqh silent_routing_scenes.py Scene1_OutcomePlateau Scene2_SilentRouting Scene3_StatelessBlockRMS
ffmpeg -y -f concat -safe 0 -i concat_list.txt -c copy 00_Silent_Routing_Full_Explainer.mp4
```
