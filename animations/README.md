# Explainer Animations

1080p60 Manim animations (with native Computer Modern LaTeX) illustrating the three core mechanisms of *Silent Routing:
How Step Supervision Trains Transformers Where Outcome Supervision Stalls*.

## Rendered Videos

| File | Duration | Scene Class | Topic |
|---|---:|---|---|
| [`00_Silent_Routing_Full_Explainer.mp4`](00_Silent_Routing_Full_Explainer.mp4) | 56s | *(all three concatenated)* | Complete three-part visual walkthrough |
| [`01_Outcome_Plateau.mp4`](01_Outcome_Plateau.mp4) | 20s | `Scene1_OutcomePlateau` | Parity chain, shared expressivity (Prop. 1), and the $\Omega(\lambda^{-n})$ outcome energy barrier (Thm. 1) |
| [`02_Silent_Routing.mp4`](02_Silent_Routing.mp4) | 22s | `Scene2_SilentRouting` | Baseline calibration $c^\star$, distractor cancellation $E_{\mathrm{dist}}(c^\star)=0$, and silent lag-$n$ routing at flat loss $H(p)$ (Thm. 2) |
| [`03_Stateless_Block_RMS.mp4`](03_Stateless_Block_RMS.mp4) | 15s | `Scene3_StatelessBlockRMS` | Hyperbolic $(q,k)$ query–key saddle and $\Theta(\varepsilon^{-5}) \to \Theta(\varepsilon^{-2})$ escape via zero-state Block-RMS SGD (Thm. 3) |

Static preview PNGs embedded in the root README live in [`previews/`](previews/).

## Re-rendering from Source

Requires Manim Community (`requirements.txt`), a LaTeX installation, and `ffmpeg`:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

manim -pqh silent_routing_scenes.py Scene1_OutcomePlateau Scene2_SilentRouting Scene3_StatelessBlockRMS

# Concatenate the three rendered scenes using concat_list.txt
ffmpeg -y -f concat -safe 0 -i concat_list.txt -c copy 00_Silent_Routing_Full_Explainer.mp4
```
