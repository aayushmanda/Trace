Yes. I would implement them as four closure experiments, in this order:

1. Direct test of Proposition 8 — highest priority.
   For fixed \(K=16\), choose several depths \(D\in\{2,4,8\}\), competence levels \(a\), and answer weights \(\beta\). Construct

$$
P_g(a)=U+a(T_g-U)
$$

exactly, then vary \(\rho\). For every point compute the actual combined-objective derivative

$$
J'_{\rho,\beta}(a)
$$

and check whether its sign flips exactly at your predicted

$$
\rho_c(a,D,\beta).
$$

Plot predicted boundary vs measured boundary. This directly validates the paper's new reliability-frontier result. 

2. Fixed-\(\rho\), varying-\(N\) — fraction versus amount.
   Pick several fixed reliabilities, e.g.

$$
\rho\in\{0.08,0.10,0.15,0.20,0.30\},
$$

and independently vary

$$
N\in\{500,1000,2000,5000,10000,20000,50000\}.
$$

For each pair measure rule-cell recovery and answer accuracy. At fixed \(\rho>1/K\), increasing \(N\) should approach the same population solution. This is the experiment that actually separates “fraction correct” from “number of examples.” Your current finite-sample study shows the gap but does not fully isolate these two variables. 

3. LoRA rerun — supporting evidence only.
   Let the current SmolLM2-135M sweep finish over all \(\rho\) and seeds, save the raw CSVs/logs/checkpoints, and regenerate the reported aggregates from those artifacts. Do not add new LoRA variants. Its purpose is simply:

$$
\text{Does noisy-process robustness/local-vs-rollout behavior survive pretrained adaptation?}
$$

The current manuscript still notes that the old LoRA aggregates lacked original logs. 

4. Rerun Table 1 — provenance closure.
   Run the five supervision conditions:

$$
\{\text{outcome, answer-first, filler, process, corrupted}\}
$$

with exactly the paper settings and five seeds. Save one CSV containing every seed rather than only means/stds. This is not a new scientific experiment; it simply makes Table 1 reproducible because the current manuscript says its original run logs are unavailable. 

Priority-wise:

$$
\boxed{
\text{Prop. 8 validation}
>
(\rho,N)\text{ grid}
>
\text{Table 1 rerun}
>
\text{LoRA}
}
$$

The first two improve the actual scientific claim. The last two mainly close reproducibility.

---

## Status as of 2026-09-12 (narrowed scope: theory fixes + 3 experiments, then stop)

Done directly (no training needed):
- Appendix A.1 embedding: replaced the ad hoc "bilinear table block" with a
  standard KM-unit ReLU hidden layer (reusing Theorem 9's own lookup unit
  unchanged) followed by an ordinary linear readout whose weights are a
  reshaping of A. No non-standard primitive remains.
- Proposition 8: added the explicit hypothesis `0<=beta<2/(K-1)` (proved
  sufficient for the K-beta*C(a,D)>0 denominator condition; the paper's own
  beta=1/D falls outside this simple sufficient range for small D but was
  checked numerically to satisfy the real condition regardless, with margin
  >=8.48 at K=16 for every D from 2 to 64). Strengthened the "directional/
  stalling frontier along one fixed path, not a claim about arbitrary SGD"
  framing right in the proposition statement, not just in the remark.
- Verified the O(e^-C) softmax-leakage claim numerically (fitted slope
  -1.00 against C for two competitor counts) and cited this in Appendix
  A.1's proof; did not claim this verifies the full composed bounds.
- LoRA rerun (item 5): the background SmolLM2-135M sweep from earlier this
  session had already finished (27/27 rows: outcome, answer_first, rho in
  {0,0.2,0.4,0.5,0.6,0.8,1.0}, x3 seeds). Recomputed the aggregates directly
  from `results/paper/smollm/smollm_trace.csv` and updated body.tex /
  appendix_results.tex to the final numbers (89.67% answer accuracy /
  89.78% exact rollout at rho=1.00; 82.49% local state / 23.78% exact
  rollout at rho=0.50 -- previously 92.11/91.44/83.75/24.44 from an earlier,
  less-complete run). Raw per-seed artifacts already archived under
  `results/paper/smollm/artifacts/`.
- Fixed-rho/varying-N (item 6): left untouched, as instructed.

Launched in background, not yet written into the paper (check
`results/paper/gpu2_progress.log` and `results/paper/gpu3_progress.log` for
completion; each stage also has its own `.log`):

- **Table 1 rerun** (item 4): `python -m src supervision --config
  configs/experiments/e1_five_condition_rerun.yaml` -- exactly Table 1's
  existing settings (boolean_circuit_8, 5 conditions, seeds 2001-2005,
  8000 steps, train 20000 / val 1000, batch 128, embed 128/4 heads/2
  layers), promoted unchanged from e1_five_condition.yaml's `confirmation:`
  block. Output: `results/paper/e1_five_condition_rerun/table1_rerun.csv`
  (per-seed rows) + `..._persist.json` (full config + summary). Running on
  CUDA_VISIBLE_DEVICES=2, queued before the D=2 reliability run below.
- **Depth x reliability** (item 3): `python -m src reliability --task
  boolean_circuit_{D} --rhos 0.5 0.7 0.8 0.9 1.0 --seeds 2001 2002 2003
  --checkpoints 8000` for D in {2,4,8} (same architecture/budget as the
  existing D=8 sweep behind Table 2/Figure 1: defaults train_size=20000,
  val_size=500, batch_size=128). Outputs:
  `results/paper/depth_reliability/boolean_circuit_{2,4,8}.csv`. D=2 queued
  on GPU 2 after the Table 1 rerun; D=4 then D=8 running sequentially on
  GPU 3. Rough estimate ~5-7 hours wall-clock total across both GPUs.

Next, once these finish: write the depth x reliability results into a new
short subsection/figure testing whether the measured frontier moves with D
the way `prop:outcome-rescue`'s stalling threshold predicts, and update
Table 1's caption to drop the "original run logs are unavailable" line
once `table1_rerun.csv` is in hand (reconciling against the existing
86.35/9.27/etc. numbers already flagged as an open discrepancy in body.tex).
Per the instruction that produced this batch: stop adding scope after
these are written up.
