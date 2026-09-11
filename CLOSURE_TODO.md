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
