# Frozen confirmation protocols (week 2)

Copies of `configs/experiments/e{1,2,3,4}_*.yaml` as of the week-2 freeze. Edit the configs/ copies if the protocol changes, then refresh these files.

| File | Claim it can support | Runnable this week | Not this week |
|---|---|---|---|
| `e1_five_condition.yaml` | Table 1 five-condition | Pilot keys (1 seed, 100 steps) | 5-seed × 8000-step confirm; interchange 75-run grid |
| `e2_architecture.yaml` | Architecture × format, 5-rate 10-seed | `plan`; 1–2 `single` cells if GPU packed | 40-job calibrate grid; 80-job confirm |
| `e3_mask_trace.yaml` | Continuation-masked traces; per-step \(\widehat P^{(t)}\) | Pilot with `skip_readout: true` | Full readout × depth grid |
| `e4_projected_kernel.yaml` | Shared-kernel projected simplex flow | CPU smoke (`python -m src projected --smoke`) | T2 proof; row-softmax `escape` as if it were T2 |

`python -m src escape` remains the **row-softmax logit** runner (`P_g=\mathrm{softmax}(A_g)`). It is not this E4 protocol.
