# paper_revision_v2

Convention for the 10 Sep 2026 revision. Do not mix with PDF Table 6 or §7 `results/executor_comparison/depth_replication/`. Do not treat pilots as Table 1 / paper numbers.

| Subdir | Experiment |
|---|---|
| `protocols/` | Frozen confirmation YAMLs (E1 five-condition, E2 5-rate 10-seed, E3 mask-trace, E4 projected kernel) |
| `e1_five_condition/` | Table 1 family **pilot** CSV/JSON (not PDF Table 1) |
| `e2_architecture/` | `plan` protocol.json; calibration singles as pipeline proof; not 80-job confirm |
| `e3_mask_trace/` | Induced mask-trace **pilot** (skip 16×52 readout) |
| `e4_projected_kernel/` | Projected simplex CPU smoke; not a T2 proof |
| `outcome_local/` | Mechanism test: outcome vs outcome+local vs process on a deep outcome net. Mixed \(\widehat P_g\) is negative transfer, not this cell. |
| `e5_reliability/` | Logged reliability \(\rho\) (CSV + persist JSON) |
| `e4_pullback/` | Optional Transformer transfer diagnostics |
| `provenance/` | Table 1 / LoRA command logs (no new 7B download) |

Week 1: measurement plumbing + ledger. Week 2: protocol freeze + tiny pilots. Confirmation grids are recorded in YAML, not launched.

**Week 1 closed.** Pullback all-step + full-vocab (tests); architecture unstable = fail (tests); provenance dir; T1 ledger + unproved T2 (`theorem_ledger.md`, `T2_CANDIDATE.md`).

**Week 2 closed except GPU grids.** Protocols in `protocols/`. E2 `plan` (5-rate, 10-seed, 40/80 jobs) + 2 CPU singles. E1/E3 pilots under `e1_five_condition/` and `e3_mask_trace/` (not paper numbers). E4 projected CPU smoke. **Blocked:** 40-job E2 calibrate and 80-job confirm (GPU 2/3 packed). See `LEFTOVER_GPU.md`.

Ledger: `Paper/theorem_ledger.md` (copy here). T2 (unproved): `Paper/T2_CANDIDATE.md` (copy here).
