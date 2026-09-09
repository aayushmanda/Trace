# Reproducing the results

## Environment

Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Every command below is run from the repository root. `experiments/_paths.py`
handles imports, so no `PYTHONPATH` is needed.

Check that all entry points import:

```bash
for f in experiments/*.py; do
  uv run python -c "import ast; ast.parse(open('$f').read())" || echo "FAILED $f"
done
```

## What runs where

Which script supports which claim is in
**[experiments/README.md](experiments/README.md)**. This file covers only how to
run them.

## Order

The experiments are independent; there is no required order. Rough cost on one
A100, at the settings used in the manuscript:

| Stage | Command | Cost |
|---|---|---|
| Smoke test | `uv run python experiments/supervision_comparison.py --tasks boolean_circuit_4 --seeds 2001 --train-size 1000 --steps 100 --batch-size 32 --workers 0` | minutes |
| Five-condition comparison | `experiments/supervision_comparison.py` with five seeds | hours |
| Reliability sweeps | `experiments/reliability_sweep.py` per task and depth | hours per task |
| Escape-time law | `experiments/escape_time_law.py` | CPU-only, deliberately small |
| Induced-rule sweeps | `experiments/run_depth_sweep.sh`, `run_condition_trajectory.sh`, `run_trace_fraction.sh` | ~1 h per sweep, shared GPU |
| Summary tables and figure | `uv run python experiments/analyze_induced.py` | seconds |

The three induced-rule runners hard-code a CUDA device (`--device cuda:N`).
Change it before running.

## Seeds

Five random streams are controlled separately: training-pool generation,
validation-pool generation, the reliability score vector, minibatch ordering and
model initialisation. Only the last varies across the reported seeds, so quoted
standard deviations are initialisation variance and not data variance. Keep this
in mind before pooling across seeds.

## What is not reproducible as shipped

Stated plainly, because these are the gaps a reviewer will find:

1. **No checkpoints are saved.** `config.py` sets `SAVE_MODELS = False`, and the
   induced-rule scripts probe in-process. Any measurement that needs a trained
   model after the fact requires a re-run with checkpointing added.
2. **The recovered scripts have not been re-run.** `mechanism_diagnostics.py`,
   `escape_time_law.py`, `competitor_support.py`, `trace_cleaning.py`,
   `trace_vs_step_corruption.py`, `copy_probe.py`, `early_acquisition.py`,
   `loss_barrier.py` and `validate_claims.py` were recovered from git history
   after having been deleted while their outputs remained in `results/`. They
   import cleanly and their CLIs are intact, but the CSVs in `results/` were
   produced by the pre-deletion versions. Re-run before citing any number from
   them as reproduced. All 26 columns of
   `results/mechanism/mechanism_summary_*.csv` are accounted for in the
   recovered `mechanism_diagnostics.py`, so the generating code for that table
   is back in the tree.
3. **Run identity.** Archived CSVs are timestamped but not hashed against a
   commit. Record the commit alongside new runs.
