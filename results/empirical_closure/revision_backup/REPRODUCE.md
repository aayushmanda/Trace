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

Run the one unit test suite in the tree (six checks on the readouts used by
`compare_executor_rules.py`, including a finite-difference gradient check):

```bash
cd experiments && uv run python -m unittest test_executor_comparison -v && cd ..
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
| Escape-time law | `experiments/escape_time_law.py` (no flags; edit the constants at the top) | CPU-only, minutes to tens of minutes |
| Induced-rule sweeps | `experiments/run_depth_sweep.sh`, `run_condition_trajectory.sh`, `run_trace_fraction.sh` | ~1 h per sweep, shared GPU |
| Summary tables and figure | `uv run python experiments/analyze_induced.py` | seconds |
| Second executor-comparison pilot | `experiments/compare_executor_rules.py --output results/executor_comparison/reproduction --depth 4 --seed 42 --steps 2000 --train-size 10000 --test-size 256 --probe-size 64 --batch-size 128 --d-model 96 --d-ff 192 --lr 0.002 --device cpu --threads 1 --backgrounds 2 --checkpoints 0 100 500 1000 2000` | ~minutes, CPU |

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
2. **Some archived CSVs predate their current generating script.**
   `mechanism_diagnostics.py`, `escape_time_law.py`, `competitor_support.py`,
   `trace_cleaning.py`, `trace_vs_step_corruption.py`, `copy_probe.py`,
   `early_acquisition.py`, `loss_barrier.py` and `validate_claims.py` were at
   one point recovered from git history after having been deleted while their
   outputs remained in `results/`; the tracked versions now carry the `_paths`
   bootstrap and all entry points in `experiments/` import without error. The
   CSVs already in `results/` were produced before that recovery, though, so
   re-run before citing a number from them as freshly reproduced. All 26
   columns of `results/mechanism/mechanism_summary_*.csv` are accounted for in
   the current `mechanism_diagnostics.py`.
3. **Run identity.** Archived CSVs are timestamped but not hashed against a
   commit. Record the commit alongside new runs.
