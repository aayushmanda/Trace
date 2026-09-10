# Reproducing the results

## Environment

Python 3.12+ and [uv](https://docs.astral.sh/uv/). Command-first paper runs: **[RUN.md](RUN.md)**.

```bash
uv sync
```

From the repository root:

```bash
python -m src help
TRACE_TQDM=0 python -m unittest discover -s tests -v
```

## Order

See **[RUN.md](RUN.md)** for the paper §7 handcoded study first, then the GPT revision extras. Claim map: **[experiments/README.md](experiments/README.md)**.

| Stage | Command |
|---|---|
| Smoke | `python -m src smoke` and `python -m src escape --smoke` |
| Table 1 | `python -m src supervision` |
| Figs 1–2 | `python -m src reliability` |
| §7 executor | `python -m src executor-depths` then `python experiments/build_executor_results.py` |
| Induced / pullback / split-verdict | `python -m src induced`, `pullback`, `split-verdict` |
| Shared kernel | `python -m src escape` |
| Figures | `python experiments/analyze_induced.py` and `python -m src analyze` |

The induced-rule shell launchers (`experiments/run_depth_sweep.sh` and siblings) hard-code a CUDA device. Change it before running.
