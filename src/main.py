"""One entry point for the whole pipeline.

    python main.py prepare                      # generate + cache every pool, once
    python main.py verify --train               # pre-flight checks
    python main.py run phase --task word_index  # the ratio sweep
    python main.py run family                   # the same sweep on every task
    python main.py run ablation                 # L_full vs. L_ans vs. L_direct
    python main.py run batch                    # rho_c(B), steps held fixed
    python main.py run tokens                   # rho_c(B), sample budget held fixed
    python main.py analyze ablation             # redraw tables and figures offline
    python main.py cache                        # what is on disk

`run` is resumable: finished runs are appended to results/<experiment>.jsonl and
skipped on a re-launch, and an identical run recorded by another experiment is
reused rather than retrained.
"""

from __future__ import annotations

import argparse
import sys

import config as C
from config import add_config_args, config_from_args
from data import VARIANTS, CacheMiss, describe_cache, prepare_cache
from sweeps import DEFAULT_OPTIONS, EXPERIMENTS, GridOptions, read_records, run
from tasks import TASKS


def _grid_options(experiment: str, args: argparse.Namespace) -> GridOptions:
    """CLI axes where given, per-experiment defaults everywhere else."""
    defaults = dict(DEFAULT_OPTIONS[experiment])
    tasks = args.tasks or ([args.task] if getattr(args, "task", None) else None)
    return GridOptions(
        tasks=tasks or defaults.get("tasks", C.TASKS_TO_TEST),
        ratios=args.ratios or C.RATIOS,
        seeds=args.seeds or defaults.get("seeds", C.SEEDS),
        batch_sizes=args.batch_sizes or C.BATCH_SIZES,
        conditions=args.conditions or ("A", "B", "C"),
        budget=args.budget,
        vary_order=args.vary_order,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -- prepare ------------------------------------------------------------
    prep = sub.add_parser("prepare", help="generate and cache the datasets (do this first)")
    prep.add_argument("--tasks", nargs="+", default=list(C.TASKS_TO_TEST), choices=list(TASKS))
    prep.add_argument("--variants", nargs="+", default=["full"], choices=list(VARIANTS),
                      help="full=condition A, answer_only=B, direct=C")
    prep.add_argument("--force", action="store_true", help="regenerate even if cached")
    add_config_args(prep)

    # -- verify -------------------------------------------------------------
    ver = sub.add_parser("verify", help="pre-flight checks on the task family")
    ver.add_argument("--mc-samples", type=int, default=200_000)
    ver.add_argument("--train", action="store_true", help="also run a training smoke test")
    ver.add_argument("--smoke-steps", type=int, default=300)
    ver.add_argument("--smoke-samples", type=int, default=20_000)
    ver.add_argument("--smoke-val", type=int, default=200)
    ver.add_argument("--seed", type=int, default=1337)
    add_config_args(ver)

    # -- run ----------------------------------------------------------------
    runp = sub.add_parser("run", help="execute an experiment")
    runp.add_argument("experiment", choices=sorted(EXPERIMENTS))
    runp.add_argument("--task", default=None, choices=list(TASKS))
    runp.add_argument("--tasks", nargs="+", default=None, choices=list(TASKS))
    runp.add_argument("--ratios", nargs="+", type=float, default=None)
    runp.add_argument("--seeds", nargs="+", type=int, default=None)
    runp.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    runp.add_argument("--conditions", nargs="+", default=None, choices=["A", "B", "C"])
    runp.add_argument("--budget", type=int, default=C.SAMPLE_BUDGET,
                      help="total samples seen, for the `tokens` experiment")
    runp.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    runp.add_argument("--no-analyze", action="store_true", help="skip the report afterwards")
    add_config_args(runp)

    # -- analyze ------------------------------------------------------------
    ana = sub.add_parser("analyze", help="tables and figures from a results file")
    ana.add_argument("experiment", choices=sorted(EXPERIMENTS))
    add_config_args(ana)

    # -- inspect ------------------------------------------------------------
    insp = sub.add_parser(
        "inspect", help="decode the exact examples a given training step consumes"
    )
    insp.add_argument("--task", default=C.DEFAULT_TASK, choices=list(TASKS))
    insp.add_argument("--variant", default="full", choices=list(VARIANTS))
    insp.add_argument("--ratio", type=float, default=1.0)
    insp.add_argument("--step", type=int, default=0)
    insp.add_argument("--limit", type=int, default=8)
    add_config_args(insp)

    # -- cache --------------------------------------------------------------
    cache = sub.add_parser("cache", help="list what is cached on disk")
    add_config_args(cache)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)

    if args.command == "prepare":
        prepare_cache(
            args.tasks,
            args.variants,
            cfg.data.n_samples,
            cfg.data.data_seed,
            cfg.data.cache_dir,
            val_examples=cfg.data.val_examples,
            val_seed=cfg.data.val_seed,
            order_length=(
                cfg.optim.steps * cfg.optim.batch_size
                if cfg.data.sampling == "epoch"
                else 0
            ),
            order_seed=cfg.data.order_seed,
            force=args.force,
        )
        return 0

    if args.command == "verify":
        from verify import run_checks

        ok = run_checks(
            cfg,
            mc_samples=args.mc_samples,
            do_train=args.train,
            steps=args.smoke_steps,
            samples=args.smoke_samples,
            val_examples=args.smoke_val,
            seed=args.seed,
        )
        return 0 if ok else 1

    if args.command == "run":
        from analysis import report
        from sweeps import build_specs, print_plan

        opts = _grid_options(args.experiment, args)
        if args.dry_run:
            specs = build_specs(args.experiment, cfg, opts)
            print_plan(args.experiment, cfg, opts, specs)
            for s in specs[:20]:
                print("   " + s.label())
            if len(specs) > 20:
                print(f"   ... and {len(specs) - 20} more")
            return 0

        try:
            store = run(args.experiment, cfg, opts, prepare=not args.no_prepare)
        except CacheMiss as e:
            print(f"\n[data ] {e}")
            return 1
        if not args.no_analyze:
            report(args.experiment, store.records, cfg.figures_dir)
        return 0

    if args.command == "analyze":
        import os

        from analysis import report

        path = os.path.join(cfg.results_dir, f"{args.experiment}.jsonl")
        report(args.experiment, read_records(path), cfg.figures_dir)
        return 0

    if args.command == "inspect":
        from data import decode_pool_rows, load_index_stream, load_pools, mix_pools
        from tasks import get_task

        task = get_task(args.task)
        pools = load_pools(task, args.variant, cfg.data.n_samples, cfg.data.data_seed,
                           cfg.data.cache_dir, device="cpu",
                           require_cache=cfg.data.require_cache)
        pool = (
            pools["direct"]
            if args.variant == "direct"
            else mix_pools(pools["correct"], pools["wrong"], args.ratio)
        )
        b = cfg.optim.batch_size
        if cfg.data.sampling == "epoch":
            stream = load_index_stream(
                len(pool), (args.step + 1) * b, cfg.data.order_seed, cfg.data.cache_dir,
                require_cache=cfg.data.require_cache,
            )
            idx = stream[args.step * b : (args.step + 1) * b][: args.limit].tolist()
        else:
            print("[note] sampling=iid draws indices at run time; showing rows 0..n")
            idx = list(range(args.limit))

        print(f"\nstep {args.step} of {args.task}/{args.variant} at rho={args.ratio} "
              f"(batch {b}, showing {len(idx)}) -- [ ] marks the supervised region\n")
        for row, (text, supervised) in zip(idx, decode_pool_rows(task, pool, idx)):
            head = text[: len(text) - len(supervised)]
            print(f"  row {row:>9}: {head!r} [{supervised!r}]")
        print()
        return 0

    if args.command == "cache":
        describe_cache(cfg.data.cache_dir)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
