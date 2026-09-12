"""python -m src <command> ..."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.seed import add_compile_bf16_flags


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(prog="python -m src", description="Trace paper experiments")
    p.add_argument("command", nargs="?", default="help",
                   choices=["help", "supervision", "reliability", "lora"])
    args, rest = p.parse_known_args(argv)
    if args.command in {None, "help"}:
        print(__doc__)
        print("""
  python -m src supervision --config configs/experiments/e1_five_condition.yaml   # Table 1
  python -m src supervision --tasks boolean_circuit_4 --seeds 2001 --steps 100 --train-size 1000
  python -m src reliability --task boolean_circuit_8 --rhos 0.8 --seeds 2001       # Figures 1-2, Table 3
  python -m src lora --help                                                       # LoRA adaptation study

Everything else (noise-threshold, credit-suppression, clean-convergence,
prop8-frontier, fraction-vs-amount, figure builders) is
`python experiments/run.py <command>`; see experiments/README.md.
""")
        return 0
    if args.command == "supervision":
        from src.eval.supervision import main as m
        m(_supervision_ns(rest))
        return 0
    if args.command == "reliability":
        from src.eval.reliability import main as m
        m(_reliability_ns(rest))
        return 0
    if args.command == "lora":
        from src.eval.lora import main as m
        m(rest)
        return 0
    return 1


def _supervision_ns(rest):
    from src.data.datasets import TARGET_BUILDERS
    from src.training.config import load_yaml
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args(rest)
    cfg = load_yaml(pre_args.config) if pre_args.config else {}
    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--tasks", nargs="+", default=list(cfg.get("tasks") or ["boolean_circuit_4"]))
    p.add_argument("--modes", nargs="+", choices=list(TARGET_BUILDERS),
                   default=list(cfg.get("modes") or ["outcome", "process"]))
    p.add_argument("--seeds", nargs="+", type=int, default=list(cfg.get("seeds") or [2001]))
    p.add_argument("--train-size", type=int, default=int(cfg.get("train_size", 1000)))
    p.add_argument("--val-size", type=int, default=int(cfg.get("val_size", 100)))
    p.add_argument("--train-seed", type=int, default=int(cfg.get("train_seed", 501)))
    p.add_argument("--val-seed", type=int, default=int(cfg.get("val_seed", 101)))
    p.add_argument("--batch-seed", type=int, default=int(cfg.get("batch_seed", 12345)))
    p.add_argument("--batch-size", type=int, default=int(cfg.get("batch_size", 32)))
    p.add_argument("--eval-batch-size", type=int, default=int(cfg.get("eval_batch_size", 64)))
    p.add_argument("--steps", type=int, default=int(cfg.get("steps", 100)))
    p.add_argument("--lr", type=float, default=float(cfg.get("lr", 3e-4)))
    p.add_argument("--weight-decay", type=float, default=float(cfg.get("weight_decay", 0.0)))
    p.add_argument("--grad-clip", type=float, default=float(cfg.get("grad_clip", 1.0)))
    p.add_argument("--embedding", type=int, default=int(cfg.get("embedding", 128)))
    p.add_argument("--heads", type=int, default=int(cfg.get("heads", 4)))
    p.add_argument("--layers", type=int, default=int(cfg.get("layers", 2)))
    p.add_argument("--dropout", type=float, default=float(cfg.get("dropout", 0.0)))
    p.add_argument("--workers", type=int, default=int(cfg.get("workers", 0)))
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=Path(cfg["output"]) if cfg.get("output") else None)
    add_compile_bf16_flags(p, cfg, from_yaml=True)
    return p.parse_args(rest)


def _reliability_ns(rest):
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="boolean_circuit_8")
    p.add_argument("--rhos", nargs="+", type=float, default=[0.8])
    p.add_argument("--seeds", nargs="+", type=int, default=[2001])
    p.add_argument("--checkpoints", nargs="+", type=int, default=[1000])
    p.add_argument("--train-size", type=int, default=20000)
    p.add_argument("--val-size", type=int, default=500)
    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--ratio-seed", type=int, default=777)
    p.add_argument("--batch-seed", type=int, default=12345)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--include-outcome", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--output", type=Path, default=None)
    add_compile_bf16_flags(p, from_yaml=True)
    return p.parse_args(rest)


if __name__ == "__main__":
    raise SystemExit(main())
