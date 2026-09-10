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
                   choices=["help", "induced", "pullback", "length", "margins", "architecture",
                            "executor", "executor-depths", "analyze", "supervision", "reliability", "smoke"])
    args, rest = p.parse_known_args(argv)
    if args.command in {None, "help"}:
        print(__doc__)
        print("commands:", ", ".join(p._option_string_actions and []))
        print("""
GPT stack
  python -m src induced --config configs/experiments/induced_rule.yaml --depth 2 --condition process --seed 2001
  python -m src pullback --depth 4 --seed 2001
  python -m src length --config configs/experiments/length_generalization.yaml
  python -m src margins --config configs/experiments/margin_histograms.yaml
  python -m src supervision --tasks boolean_circuit_4 --seeds 2001 --steps 100 --train-size 1000
  python -m src reliability --task boolean_circuit_8 --rhos 0.8 --seeds 2001
  python -m src analyze
  python -m src smoke

Handcoded / semantic-token stack
  python -m src architecture plan|calibrate|confirm|summarize --config configs/experiments/architecture_controls.yaml
  python -m src executor --config configs/experiments/executor_comparison.yaml
  python -m src executor-depths --config configs/experiments/executor_depths.yaml
""")
        return 0
    if args.command == "architecture":
        from src.eval.architecture_controls import main as m
        sys.argv = ["architecture", *rest]
        m()
        return 0
    if args.command == "executor":
        from src.eval.executor_comparison import main as m
        sys.argv = ["executor", *rest]
        m()
        return 0
    if args.command == "executor-depths":
        from src.eval.executor_depths import main as m
        sys.argv = ["executor-depths", *rest]
        m()
        return 0
    if args.command == "induced":
        from src.cli import run_induced
        ns = _induced_ns(rest)
        run_induced(ns)
        return 0
    if args.command == "pullback":
        from src.cli import run_pullback
        ns = _pullback_ns(rest)
        run_pullback(ns)
        return 0
    if args.command == "length":
        from src.cli import run_length
        ns = _length_ns(rest)
        run_length(ns)
        return 0
    if args.command == "margins":
        from src.cli import run_margins
        ns = _margins_ns(rest)
        run_margins(ns)
        return 0
    if args.command == "analyze":
        from src.eval.analyze_revision import main as m
        m()
        return 0
    if args.command == "supervision":
        from src.eval.supervision import main as m
        m(_supervision_ns(rest))
        return 0
    if args.command == "reliability":
        from src.eval.reliability import main as m
        m(_reliability_ns(rest))
        return 0
    if args.command == "smoke":
        from src.cli import run_induced
        ns = _induced_ns(["--config", "configs/experiments/smoke.yaml", "--depth", "2",
                          "--condition", "process", "--seed", "2001"])
        run_induced(ns)
        return 0
    return 1


def _induced_ns(rest):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/induced_rule.yaml")
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--condition", default=None, choices=["both", "outcome", "process"])
    p.add_argument("--trace-fraction", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--probe-size", type=int, default=None)
    p.add_argument("--probe-step", type=int, default=1)
    p.add_argument("--probe-steps", type=int, nargs="+", default=None)
    p.add_argument("--with-pullback", action="store_true")
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-embd", type=int, default=None)
    p.add_argument("--n-head", type=int, default=None)
    p.add_argument("--n-layer", type=int, default=None)
    p.add_argument("--checkpoints", type=int, nargs="+", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--dump-tables", default=None)
    add_compile_bf16_flags(p, from_yaml=False)
    return p.parse_args(rest)


def _pullback_ns(rest):
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--train-size", type=int, default=60000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--probe-size", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--checkpoints", type=int, nargs="+", default=[0, 500, 2000, 4000])
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="results/pullback.csv")
    add_compile_bf16_flags(p, from_yaml=False)
    return p.parse_args(rest)


def _length_ns(rest):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/length_generalization.yaml")
    p.add_argument("--train-depth", type=int, default=None)
    p.add_argument("--eval-depths", nargs="+", type=int, default=None)
    p.add_argument("--modes", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch-seed", type=int, default=12345)
    p.add_argument("--device", default=None)
    p.add_argument("--ckpt-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    add_compile_bf16_flags(p, from_yaml=False)
    return p.parse_args(rest)


def _margins_ns(rest):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiments/margin_histograms.yaml")
    p.add_argument("--task", default=None)
    p.add_argument("--rho", type=float, default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    p.add_argument("--train-size", type=int, default=None)
    p.add_argument("--val-size", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding", type=int, default=None)
    p.add_argument("--heads", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--batch-seed", type=int, default=12345)
    p.add_argument("--device", default=None)
    p.add_argument("--out", type=Path, default=None)
    add_compile_bf16_flags(p, from_yaml=False)
    return p.parse_args(rest)


def _supervision_ns(rest):
    from src.data.datasets import TARGET_BUILDERS
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", default=["boolean_circuit_4"])
    p.add_argument("--modes", nargs="+", choices=list(TARGET_BUILDERS), default=["outcome", "process"])
    p.add_argument("--seeds", nargs="+", type=int, default=[2001])
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--val-size", type=int, default=100)
    p.add_argument("--train-seed", type=int, default=501)
    p.add_argument("--val-seed", type=int, default=101)
    p.add_argument("--batch-seed", type=int, default=12345)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--embedding", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default=None)
    add_compile_bf16_flags(p, from_yaml=True)
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
    p.add_argument("--output", type=Path, default=None)
    add_compile_bf16_flags(p, from_yaml=True)
    return p.parse_args(rest)


if __name__ == "__main__":
    raise SystemExit(main())
