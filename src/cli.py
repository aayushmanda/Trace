"""Paper + diagnostic entrypoints: python -m src <command>."""
from argparse import Namespace
from pathlib import Path

from src.training.config import load_experiment
from src.training.seed import default_device

ROOT = Path(__file__).resolve().parents[1]


def _fill(args, cfg, **defaults):
    train = cfg.get("train") or {}
    model = cfg.get("model") or {}
    for key, value in {**train, **model, **{k: v for k, v in cfg.items() if not isinstance(v, dict)}}.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    for key, value in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)
    if getattr(args, "device", None) is None:
        args.device = default_device()
    args.workers = getattr(args, "workers", 0) or 0
    args.batch_seed = getattr(args, "batch_seed", 12345)
    args.weight_decay = getattr(args, "weight_decay", 0.0)
    args.grad_clip = getattr(args, "grad_clip", 1.0)
    args.dropout = getattr(args, "dropout", 0.0)
    from src.training.seed import configure_device
    configure_device(args.device, compile=getattr(args, "compile", None), bf16=getattr(args, "bf16", None))
    return args


def run_induced(ns):
    from src.eval.induced_rule import run
    args, cfg = load_experiment(ns.config, ns)
    args.condition = args.condition or "process"
    args.trace_fraction = getattr(args, "trace_fraction", None) or 0.5
    args.probe_step = getattr(args, "probe_step", None) or 1
    _fill(args, cfg, depth=4, seed=2001, train_size=30000, val_size=800, probe_size=300,
          batch_size=128, lr=3e-4, n_embd=128, n_head=4, n_layer=4,
          checkpoints=[0, 500, 2000, 4000], out="results/revision/induced_rule.csv")
    if getattr(args, "probe_steps", None) is None:
        d = int(args.depth)
        args.probe_steps = [1, max(1, d // 2), d]
    if not getattr(args, "with_pullback", False):
        args.with_pullback = bool(cfg.get("with_pullback", False))
    run(args)


def run_pullback(ns):
    from src.eval.pullback import run
    ns.workers = 0
    ns.batch_seed = 12345
    ns.weight_decay = 0.0
    if ns.device is None:
        ns.device = default_device()
    run(ns)


def run_length(ns):
    from src.eval.paper_runs import run_length
    args, cfg = load_experiment(ns.config, ns)
    _fill(args, cfg, train_depth=8, eval_depths=[8, 10, 12, 16],
          modes=["outcome", "process"], seeds=[2001, 2002, 2003],
          train_size=20000, val_size=500, steps=4000, batch_size=128, lr=3e-4,
          embedding=128, heads=4, layers=2, eval_batch_size=128)
    args.out = Path(args.out or cfg.get("out", ROOT / "results/revision/length_generalization.csv"))
    args.ckpt_dir = Path(getattr(args, "ckpt_dir", None) or cfg.get("ckpt_dir", ROOT / "results/revision/length_ckpts"))
    run_length(args)


def run_margins(ns):
    from src.eval.paper_runs import run_margins
    args, cfg = load_experiment(ns.config, ns)
    _fill(args, cfg, task="boolean_circuit_8", rho=0.80, seeds=[2001, 2002, 2003],
          train_size=20000, val_size=500, steps=4000, batch_size=128, lr=3e-4,
          embedding=128, heads=4, layers=2, eval_batch_size=128)
    args.out = Path(args.out or cfg.get("out", ROOT / "results/revision/mmin_histograms.csv"))
    run_margins(args)


def run_architecture(ns):
    from src.eval.architecture_controls import main as arch_main
    arch_main(ns.argv)


def run_executor(ns):
    from src.eval.executor_comparison import main as ex_main
    ex_main(ns.argv)


def run_analyze(ns):
    from src.eval.analyze_revision import main
    main()


def run_supervision(ns):
    from src.eval.supervision import main
    main(ns)


def run_reliability(ns):
    from src.eval.reliability import main
    main(ns)
