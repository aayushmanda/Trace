"""Single public CLI for paper figure builders and closure checks.

Invoke as `python experiments/run.py <command> ...` (not as an executable).
Implementations live in `src/experiments/`. Training entry points that already
have `python -m src <command>` stay there.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# command -> (module, one-line help)
COMMANDS = {
    "tabular-sampling": (
        "src.experiments.closure_tabular_sampling",
        "CPU: App. I 1/K vs 1/2 under uniform vs family gate sampling",
    ),
    "noise-threshold": (
        "src.experiments.noise_threshold",
        "App. I tabular reliability threshold (Fig. 3, Tables 5–6)",
    ),
    "family-sampling": (
        "src.experiments.noise_threshold_family_sampling",
        "App. I frontier under family-first gate sampling",
    ),
    "clean-convergence": (
        "src.experiments.clean_convergence",
        "Cor. 5: GD trajectory converging to T_g under clean process supervision",
    ),
    "credit-suppression": (
        "src.experiments.credit_suppression",
        "Thm 2 residuals, Thm 4 both halves, Cor. 2 exponent",
    ),
    "prop8-frontier": (
        "src.experiments.prop8_frontier",
        "Reliability-frontier diagnostic: measured vs. predicted rho_c(a,D,beta)",
    ),
    "fraction-vs-amount": (
        "src.experiments.fraction_vs_amount",
        "Fixed-rho, varying-N grid separating fraction from corpus size",
    ),
    "plot-paper-figures": (
        "src.experiments.plot_paper_figures",
        "Rebuild paper figures from archived CSVs",
    ),
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python experiments/run.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )
    p.add_argument(
        "command",
        nargs="?",
        choices=list(COMMANDS),
        help="subcommand; omit with --help to list them",
    )
    p.add_argument("-h", "--help", action="store_true")
    p.epilog = "commands:\n" + "\n".join(
        f"  {name:<24} {help_}" for name, (_, help_) in COMMANDS.items()
    )
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = _parser()
    args, rest = p.parse_known_args(argv)
    if args.command is None:
        p.print_help()
        return 0
    if args.help:
        rest = ["--help", *rest]
    mod_name, _ = COMMANDS[args.command]
    mod = importlib.import_module(mod_name)
    result = mod.main(rest)
    return 0 if result is None else int(result)


if __name__ == "__main__":
    raise SystemExit(main())
