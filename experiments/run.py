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
    "transformer-law": (
        "src.experiments.closure_transformer_law",
        "PRIMARY GPU: symmetric vs coherent corruption on the D-block Transformer",
    ),
    "table2": (
        "src.experiments.closure_table2",
        "CPU: Cor. 2 / Table 2 pre-registered check against recovered.csv",
    ),
    "tabular-sampling": (
        "src.experiments.closure_tabular_sampling",
        "CPU: App. I 1/K vs 1/2 under uniform vs family gate sampling",
    ),
    "kernel": (
        "src.experiments.closure_kernel",
        "CPU: Thm 4 / Thm 2 / Cor. 3 in the shared kernel",
    ),
    "handcoded": (
        "src.experiments.closure_handcoded",
        "Oracle patches + init Def 20 on HandcodedOutcomeTransformer",
    ),
    "notebook-closure": (
        "src.experiments.notebook_closure_run",
        "CPU kernel + Thm 1 patches / Def 20 / C_t at notebook depth",
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
        "CPU: Cor. 5 clean process GD trajectory toward T_g",
    ),
    "credit-suppression": (
        "src.experiments.credit_suppression",
        "Thm 2 residuals, Thm 4 both halves, Cor. 2 exponent",
    ),
    "clean-convergence": (
        "src.experiments.clean_convergence",
        "Cor. 5: GD trajectory converging to T_g under clean process supervision",
    ),
    "recover-mechanism-deep": (
        "src.experiments.recover_mechanism_deep",
        "Recover Tables 2–3 and escape times from crashed-run logs",
    ),
    "oracle-alignment": (
        "src.experiments.oracle_alignment",
        "Gradient cosine against constructed oracle (documented negative)",
    ),
    "gradient-transfer": (
        "src.experiments.gradient_transfer",
        "Init-point transfer of process vs outcome gradients",
    ),
    "analyze-induced": (
        "src.experiments.analyze_induced",
        "Tables + induced_rule.pdf from archived CSVs",
    ),
    "build-executor-results": (
        "src.experiments.build_executor_results",
        "Figures 4–5 and Table 7 TeX from depth_replication",
    ),
    "build-rule-credit-table": (
        "src.experiments.build_rule_credit_table",
        "Rule-credit CSV from depth_replication checkpoints",
    ),
    "fit-credit-exponent": (
        "src.experiments.fit_credit_exponent",
        "Fit credit exponent from rule_credit_table.csv",
    ),
    "plot-paper-figures": (
        "src.experiments.plot_paper_figures",
        "Rebuild paper figures from archived CSVs",
    ),
    "plot-reliability-panels": (
        "src.experiments.plot_reliability_panels",
        "Four-panel reliability figure (RM16 + Boolean-8)",
    ),
    "summarize-alignment": (
        "src.experiments.summarize_alignment",
        "Aggregate results/oracle_alignment/summary.json",
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
