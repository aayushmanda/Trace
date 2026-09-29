"""python -m src supervision | reliability | lora"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELP = """
  python -m src supervision --config configs/experiments/e1_five_condition.yaml
  python -m src reliability --task boolean_circuit_8 --rhos 0.8 --seeds 2001
  python -m src lora --config configs/experiments/lora_transfer.yaml
  python -m src align --help
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else "help"
    rest = argv[1:]
    if cmd in {"help", "-h", "--help"}:
        print(HELP)
        return 0
    if cmd == "supervision":
        from src.runs import supervision_main as run
        run(rest)
        return 0
    if cmd == "reliability":
        from src.runs import reliability_main as run
        run(rest)
        return 0
    if cmd == "lora":
        from src.lora import main as run
        run(rest)
        return 0
    if cmd == "align":
        from src.runs import align_main as run
        run(rest)
        return 0
    raise SystemExit(f"unknown command {cmd!r}. {HELP}")


if __name__ == "__main__":
    raise SystemExit(main())
