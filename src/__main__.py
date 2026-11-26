"""python -m src supervision | lora | shuffled"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

HELP = """
  python -m src supervision --config configs/experiments/boolean8_gpt.yaml
  python -m src supervision --config configs/experiments/register16_gpt.yaml
  python -m src lora --config configs/experiments/lora_transfer.yaml
  python -m src shuffled train --K 3 --condition process --seed 2001 --out results/shuffled/bbh/bbh_K3_process.json
  python -m src shuffled zeroshot --out results/shuffled/zeroshot_qwen05.json
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
    elif cmd == "lora":
        from src.lora import main as run
    elif cmd == "shuffled":
        from src.shuffled import main as run
    else:
        raise SystemExit(f"unknown command {cmd!r}. {HELP}")
    run(rest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
