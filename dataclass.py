from dataclasses import dataclass
import random
from typing import Callable, Optional
from tokenizer import CharTokenizer
import hashlib
import re
import string

ANSWER_SEP = " : "

DATASET_SEED = 12345
VAL_SEED = 9999

DATASET_VERSION = "1.0.0"

GLOBAL_CHARS = string.ascii_lowercase + string.digits + " ;:->+=*\n"
GLOBAL_TOKENIZER = CharTokenizer(GLOBAL_CHARS)


@dataclass(frozen=True)
class Instance:
    prompt: str
    correct_trace: str
    wrong_trace: str
    gold: str


@dataclass
class Task:
    name: str
    block_size: int
    max_new_tokens: int
    sample: Callable[[], Instance]
    chance_acc: float
    ceiling_acc: float
    description: str = ""
    answer_pattern: str = r"-?\d+"
    bayes_prob: Optional[Callable[[Instance], float]] = None

    @property
    def tokenizer(self) -> CharTokenizer:
        return GLOBAL_TOKENIZER

    def render(self, inst: Instance, mode: str = "correct_think"):
        if mode == "correct_think":
            return inst.prompt, f" {inst.correct_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "wrong_think":
            return inst.prompt, f" {inst.wrong_trace}{ANSWER_SEP}{inst.gold}"
        if mode == "no_think":
            return inst.prompt, f" {inst.gold}"
        raise ValueError(f"unknown mode {mode}")

    def context(self, inst: Instance, mode: str) -> str:
        if mode == "free":
            return inst.prompt
        if mode == "forced_correct":
            return f"{inst.prompt} {inst.correct_trace}{ANSWER_SEP}"
        if mode == "forced_wrong":
            return f"{inst.prompt} {inst.wrong_trace}{ANSWER_SEP}"
        if mode == "direct":
            return f"{inst.prompt} "
        raise ValueError(f"unknown eval mode {mode}")

    def extract_answer(self, generated_tail: str, first: bool = False) -> Optional[str]:
        tail = generated_tail.split("\n")[0]
        segment = tail.split(":")[-1] if ":" in tail else tail
        found = re.findall(self.answer_pattern, segment)
        if not found:
            return None
        return found[0] if first else found[-1]

    def fingerprint(self, n: int = 64) -> str:
        state = random.getstate()
        random.seed(12345)
        h = hashlib.sha1()
        h.update(DATASET_VERSION.encode())
        h.update(self.name.encode())
        h.update(f"{DATASET_SEED}|{VAL_SEED}".encode())
        for _ in range(n):
            inst = self.sample()
            for mode in ("correct_think", "wrong_think"):
                prompt, answer = self.render(inst, mode)
                h.update(f"{prompt}{answer}\n".encode())
        random.setstate(state)
        h.update(f"{self.block_size}|{''.join(self.tokenizer.chars)}".encode())
        return h.hexdigest()[:10]

if __name__ == "__main__":
    # Example usage
    task = Task(
        name="example_task",
        block_size=128,
        max_new_tokens=32,
        sample=lambda: Instance(
            prompt="What is 2 + 2?",
            correct_trace="2 + 2 = 4",
            wrong_trace="2 + 2 = 5",
            gold="4"
        ),
        chance_acc=0.5,
        ceiling_acc=1.0,
        description="An example task for demonstration purposes."
    )
    print(f"Task fingerprint: {task.fingerprint()}")