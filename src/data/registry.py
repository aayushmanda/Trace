from typing import Dict

from src.data.boolean_circuit_tasks import (
    BOOLEAN_CIRCUIT_BLOCK_SIZE,
    BOOLEAN_CIRCUIT_MAX_NEW_TOKENS,
    BOOLEAN_CIRCUIT_SAMPLERS,
)
from src.data.dataclass import Task
from src.data.hard_word_index_tasks import (
    HARD_WORD_INDEX_BLOCK_SIZE,
    HARD_WORD_INDEX_MAX_NEW_TOKENS,
    HARD_WORD_INDEX_SAMPLERS,
)
from src.data.local_machine_tasks import (
    GRID_WALK_BLOCK_SIZE,
    GRID_WALK_MAX_NEW_TOKENS,
    GRID_WALK_SAMPLERS,
    QUEUE_MACHINE_BLOCK_SIZE,
    QUEUE_MACHINE_MAX_NEW_TOKENS,
    QUEUE_MACHINE_SAMPLERS,
    TAPE_MACHINE_BLOCK_SIZE,
    TAPE_MACHINE_MAX_NEW_TOKENS,
    TAPE_MACHINE_SAMPLERS,
)
from src.data.sequential_tasks import (
    MODULAR_PROGRAM_SAMPLERS,
    REGISTER_MACHINE_SAMPLERS,
    STACK_MACHINE_SAMPLERS,
)
from src.data.state_machine_tasks import STATE_MACHINE_SAMPLERS
from src.data.task import (
    _sample_count_char,
    _sample_multiply,
    _sample_sort_letters,
    _sample_word_index,
)

TASKS: Dict[str, Task] = {
    "word_index": Task(
        name="word_index", block_size=128, max_new_tokens=60,
        sample=_sample_word_index, chance_acc=0.10019529082029081, ceiling_acc=1.0,
        description="report the index of a queried letter",
    ),
    "multiply": Task(
        name="multiply", block_size=64, max_new_tokens=48,
        sample=_sample_multiply, chance_acc=0.00183, ceiling_acc=1.0,
        description="two-digit multiplication via partial products",
    ),
    "count_char": Task(
        name="count_char", block_size=76, max_new_tokens=52,
        sample=_sample_count_char, chance_acc=0.23124, ceiling_acc=1.0,
        description="count occurrences of a letter (running tally)",
    ),
    "sort_letters": Task(
        name="sort_letters", block_size=160, max_new_tokens=115,
        sample=_sample_sort_letters, chance_acc=0.125, ceiling_acc=1.0,
        description="which original index lands at sorted rank k",
        answer_pattern=r"\d+",
    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(f"Unknown task {name!r}. Available: {list(TASKS)}")
    return TASKS[name]


TASKS.update({
    f"state_machine_{steps}": Task(
        name=f"state_machine_{steps}", block_size=256, max_new_tokens=128,
        sample=sampler, chance_acc=1 / 16, ceiling_acc=1.0,
        description=f"execute a random state machine for {steps} transitions",
        answer_pattern=r"\d+",
    )
    for steps, sampler in STATE_MACHINE_SAMPLERS.items()
})
TASKS.update({
    f"modular_program_{steps}": Task(
        name=f"modular_program_{steps}", block_size=256, max_new_tokens=208,
        sample=sampler, chance_acc=1 / 17, ceiling_acc=1.0,
        description=f"execute {steps} arithmetic operations modulo 17",
        answer_pattern=r"\d+",
    )
    for steps, sampler in MODULAR_PROGRAM_SAMPLERS.items()
})
TASKS.update({
    f"register_machine_{steps}": Task(
        name=f"register_machine_{steps}", block_size=192, max_new_tokens=128,
        sample=sampler, chance_acc=1 / 17, ceiling_acc=1.0,
        description=f"execute {steps} updates on two registers modulo 17",
        answer_pattern=r"\d+",
    )
    for steps, sampler in REGISTER_MACHINE_SAMPLERS.items()
})
TASKS.update({
    f"stack_machine_{steps}": Task(
        name=f"stack_machine_{steps}", block_size=384, max_new_tokens=320,
        sample=sampler, chance_acc=1 / 17, ceiling_acc=1.0,
        description=f"execute {steps} bounded push/pop operations",
        answer_pattern=r"\d+",
    )
    for steps, sampler in STACK_MACHINE_SAMPLERS.items()
})
TASKS.update({
    f"word_index_len{length}": Task(
        name=f"word_index_len{length}",
        block_size=HARD_WORD_INDEX_BLOCK_SIZE,
        max_new_tokens=HARD_WORD_INDEX_MAX_NEW_TOKENS[length],
        sample=sampler, chance_acc=1 / length, ceiling_acc=1.0,
        description=f"first index with a fixed word length of {length}",
        answer_pattern=r"\d+",
    )
    for length, sampler in HARD_WORD_INDEX_SAMPLERS.items()
})
TASKS.update({
    f"tape_machine_{steps}": Task(
        name=f"tape_machine_{steps}",
        block_size=TAPE_MACHINE_BLOCK_SIZE,
        max_new_tokens=TAPE_MACHINE_MAX_NEW_TOKENS[steps],
        sample=sampler, chance_acc=1 / 10, ceiling_acc=1.0,
        description=f"execute {steps} head-move/write ops on an 8-cell decimal tape",
        answer_pattern=r"\d+",
    )
    for steps, sampler in TAPE_MACHINE_SAMPLERS.items()
})
TASKS.update({
    f"queue_machine_{steps}": Task(
        name=f"queue_machine_{steps}",
        block_size=QUEUE_MACHINE_BLOCK_SIZE,
        max_new_tokens=QUEUE_MACHINE_MAX_NEW_TOKENS[steps],
        sample=sampler, chance_acc=1 / 17, ceiling_acc=1.0,
        description=f"execute {steps} bounded FIFO enqueue/dequeue operations",
        answer_pattern=r"\d+",
    )
    for steps, sampler in QUEUE_MACHINE_SAMPLERS.items()
})
TASKS.update({
    f"grid_walk_{steps}": Task(
        name=f"grid_walk_{steps}",
        block_size=GRID_WALK_BLOCK_SIZE,
        max_new_tokens=GRID_WALK_MAX_NEW_TOKENS[steps],
        sample=sampler, chance_acc=1 / 16, ceiling_acc=1.0,
        description=f"walk {steps} steps on a 4x4 grid with blocked cells",
        answer_pattern=r"\d+",
    )
    for steps, sampler in GRID_WALK_SAMPLERS.items()
})
TASKS.update({
    f"boolean_circuit_{depth}": Task(
        name=f"boolean_circuit_{depth}",
        block_size=BOOLEAN_CIRCUIT_BLOCK_SIZE,
        max_new_tokens=BOOLEAN_CIRCUIT_MAX_NEW_TOKENS[depth],
        sample=sampler, chance_acc=1 / 16, ceiling_acc=1.0,
        description=f"execute a reversible four-bit Boolean circuit with {depth} gates",
        answer_pattern=r"[01]{4}",
    )
    for depth, sampler in BOOLEAN_CIRCUIT_SAMPLERS.items()
})
