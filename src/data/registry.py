"""Tasks the paper trains in the character-token Transformer.

Boolean circuits, the register machine, and the state machine, including
the one-wrong-rule variants of the first two. Counting, shell, and lights
live in handcoded/lettertrace.py.
"""
from src.data.boolean_circuit_tasks import (
    BOOLEAN_CIRCUIT_BLOCK_SIZE,
    BOOLEAN_CIRCUIT_MAX_NEW_TOKENS,
    BOOLEAN_CIRCUIT_SAMPLERS,
    make_boolean_circuit_sampler_coherent,
)
from src.data.dataclass import Task
from src.data.sequential_tasks import (
    MODULUS,
    REGISTER_MACHINE_SAMPLERS,
    make_register_machine_sampler_coherent,
)
from src.data.state_machine_tasks import STATE_MACHINE_SAMPLERS

TASKS: dict[str, Task] = {
    f"boolean_circuit_{depth}": Task(
        name=f"boolean_circuit_{depth}",
        block_size=BOOLEAN_CIRCUIT_BLOCK_SIZE,
        max_new_tokens=BOOLEAN_CIRCUIT_MAX_NEW_TOKENS[depth],
        sample=sampler, chance_acc=1 / 16, ceiling_acc=1.0,
        description=f"execute a reversible four-bit Boolean circuit with {depth} gates",
        answer_pattern=r"[01]{4}",
    )
    for depth, sampler in BOOLEAN_CIRCUIT_SAMPLERS.items()
}
TASKS["boolean_circuit_8_coherent"] = Task(
    name="boolean_circuit_8_coherent",
    block_size=BOOLEAN_CIRCUIT_BLOCK_SIZE,
    max_new_tokens=BOOLEAN_CIRCUIT_MAX_NEW_TOKENS[8],
    sample=make_boolean_circuit_sampler_coherent(8), chance_acc=1 / 16, ceiling_acc=1.0,
    description="depth-8 Boolean circuit, corrupted traces follow one fixed wrong permutation per gate",
    answer_pattern=r"[01]{4}",
)
TASKS.update({
    f"register_machine_{steps}": Task(
        name=f"register_machine_{steps}", block_size=192, max_new_tokens=128,
        sample=sampler, chance_acc=1 / MODULUS, ceiling_acc=1.0,
        description=f"execute {steps} updates on two registers modulo 17",
        answer_pattern=r"\d+",
    )
    for steps, sampler in REGISTER_MACHINE_SAMPLERS.items()
})
TASKS["register_machine_16_coherent"] = Task(
    name="register_machine_16_coherent", block_size=192, max_new_tokens=128,
    sample=make_register_machine_sampler_coherent(16), chance_acc=1 / MODULUS, ceiling_acc=1.0,
    description="16-step register machine, corrupted traces follow one fixed wrong permutation per instruction",
    answer_pattern=r"\d+",
)
TASKS.update({
    f"state_machine_{steps}": Task(
        name=f"state_machine_{steps}", block_size=256, max_new_tokens=128,
        sample=sampler, chance_acc=1 / 16, ceiling_acc=1.0,
        description=f"execute a random state machine for {steps} transitions",
        answer_pattern=r"\d+",
    )
    for steps, sampler in STATE_MACHINE_SAMPLERS.items()
})


def get_task(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        known = ", ".join(sorted(TASKS))
        raise KeyError(f"unknown task {name!r}; known: {known}") from None
