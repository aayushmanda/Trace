# registry.py

from typing import Dict



from src.task import (
    _sample_word_index,
    _sample_multiply,
    _sample_count_char,
    # _sample_graph_path,
    _sample_sort_letters

)



from src.dataclass import Task

TASKS: Dict[str, Task] = {

    "word_index": Task(
        name="word_index",
        block_size=128,
        max_new_tokens=50,
        sample=_sample_word_index,
        chance_acc=0.15652,
        ceiling_acc=0.89259,
        description="report the index of a queried letter",

    ),

    "multiply": Task(
        name="multiply",
        block_size=64,
        max_new_tokens=48,
        sample=_sample_multiply,
        chance_acc=0.00183,
        ceiling_acc=1.0,
        description="two-digit multiplication via partial products",
    ),

    "count_char": Task(
        name="count_char",
        block_size=76,
        max_new_tokens=52,
        sample=_sample_count_char,
        chance_acc=0.23124,
        ceiling_acc=1.0,
        description="count occurrences of a letter (running tally)",
    ),

    # "graph_path": Task(
    #     name="graph_path",
    #     block_size=128,
    #     max_new_tokens=64,
    #     sample=_sample_graph_path,
    #     chance_acc=0.18,
    #     ceiling_acc=1.0,
    #     description="shortest path length in a small undirected graph",
    #     answer_pattern=r"-?\d+",
    # ),

    "sort_letters": Task(
        name="sort_letters",
        block_size=160,
        max_new_tokens=115,
        sample=_sample_sort_letters,
        chance_acc=0.125,
        ceiling_acc=1.0,
        description="which original index lands at sorted rank k",
        answer_pattern=r"\d+",

    ),
}


def get_task(name: str) -> Task:
    if name not in TASKS:
        raise KeyError(
            f"Unknown task {name!r}. "
            f"Available tasks: {list(TASKS.keys())}"
        )

    return TASKS[name]


if __name__ == "__main__":

    task = TASKS["word_index"]

    print(f"Task: {task.name}")
    print(f"Description: {task.description}")
    print(f"Chance Accuracy: {task.chance_acc}")
    print(f"Ceiling Accuracy: {task.ceiling_acc}")
    print(f"Sample Instance: {task.sample()}")