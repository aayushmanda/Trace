"""Semantic-token Boolean-circuit executors (tutorial + paper stack)."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from handcoded import config
from handcoded.config import configure, current, reset, load_config, make_checkpoints
from handcoded.gates import (
    Circuit, apply_gate, bits, gate_families, make_circuits, make_gate_names, phi,
    sample_example, sample_gate, state_text,
)

# Hyperparameters and GATES are re-exported LAZILY: `configure()` rebinds them, and a
# plain `from handcoded import N_BITS` would otherwise freeze the value at import time.
_LIVE = {
    "BATCH_SEED", "DATA_SEED", "D_FF", "D_MODEL", "DEPTH", "LR", "MODEL_SEED", "N_BITS",
    "N_HEADS", "N_STATES", "STEPS", "TEST_SEED", "TEST_SIZE", "TRAIN_SIZE", "BATCH_SIZE",
}


def __getattr__(name):
    if name in _LIVE:
        return getattr(config, name)
    if name == "GATES":
        from handcoded.gates import make_gate_names
        return make_gate_names()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
from handcoded.tokenizer import CircuitTokenizer, make_tokenizer
from handcoded.data import LanguageBatch, encode_dataset, language_model_loss, make_batch_schedule
from handcoded.models import (
    FixedAttentionHead, LearnedOneLayerTransformer, UnifiedBlock, UnifiedExecutor,
    attach_local_heads, build_fixed_outcome, build_fixed_process, build_random_learned_model,
    build_random_trainable_unified, make_random_trainable_copy,
    unified_block_grad_norms, unified_solution_distance,
    write_outcome_solution_, write_process_solution_,
)
from handcoded.eval import (
    circuit_answer_matrix, evaluate_checkpoint, free_run_metrics, generate, generated_answer,
    inspect_fixed_circuit, make_circuit_prompts, make_generation_evaluation, strip_after_eos,
    gold_answer_matrix,
)
from handcoded.train import run_architecture_experiment, run_experiment, train_one_model, train_step
from handcoded.animate import (
    animate_all_circuits, animate_architecture_dynamics, animate_training_dynamics, export_training_animation, save_training_mp4,
)
from src.plot_style import apply_style

__all__ = [
    "BATCH_SEED", "DATA_SEED", "D_FF", "D_MODEL", "DEPTH", "LR", "MODEL_SEED", "N_BITS", "N_HEADS",
    "N_STATES", "STEPS", "TEST_SEED", "TEST_SIZE", "TRAIN_SIZE", "BATCH_SIZE", "load_config",
    "make_checkpoints", "configure", "current", "reset", "Circuit", "GATES", "gate_families", "apply_gate", "bits", "make_circuits", "make_gate_names",
    "phi", "sample_example", "sample_gate", "state_text", "CircuitTokenizer", "make_tokenizer",
    "LanguageBatch", "encode_dataset", "language_model_loss", "make_batch_schedule", "generate",
    "strip_after_eos", "FixedAttentionHead",
    "LearnedOneLayerTransformer", "attach_local_heads",
    "build_random_learned_model", "UnifiedBlock", "UnifiedExecutor",
    "build_fixed_process", "build_fixed_outcome", "build_random_trainable_unified",
    "unified_solution_distance", "unified_block_grad_norms",
    "write_process_solution_", "write_outcome_solution_",
    "make_random_trainable_copy", "circuit_answer_matrix", "evaluate_checkpoint", "free_run_metrics",
    "generated_answer", "inspect_fixed_circuit", "make_circuit_prompts", "make_generation_evaluation",
    "gold_answer_matrix",
    "run_architecture_experiment", "run_experiment", "train_one_model", "train_step",
    "animate_all_circuits", "animate_architecture_dynamics", "animate_training_dynamics", "export_training_animation", "save_training_mp4", "apply_style",
]
