"""Semantic-token Boolean-circuit executors (tutorial + paper stack)."""
from handcoded.config import (
    BATCH_SEED, DATA_SEED, D_FF, D_MODEL, DEPTH, LR, MODEL_SEED, N_BITS, N_HEADS,
    N_STATES, STEPS, TEST_SEED, TEST_SIZE, TRAIN_SIZE, BATCH_SIZE, load_config, make_checkpoints,
)
from handcoded.gates import (
    Circuit, GATES, apply_gate, bits, make_circuits, make_gate_names, phi, sample_example, sample_gate, state_text,
)
from handcoded.tokenizer import CircuitTokenizer, make_tokenizer
from handcoded.data import LanguageBatch, encode_dataset, language_model_loss, make_batch_schedule
from handcoded.generate import generate, strip_after_eos
from handcoded.models import (
    FixedAttentionHead, FixedOutcomeBlock, HandcodedOutcomeTransformer, HandcodedProcessTransformer,
    LearnedOneLayerTransformer, build_random_learned_model, build_random_trainable_outcome_architecture,
    build_random_trainable_process_architecture, make_random_trainable_copy,
)
from handcoded.eval import (
    circuit_answer_matrix, evaluate_checkpoint, free_run_metrics, generated_answer,
    inspect_fixed_circuit, make_circuit_prompts, make_generation_evaluation,
)
from handcoded.train import run_architecture_experiment, run_experiment, train_one_model, train_step
from handcoded.animate import animate_training_dynamics, export_training_animation, save_training_mp4
from handcoded.plotting import apply_style

__all__ = [
    "BATCH_SEED", "DATA_SEED", "D_FF", "D_MODEL", "DEPTH", "LR", "MODEL_SEED", "N_BITS", "N_HEADS",
    "N_STATES", "STEPS", "TEST_SEED", "TEST_SIZE", "TRAIN_SIZE", "BATCH_SIZE", "load_config",
    "make_checkpoints", "Circuit", "GATES", "apply_gate", "bits", "make_circuits", "make_gate_names",
    "phi", "sample_example", "sample_gate", "state_text", "CircuitTokenizer", "make_tokenizer",
    "LanguageBatch", "encode_dataset", "language_model_loss", "make_batch_schedule", "generate",
    "strip_after_eos", "FixedAttentionHead", "FixedOutcomeBlock", "HandcodedOutcomeTransformer",
    "HandcodedProcessTransformer", "LearnedOneLayerTransformer", "build_random_learned_model",
    "build_random_trainable_outcome_architecture", "build_random_trainable_process_architecture",
    "make_random_trainable_copy", "circuit_answer_matrix", "evaluate_checkpoint", "free_run_metrics",
    "generated_answer", "inspect_fixed_circuit", "make_circuit_prompts", "make_generation_evaluation",
    "run_architecture_experiment", "run_experiment", "train_one_model", "train_step",
    "animate_training_dynamics", "export_training_animation", "save_training_mp4", "apply_style",
]
