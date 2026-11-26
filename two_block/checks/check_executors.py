"""Independent rebuild of the Proposition 1 exact executors from Appendix B (app:architectures).

Constructs both the step-by-step and direct-answer executors from scratch without calling any
construction helper in counting.py, and verifies entry-for-entry tensor equality against
counting.handcoded_params().

    uv run python two_block/checks/check_executors.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

TWO_BLOCK_DIR = Path(__file__).resolve().parents[1]
if str(TWO_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(TWO_BLOCK_DIR))

import counting as lt  # noqa: E402


def _build_register_map(alphabet: int, num_states: int, seq_len: int) -> tuple[dict[str, list[int]], int]:
    """Allocate contiguous coordinate index lists for every residual slot in Appendix B."""
    slot_sizes = {
        "tl": alphabet,
        "tv": num_states,
        "pos": seq_len,
        "qry": alphabet,
        "match": 1,
        "prev": num_states,
        "mt": 1,
        "cnt": 1,
        "s0": num_states,
        "out": num_states,
    }
    registers: dict[str, list[int]] = {}
    offset = 0
    for name, size in slot_sizes.items():
        registers[name] = list(range(offset, offset + size))
        offset += size
    return registers, offset


def build_executors(task: str, word_len: int, alphabet: int, num_states: int) -> dict[str, dict[str, torch.Tensor]]:
    """Build exact {'process', 'outcome'} parameter dictionaries directly from Appendix B formulas."""
    seq_len = 2 * word_len + 6
    reg, d_model = _build_register_map(alphabet, num_states, seq_len)
    d_mlp = max(alphabet, 3 * num_states * (word_len + 1), 2 * word_len + 1)
    vocab_size = alphabet + num_states + 3
    colon_token = alphabet + num_states + 1
    eos_token = alphabet + num_states + 2

    gamma = max(16.0, 3.0 * math.log(8.0 * (word_len + 3)))
    routing_scale = gamma * math.sqrt(seq_len)

    def make_zeros() -> dict[str, torch.Tensor]:
        params = {
            "wte": torch.zeros(vocab_size, d_model),
            "wpe": torch.zeros(seq_len, d_model),
            "readout": torch.zeros(vocab_size, d_model),
        }
        for block in (0, 1):
            params[f"{block}.q"] = torch.zeros(3, seq_len, d_model)
            params[f"{block}.k"] = torch.zeros(3, seq_len, d_model)
            params[f"{block}.v"] = torch.zeros(3, d_model, d_model)
            params[f"{block}.mlp_in"] = torch.zeros(d_mlp, d_model)
            params[f"{block}.mlp_b"] = torch.zeros(d_mlp)
            params[f"{block}.mlp_out"] = torch.zeros(d_model, d_mlp)
        return params

    def route(
        params: dict[str, torch.Tensor],
        block: int,
        head: int,
        sources: dict[int, list[int]],
        fallback: int,
    ) -> None:
        for key_pos in range(seq_len):
            params[f"{block}.k"][head, key_pos, reg["pos"][key_pos]] = 1.0
        for query_pos in range(seq_len):
            for key_pos in sources.get(query_pos, [fallback]):
                params[f"{block}.q"][head, key_pos, reg["pos"][query_pos]] = routing_scale

    def copy_slot(
        params: dict[str, torch.Tensor],
        block: int,
        head: int,
        dst_slot: str,
        src_slot: str,
        scale: float = 1.0,
    ) -> None:
        for dst_idx, src_idx in zip(reg[dst_slot], reg[src_slot]):
            params[f"{block}.v"][head, dst_idx, src_idx] = scale

    def build_shared_stem() -> dict[str, torch.Tensor]:
        params = make_zeros()
        for letter in range(alphabet):
            params["wte"][letter, reg["tl"][letter]] = 1.0
        for val in range(num_states):
            params["wte"][alphabet + val, reg["tv"][val]] = 1.0
        for pos in range(seq_len):
            params["wpe"][pos, reg["pos"][pos]] = 1.0

        route(params, 0, 0, {1 + t: [1] for t in range(1, word_len + 1)}, fallback=0)
        copy_slot(params, 0, 0, "qry", "tl")
        for letter in range(alphabet):
            params["0.mlp_in"][letter, reg["tl"][letter]] = 1.0
            params["0.mlp_in"][letter, reg["qry"][letter]] = 1.0
            params["0.mlp_b"][letter] = -1.5
            params["0.mlp_out"][reg["match"][0], letter] = 2.0
        return params

    def attach_readout(params: dict[str, torch.Tensor], colon_pos: int, eos_pos: int) -> dict[str, torch.Tensor]:
        for val in range(num_states):
            params["readout"][alphabet + val, reg["out"][val]] = 20.0
        params["readout"][colon_token, reg["pos"][colon_pos]] = 20.0
        params["readout"][eos_token, reg["pos"][eos_pos]] = 20.0
        return params

    # --- Step-by-step executor (theta*_step): position n+1+t predicts s_t ---
    step_params = build_shared_stem()
    prev_sources = {word_len + 2: [0], **{word_len + 1 + t: [word_len + 1 + t] for t in range(2, word_len + 1)}}
    route(step_params, 1, 0, prev_sources, fallback=1)
    copy_slot(step_params, 1, 0, "prev", "tv")
    route(step_params, 1, 1, {word_len + 1 + t: [1 + t] for t in range(1, word_len + 1)}, fallback=0)
    copy_slot(step_params, 1, 1, "mt", "match")
    route(step_params, 1, 2, {2 * word_len + 3: [2 * word_len + 2]}, fallback=1)
    copy_slot(step_params, 1, 2, "out", "tv")

    w_in, b_in, w_out = step_params["1.mlp_in"], step_params["1.mlp_b"], step_params["1.mlp_out"]
    if task == "count":
        for state_val in range(num_states):
            for match_bit in (0, 1):
                unit = 2 * state_val + match_bit
                w_in[unit, reg["prev"][state_val]] = 1.0
                w_in[unit, reg["mt"][0]] = float(2 * match_bit - 1)
                b_in[unit] = -0.5 - match_bit
                w_out[reg["out"][(state_val + match_bit) % num_states], unit] = 2.0
    else:
        unit = 0
        for state_val in range(1, num_states):
            w_in[unit, reg["prev"][state_val]] = 1.0
            b_in[unit] = -0.5
            w_out[reg["out"][state_val], unit] = 2.0
            unit += 1
        w_in[unit, reg["prev"][0]] = 1.0
        w_in[unit, reg["mt"][0]] = -1.0
        b_in[unit] = -0.5
        w_out[reg["out"][0], unit] = 2.0
        unit += 1
        for step_idx in range(1, word_len + 1):
            w_in[unit, reg["prev"][0]] = 1.0
            w_in[unit, reg["mt"][0]] = 1.0
            w_in[unit, reg["pos"][word_len + 1 + step_idx]] = 1.0
            b_in[unit] = -2.5
            w_out[reg["out"][step_idx], unit] = 2.0
            unit += 1
    step_params = attach_readout(step_params, colon_pos=2 * word_len + 2, eos_pos=2 * word_len + 4)

    # --- Direct outcome executor (theta*_out): position n+3 (':') predicts s_n ---
    out_params = build_shared_stem()
    ans_pos = word_len + 3
    if task == "count":
        route(out_params, 1, 0, {ans_pos: list(range(2, word_len + 2))}, fallback=0)
        copy_slot(out_params, 1, 0, "cnt", "match", scale=float(word_len))
        route(out_params, 1, 1, {ans_pos: [0]}, fallback=1)
        copy_slot(out_params, 1, 1, "s0", "tv")
        gate = float(word_len + 2)
        hat_coefs = {1: 1.0, 0: -2.0, -1: 1.0}
        for s0_val in range(num_states):
            for count_val in range(word_len + 1):
                for offset_idx, delta in enumerate((1, 0, -1)):
                    unit = 3 * ((word_len + 1) * s0_val + count_val) + offset_idx
                    out_params["1.mlp_in"][unit, reg["cnt"][0]] = 1.0
                    out_params["1.mlp_in"][unit, reg["s0"][s0_val]] = gate
                    out_params["1.mlp_b"][unit] = delta - count_val - gate
                    out_params["1.mlp_out"][reg["out"][(s0_val + count_val) % num_states], unit] = hat_coefs[delta]
    else:
        w_q, w_k = out_params["1.q"][0], out_params["1.k"][0]
        w_k[0, reg["match"][0]] = 1.0
        for pos in range(seq_len):
            w_k[1, reg["pos"][pos]] = float(-pos)
        w_k[2, reg["pos"][1]] = 1.0
        w_q[0, reg["pos"][ans_pos]] = routing_scale * (word_len + 3)
        w_q[1, reg["pos"][ans_pos]] = routing_scale
        for pos in range(seq_len):
            if pos != ans_pos:
                w_q[2, reg["pos"][pos]] = routing_scale
        for step_idx in range(1, word_len + 1):
            out_params["1.v"][0, reg["out"][step_idx], reg["pos"][1 + step_idx]] = 1.0
        out_params["1.v"][0, reg["out"][0], reg["pos"][0]] = 1.0
    out_params = attach_readout(out_params, colon_pos=word_len + 2, eos_pos=word_len + 4)

    return {"process": step_params, "outcome": out_params}


def main() -> int:
    test_cases = (
        ("count", 5, 3),
        ("count", 8, 2),
        ("count", 16, 3),
        ("first", 6, 2),
        ("first", 12, 2),
    )
    all_match = True
    for task, word_len, mod in test_cases:
        cfg = lt.configure(task=task, word_len=word_len, mod=mod, device="cpu")
        reference = lt.handcoded_params(cfg)
        rebuilt = build_executors(task, word_len, cfg.alphabet, cfg.num_values)

        case_ok = True
        for mode in ("process", "outcome"):
            for key, ref_tensor in reference[mode].items():
                rebuilt_tensor = rebuilt[mode][key].float()
                if not torch.equal(ref_tensor.cpu().float(), rebuilt_tensor):
                    case_ok = False
                    all_match = False
                    diff = (ref_tensor.cpu().float() - rebuilt_tensor).abs()
                    print(
                        f"MISMATCH {task} n={word_len} {mode} {key}: "
                        f"max |diff| = {diff.max().item():.3g} at index {diff.argmax().item()}"
                    )
        print(f"{task:5s} n={word_len:2d} K={cfg.num_values:2d}: {'identical' if case_ok else 'differs'}")
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
