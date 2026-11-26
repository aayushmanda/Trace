"""Exact float64 population GD exit times inside the outcome plateau (Theorem 1, Table 5, Appendix G.1).

Computes exit times from the certified initialization ball B_eps (eps = 0.002) and macroscopic radius
r = 0.1 (eps = 0.02) on modulo-2 counting (p = 1/4, lambda = 1/2) over all 2^n words under both step
and outcome cross-entropy objectives.

Examples:
    REGIME_ZERO_READOUT=1 REGIME_STOP=ball REGIME_STEP=1e-6 REGIME_SEED=0 \\
        uv run python two_block/population_gd.py 0.002
    uv run python two_block/population_gd.py 0.02 --zero-readout --n 4 6 8
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class RegimeConfig:
    """Architecture and data law for exact float64 population gradient flow."""

    alphabet_size: int = 2
    num_states: int = 2
    d_model: int = 16
    num_heads: int = 3
    head_dim: int = 8
    d_mlp: int = 16
    max_seq_len: int = 40
    p_match: float = 0.25
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def sep_token(self) -> int:
        return self.alphabet_size + self.num_states

    @property
    def colon_token(self) -> int:
        return self.alphabet_size + self.num_states + 1

    @property
    def vocab_size(self) -> int:
        return self.alphabet_size + self.num_states + 3


DEFAULT_CONFIG = RegimeConfig()


def sample_unit_params(seed: int = 0, cfg: RegimeConfig = DEFAULT_CONFIG) -> dict[str, torch.Tensor]:
    """Sample balanced unit-scale parameter direction theta_tilde with |b_m| >= 1."""
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen)

    params: dict[str, torch.Tensor] = {
        "wte": randn(cfg.vocab_size, cfg.d_model),
        "wpe": randn(cfg.max_seq_len, cfg.d_model),
        "R": randn(cfg.num_states, cfg.d_model),
    }
    for block in range(2):
        params[f"{block}.q"] = randn(cfg.num_heads, cfg.head_dim, cfg.d_model)
        params[f"{block}.k"] = randn(cfg.num_heads, cfg.head_dim, cfg.d_model)
        params[f"{block}.v"] = randn(cfg.num_heads, cfg.d_model, cfg.d_model)
        params[f"{block}.mi"] = randn(cfg.d_mlp, cfg.d_model)
        params[f"{block}.mo"] = randn(cfg.d_model, cfg.d_mlp)
        raw_bias = randn(cfg.d_mlp)
        params[f"{block}.mb"] = torch.sign(raw_bias) * (1.0 + raw_bias.abs())

    return {name: tensor.to(cfg.device) for name, tensor in params.items()}


def build_exact_dataset(word_len: int, cfg: RegimeConfig = DEFAULT_CONFIG) -> dict[str, torch.Tensor]:
    """Enumerate all A^n words and K initial states with exact probabilities (query letter = 0)."""
    words = torch.tensor(list(itertools.product(range(cfg.alphabet_size), repeat=word_len)))
    matches = (words == 0).long()
    match_counts = matches.sum(dim=1)
    probs = (cfg.p_match ** match_counts * (1.0 - cfg.p_match) ** (word_len - match_counts)).double()
    probs = probs / cfg.num_states

    n_words = len(words)
    words = words.repeat(cfg.num_states, 1)
    matches = matches.repeat(cfg.num_states, 1)
    probs = probs.repeat(cfg.num_states)

    s0 = torch.arange(cfg.num_states).repeat_interleave(n_words)
    states = (s0[:, None] + matches.cumsum(dim=1)) % cfg.num_states

    n_rows = len(words)
    prefix = torch.cat(
        [
            (cfg.alphabet_size + s0)[:, None],
            torch.zeros(n_rows, 1, dtype=torch.long),
            words,
            torch.full((n_rows, 1), cfg.sep_token, dtype=torch.long),
        ],
        dim=1,
    )
    out_ids = torch.cat([prefix, torch.full((n_rows, 1), cfg.colon_token, dtype=torch.long)], dim=1)
    step_ids = torch.cat([prefix, cfg.alphabet_size + states[:, : word_len - 1]], dim=1)

    return {
        "prob": probs.to(cfg.device),
        "out_ids": out_ids.to(cfg.device),
        "out_tgt": states[:, -1].to(cfg.device),
        "step_ids": step_ids.to(cfg.device),
        "step_tgt": states[:, 1:].to(cfg.device),
    }


def forward_residual(
    params: dict[str, torch.Tensor],
    token_ids: torch.Tensor,
    cfg: RegimeConfig = DEFAULT_CONFIG,
    *,
    count_switching: bool = False,
    sign_log: list[torch.Tensor] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    """Run the two-block normalization-free causal Transformer up to the final residual stream."""
    seq_len = token_ids.shape[1]
    hidden = params["wte"][token_ids] + params["wpe"][:seq_len]
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=token_ids.device).triu(1)
    scale = math.sqrt(cfg.head_dim)
    switching_units = 0

    for block in range(2):
        q = torch.einsum("btw,hdw->bhtd", hidden, params[f"{block}.q"])
        k = torch.einsum("btw,hdw->bhtd", hidden, params[f"{block}.k"])
        v = torch.einsum("btw,hvw->bhtv", hidden, params[f"{block}.v"])
        scores = (q @ k.transpose(-1, -2) / scale).masked_fill(causal_mask, -math.inf)
        hidden = hidden + (scores.softmax(dim=-1) @ v).sum(dim=1)

        pre_act = hidden @ params[f"{block}.mi"].T + params[f"{block}.mb"]
        if sign_log is not None:
            sign_log.append(pre_act.detach() > 0)
        if count_switching:
            active = pre_act > 0
            flat = active.flatten(0, 1)
            switching_units += int((flat.any(dim=0) != flat.all(dim=0)).sum())
        hidden = hidden + F.relu(pre_act) @ params[f"{block}.mo"].T

    return (hidden, switching_units) if count_switching else hidden


def population_loss(
    params: dict[str, torch.Tensor],
    dataset: dict[str, torch.Tensor],
    mode: str,
    word_len: int,
    cfg: RegimeConfig = DEFAULT_CONFIG,
    *,
    chunk_size: int = 1 << 15,
    reference_signs: list[torch.Tensor] | None = None,
) -> torch.Tensor | tuple[torch.Tensor, int]:
    """Compute exact probability-weighted cross-entropy (optionally tracking ReLU sign flips)."""
    total_loss: torch.Tensor | float = 0.0
    sign_flips = 0
    sign_idx = 0
    num_rows = len(dataset["prob"])

    for start in range(0, num_rows, chunk_size):
        end = start + chunk_size
        weights = dataset["prob"][start:end]
        chunk_signs: list[torch.Tensor] | None = [] if reference_signs is not None else None

        if mode == "outcome":
            resid = forward_residual(params, dataset["out_ids"][start:end], cfg, sign_log=chunk_signs)
            assert isinstance(resid, torch.Tensor)
            logits = resid[:, -1] @ params["R"].T
            per_example_ce = F.cross_entropy(logits, dataset["out_tgt"][start:end], reduction="none")
        else:
            resid = forward_residual(params, dataset["step_ids"][start:end], cfg, sign_log=chunk_signs)
            assert isinstance(resid, torch.Tensor)
            logits = resid[:, word_len + 3 :] @ params["R"].T
            targets = dataset["step_tgt"][start:end].flatten()
            per_example_ce = F.cross_entropy(logits.flatten(0, 1), targets, reduction="none")
            per_example_ce = per_example_ce.view(len(weights), -1).mean(dim=1)

        total_loss = total_loss + (weights * per_example_ce).sum()
        if reference_signs is not None and chunk_signs is not None:
            for sign_tensor in chunk_signs:
                if len(reference_signs) <= sign_idx:
                    reference_signs.append(sign_tensor)
                else:
                    sign_flips += int((sign_tensor != reference_signs[sign_idx]).sum())
                sign_idx += 1

    assert isinstance(total_loss, torch.Tensor)
    return (total_loss, sign_flips) if reference_signs is not None else total_loss


def count_switching_units(
    params: dict[str, torch.Tensor],
    dataset: dict[str, torch.Tensor],
    mode: str,
    cfg: RegimeConfig = DEFAULT_CONFIG,
) -> int:
    """Count ReLU units that switch sign across the evaluated token positions."""
    key = "out_ids" if mode == "outcome" else "step_ids"
    with torch.no_grad():
        _, switching = forward_residual(params, dataset[key][: 1 << 15], cfg, count_switching=True)
    return int(switching)


def certified_ball_radius(unit_params: dict[str, torch.Tensor], cfg: RegimeConfig = DEFAULT_CONFIG) -> float:
    """Return Theorem 1 ball radius r_0 = min(1/8, c_emb/4) in units of eps * ||theta_tilde||."""
    state_embeddings = unit_params["wte"][cfg.alphabet_size : cfg.alphabet_size + cfg.num_states]
    c_emb = float(torch.linalg.svdvals(state_embeddings).min())
    unit_norm = _tensor_dict_norm(unit_params)
    return min(0.125, c_emb / 4.0) / unit_norm


def _tensor_dict_norm(tensors: dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float((t ** 2).sum()) for t in tensors.values()))


def _grad_norm(grads: tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(sum(float((g ** 2).sum()) for g in grads))


def run_population_gd(
    word_len: int,
    sigma: float,
    mode: str,
    unit_params: dict[str, torch.Tensor],
    *,
    stop_radius: float = 0.1,
    max_steps: int = 20_000,
    step_len: float = 1e-3,
    track_radii: list[float] | None = None,
    cfg: RegimeConfig = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Integrate normalized population GD until ||theta - theta_0|| / (sigma ||theta_tilde||) >= stop_radius."""
    dataset = build_exact_dataset(word_len, cfg)
    unit_norm = _tensor_dict_norm(unit_params)
    scale_norm = sigma * unit_norm

    params = {name: (sigma * tensor).clone().requires_grad_() for name, tensor in unit_params.items()}
    theta0 = {name: tensor.detach().clone() for name, tensor in params.items()}
    radii = sorted(set(track_radii or []) | {stop_radius})

    reference_signs: list[torch.Tensor] = []
    max_flips = 0
    initial_loss: float | None = None
    elapsed_time = 0.0

    record: dict[str, Any] = {
        "n": word_len,
        "sigma": sigma,
        "mode": mode,
        "switch_start": count_switching_units(params, dataset, mode, cfg),
        "T_r": {},
        "relu_flips": {},
    }

    for step_idx in range(max_steps):
        loss_val, flips = population_loss(params, dataset, mode, word_len, cfg, reference_signs=reference_signs)
        max_flips = max(max_flips, flips)
        grads = torch.autograd.grad(loss_val, list(params.values()))
        grad_norm = _grad_norm(grads)

        if step_idx == 0:
            initial_loss = loss_val.detach().item()
            record["grad0_over_sigma"] = grad_norm / sigma

        dt = step_len * scale_norm / grad_norm
        with torch.no_grad():
            for param, grad in zip(params.values(), grads):
                param -= dt * grad
        elapsed_time += dt

        rel_dist = math.sqrt(sum(float(((params[k] - theta0[k]) ** 2).sum()) for k in params)) / scale_norm
        for radius in radii:
            if rel_dist >= radius and radius not in record["T_r"]:
                with torch.no_grad():
                    _, exit_flips = population_loss(
                        params, dataset, mode, word_len, cfg, reference_signs=reference_signs
                    )
                    max_flips = max(max_flips, exit_flips)
                record["T_r"][radius] = elapsed_time
                record["relu_flips"][radius] = max_flips

        if rel_dist >= stop_radius:
            with torch.no_grad():
                final_loss = float(population_loss(params, dataset, mode, word_len, cfg))
            assert initial_loss is not None
            record.update(
                T=elapsed_time,
                steps=step_idx + 1,
                drop_over_sigma2=(initial_loss - final_loss) / (sigma ** 2),
                gap_over_sigma2=(math.log(cfg.num_states) - final_loss) / (sigma ** 2),
                switch_end=count_switching_units(params, dataset, mode, cfg),
            )
            break

    return record


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    env_n = [int(x) for x in os.environ.get("REGIME_N", "4,6,8,10,12,14,16").split(",")]
    parser = argparse.ArgumentParser(description="Exact float64 population GD exit times (Theorem 1 / Table 5).")
    parser.add_argument("sigmas", nargs="*", type=float, help="Initialization scales eps (default: 0.02 0.005 0.002).")
    parser.add_argument("--n", nargs="+", type=int, default=env_n, help="Sequence lengths to evaluate.")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("REGIME_SEED", "0")))
    parser.add_argument("--step-len", type=float, default=float(os.environ.get("REGIME_STEP", "1e-3")))
    parser.add_argument(
        "--zero-readout",
        action="store_true",
        default=os.environ.get("REGIME_ZERO_READOUT") == "1",
        help="Zero the initial readout matrix V(0) = 0.",
    )
    parser.add_argument(
        "--stop",
        choices=("ball", "macro"),
        default="ball" if os.environ.get("REGIME_STOP") == "ball" else "macro",
        help="Stop at certified ball radius r_0 ('ball') or macroscopic radius 0.1 ('macro').",
    )
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("REGIME_OUT", "results/theorem_regime/both.json")))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sigmas = args.sigmas or [0.02, 0.005, 0.002]
    cfg = DEFAULT_CONFIG

    unit_params = sample_unit_params(args.seed, cfg)
    if args.zero_readout:
        unit_params["R"].zero_()

    r_ball = certified_ball_radius(unit_params, cfg)
    stop_radius = r_ball if args.stop == "ball" else 0.1
    max_steps = int(20_000 * 1e-3 / args.step_len)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for sigma in sigmas:
        for word_len in args.n:
            for mode in ("step", "outcome"):
                rec = run_population_gd(
                    word_len=word_len,
                    sigma=sigma,
                    mode=mode,
                    unit_params=unit_params,
                    stop_radius=stop_radius,
                    max_steps=max_steps,
                    step_len=args.step_len,
                    track_radii=[r_ball],
                    cfg=cfg,
                )
                rec.update(seed=args.seed, zero_readout=args.zero_readout, step_len=args.step_len)
                results.append(rec)

                delta = 0.5 ** word_len + sigma * 0.5 ** (word_len - 1)
                t_exit = rec.get("T", float("nan"))
                radii_str = ", ".join(f"{r:.2e}: {val:.3e}" for r, val in rec["T_r"].items())
                print(
                    f"sigma={sigma} n={word_len:2d} {mode:8s} "
                    f"grad/sigma={rec['grad0_over_sigma']:.3e}  "
                    f"T({stop_radius:.2g})={t_exit:.3e}  "
                    f"grad/(sigma*(sigma^2+delta_n))={rec['grad0_over_sigma'] / (sigma ** 2 + delta):.3e}  "
                    f"T*(sigma^2+delta_n)={t_exit * (sigma ** 2 + delta):.2f}  "
                    f"switching {rec['switch_start']}->{rec.get('switch_end')}  "
                    f"T(r) {{{radii_str}}}  relu flips {rec['relu_flips']}",
                    flush=True,
                )
                args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
