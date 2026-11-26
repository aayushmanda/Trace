"""Numerical checks of score-degree and credit bounds at initialization (Appendices C, G.1, G.3).

Evaluates exact float64 expectations over all words and start states in the setting of population_gd.py:
    uv run python two_block/checks/check_credit.py rule          # prior / rule / outcome credit and eps-exponents
    uv run python two_block/checks/check_credit.py interaction   # outcome credit by ANOVA degree vs. rule credit
    uv run python two_block/checks/check_credit.py query         # rule credit: fixed query vs. uniform query
    uv run python two_block/checks/check_credit.py structured    # eps^6 rule coefficient vs. closed form
    uv run python two_block/checks/check_credit.py setsums [n] [sigma1 sigma2 sigma3]  # Jacobian set-sums T_r
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import torch
from torch.func import jvp

TWO_BLOCK_DIR = Path(__file__).resolve().parents[1]
if str(TWO_BLOCK_DIR) not in sys.path:
    sys.path.insert(0, str(TWO_BLOCK_DIR))

import population_gd as rb  # noqa: E402

RESULTS_DIR = Path("results/theorem_regime")


def _grad_norm(grads: tuple[torch.Tensor, ...]) -> float:
    return math.sqrt(sum(float((g ** 2).sum()) for g in grads))


def _transition_matrix(p_match: float, cfg: rb.RegimeConfig) -> tuple[torch.Tensor, torch.Tensor]:
    eye = torch.eye(cfg.num_states, device=cfg.device)
    tbar = (1.0 - p_match) * eye + p_match * eye.roll(1, dims=1)
    return eye, tbar


# -----------------------------------------------------------------------------
# 1. rule: prior, rule, and outcome credit norms and sigma-exponents
# -----------------------------------------------------------------------------


def compute_rule_credits(
    word_len: int,
    sigma: float,
    unit_params: dict[str, torch.Tensor],
    cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG,
) -> dict[str, float]:
    dataset = rb.build_exact_dataset(word_len, cfg)
    params = {k: (sigma * v).clone().requires_grad_() for k, v in unit_params.items()}
    eye, tbar = _transition_matrix(cfg.p_match, cfg)

    prev_states = dataset["step_ids"][:, word_len + 3 :] - cfg.alphabet
    step_resid = rb.forward_residual(params, dataset["step_ids"], cfg)
    assert isinstance(step_resid, torch.Tensor)
    step_logits = step_resid[:, word_len + 3 :] @ params["R"].T

    uniform = 1.0 / cfg.num_states
    prior_target = tbar[prev_states] - uniform
    full_target = eye[dataset["step_tgt"]] - uniform
    weights = dataset["prob"][:, None, None]

    out: dict[str, float] = {}
    for name, target in (("prior", prior_target), ("rule", full_target - prior_target)):
        phi = (weights * step_logits * target).sum() / step_logits.shape[1]
        out[name] = _grad_norm(torch.autograd.grad(phi, list(params.values()), retain_graph=True))

    out_resid = rb.forward_residual(params, dataset["out_ids"], cfg)
    assert isinstance(out_resid, torch.Tensor)
    out_logits = out_resid[:, -1] @ params["R"].T
    out_phi = (dataset["prob"][:, None] * out_logits * (eye[dataset["out_tgt"]] - uniform)).sum()
    out["outcome"] = _grad_norm(torch.autograd.grad(out_phi, list(params.values())))
    return out


def cmd_rule() -> None:
    cfg = rb.DEFAULT_CONFIG
    records: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        unit_params = rb.sample_unit_params(seed, cfg)
        for word_len in (4, 6, 8, 10, 12, 14):
            by_sigma = {s: compute_rule_credits(word_len, s, unit_params, cfg) for s in (0.005, 0.01, 0.02)}
            exponents = {
                key: math.log(by_sigma[0.02][key] / by_sigma[0.005][key]) / math.log(4.0)
                for key in ("prior", "rule", "outcome")
            }
            records.append({"seed": seed, "n": word_len, "credits": by_sigma, "exponent": exponents})
            c = by_sigma[0.02]
            print(
                f"draw {seed} n={word_len:2d} sigma=0.02: "
                f"prior {c['prior']:.3e}  rule {c['rule']:.3e}  outcome {c['outcome']:.3e}  "
                f"rule*n/sigma^6 {c['rule'] * word_len / 0.02 ** 6:.3g}  "
                f"exponents prior {exponents['prior']:.2f} rule {exponents['rule']:.2f} "
                f"outcome {exponents['outcome']:.2f}",
                flush=True,
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rule_credit.json").write_text(json.dumps(records, indent=2))


# -----------------------------------------------------------------------------
# 2. interaction: outcome credit by Efron-Stein degree vs. step rule credit
# -----------------------------------------------------------------------------


def efron_stein_degree_parts(
    target_rows: torch.Tensor,
    word_len: int,
    cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG,
) -> list[torch.Tensor]:
    """Decompose target_rows of shape (K * 2^n, K) into exact degree-r ANOVA components."""
    letter_probs = torch.tensor([cfg.p_match, 1.0 - cfg.p_match], device=target_rows.device)
    grid = target_rows.view(cfg.num_states, *([2] * word_len), cfg.num_states)
    parts: list[torch.Tensor] = [grid]

    for pos in range(word_len):
        axis = 1 + pos
        weight_shape = [1] * axis + [2] + [1] * (word_len - pos)
        w = letter_probs.view(*weight_shape)
        next_parts: list[torch.Tensor | None] = [None] * (len(parts) + 1)
        for deg, tensor in enumerate(parts):
            mean = (tensor * w).sum(dim=axis, keepdim=True).expand_as(tensor)
            prev = next_parts[deg]
            next_parts[deg] = mean if prev is None else prev + mean
            next_parts[deg + 1] = tensor - mean
        parts = [p for p in next_parts if p is not None]

    return [part.reshape(-1, cfg.num_states) for part in parts]


def compute_interaction_credits(
    word_len: int,
    sigma: float,
    unit_params: dict[str, torch.Tensor],
    cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG,
) -> dict[str, float]:
    dataset = rb.build_exact_dataset(word_len, cfg)
    params = {k: (sigma * v).clone().requires_grad_() for k, v in unit_params.items()}
    eye = torch.eye(cfg.num_states, device=cfg.device)
    parts = efron_stein_degree_parts(eye[dataset["out_tgt"]] - 1.0 / cfg.num_states, word_len, cfg)

    out_resid = rb.forward_residual(params, dataset["out_ids"], cfg)
    assert isinstance(out_resid, torch.Tensor)
    out_logits = out_resid[:, -1] @ params["R"].T
    weights = dataset["prob"][:, None]

    targets: dict[str, torch.Tensor] = {f"r{r}": parts[r] for r in range(min(4, word_len + 1))}
    targets["ge2"] = sum(parts[2:], torch.zeros_like(parts[0]))
    targets["all"] = sum(parts, torch.zeros_like(parts[0]))

    out = {
        name: _grad_norm(torch.autograd.grad((weights * out_logits * tgt).sum(), list(params.values()), retain_graph=True))
        for name, tgt in targets.items()
    }
    out["rule"] = compute_rule_credits(word_len, sigma, unit_params, cfg)["rule"]
    return out


def cmd_interaction() -> None:
    cfg = rb.DEFAULT_CONFIG
    lam = 1.0 - 2.0 * cfg.p_match
    records: list[dict[str, Any]] = []
    for seed in (0, 1, 2):
        unit_params = rb.sample_unit_params(seed, cfg)
        for word_len in (4, 6, 8, 10, 12, 14):
            by_sigma = {s: compute_interaction_credits(word_len, s, unit_params, cfg) for s in (0.005, 0.02)}
            exponents = {
                key: math.log(by_sigma[0.02][key] / by_sigma[0.005][key]) / math.log(4.0)
                for key in by_sigma[0.02]
            }
            records.append({"seed": seed, "n": word_len, "credits": by_sigma, "exponent": exponents})
            c = by_sigma[0.02]
            print(
                f"draw {seed} n={word_len:2d} sigma=0.02: outcome all {c['all']:.2e} "
                f"r0 {c['r0']:.2e} r1 {c['r1']:.2e} r2 {c['r2']:.2e} r3 {c['r3']:.2e} >=2 {c['ge2']:.2e} "
                f"| rule {c['rule']:.2e} | rule/(>=2) {c['rule'] / c['ge2']:.3g} "
                f"| (>=2)/(sigma^6 lam^(n-2)) {c['ge2'] / (0.02 ** 6 * lam ** (word_len - 2)):.3g} "
                f"| exps r2 {exponents['r2']:.2f} r3 {exponents['r3']:.2f} "
                f">=2 {exponents['ge2']:.2f} rule {exponents['rule']:.2f}",
                flush=True,
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "interaction_credit.json").write_text(json.dumps(records, indent=2))


# -----------------------------------------------------------------------------
# 3. query: fixed vs. uniform query rule credit (|Sigma| = 4)
# -----------------------------------------------------------------------------


def _build_query_dataset(
    word_len: int,
    queries: list[int],
    cfg: rb.RegimeConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    words = torch.tensor(list(itertools.product(range(cfg.alphabet_size), repeat=word_len)))
    n_words = len(words)
    id_chunks: list[torch.Tensor] = []
    tgt_chunks: list[torch.Tensor] = []
    prob_chunks: list[torch.Tensor] = []
    row_prob = 1.0 / (n_words * cfg.num_states * len(queries))

    for query in queries:
        matches = (words == query).long()
        for s0 in range(cfg.num_states):
            states = (s0 + matches.cumsum(dim=1)) % cfg.num_states
            prefix = torch.cat(
                [
                    torch.full((n_words, 1), cfg.alphabet_size + s0, dtype=torch.long),
                    torch.full((n_words, 1), query, dtype=torch.long),
                    words,
                    torch.full((n_words, 1), cfg.sep_token, dtype=torch.long),
                ],
                dim=1,
            )
            id_chunks.append(torch.cat([prefix, cfg.alphabet_size + states[:, : word_len - 1]], dim=1))
            tgt_chunks.append(states[:, 1:])
            prob_chunks.append(torch.full((n_words,), row_prob))

    return (
        torch.cat(id_chunks).to(cfg.device),
        torch.cat(tgt_chunks).to(cfg.device),
        torch.cat(prob_chunks).to(cfg.device),
    )


def compute_query_rule_credit(
    word_len: int,
    sigma: float,
    unit_params: dict[str, torch.Tensor],
    queries: list[int],
    cfg: rb.RegimeConfig,
) -> float:
    ids, targets, probs = _build_query_dataset(word_len, queries, cfg)
    params = {k: (sigma * v).clone().requires_grad_() for k, v in unit_params.items()}
    eye, tbar = _transition_matrix(1.0 / cfg.alphabet_size, cfg)
    rule_target = eye[targets] - tbar[ids[:, word_len + 3 :] - cfg.alphabet_size]

    phi: torch.Tensor | float = 0.0
    chunk_size = 1 << 16
    for start in range(0, len(probs), chunk_size):
        end = start + chunk_size
        resid = rb.forward_residual(params, ids[start:end], cfg)
        assert isinstance(resid, torch.Tensor)
        logits = resid[:, word_len + 3 :] @ params["R"].T
        phi = phi + (probs[start:end, None, None] * logits * rule_target[start:end]).sum() / logits.shape[1]

    assert isinstance(phi, torch.Tensor)
    return _grad_norm(torch.autograd.grad(phi, list(params.values())))


def cmd_query() -> None:
    cfg = rb.RegimeConfig(alphabet_size=4)
    records: list[dict[str, Any]] = []
    for seed in (0, 1):
        unit_params = rb.sample_unit_params(seed, cfg)
        for word_len in (4, 5, 6, 7, 8):
            row: dict[str, Any] = {}
            for label, queries in (("fixed", [0]), ("random", list(range(cfg.alphabet_size)))):
                vals = {s: compute_query_rule_credit(word_len, s, unit_params, queries, cfg) for s in (0.01, 0.02)}
                row[label] = {"R": vals[0.02], "expo": math.log(vals[0.02] / vals[0.01]) / math.log(2.0)}
            records.append({"seed": seed, "n": word_len, **row})
            fx, rnd = row["fixed"], row["random"]
            print(
                f"draw {seed} n={word_len}: "
                f"fixed q R={fx['R']:.3e} nR/s^6={word_len * fx['R'] / 0.02 ** 6:.1f} exp {fx['expo']:.2f} | "
                f"random q R={rnd['R']:.3e} nR/s^6={word_len * rnd['R'] / 0.02 ** 6:.2f} "
                f"n^2R/s^6={word_len * word_len * rnd['R'] / 0.02 ** 6:.1f} exp {rnd['expo']:.2f}",
                flush=True,
            )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "rule_credit_query.json").write_text(json.dumps(records, indent=2))


# -----------------------------------------------------------------------------
# 4. structured: closed-form eps^6 step rule-credit check (Appendix G.3)
# -----------------------------------------------------------------------------


def make_structured_unit_params(cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG) -> dict[str, torch.Tensor]:
    """Explicit theta_tilde where a single Block-2 head attains the exact eps^6 rule coefficient."""
    params = {k: torch.zeros_like(v) for k, v in rb.sample_unit_params(0, cfg).items()}
    for state in range(cfg.num_states):
        params["wte"][cfg.alphabet_size + state, 0] = (-1.0) ** state
    params["wte"][0, 1] = 1.0
    params["1.q"][0, 0, 0] = 1.0
    params["1.k"][0, 0, 1] = 1.0
    params["1.v"][0, 2, 1] = 1.0
    for block in range(2):
        params[f"{block}.mb"][:] = -1.0
    return params


def structured_measured(
    word_len: int,
    sigma: float,
    unit_params: dict[str, torch.Tensor],
    cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG,
) -> list[float]:
    dataset = rb.build_exact_dataset(word_len, cfg)
    params = {k: (sigma * v).clone().requires_grad_() for k, v in unit_params.items()}
    eye, tbar = _transition_matrix(cfg.p_match, cfg)

    resid = rb.forward_residual(params, dataset["step_ids"], cfg)
    assert isinstance(resid, torch.Tensor)
    logits = resid[:, word_len + 3 :] @ params["R"].T
    rule_target = eye[dataset["step_tgt"]] - tbar[dataset["step_ids"][:, word_len + 3 :] - cfg.alphabet_size]
    phi = (dataset["prob"][:, None, None] * logits * rule_target).sum() / logits.shape[1]
    return torch.autograd.grad(phi, params["R"])[0][:, 2].tolist()


def structured_closed_form(word_len: int, sigma: float, cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG) -> list[float]:
    p = cfg.p_match
    total = 0.0
    for step_idx in range(2, word_len + 1):
        prefix_len = word_len + 2 + step_idx
        total += p * (1.0 - p) / prefix_len * (1.0 - (3.0 + 2.0 * (word_len - 1) * p) / prefix_len)
    weights = [((-1.0) ** (k - 1) - (-1.0) ** k) / cfg.num_states for k in range(cfg.num_states)]
    prefactor = sigma ** 6 / math.sqrt(cfg.head_dim) * total / (word_len - 1)
    return [prefactor * w for w in weights]


def cmd_structured() -> None:
    cfg = rb.DEFAULT_CONFIG
    unit_params = make_structured_unit_params(cfg)
    for word_len in (2, 4, 6, 8, 10, 12):
        for sigma in (0.05, 0.02):
            got = structured_measured(word_len, sigma, unit_params, cfg)
            want = structured_closed_form(word_len, sigma, cfg)
            print(
                f"n={word_len:2d} sigma={sigma}: "
                f"measured {got[0]:+.6e} {got[1]:+.6e}   "
                f"closed form {want[0]:+.6e} {want[1]:+.6e}   "
                f"ratio {got[0] / want[0]:.6f}",
                flush=True,
            )


# -----------------------------------------------------------------------------
# 5. setsums: Jacobian set-sums T_r by degree (Proposition prop:score-degree)
# -----------------------------------------------------------------------------

SETSUM_ALPHABET = 4
SETSUM_D_MODEL = 16
SETSUM_HEADS = 3
SETSUM_HEAD_DIM = 8
SETSUM_D_MLP = 16
SETSUM_DIRS = 8


def _build_setsum_models(
    word_len: int,
    seed: int = 0,
    cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG,
) -> tuple[
    tuple[dict[str, torch.Tensor], Callable[[dict[str, torch.Tensor]], torch.Tensor]],
    tuple[dict[str, torch.Tensor], Callable[[dict[str, torch.Tensor]], torch.Tensor]],
]:
    dev = cfg.device
    gen = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen)

    words = torch.tensor(list(itertools.product(range(SETSUM_ALPHABET), repeat=word_len)))
    n_words = len(words)

    # Single-layer attention model
    ids_1l = torch.cat(
        [
            torch.full((n_words, 1), SETSUM_ALPHABET, dtype=torch.long),
            words,
            torch.full((n_words, 1), SETSUM_ALPHABET + 1, dtype=torch.long),
        ],
        dim=1,
    ).to(dev)
    params_1l = {
        "E": randn(SETSUM_ALPHABET + 2, SETSUM_D_MODEL),
        "P": randn(word_len + 2, SETSUM_D_MODEL),
        "Wq": randn(SETSUM_D_MODEL, SETSUM_D_MODEL),
        "Wk": randn(SETSUM_D_MODEL, SETSUM_D_MODEL),
        "Wv": randn(SETSUM_D_MODEL, SETSUM_D_MODEL),
        "R": randn(cfg.num_states, SETSUM_D_MODEL),
    }

    def forward_1l(p: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = p["E"][ids_1l] + p["P"][None]
        q = hidden[:, -1] @ p["Wq"].T
        k = hidden @ p["Wk"].T
        v = hidden @ p["Wv"].T
        attn = torch.softmax((k @ q.unsqueeze(-1)).squeeze(-1) / math.sqrt(SETSUM_D_MODEL), dim=-1)
        return (hidden[:, -1] + (attn.unsqueeze(-1) * v).sum(dim=1)) @ p["R"].T

    # Two-block causal Transformer
    vocab_size = SETSUM_ALPHABET + cfg.num_states + 3
    seq_len = word_len + 4
    ids_2b = torch.cat(
        [
            torch.full((n_words, 1), SETSUM_ALPHABET, dtype=torch.long),
            torch.zeros(n_words, 1, dtype=torch.long),
            words,
            torch.full((n_words, 1), SETSUM_ALPHABET + cfg.num_states, dtype=torch.long),
            torch.full((n_words, 1), SETSUM_ALPHABET + cfg.num_states + 1, dtype=torch.long),
        ],
        dim=1,
    ).to(dev)
    params_2b: dict[str, torch.Tensor] = {
        "wte": randn(vocab_size, SETSUM_D_MODEL),
        "wpe": randn(seq_len, SETSUM_D_MODEL),
        "R": randn(cfg.num_states, SETSUM_D_MODEL),
    }
    for block in range(2):
        params_2b[f"{block}.q"] = randn(SETSUM_HEADS, SETSUM_HEAD_DIM, SETSUM_D_MODEL)
        params_2b[f"{block}.k"] = randn(SETSUM_HEADS, SETSUM_HEAD_DIM, SETSUM_D_MODEL)
        params_2b[f"{block}.v"] = randn(SETSUM_HEADS, SETSUM_D_MODEL, SETSUM_D_MODEL)
        params_2b[f"{block}.mi"] = randn(SETSUM_D_MLP, SETSUM_D_MODEL)
        params_2b[f"{block}.mo"] = randn(SETSUM_D_MODEL, SETSUM_D_MLP)
        raw_bias = randn(SETSUM_D_MLP)
        params_2b[f"{block}.mb"] = torch.sign(raw_bias) * (1.0 + raw_bias.abs())
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=dev).triu(1)

    def forward_2b(p: dict[str, torch.Tensor]) -> torch.Tensor:
        hidden = p["wte"][ids_2b] + p["wpe"]
        for block in range(2):
            q = torch.einsum("btw,hdw->bhtd", hidden, p[f"{block}.q"])
            k = torch.einsum("btw,hdw->bhtd", hidden, p[f"{block}.k"])
            v = torch.einsum("btw,hvw->bhtv", hidden, p[f"{block}.v"])
            scores = (q @ k.transpose(-1, -2) / math.sqrt(SETSUM_HEAD_DIM)).masked_fill(causal_mask, -math.inf)
            hidden = hidden + (scores.softmax(dim=-1) @ v).sum(dim=1)
            hidden = hidden + torch.relu(hidden @ p[f"{block}.mi"].T + p[f"{block}.mb"]) @ p[f"{block}.mo"].T
        return hidden[:, -1] @ p["R"].T

    return (
        ({k: v.to(dev) for k, v in params_1l.items()}, forward_1l),
        ({k: v.to(dev) for k, v in params_2b.items()}, forward_2b),
    )


def compute_setsums(
    unit_params: dict[str, torch.Tensor],
    model_fn: Callable[[dict[str, torch.Tensor]], torch.Tensor],
    word_len: int,
    sigma: float,
    seed: int = 1,
    cfg: rb.RegimeConfig = rb.DEFAULT_CONFIG,
) -> list[float]:
    dev = cfg.device
    gen = torch.Generator(device=dev).manual_seed(seed)
    basis_raw = torch.cat(
        [
            torch.ones(SETSUM_ALPHABET, 1),
            torch.randn(SETSUM_ALPHABET, SETSUM_ALPHABET - 1, generator=torch.Generator().manual_seed(1)),
        ],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(basis_raw)
    q_mat = (q_mat.T * torch.sign(q_mat.T[:, :1])).to(dev)

    multi_idx = torch.tensor(list(itertools.product(range(SETSUM_ALPHABET), repeat=word_len)), device=dev)
    bit_weights = 2 ** torch.arange(word_len, device=dev)
    support = ((multi_idx != 0).long() * bit_weights).sum(dim=1)

    scaled_params = {k: sigma * v for k, v in unit_params.items()}
    energy_by_subset = torch.zeros(2 ** word_len, device=dev)

    for _ in range(SETSUM_DIRS):
        direction = {k: torch.randn(v.shape, generator=gen, device=dev) for k, v in scaled_params.items()}
        _, tangent = jvp(model_fn, (scaled_params,), (direction,))
        coeffs = tangent.reshape((SETSUM_ALPHABET,) * word_len + (-1,))
        for axis in range(word_len):
            coeffs = torch.tensordot(q_mat, coeffs, dims=([1], [axis])).movedim(0, axis)
        subset_e = (coeffs.reshape(SETSUM_ALPHABET ** word_len, -1) ** 2).sum(dim=1)
        subset_e = subset_e / (SETSUM_ALPHABET ** word_len * SETSUM_DIRS)
        energy_by_subset.index_add_(0, support, subset_e)

    subset_sizes = torch.tensor([bin(mask).count("1") for mask in range(2 ** word_len)], device=dev)
    return [float(energy_by_subset[subset_sizes == r].sqrt().sum()) for r in range(word_len + 1)]


def cmd_setsums(word_len: int, sigmas: list[float]) -> None:
    (params_1l, fn_1l), (params_2b, fn_2b) = _build_setsum_models(word_len)
    for label, params, fn in (("one layer ", params_1l, fn_1l), ("two blocks", params_2b, fn_2b)):
        totals = {s: compute_setsums(params, fn, word_len, s) for s in sigmas}
        for degree in range(1, min(6, word_len + 1)):
            e1 = math.log(totals[sigmas[1]][degree] / totals[sigmas[0]][degree]) / math.log(sigmas[1] / sigmas[0])
            e2 = math.log(totals[sigmas[2]][degree] / totals[sigmas[1]][degree]) / math.log(sigmas[2] / sigmas[1])
            print(
                f"{label} n={word_len} r={degree}:  "
                f"T_r(sigma={sigmas[0]})={totals[sigmas[0]][degree]:.3e}  "
                f"sigma-exponent {e1:.2f}, {e2:.2f}",
                flush=True,
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score-degree and credit checks (Appendices C, G.1, G.3).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("rule", help="Prior, rule, and outcome credit norms and sigma-exponents.")
    sub.add_parser("interaction", help="Outcome credit by Efron-Stein degree vs. rule credit.")
    sub.add_parser("query", help="Rule credit with fixed query vs. uniform query (|Sigma|=4).")
    sub.add_parser("structured", help="Structured sigma^6 rule coefficient vs. closed form.")
    ss = sub.add_parser("setsums", help="Jacobian set-sums T_r by degree.")
    ss.add_argument("n", nargs="?", type=int, default=6)
    ss.add_argument("sigmas", nargs="*", type=float, default=[0.1, 0.2, 0.3])

    args = parser.parse_args(argv)
    if args.cmd == "rule":
        cmd_rule()
    elif args.cmd == "interaction":
        cmd_interaction()
    elif args.cmd == "query":
        cmd_query()
    elif args.cmd == "structured":
        cmd_structured()
    elif args.cmd == "setsums":
        sigmas = args.sigmas if len(args.sigmas) >= 3 else [0.1, 0.2, 0.3]
        cmd_setsums(args.n, sigmas[:3])


if __name__ == "__main__":
    main()
