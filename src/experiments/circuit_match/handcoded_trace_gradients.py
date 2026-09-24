"""Matched clean vs wrong process-trace loss and QK/OV/MLP gradient probes.

Same prompt x; tau+ is the gold process continuation and tau- is a matched
corruption of the D state successors (Paper 1 wrong_trace), keeping gold
answer after COLON. Gradients of full continuation CE are split into the
constructed circuit groups (all heads' query/key, all heads' value, MLP
mlp_in/mlp_bias/mlp_out) and projected onto the star parameters d*.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random

import sys

import matplotlib

# Force a headless backend for plain script runs only. This module is also
# imported from the tutorial notebook for `encode_process_states`; calling
# matplotlib.use("Agg") there silently switches the kernel off the inline
# backend, and every later plt figure stops rendering with no error.
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from handcoded.data import LanguageBatch, encode_dataset
from handcoded.gates import phi
from src.data.boolean_circuit_tasks import _coherent_wrong_mask
from src.plot_style import apply_style

GROUPS = ("QK", "OV", "MLP")
RHO_GRID = (0.0, 0.3, 0.5, 1.0)
#: step sizes for the finite-step functional probe (see functional_probe).
FUNCTIONAL_ETAS = (1e-3, 1e-2)
CORRUPTION_SEED = 4242
EPS = 1e-12
STAR_D_MIN_NORM = 1e-8


@dataclass(frozen=True)
class ProcessSlices:
    """Teacher-forced target indices for one process continuation (D=4 → T=16)."""

    depth: int
    gate: tuple[int, ...]
    state: tuple[int, ...]
    answer: tuple[int, ...]
    syntax: tuple[int, ...]
    continuation: tuple[int, ...]

    def as_dict(self) -> dict:
        return {
            "depth": self.depth,
            "gate": list(self.gate),
            "state": list(self.state),
            "answer": list(self.answer),
            "syntax": list(self.syntax),
            "continuation": list(self.continuation),
        }


def continuation_slices(depth: int) -> ProcessSlices:
    """Map process tokens onto target indices of `encode_dataset(..., "process")`.

    Sequence (input positions, length 3D+4): 0 s0; 1..D prompt gates; D+1 SEP;
    then pairs (g_k, s_k) for k=1..D; then COLON, answer s_D, with EOS as the
    final next-token target. Loss is next-token CE, so group k is predicted at
    the index of the *previous* token.

    D=4: gate p_k at (5,7,9,11); state r_k at (6,8,10,12); COLON at 13;
    answer at 14; EOS at 15. SEP is the last prompt token and is masked out of CE.
    """
    if depth < 1:
        raise ValueError("depth must be positive")
    first = depth + 1  # predict g_1 from SEP
    gate = tuple(first + 2 * k for k in range(depth))
    state = tuple(first + 2 * k + 1 for k in range(depth))
    colon = first + 2 * depth
    answer = (colon + 1,)
    eos = colon + 2
    syntax = (colon, eos)
    continuation = tuple(range(first, eos + 1))
    return ProcessSlices(
        depth=depth, gate=gate, state=state, answer=answer,
        syntax=syntax, continuation=continuation,
    )


def assert_d4_continuation_slices() -> ProcessSlices:
    """Self-check of the D=4 teacher-forced index map used in NOTES and plots."""
    slices = continuation_slices(4)
    expected = ProcessSlices(
        depth=4,
        gate=(5, 7, 9, 11),
        state=(6, 8, 10, 12),
        answer=(14,),
        syntax=(13, 15),
        continuation=tuple(range(5, 16)),
    )
    if slices != expected:
        raise AssertionError(f"continuation_slices(4)={slices.as_dict()} != {expected.as_dict()}")
    return slices


def corrupt_process_states(circuit, mode: str, rng: random.Random, n_bits: int) -> list[int]:
    """Wrong successor chain on the same (start, gates), Paper 1 laws.

    Symmetric: at each step, from the *current already-corrupted* state, the
    true next is phi(current, gate); sample uniformly among the K-1 others.
    Coherent: wrong_next = phi(current, gate) XOR _coherent_wrong_mask(gate).
    """
    n_states = 2 ** n_bits
    wrong_state = int(circuit.start)
    out = []
    if mode == "symmetric":
        for gate in circuit.gates:
            true_next = phi(wrong_state, gate, n_bits)
            choices = [v for v in range(n_states) if v != true_next]
            wrong_state = rng.choice(choices)
            out.append(wrong_state)
        return out
    if mode == "coherent":
        for gate in circuit.gates:
            true_next = phi(wrong_state, gate, n_bits)
            wrong_state = true_next ^ _coherent_wrong_mask(gate)
            out.append(wrong_state)
        return out
    raise ValueError(f"unknown corruption {mode!r}; expected 'symmetric' or 'coherent'")


def encode_process_states(circuits, tokenizer, states_list, answers) -> LanguageBatch:
    """Process teacher-forcing with explicit state traces and (gold) answers."""
    inputs, targets = [], []
    for circuit, states, answer in zip(circuits, states_list, answers):
        if len(states) != len(circuit.gates):
            raise ValueError("state trace length must equal depth")
        prompt = tokenizer.prompt(circuit)
        trace = []
        for gate, state in zip(circuit.gates, states):
            trace.extend([tokenizer.ids[gate], int(state)])
        full = prompt + trace + [tokenizer.colon, int(answer), tokenizer.eos]
        inputs.append(full[:-1])
        target = full[1:]
        target[: len(prompt) - 1] = [-100] * (len(prompt) - 1)
        targets.append(target)
    return LanguageBatch(torch.tensor(inputs), torch.tensor(targets))


def make_matched_eval_batches(
    circuits, tokenizer, *, corruption: str = "symmetric",
    seed: int = CORRUPTION_SEED, device=None, n_bits: int | None = None,
) -> tuple[LanguageBatch, LanguageBatch]:
    """Fixed matched (τ+, τ-) batches: same prompts, gold answer after COLON."""
    if not circuits:
        raise ValueError("need at least one circuit for matched eval batches")
    if n_bits is None:
        n_bits = int(math.log2(tokenizer.n_states))
    plus = encode_dataset(circuits, tokenizer, "process")
    rng = random.Random(seed)
    wrong_states = [corrupt_process_states(c, corruption, rng, n_bits) for c in circuits]
    gold = [c.answer for c in circuits]
    minus = encode_process_states(circuits, tokenizer, wrong_states, gold)
    slices = continuation_slices(len(circuits[0].gates))
    prompt_len = slices.depth + 2
    if not torch.equal(plus.inputs[:, :prompt_len], minus.inputs[:, :prompt_len]):
        raise RuntimeError("matched batches must share the prompt (including SEP)")
    if not torch.equal(plus.targets[:, list(slices.gate)], minus.targets[:, list(slices.gate)]):
        raise RuntimeError("matched batches must share gate-copy targets")
    if not torch.equal(plus.targets[:, list(slices.answer)], minus.targets[:, list(slices.answer)]):
        raise RuntimeError("matched batches must share the gold answer target")
    if torch.equal(plus.targets[:, list(slices.state)], minus.targets[:, list(slices.state)]):
        raise RuntimeError("wrong-trace state targets must differ from gold")
    if device is not None:
        plus, minus = plus.to(device), minus.to(device)
    return plus, minus


def _group_tensors(model) -> dict[str, list[torch.Tensor]]:
    """Flattening order: heads 0..H-1 of query then key; then value; then MLP."""
    qk, ov = [], []
    for head in model.heads:
        qk.extend([head.query, head.key])
        ov.append(head.value)
    mlp = [model.mlp_in, model.mlp_bias, model.mlp_out]
    return {"QK": qk, "OV": ov, "MLP": mlp}


def star_direction_vectors(star_model) -> dict[str, torch.Tensor]:
    """d_r* = concatenated constructed parameters, same order as learned grads."""
    out = {}
    for name, tensors in _group_tensors(star_model).items():
        flat = torch.cat([t.detach().float().reshape(-1).cpu() for t in tensors])
        out[name] = flat
    return out


def check_star_dirs(star_dirs: dict[str, torch.Tensor], min_norm: float = STAR_D_MIN_NORM) -> dict[str, float]:
    """Print ||d_r*|| and raise if any constructed group is numerically zero."""
    norms = {}
    for name in GROUPS:
        n = float(star_dirs[name].detach().float().norm())
        norms[name] = n
        print(f"star d* {name} norm={n:.6g}", flush=True)
        if n < min_norm:
            raise RuntimeError(f"star d* group {name} has norm {n} < {min_norm}")
    return norms


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    nx, ny = float(x.norm()), float(y.norm())
    if nx < EPS and ny < EPS:
        return 1.0
    if nx < EPS or ny < EPS:
        return 0.0
    return float(x @ y) / (nx * ny)


def _token_nll(model, batch) -> torch.Tensor:
    logits = model(batch.inputs)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        batch.targets.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).view_as(batch.targets)


def _mean_at(nll: torch.Tensor, index: tuple[int, ...]) -> torch.Tensor:
    return nll[:, list(index)].mean()


def _subset_losses(nll: torch.Tensor, slices: ProcessSlices) -> dict[str, torch.Tensor]:
    return {
        "L": _mean_at(nll, slices.continuation),
        "L_gate": _mean_at(nll, slices.gate),
        "L_state": _mean_at(nll, slices.state),
        "L_answer": _mean_at(nll, slices.answer),
        "L_syntax": _mean_at(nll, slices.syntax),
    }


def _grad_groups(loss: torch.Tensor, groups: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    names = list(groups)
    params = [p for name in names for p in groups[name]]
    grads = torch.autograd.grad(loss, params, create_graph=False, allow_unused=True)
    out = {}
    i = 0
    for name in names:
        pieces = []
        for param, grad in zip(groups[name], grads[i: i + len(groups[name])]):
            if grad is None:
                pieces.append(torch.zeros(param.numel(), device=param.device, dtype=torch.float32))
            else:
                pieces.append(grad.detach().float().reshape(-1))
        out[name] = torch.cat(pieces)
        i += len(groups[name])
    return out


def _rho_tag(rho: float) -> str:
    if abs(rho - round(rho)) < 1e-12:
        return f"rho{int(round(rho))}"
    return "rho" + f"{rho:.2f}".replace(".", "").rstrip("0")


def _polarity_metrics(model, batch, slices: ProcessSlices, groups) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    nll = _token_nll(model, batch)
    losses = _subset_losses(nll, slices)
    grads = _grad_groups(losses["L"], groups)
    row = {key: float(val.detach()) for key, val in losses.items()}
    return row, grads


def _eta_tag(eta: float) -> str:
    return "eta" + f"{eta:g}".replace(".", "").replace("-", "m")


def _flat_params(tensors) -> torch.Tensor:
    return torch.cat([t.detach().float().reshape(-1) for t in tensors])


def _apply_group_delta(tensors, flat_delta: torch.Tensor) -> None:
    """In-place theta_r += flat_delta, chunked back to each tensor's shape."""
    offset = 0
    with torch.no_grad():
        for tensor in tensors:
            size = tensor.numel()
            tensor.add_(flat_delta[offset: offset + size].view_as(tensor).to(tensor.dtype))
            offset += size


def functional_probe(model, groups, star_dirs, g_plus, g_minus, batch_plus,
                     slices: ProcessSlices, etas=FUNCTIONAL_ETAS) -> dict[str, float]:
    """What a trace's gradient actually does, as opposed to where it points.

    alpha_r^pm = -<g_r^pm, theta*_r> projects onto the constructed parameter
    vector itself, so its sign says the update grows the component of theta_r
    along theta*_r.  It does NOT say the update shortens ||theta_r - theta*_r||,
    which needs the gap direction, and it says nothing about competence.  This
    adds both of the quantities that do:

      beta_r^pm  = <-g_r^pm, (theta*_r - theta_r)> / ||theta*_r - theta_r||,
                   positive iff the step reduces the distance to theta*_r to
                   first order;
      proj_r^pm  = <g_r^pm, h_r> / ||h_r||^2 with h_r = grad_r L_state(clean).
                   A step -eta*g changes the clean state loss by
                   -eta*<g, h> + o(eta), so POSITIVE proj means the step
                   improves clean local competence.  Same convention and
                   normalisation as the Sec. 5.2 alignment diagnostic;
      dLclean    = L_state(clean; theta - eta g_r^pm) - L_state(clean; theta),
                   the same effect measured rather than linearized, with the
                   update restricted to group r.

    Negative dLclean means a step built from that trace improves the model's
    clean next-state predictions.
    """
    row: dict[str, float] = {}
    with torch.enable_grad():
        clean_state_grads = _grad_groups(_mean_at(_token_nll(model, batch_plus), slices.state), groups)
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        base = float(_mean_at(_token_nll(model, batch_plus), slices.state))
    row["L_state_clean_base"] = base

    for name in GROUPS:
        tensors = groups[name]
        theta = _flat_params(tensors).to(g_plus[name].device)
        star = star_dirs[name].to(device=theta.device, dtype=theta.dtype)
        gap = star - theta
        gap_norm = float(gap.norm())
        h = clean_state_grads[name].to(theta.device)
        h_sq = float(h @ h)
        row[f"gap_{name}_norm"] = gap_norm
        row[f"h_{name}_norm"] = float(h.norm())
        saved = [t.detach().clone() for t in tensors]
        for tag, grad in (("plus", g_plus[name]), ("minus", g_minus[name])):
            row[f"beta_{name}_{tag}"] = float(-(grad @ gap) / gap_norm) if gap_norm > EPS else 0.0
            row[f"proj_{name}_{tag}"] = float((grad @ h) / h_sq) if h_sq > EPS else 0.0
            for eta in etas:
                _apply_group_delta(tensors, -eta * grad)
                with torch.no_grad():
                    moved = float(_mean_at(_token_nll(model, batch_plus), slices.state))
                with torch.no_grad():
                    for tensor, original in zip(tensors, saved):
                        tensor.copy_(original)
                row[f"dLclean_{name}_{tag}_{_eta_tag(eta)}"] = moved - base
    model.zero_grad(set_to_none=True)
    return row


def matched_loss_and_grads(model, star_dirs, batch_plus, batch_minus, slices: ProcessSlices) -> dict[str, float]:
    """Full continuation CE, token-group CE, and QK/OV/MLP grad geometry on matched pairs.

    Sign of α: α_r^± = -<g_r^±, θ*_r>, a projection onto the constructed
    parameter vector itself. Positive α means a step along -g_r^± increases the
    component of θ_r along θ*_r. It does NOT mean the step shortens
    ||θ_r - θ*_r|| (that is β, see functional_probe) nor that it improves clean
    local competence (that is proj/dLclean, also in functional_probe).
    """
    model.eval()
    groups = _group_tensors(model)
    with torch.enable_grad():
        plus_loss, g_plus = _polarity_metrics(model, batch_plus, slices, groups)
        minus_loss, g_minus = _polarity_metrics(model, batch_minus, slices, groups)
    model.zero_grad(set_to_none=True)

    row: dict[str, float] = {
        "L_plus": plus_loss["L"],
        "L_minus": minus_loss["L"],
        "L_gate_plus": plus_loss["L_gate"],
        "L_gate_minus": minus_loss["L_gate"],
        "L_state_plus": plus_loss["L_state"],
        "L_state_minus": minus_loss["L_state"],
        "L_answer_plus": plus_loss["L_answer"],
        "L_answer_minus": minus_loss["L_answer"],
        "L_syntax_plus": plus_loss["L_syntax"],
        "L_syntax_minus": minus_loss["L_syntax"],
        "delta_L_trace": minus_loss["L_state"] - plus_loss["L_state"],
        "n_probe": float(batch_plus.inputs.shape[0]),
    }

    for name in GROUPS:
        gp, gm = g_plus[name], g_minus[name]
        d_star = star_dirs[name].to(device=gp.device, dtype=gp.dtype)
        row[f"g_{name}_plus_norm"] = float(gp.norm())
        row[f"g_{name}_minus_norm"] = float(gm.norm())
        row[f"cos_{name}"] = _cosine(gp, gm)
        row[f"alpha_{name}_plus"] = float(-(gp @ d_star))
        row[f"alpha_{name}_minus"] = float(-(gm @ d_star))
        row[f"star_d_{name}_norm"] = float(d_star.norm())
        a_plus, a_minus = row[f"alpha_{name}_plus"], row[f"alpha_{name}_minus"]
        for rho in RHO_GRID:
            row[f"alpha_{name}_{_rho_tag(rho)}"] = rho * a_plus + (1.0 - rho) * a_minus

    row.update(functional_probe(model, groups, star_dirs, g_plus, g_minus,
                                batch_plus, slices))

    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite probe value {key}={value}")
        if key.startswith("cos_") and not (-1.0 - 1e-5 <= value <= 1.0 + 1e-5):
            raise RuntimeError(f"cosine {key}={value} outside [-1, 1]")
    return row


def grad_match_columns() -> tuple[str, ...]:
    cols = [
        "step", "seed", "corruption", "n_probe",
        "L_plus", "L_minus",
        "L_gate_plus", "L_gate_minus",
        "L_state_plus", "L_state_minus",
        "L_answer_plus", "L_answer_minus",
        "L_syntax_plus", "L_syntax_minus",
        "delta_L_trace",
    ]
    for name in GROUPS:
        cols += [
            f"g_{name}_plus_norm", f"g_{name}_minus_norm", f"cos_{name}",
            f"alpha_{name}_plus", f"alpha_{name}_minus", f"star_d_{name}_norm",
        ]
        cols += [f"alpha_{name}_{_rho_tag(rho)}" for rho in RHO_GRID]
        cols += [f"gap_{name}_norm", f"h_{name}_norm"]
        for tag in ("plus", "minus"):
            cols += [f"beta_{name}_{tag}", f"proj_{name}_{tag}"]
            cols += [f"dLclean_{name}_{tag}_{_eta_tag(eta)}" for eta in FUNCTIONAL_ETAS]
    cols.append("L_state_clean_base")
    return tuple(cols)


def extract_grad_match_rows(rows: list[dict]) -> list[dict]:
    keys = grad_match_columns()
    return [{k: row[k] for k in keys if k in row} for row in rows]


def draw_trace_gradients(frame, path: Path) -> None:
    apply_style()
    grouped = frame.groupby("step", as_index=False).mean(numeric_only=True)
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.0))

    ax = axes[0]
    ax.plot(grouped["step"], grouped["L_state_plus"], "-", color="#087eaa", linewidth=2.0, label=r"$L_{\mathrm{state}}^+$")
    ax.plot(grouped["step"], grouped["L_state_minus"], "--", color="#c56b08", linewidth=2.0, label=r"$L_{\mathrm{state}}^-$")
    ax.plot(grouped["step"], grouped["delta_L_trace"], ":", color="#6a4c93", linewidth=2.0, label=r"$\Delta L_{\mathrm{trace}}$")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy")
    ax.set_title(r"(a) $L_{\mathrm{state}}^\pm$ and $\Delta L_{\mathrm{trace}}$")
    ax.legend(frameon=True, loc="best", fontsize=9)

    ax = axes[1]
    colors = {"QK": "#087eaa", "OV": "#c56b08", "MLP": "#2a9d8f"}
    for name in GROUPS:
        ax.plot(grouped["step"], grouped[f"cos_{name}"], "-", color=colors[name], linewidth=2.0, label=rf"$\cos_{{{name}}}$")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel("cosine")
    ax.set_title(r"(b) $\cos(g^+, g^-)$")
    ax.set_ylim(-1.05, 1.05)
    ax.legend(frameon=True, loc="best", fontsize=9)

    ax = axes[2]
    ax.plot(grouped["step"], grouped["alpha_QK_plus"], "-", color="#087eaa", linewidth=2.0, label=r"$\alpha_{QK}^+$")
    ax.plot(grouped["step"], grouped["alpha_QK_minus"], "--", color="#087eaa", linewidth=2.0, label=r"$\alpha_{QK}^-$")
    ax.plot(grouped["step"], grouped["alpha_MLP_plus"], "-", color="#2a9d8f", linewidth=2.0, label=r"$\alpha_{MLP}^+$")
    ax.plot(grouped["step"], grouped["alpha_MLP_minus"], "--", color="#2a9d8f", linewidth=2.0, label=r"$\alpha_{MLP}^-$")
    ax.axhline(0.0, color="#444444", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\alpha=-\langle g, d^\star\rangle$")
    ax.set_title(r"(c) $\alpha^\pm$ for QK and MLP")
    ax.legend(frameon=True, loc="best", fontsize=9)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)


def notes_section(slices: ProcessSlices, corruption: str) -> list[str]:
    d4 = continuation_slices(4)
    return [
        "## Matched clean vs wrong trace (loss + gradients)",
        "",
        r"Same prompt \(x\); two counterfactual process continuations. Training is still clean process CE.",
        r"- \(\tau^+(x)\): gold process `g_1,s_1,...,g_D,s_D,COLON,s_D,EOS`.",
        r"- \(\tau^-(x)\): Paper 1 `wrong_trace` on the same gates (matched), gold answer after COLON.",
        r"  Gate copies, COLON, gold \(s_D\), and EOS are shared; only the \(D\) state successors differ.",
        f"- Corruption law (this run): `{corruption}`. `symmetric` = independent wrong successor at each step;",
        "  `coherent` = one fixed competing rule per gate via `_coherent_wrong_mask` (XOR mask, Paper 1).",
        "",
        "CE is next-token on continuation positions only (prompt, including SEP, is `ignore_index=-100`).",
        r"Group means: \(L_{\mathrm{gate}}, L_{\mathrm{state}}, L_{\mathrm{answer}}, L_{\mathrm{syntax}}\)",
        r"with \(L_{\mathrm{state}}=-\frac1D\sum_k \log p(s_k\mid c_k)\) and",
        r"\(L_{\mathrm{answer}}=-\log p(s_D\mid c_{\mathrm{answer}})\).",
        r"Full continuation CE is \(L^\pm\) (mean over all \(2D+3\) continuation tokens).",
        r"\(\Delta L_{\mathrm{trace}}=L_{\mathrm{state}}^--L_{\mathrm{state}}^+\). Expect \(\approx 0\) at init, \(>0\) as the executor forms.",
        "",
        "The probe is eval-only at the existing circuit-match checkpoints. Wrong traces are never",
        "trained on; there is no outcome-train twin.",
        "",
        "### Token-index mapping (teacher-forced targets, D=4, T=16)",
        "",
        "- input: 0 `s0`; 1–4 prompt gates; 5 SEP; 6 `g1`; 7 `s1`; …; 12 `g4`; 13 `s4`; 14 COLON; 15 answer `s4`. EOS is the target of position 15.",
        rf"- gate copies \(p_k\): `{list(d4.gate)}` (predict `g_k`)",
        rf"- state successors \(r_k\): `{list(d4.state)}` (predict `s_k`)",
        rf"- syntax COLON, EOS: `{list(d4.syntax)}`  (SEP is prompt index 5 and is **not** in the loss)",
        rf"- answer after COLON: `{list(d4.answer)}`",
        rf"- all continuation: `{list(d4.continuation)}`",
        rf"- this run (D={slices.depth}): gate `{list(slices.gate)}`, state `{list(slices.state)}`, "
        rf"syntax `{list(slices.syntax)}`, answer `{list(slices.answer)}`.",
        "",
        "### Gradients and star projection",
        "",
        r"On the same matched batch: \(g^+=\nabla_\theta L^+\), \(g^-=\nabla_\theta L^-\) (full continuation CE).",
        "Groups, concatenated in this flattening order:",
        "- QK: every head's `query` then `key` (heads 0..3, including the unused construction head)",
        "- OV: every head's `value`",
        "- MLP: `mlp_in`, `mlp_bias`, `mlp_out`",
        r"Log \(\|g_r^+\|\), \(\|g_r^-\|\), and \(\cos_r=\langle g_r^+,g_r^-\rangle/(\|g_r^+\|\|g_r^-\|)\).",
        r"Hypothesis (not asserted): \(\cos_{QK}>0\), \(\cos_{OV}\gtrsim 0\), \(\cos_{MLP}<0\).",
        "",
        r"Star direction \(d_r^\star\) is the **vectorized constructed parameters** (not \(C_{QK}/C_{OV}\))",
        "in the same flattening order. Sign convention:",
        r"\[\alpha_r^\pm=-\langle g_r^\pm, d_r^\star\rangle.\]",
        r"\(\alpha>0\) means the gradient decreases loss in the direction of the true circuit",
        "(gradient descent would move that group toward star). Mixture",
        r"\(\alpha_r(\rho)=\rho\alpha_r^++(1-\rho)\alpha_r^-\) for \(\rho\in\{0,0.3,0.5,1\}\) is a derived column (no extra backward).",
        "",
    ]
