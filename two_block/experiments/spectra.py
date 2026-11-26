"""Jacobian/ANOVA spectrum measurements and single-head untied routing (Appendices G.1-G.3).

Subcommands:
    uv run python two_block/experiments/spectra.py spectrum task=count word_len=8 mod=2   # App. G.1 gradient decay
    uv run python two_block/experiments/spectra.py answers                                # App. G.2 Boolean-8 / register ANOVA
    uv run python two_block/experiments/spectra.py exact site=answer dirs=16              # App. G.1 exact Jacobian degree split
    uv run python two_block/experiments/spectra.py reach --format step --n 6 8 --sigma 0.3 --lr 2 --steps 300000
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import math
import sys
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jacrev, jvp, vmap

TWO_BLOCK_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = TWO_BLOCK_DIR.parent
for _path in (str(TWO_BLOCK_DIR), str(ROOT_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import counting as lt  # noqa: E402


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _parse_kv_dataclass(cls: type, argv: list[str], string_fields: frozenset[str]) -> Any:
    valid = {f.name for f in fields(cls)}
    kwargs: dict[str, Any] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Expected key=value argument, got {arg!r}")
        key, raw = arg.split("=", 1)
        if key not in valid:
            raise KeyError(f"Unknown setting {key!r}; valid keys: {sorted(valid)}")
        kwargs[key] = raw if key in string_fields else ast.literal_eval(raw)
    return cls(**kwargs)


def _orthonormal_letter_basis(alphabet: int, device: str | torch.device) -> torch.Tensor:
    """Orthonormal basis for L2(Uniform({0..A-1})) with constant column 0."""
    raw = torch.cat(
        [
            torch.ones(alphabet, 1),
            torch.randn(alphabet, alphabet - 1, generator=torch.Generator().manual_seed(1)),
        ],
        dim=1,
    )
    q_mat = torch.linalg.qr(raw)[0]
    return (q_mat * math.sqrt(alphabet) * torch.sign(q_mat[0, 0])).to(device)


# -----------------------------------------------------------------------------
# 1. spectrum: interaction spectra & answer-dependent gradients (Appendix G.1)
# -----------------------------------------------------------------------------


@dataclass
class SpectrumConfig:
    task: str = "count"
    alphabet: int = 4
    word_len: int = 8
    mod: int = 2
    n_blocks: int = 2
    init_std: float = 0.02
    seed: int = 42
    rhos: tuple[float, ...] = tuple(i / 10.0 for i in range(11))
    pairs: int = 512
    chunk: int = 32
    grad_prompts: int = 4_096
    exact: bool = True
    max_exact: int = 2_000_000
    trace_rhos: tuple[float, ...] = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
    out: str = ""
    ckpt: str = ""
    site: str = "answer"
    layout: str = "block"
    zero_readout: bool = False
    grads_only: bool = False
    match_p: float | None = None
    device: str = ""

    def __post_init__(self) -> None:
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rhos = tuple(self.rhos)
        self.trace_rhos = tuple(self.trace_rhos)
        if self.task not in ("count", "first"):
            raise ValueError(f"Unsupported task {self.task!r}")
        if self.site not in ("answer", "state"):
            raise ValueError(f"Unsupported site {self.site!r}")
        if self.layout not in ("block", "stream"):
            raise ValueError(f"Unsupported layout {self.layout!r}")
        if self.site == "state" and self.task != "count":
            raise ValueError("site='state' is only defined for task='count'")


def _load_or_init_lettertrace_theta(
    lt_cfg: lt.Config,
    ckpt: str,
    zero_readout: bool,
) -> tuple[dict[str, torch.Tensor], str]:
    torch.manual_seed(lt_cfg.seed)
    theta = {k: (torch.randn_like(t) * lt_cfg.init_std).to(lt_cfg.device) for k, t in lt.zero_params(lt_cfg).items()}
    if ckpt:
        saved = torch.load(ckpt, map_location=lt_cfg.device)
        scfg = saved["config"]
        for key in ("task", "word_len", "alphabet", "mod", "n_blocks"):
            assert scfg[key] == getattr(lt_cfg, key), f"Checkpoint has {key}={scfg[key]}, run has {getattr(lt_cfg, key)}"
        assert scfg.get("layout", lt_cfg.layout) == lt_cfg.layout
        assert set(saved["params"]) == set(theta)
        theta = {k: saved["params"][k].to(lt_cfg.device).float() for k in theta}
        return theta, f"{saved['mode']} step {saved['step']}"
    if zero_readout:
        theta["readout"].zero_()
    return theta, "init"


def cmd_spectrum(argv: list[str]) -> None:
    cfg: SpectrumConfig = _parse_kv_dataclass(
        SpectrumConfig, argv, frozenset({"task", "device", "out", "ckpt", "site", "layout"})
    )
    lt_cfg = lt.Config(
        task=cfg.task,
        alphabet=cfg.alphabet,
        word_len=cfg.word_len,
        mod=cfg.mod,
        n_blocks=cfg.n_blocks,
        init_std=cfg.init_std,
        seed=cfg.seed,
        layout=cfg.layout,
        device=cfg.device,
    )
    n, alphabet, num_vals = lt_cfg.word_len, lt_cfg.alphabet, lt_cfg.num_values
    p_match = (1.0 / alphabet) if cfg.match_p is None else float(cfg.match_p)
    assert 0.0 < p_match < 1.0, p_match

    theta, step_label = _load_or_init_lettertrace_theta(lt_cfg, cfg.ckpt, cfg.zero_readout)
    n_params = sum(t.numel() for t in theta.values())
    rng = np.random.default_rng(cfg.seed + 1)

    pred_positions = np.arange(1, n + 1) + (n + 1 if cfg.layout == "block" else np.arange(1, n + 1))
    query_pos = n + 3 if cfg.site == "answer" else int(pred_positions[-1])

    def compute_answer(s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> np.ndarray:
        matches = word == q[:, None]
        if cfg.task == "count":
            return (s0 + matches.sum(axis=1)) % cfg.mod
        return np.where(matches.any(axis=1), matches.argmax(axis=1) + 1, 0)

    def sample_prompts(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = rng.integers(alphabet, size=count)
        word = rng.integers(alphabet, size=(count, n))
        s0 = rng.integers(cfg.mod, size=count) if cfg.task == "count" else np.zeros(count, dtype=int)
        return s0, q, word

    def make_outcome_ids(s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> torch.Tensor:
        m = len(q)
        cols = [
            alphabet + s0[:, None],
            q[:, None],
            word,
            np.full((m, 1), lt_cfg.sep_token),
            np.full((m, 1), lt_cfg.colon_token),
        ]
        return torch.tensor(np.concatenate(cols, axis=1), device=cfg.device)

    def make_trace_ids(s0: np.ndarray, q: np.ndarray, word: np.ndarray, shown: np.ndarray) -> torch.Tensor:
        m, k = shown.shape
        if cfg.layout == "block":
            cols = [alphabet + s0[:, None], q[:, None], word, np.full((m, 1), lt_cfg.sep_token), alphabet + shown]
        else:
            cols = [alphabet + s0[:, None], q[:, None]]
            for t in range(n):
                cols.append(word[:, t : t + 1])
                if t < k:
                    cols.append(alphabet + shown[:, t : t + 1])
        return torch.tensor(np.concatenate(cols, axis=1), device=cfg.device)

    def make_process_rows(s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> tuple[torch.Tensor, np.ndarray]:
        matches = word == q[:, None]
        state_list: list[np.ndarray] = []
        s = s0.copy()
        for t in range(n):
            s = (s + matches[:, t]) % cfg.mod if cfg.task == "count" else np.where((s == 0) & matches[:, t], t + 1, s)
            state_list.append(s.copy())
        states = np.stack(state_list, axis=1)
        return make_trace_ids(s0, q, word, states), states

    def sample_state_prompts(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        s0, q, word = sample_prompts(count)
        shown = rng.integers(cfg.mod, size=(count, n - 1))
        return s0, q, word, shown

    def value_logits(params: dict[str, torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        return lt.forward(params, ids[None], lt_cfg)[0, query_pos, alphabet : alphabet + num_vals]

    batched_jac = vmap(jacrev(value_logits), in_dims=(None, 0))

    def flat_jacobian(ids: torch.Tensor) -> torch.Tensor:
        jac_dict = batched_jac(theta, ids)
        return torch.cat([t.reshape(ids.shape[0], num_vals, -1) for t in jac_dict.values()], dim=2)

    def measure_stability(rho: float) -> tuple[float, float, float, float]:
        c_chunks, d_chunks = [], []
        for start in range(0, cfg.pairs, cfg.chunk):
            m = min(cfg.chunk, cfg.pairs - start)
            if cfg.site == "answer":
                s0, q, word = sample_prompts(m)
                to_ids = lambda w: make_outcome_ids(s0, q, w)
            else:
                s0, q, word, shown = sample_state_prompts(m)
                to_ids = lambda w: make_trace_ids(s0, q, w, shown)
            fresh = rng.integers(alphabet, size=(m, n))
            word_rho = np.where(rng.random((m, n)) < rho, word, fresh)

            j_base = flat_jacobian(to_ids(word)).detach()
            j_rho = flat_jacobian(to_ids(word_rho)).detach()
            j_zero = flat_jacobian(to_ids(fresh)).detach()
            c_chunks.append((j_base * j_rho).sum((1, 2)).double().cpu())
            d_chunks.append((j_base * (j_rho - j_zero)).sum((1, 2)).double().cpu())

        c_all, d_all = torch.cat(c_chunks), torch.cat(d_chunks)
        return (
            c_all.mean().item(),
            c_all.std().item() / math.sqrt(len(c_all)),
            d_all.mean().item(),
            d_all.std().item() / math.sqrt(len(d_all)),
        )

    def compute_target_spectrum() -> np.ndarray:
        if cfg.site == "state":
            e0, e1 = np.eye(num_vals)[0], np.eye(num_vals)[1 % num_vals]
            a0 = (1.0 - p_match) * e0 + p_match * e1 - 1.0 / num_vals
            out = np.zeros(n + 1)
            out[0] = float((a0 ** 2).sum())
            out[1] = float(p_match * (1.0 - p_match) * ((e1 - e0) ** 2).sum())
            return out

        assert n <= 18, "Exact answer spectrum requires n <= 18"
        bits = (np.arange(2 ** n)[:, None] >> np.arange(n)[::-1]) & 1
        y = compute_answer(np.zeros(2 ** n, dtype=int), np.zeros(2 ** n, dtype=int), 1 - bits)
        grid = (np.eye(num_vals)[y] - 1.0 / num_vals).reshape((2,) * n + (num_vals,))
        scale = math.sqrt(p_match * (1.0 - p_match))
        for axis in range(n):
            f0 = np.take(grid, 0, axis=axis)
            f1 = np.take(grid, 1, axis=axis)
            grid = np.stack([(1.0 - p_match) * f0 + p_match * f1, scale * (f1 - f0)], axis=axis)
        energies = (grid ** 2).sum(axis=-1).reshape(-1)
        degrees = bits.sum(axis=1)
        return np.array([energies[degrees == r].sum() for r in range(n + 1)])

    def enumerate_all_prompts() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        n_s0 = cfg.mod if cfg.task == "count" else 1
        total = n_s0 * alphabet * (alphabet ** n)
        if not cfg.exact or total > cfg.max_exact:
            return None
        idx = np.arange(total)
        word = (idx[:, None] // (alphabet ** np.arange(n)[::-1])) % alphabet
        q = (idx // (alphabet ** n)) % alphabet
        s0 = idx // (alphabet ** (n + 1)) if cfg.task == "count" else np.zeros(total, dtype=int)
        return s0, q, word

    all_prompts = enumerate_all_prompts()

    def density_ratio(q: np.ndarray, word: np.ndarray) -> torch.Tensor | None:
        if cfg.match_p is None:
            return None
        n_matches = (word == np.asarray(q)[:, None]).sum(axis=1)
        log_w = n_matches * math.log(alphabet * p_match) + (n - n_matches) * math.log(
            alphabet * (1.0 - p_match) / (alphabet - 1)
        )
        return torch.tensor(np.exp(log_w), device=cfg.device, dtype=torch.float64)

    def weighted_mean(values: torch.Tensor, q: np.ndarray, word: np.ndarray) -> torch.Tensor:
        weights = density_ratio(q, word)
        if weights is None:
            return values.mean()
        return (values.double() * weights).sum() / values.shape[0]

    def population_grad_vec(scalar_fn: Callable[..., torch.Tensor], m_total: int, batch: int = 256) -> torch.Tensor:
        chunks: list[torch.Tensor] = []
        total = len(all_prompts[0]) if all_prompts is not None else m_total
        step = 4_096 if all_prompts is not None else batch
        for start in range(0, total, step):
            if all_prompts is not None:
                batch_args = tuple(arr[start : start + step] for arr in all_prompts)
            else:
                batch_args = sample_prompts(min(batch, m_total - start))
            params = {k: t.clone().requires_grad_() for k, t in theta.items()}
            scalar = scalar_fn(params, *batch_args) * (len(batch_args[0]) / total)
            grads = torch.autograd.grad(scalar, list(params.values()))
            chunks.append(torch.cat([g.reshape(-1) for g in grads]).double())
        return torch.stack(chunks).sum(dim=0)

    def outcome_credit_scalar(params: dict[str, torch.Tensor], s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> torch.Tensor:
        logits = lt.forward(params, make_outcome_ids(s0, q, word), lt_cfg)[:, n + 3, alphabet : alphabet + num_vals]
        one_hot = torch.eye(num_vals, device=cfg.device)[torch.tensor(compute_answer(s0, q, word), device=cfg.device)]
        return weighted_mean(((one_hot - 1.0 / num_vals) * logits).sum(dim=1), q, word)

    def outcome_residual_scalar(params: dict[str, torch.Tensor], s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> torch.Tensor:
        logits = lt.forward(params, make_outcome_ids(s0, q, word), lt_cfg)[:, n + 3]
        probs = logits.softmax(dim=-1)[:, alphabet : alphabet + num_vals].detach()
        return ((probs - 1.0 / num_vals) * logits[:, alphabet : alphabet + num_vals]).sum(dim=1).mean()

    def outcome_full_loss(params: dict[str, torch.Tensor], s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> torch.Tensor:
        logits = lt.forward(params, make_outcome_ids(s0, q, word), lt_cfg)[:, n + 3]
        targets = torch.tensor(alphabet + compute_answer(s0, q, word), device=cfg.device)
        return F.cross_entropy(logits, targets)

    def outcome_nonvalue_scalar(params: dict[str, torch.Tensor], s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> torch.Tensor:
        logits = lt.forward(params, make_outcome_ids(s0, q, word), lt_cfg)[:, n + 3]
        probs = logits.softmax(dim=-1).detach()
        mask = torch.ones(lt_cfg.vocab_size, device=cfg.device)
        mask[alphabet : alphabet + num_vals] = 0.0
        return (probs * mask * logits).sum(dim=1).mean()

    def process_credit_scalar(params: dict[str, torch.Tensor], s0: np.ndarray, q: np.ndarray, word: np.ndarray) -> torch.Tensor:
        ids, states = make_process_rows(s0, q, word)
        logits = lt.forward(params, ids, lt_cfg)[:, pred_positions, alphabet : alphabet + num_vals]
        one_hot = torch.eye(num_vals, device=cfg.device)[torch.tensor(states, device=cfg.device)]
        return weighted_mean(((one_hot - 1.0 / num_vals) * logits).sum(dim=-1).mean(dim=-1), q, word)

    lam: float | None = None
    if cfg.task == "count":
        omega = np.exp(2j * np.pi / cfg.mod)
        lam = float(max(abs(1.0 - p_match + p_match * omega ** j) for j in range(1, cfg.mod)))
    print(
        f"task={cfg.task} A={alphabet} n={n} p={p_match:g}"
        + (f" K={cfg.mod} lambda={lam:.4f}" if lam is not None else "")
        + f"  L={cfg.n_blocks} width={lt_cfg.model_width} params={n_params:,}  "
        + f"site={cfg.site}  layout={cfg.layout}  weights: {step_label}"
        + (" (zero readout)" if cfg.zero_readout and not cfg.ckpt else "")
    )

    a_r = compute_target_spectrum()
    if cfg.task == "count" and cfg.site == "answer":
        omega = np.exp(2j * np.pi / cfg.mod)
        g2 = [abs(1.0 - p_match + p_match * omega ** j) ** 2 for j in range(1, cfg.mod)]
        closed = [math.comb(n, r) * sum(x ** (n - r) * (1.0 - x) ** r for x in g2) / cfg.mod for r in range(n + 1)]
        assert np.allclose(a_r, closed), "Answer spectrum disagrees with closed form"
    print("target spectrum A_r (exact):", np.round(a_r, 5).tolist(), f" total {a_r.sum():.4f}")

    stability_rows: list[tuple[float, float, float, float]] | None = None
    w_r: np.ndarray | None = None
    a_fit = b_fit = hi_share = hi_share_se = best_bound = best_upper = rho_best = None

    if cfg.grads_only:
        print("grads_only: skipping C_J / W_r")
    else:
        stability_rows = [measure_stability(r) for r in cfg.rhos]
        print("\n  rho    C_J(rho)     s.e.      D(rho)=C_J(rho)-C_J(0)   s.e.")
        for r, (c_val, c_se, d_val, d_se) in zip(cfg.rhos, stability_rows):
            print(f"  {r:4.2f}   {c_val:.4e}  {c_se:.1e}   {d_val:+.3e}              {d_se:.1e}")

        pos_rhos = [r for r in cfg.rhos if r > 0]
        pos_rows = [row for r, row in zip(cfg.rhos, stability_rows) if r > 0]
        design = np.array([[r ** k for k in range(1, n + 1)] for r in pos_rhos])
        y_vals = np.array([row[2] for row in pos_rows])
        try:
            from scipy.optimize import nnls

            w_hi = nnls(design, y_vals)[0]
        except ImportError:
            w_hi = np.clip(np.linalg.lstsq(design, y_vals, rcond=None)[0], 0.0, None)

        w0 = stability_rows[0][0]
        w_r = np.concatenate([[w0], w_hi])

        rr = np.array(pos_rhos)
        dd = y_vals
        inv_se = 1.0 / np.maximum(np.array([row[3] for row in pos_rows]), 1e-30)
        x_quad = np.c_[rr, rr ** 2] * inv_se[:, None]
        coef = np.linalg.lstsq(x_quad, dd * inv_se, rcond=None)[0]
        cov = np.linalg.inv(x_quad.T @ x_quad)
        a_fit, b_fit = float(coef[0]), float(coef[1])
        grad_ratio = np.array([-b_fit, a_fit]) / (a_fit + b_fit) ** 2
        hi_share = float(b_fit / (a_fit + b_fit))
        hi_share_se = float(np.sqrt(grad_ratio @ cov @ grad_ratio))

        print("\nmodel spectrum W_r:", [f"{val:.2e}" for val in w_r])
        print(f"share of Jacobian energy at order 0: {w0 / stability_rows[-1][0]:.4f}")
        print(
            f"letter-dependent energy: order 1 ~ {a_fit:.3e}, orders >= 2 ~ {b_fit:.3e}; "
            f"order >= 2 share {hi_share:+.2f} +- {hi_share_se:.2f}"
        )

        candidates = []
        for r, row in zip(pos_rhos, pos_rows):
            tail = sum(a_r[k] * (r ** -k) for k in range(1, n + 1))
            b_nom = math.sqrt(w0 * a_r[0]) + math.sqrt(max(row[2], 0.0) * tail)
            b_hi = math.sqrt(w0 * a_r[0]) + math.sqrt(max(row[2] + 2.0 * row[3], 0.0) * tail)
            candidates.append((b_nom, b_hi, r))
        best_bound, best_upper, rho_best = min(candidates)
        print(f"\nbound: {best_bound:.3e}  (D+2se: {best_upper:.3e}; best rho = {rho_best:.2f})")

    mode_desc = "exact over all prompts" if all_prompts is not None else f"Monte Carlo, {cfg.grad_prompts} prompts"
    grads_summary: dict[str, float] = {}
    if cfg.site == "answer":
        g_out = population_grad_vec(outcome_credit_scalar, cfg.grad_prompts)
        g_proc = population_grad_vec(process_credit_scalar, cfg.grad_prompts)
        grads_summary = {"grad_outcome": g_out.norm().item(), "grad_process": g_proc.norm().item()}
        ratio = grads_summary["grad_process"] / max(grads_summary["grad_outcome"], 1e-30)
        print(f"gradients: {mode_desc}")
        print(f"  outcome ||E[J_v^T(e_y-u)]||:    {grads_summary['grad_outcome']:.3e}")
        print(f"  step    ||E_t E[J_t^T(e_st-u)]||: {grads_summary['grad_process']:.3e}   (ratio {ratio:.1f}x)")
        if not cfg.grads_only:
            g_res = population_grad_vec(outcome_residual_scalar, cfg.grad_prompts)
            g_nv = population_grad_vec(outcome_nonvalue_scalar, cfg.grad_prompts)
            g_full = population_grad_vec(outcome_full_loss, cfg.grad_prompts)
            split_err = (g_full - (g_res - g_out + g_nv)).norm().item() / max(g_full.norm().item(), 1e-30)
            grads_summary.update(
                grad_residual=g_res.norm().item(),
                grad_nonvalue=g_nv.norm().item(),
                grad_full=g_full.norm().item(),
                split_error=split_err,
            )
            print(f"  full answer-token gradient:       {grads_summary['grad_full']:.3e}   (split err {split_err:.1e})")
    else:
        vecs = []
        for start in range(0, cfg.grad_prompts, 256):
            batch_s0, batch_q, batch_w, batch_shown = sample_state_prompts(min(256, cfg.grad_prompts - start))
            params = {k: t.clone().requires_grad_() for k, t in theta.items()}
            logits = lt.forward(params, make_trace_ids(batch_s0, batch_q, batch_w, batch_shown), lt_cfg)[
                :, query_pos, alphabet : alphabet + num_vals
            ]
            prev = batch_shown[:, -1] if n > 1 else batch_s0
            tgt = (prev + (batch_w[:, -1] == batch_q)) % cfg.mod
            one_hot = torch.eye(num_vals, device=cfg.device)[torch.tensor(tgt, device=cfg.device)]
            sc = ((one_hot - 1.0 / num_vals) * logits).sum(dim=1).mean()
            grads = torch.autograd.grad(sc, list(params.values()))
            vecs.append(torch.cat([g.reshape(-1) for g in grads]).double())
        stacked = torch.stack(vecs)
        mean_vec = stacked.mean(dim=0)
        se_val = math.sqrt(((stacked - mean_vec) ** 2).sum().item() / (len(stacked) * (len(stacked) - 1)))
        grads_summary = {"grad_state": mean_vec.norm().item(), "grad_state_se": se_val}
        print(f"state-position ||E[J^T(e_target-u)]||: {grads_summary['grad_state']:.3e} (s.e. {se_val:.1e})")

    if cfg.out:
        payload: dict[str, Any] = {
            "task": cfg.task,
            "A": alphabet,
            "n": n,
            "K": cfg.mod,
            "L": cfg.n_blocks,
            "seed": cfg.seed,
            "init_std": cfg.init_std,
            "match_p": p_match,
            "lambda_theory": lam,
            "layout": cfg.layout,
            "A_r": a_r.tolist(),
            "gradients": mode_desc,
            "site": cfg.site,
            "weights": step_label,
            "ckpt": cfg.ckpt,
            "zero_readout": cfg.zero_readout,
            "grads_only": cfg.grads_only,
            **grads_summary,
        }
        if stability_rows is not None and w_r is not None:
            payload.update(
                rhos=list(cfg.rhos),
                C_J=[r[0] for r in stability_rows],
                C_J_se=[r[1] for r in stability_rows],
                D=[r[2] for r in stability_rows],
                D_se=[r[3] for r in stability_rows],
                W_r=w_r.tolist(),
                bound=best_bound,
                bound_upper=best_upper,
                rho_best=rho_best,
                W1_fit=a_fit,
                Whi_fit=b_fit,
                hi_share=hi_share,
                hi_share_se=hi_share_se,
            )
        out_path = Path(cfg.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2))


# -----------------------------------------------------------------------------
# 2. answers: exact ANOVA spectra of Boolean-8 & Register mod 17 (Appendix G.2)
# -----------------------------------------------------------------------------


def _markov_mixing_rate(transitions: list[np.ndarray], weights: list[float]) -> float:
    num_states = transitions[0].shape[0]
    avg_op = sum(w * t for w, t in zip(weights, transitions))
    proj = np.eye(num_states) - np.ones((num_states, num_states)) / num_states
    return float(np.linalg.svd(proj @ avg_op @ proj, compute_uv=False)[0])


def _exact_automaton_spectrum(
    transitions: list[np.ndarray],
    weights: list[float],
    horizon: int,
    readout: np.ndarray,
) -> np.ndarray:
    """Dynamic programming over second moments giving exact degree-r ANOVA energies A_r."""
    num_states = transitions[0].shape[0]
    avg_op = sum(w * t for w, t in zip(weights, transitions))
    covs = [np.zeros((num_states, num_states)) for _ in range(horizon + 1)]
    covs[0] = (np.eye(num_states) - np.ones((num_states, num_states)) / num_states) / num_states

    for step_idx in range(horizon):
        next_covs = [np.zeros((num_states, num_states)) for _ in range(horizon + 1)]
        for deg in range(step_idx + 1):
            cur = covs[deg]
            avg_next = avg_op.T @ cur @ avg_op
            next_covs[deg] += avg_next
            next_covs[deg + 1] += sum(w * (t.T @ cur @ t) for w, t in zip(weights, transitions)) - avg_next
        covs = next_covs

    return np.array([float(np.trace(readout.T @ cov @ readout)) for cov in covs])


def cmd_answers(_argv: list[str]) -> None:
    # Boolean-8 (A_16 on {0,1}^4)
    n_bits = 4
    n_states = 1 << n_bits

    def to_bits(val: int) -> list[int]:
        return [int(b) for b in f"{val:0{n_bits}b}"]

    def from_bits(bits: list[int]) -> int:
        return int("".join(map(str, bits)), 2)

    def apply_gate(state_bits: list[int], gate: tuple[Any, ...]) -> list[int]:
        out = state_bits.copy()
        op = gate[0]
        if op == "x":
            out[gate[1]] ^= 1
        elif op == "c":
            out[gate[2]] ^= out[gate[1]]
        elif op == "s":
            out[gate[1]], out[gate[2]] = out[gate[2]], out[gate[1]]
        else:
            out[gate[3]] ^= out[gate[1]] & out[gate[2]]
        return out

    gate_law: list[tuple[tuple[Any, ...], float]] = [(("x", i), 0.25 / 4) for i in range(n_bits)]
    for op in ("c", "s"):
        gate_law.extend(((op, a, b), 0.25 / 12) for a, b in itertools.permutations(range(n_bits), 2))
    gate_law.extend((("t", a, b, c), 0.25 / 24) for a, b, c in itertools.permutations(range(n_bits), 3))

    bool_ops: list[np.ndarray] = []
    bool_weights: list[float] = []
    for gate, weight in gate_law:
        mat = np.zeros((n_states, n_states))
        for s in range(n_states):
            mat[s, from_bits(apply_gate(to_bits(s), gate))] = 1.0
        bool_ops.append(mat)
        bool_weights.append(weight)

    bool_spec = _exact_automaton_spectrum(bool_ops, bool_weights, 8, np.eye(n_states))
    print(f"boolean lambda {_markov_mixing_rate(bool_ops, bool_weights):.4f}")
    print(
        f"boolean D=8  total {bool_spec.sum():.4f} (1-1/K={1 - 1 / n_states:.4f})  "
        f"A_r: {np.round(bool_spec, 4).tolist()}  "
        f"share r>=4: {bool_spec[4:].sum() / bool_spec.sum():.3f}  "
        f"share r<=1: {bool_spec[:2].sum() / bool_spec.sum():.4f}"
    )

    # Modulo-17 two-register machine (readout: x register)
    mod_p = 17
    reg_states = mod_p * mod_p
    reg_ops: list[np.ndarray] = []
    for op_code in ("a", "b", "c", "d", "e"):
        mat = np.zeros((reg_states, reg_states))
        for x in range(mod_p):
            for y in range(mod_p):
                if op_code == "a":
                    nx, ny = (x + y) % mod_p, y
                elif op_code == "b":
                    nx, ny = x, (x + y) % mod_p
                elif op_code == "c":
                    nx, ny = y, x
                elif op_code == "d":
                    nx, ny = (x + 1) % mod_p, y
                else:
                    nx, ny = x, (y + 1) % mod_p
                mat[x * mod_p + y, nx * mod_p + ny] = 1.0
        reg_ops.append(mat)

    readout = np.zeros((reg_states, mod_p))
    for x in range(mod_p):
        for y in range(mod_p):
            readout[x * mod_p + y, x] = 1.0

    reg_spec = _exact_automaton_spectrum(reg_ops, [0.2] * 5, 16, readout)
    mean_order = float((np.arange(17) * reg_spec).sum() / reg_spec.sum())
    print(f"register lambda {_markov_mixing_rate(reg_ops, [0.2] * 5):.4f}")
    print(
        f"register D=16 x-readout total {reg_spec.sum():.4f} (1-1/17={1 - 1 / mod_p:.4f})  "
        f"A_r: {np.round(reg_spec, 4).tolist()}  "
        f"share r>=8: {reg_spec[8:].sum() / reg_spec.sum():.3f}  "
        f"share r<=1: {reg_spec[:2].sum() / reg_spec.sum():.4f}  mean order {mean_order:.2f}"
    )


# -----------------------------------------------------------------------------
# 3. exact: exact letter-order Jacobian spectrum via forward-mode JVPs (App. G.1)
# -----------------------------------------------------------------------------


@dataclass
class ExactSpectrumConfig:
    ckpt: str = ""
    site: str = "answer"
    every_k: int = 2
    shown_draws: int = 2
    dirs: int = 16
    seed: int = 0
    init_std: float = 0.02
    init_seed: int = 42
    n: int = 8
    A: int = 4
    mod: int = 2
    n_blocks: int = 2
    device: str = ""

    def __post_init__(self) -> None:
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


def cmd_exact(argv: list[str]) -> None:
    cfg: ExactSpectrumConfig = _parse_kv_dataclass(
        ExactSpectrumConfig, argv, frozenset({"ckpt", "device", "site"})
    )
    torch.set_default_dtype(torch.float64)

    lt_cfg = lt.Config(
        task="count",
        word_len=cfg.n,
        alphabet=cfg.A,
        mod=cfg.mod,
        n_blocks=cfg.n_blocks,
        init_std=cfg.init_std,
        seed=cfg.init_seed,
        device=cfg.device,
    )
    n, alphabet, num_vals = cfg.n, cfg.A, cfg.mod
    q_pos = n + 3

    saved: dict[str, Any] | None = None
    if cfg.ckpt:
        saved = torch.load(cfg.ckpt, map_location=cfg.device)
        scfg = saved["config"]
        assert (scfg["task"], scfg["word_len"], scfg["alphabet"], scfg["mod"], scfg["n_blocks"]) == (
            "count",
            n,
            alphabet,
            cfg.mod,
            cfg.n_blocks,
        ), scfg
        theta = {k: t.to(cfg.device).double() for k, t in saved["params"].items()}
        label = f"{saved['mode']} step {saved['step']}"
    else:
        torch.manual_seed(cfg.init_seed)
        theta = {
            k: (torch.randn(t.shape, dtype=torch.float32) * cfg.init_std).double().to(cfg.device)
            for k, t in lt.zero_params(lt_cfg).items()
        }
        label = f"init std={cfg.init_std} seed={cfg.init_seed}"

    words = torch.tensor((np.arange(alphabet ** n)[:, None] // (alphabet ** np.arange(n)[::-1])) % alphabet, device=cfg.device)
    q_basis = _orthonormal_letter_basis(alphabet, cfg.device)
    order_idx = (words != 0).sum(dim=1)

    def subset_energy(values: torch.Tensor) -> torch.Tensor:
        coeffs = values.reshape((alphabet,) * n + (num_vals,))
        for axis in range(n):
            coeffs = torch.tensordot(coeffs, q_basis, dims=([axis], [0])) / alphabet
            coeffs = coeffs.movedim(-1, axis)
        return (coeffs ** 2).sum(dim=-1).reshape(-1)

    def order_energy(values: torch.Tensor) -> torch.Tensor:
        return torch.zeros(n + 1, device=cfg.device).index_add_(0, order_idx, subset_energy(values))

    def answer_ids(s0: int, q: int) -> torch.Tensor:
        m = len(words)
        return torch.cat(
            [
                torch.full((m, 1), alphabet + s0, device=cfg.device),
                torch.full((m, 1), q, device=cfg.device),
                words,
                torch.full((m, 1), lt_cfg.sep_token, device=cfg.device),
                torch.full((m, 1), lt_cfg.colon_token, device=cfg.device),
            ],
            dim=1,
        )

    def answer_logits(params: dict[str, torch.Tensor], ids: torch.Tensor) -> torch.Tensor:
        return lt.forward(params, ids, lt_cfg)[:, q_pos, alphabet : alphabet + num_vals]

    if cfg.site == "answer":
        gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
        w_r = torch.zeros(n + 1, device=cfg.device)
        a_r = torch.zeros(n + 1, device=cfg.device)
        grad_phi = {k: torch.zeros_like(t) for k, t in theta.items()}
        combos = [(s0, q) for s0 in range(cfg.mod) for q in range(alphabet)]

        for s0, q in combos:
            ids = answer_ids(s0, q)
            y = (s0 + (words == q).sum(dim=1)) % cfg.mod
            centered_target = torch.eye(num_vals, device=cfg.device)[y] - 1.0 / num_vals
            a_r += order_energy(centered_target) / len(combos)

            for _ in range(cfg.dirs):
                direction = {k: torch.randn(t.shape, generator=gen, device=cfg.device) for k, t in theta.items()}
                _, tangent = jvp(lambda p: answer_logits(p, ids), (theta,), (direction,))
                w_r += order_energy(tangent) / (len(combos) * cfg.dirs)

            params = {k: t.clone().requires_grad_() for k, t in theta.items()}
            phi = (centered_target * answer_logits(params, ids)).sum(dim=1).mean() / len(combos)
            grads = torch.autograd.grad(phi, list(params.values()))
            combo_norm = len(combos) * math.sqrt(sum((g ** 2).sum().item() for g in grads))
            print(f"s0={s0} q={q} per-combo ||grad Phi|| {combo_norm:.4e}")
            for k, g in zip(params, grads):
                grad_phi[k] += g

        w_np, a_np = w_r.cpu().numpy(), a_r.cpu().numpy()
        g_norm = math.sqrt(sum((t ** 2).sum().item() for t in grad_phi.values()))
        w_tot = float(w_np.sum())
        print(f"== {label}")
        print("W_r / W :", " ".join(f"{x:.3e}" for x in w_np / w_tot), f"  (W = {w_tot:.4e})")
        print("A_r     :", " ".join(f"{x:.3e}" for x in a_np), f"  (sum = {a_np.sum():.4f})")
        print(f"order-0 share {w_np[0] / w_tot:.6f}   H_0/W {w_np[1:].sum() / w_tot:.3e}   H_1/W {w_np[2:].sum() / w_tot:.3e}")
        print(f"||grad Phi|| exact {g_norm:.4e}   sum_r sqrt(A_r W_r) {np.sqrt(a_np * w_np).sum():.4e}")
        for q_cut in range(3):
            lo = math.sqrt(a_np[: q_cut + 1].sum() * w_np[: q_cut + 1].sum())
            hi = math.sqrt(a_np[q_cut + 1 :].sum() * w_np[q_cut + 1 :].sum())
            print(f"  q={q_cut}: sqrt(A<=q W<=q) {lo:.3e}   sqrt(A>q H_q) {hi:.3e}   bound {lo + hi:.3e}")
        return

    # every_k site
    if saved is not None:
        assert saved["config"].get("every_k", 1) == cfg.every_k
    k_step = cfg.every_k
    n_chunks = n // k_step
    rng = np.random.default_rng(cfg.seed)
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)

    def state_ids(s0: int, q: int, shown: np.ndarray) -> torch.Tensor:
        m = len(words)
        cols = [
            torch.full((m, 1), alphabet + s0, device=cfg.device),
            torch.full((m, 1), q, device=cfg.device),
            words,
            torch.full((m, 1), lt_cfg.sep_token, device=cfg.device),
        ]
        cols.extend(torch.full((m, 1), alphabet + int(v), device=cfg.device) for v in shown)
        return torch.cat(cols, dim=1)

    rows = []
    for chunk_idx in range(1, n_chunks + 1):
        window = list(range((chunk_idx - 1) * k_step, chunk_idx * k_step))
        nz = words != 0
        outside_axes = [t for t in range(n) if t not in window]
        in_win_only = nz[:, outside_axes].sum(dim=1) == 0
        win_order = nz[:, window].sum(dim=1)
        masks = {f"win_order{r}": (in_win_only & (win_order == r)) for r in range(k_step + 1)}
        masks["outside"] = ~in_win_only

        w_acc = {name: 0.0 for name in masks}
        a_acc = {name: 0.0 for name in masks}
        total_w = 0.0
        count = 0

        for s0 in range(cfg.mod):
            for q in range(alphabet):
                for _ in range(cfg.shown_draws if chunk_idx > 1 else 1):
                    shown = rng.integers(cfg.mod, size=chunk_idx - 1)
                    prev = int(shown[-1]) if chunk_idx > 1 else s0
                    ids = state_ids(s0, q, shown)
                    pos = n + 2 + (chunk_idx - 1)
                    y = (prev + (words[:, window] == q).sum(dim=1)) % cfg.mod
                    ea = subset_energy(torch.eye(num_vals, device=cfg.device)[y] - 1.0 / num_vals)
                    for name, mask in masks.items():
                        a_acc[name] += ea[mask].sum().item()
                    for _ in range(cfg.dirs):
                        direction = {kk: torch.randn(t.shape, generator=gen, device=cfg.device) for kk, t in theta.items()}
                        _, tangent = jvp(
                            lambda p: lt.forward(p, ids, lt_cfg)[:, pos, alphabet : alphabet + num_vals],
                            (theta,),
                            (direction,),
                        )
                        energy = subset_energy(tangent)
                        total_w += energy.sum().item() / cfg.dirs
                        for name, mask in masks.items():
                            w_acc[name] += energy[mask].sum().item() / cfg.dirs
                    count += 1
        rows.append(
            (chunk_idx, {m: v / total_w for m, v in w_acc.items()}, {m: v / count for m, v in a_acc.items()}, total_w / count)
        )

    print(f"== {label}  every_k={k_step}")
    for chunk_idx, w_share, a_share, w_mean in rows:
        w_str = " ".join(f"{m}={v:.3e}" for m, v in w_share.items())
        a_str = " ".join(f"{m}={v:.3e}" for m, v in a_share.items() if v > 1e-12)
        print(f"v{chunk_idx}: W share {w_str}  | A {a_str}  | W={w_mean:.3e}")


# -----------------------------------------------------------------------------
# 4. reach: single-head untied routing (Appendix G.3)
# -----------------------------------------------------------------------------


def _run_single_reach(
    word_len: int,
    num_states: int,
    alphabet: int,
    sigma: float,
    seed: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dev = torch.device(args.device)
    torch.manual_seed(seed)
    sep_token = alphabet + num_states
    vocab_size = alphabet + num_states + 1
    seq_cap = 2 * word_len + 6
    d_model = 2 * alphabet + 4 * num_states + seq_cap + 3
    d_mlp = max(alphabet, 3 * num_states * (word_len + 1), 2 * word_len + 1)
    num_heads = 3
    trained_suffixes = {"q", "k", "rb", "readout"}

    def sample_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        words = torch.randint(alphabet, (batch_size, word_len))
        matches = (words == 0).long()
        s0 = torch.randint(num_states, (batch_size,))
        states = (s0[:, None] + matches.cumsum(dim=1)) % num_states
        ids = torch.cat(
            [
                alphabet + s0[:, None],
                torch.zeros(batch_size, 1, dtype=torch.long),
                words,
                torch.full((batch_size, 1), sep_token, dtype=torch.long),
                alphabet + states[:, :-1],
            ],
            dim=1,
        )
        if args.format == "outcome":
            return ids[:, : word_len + 3].to(dev), states[:, -1:].to(dev)
        return ids.to(dev), states.to(dev)

    x_train, y_train = sample_batch(args.batch)
    x_eval, y_eval = sample_batch(4 * args.batch)
    seq_len = x_train.shape[1]
    pred_pos = (
        torch.tensor([word_len + 2], device=dev)
        if args.format == "outcome"
        else torch.arange(word_len + 2, 2 * word_len + 2, device=dev)
    )

    def scaled_randn(fan_in: int, *shape: int) -> torch.Tensor:
        return sigma * torch.randn(*shape, device=dev) / math.sqrt(fan_in)

    theta: dict[str, torch.Tensor] = {
        "wte": scaled_randn(d_model, vocab_size, d_model),
        "wpe": scaled_randn(d_model, seq_cap, d_model),
        "readout": scaled_randn(d_model, num_states, d_model),
    }
    for block in range(2):
        theta[f"{block}.q"] = scaled_randn(d_model, num_heads, seq_cap, d_model)
        theta[f"{block}.k"] = scaled_randn(d_model, num_heads, seq_cap, d_model)
        theta[f"{block}.v"] = scaled_randn(d_model, num_heads, d_model, d_model)
        theta[f"{block}.rb"] = torch.zeros(num_heads, seq_cap, device=dev)
        theta[f"{block}.mlp_in"] = scaled_randn(d_model, d_mlp, d_model)
        theta[f"{block}.mlp_out"] = scaled_randn(d_mlp, d_model, d_mlp)
        signs = torch.randint(2, (d_mlp,), device=dev) * 2 - 1
        theta[f"{block}.mlp_b"] = sigma * signs * (1.0 + torch.rand(d_mlp, device=dev))

    trained_params = [t.requires_grad_() for name, t in theta.items() if name.split(".")[-1] in trained_suffixes]
    lag_index = (torch.arange(seq_len, device=dev)[:, None] - torch.arange(seq_len, device=dev)[None, :]).clamp(min=0)
    causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=dev).triu(1)
    scale = math.sqrt(seq_cap)

    def forward_logits(ids: torch.Tensor) -> torch.Tensor:
        hidden = theta["wte"][ids] + theta["wpe"][:seq_len]
        for block in range(2):
            q = torch.einsum("btw,hdw->bhtd", hidden, theta[f"{block}.q"])
            k = torch.einsum("btw,hdw->bhtd", hidden, theta[f"{block}.k"])
            v = torch.einsum("btw,hvw->bhtv", hidden, theta[f"{block}.v"])
            scores = q @ k.transpose(-1, -2) / scale + theta[f"{block}.rb"][:, lag_index]
            hidden = hidden + (scores.masked_fill(causal_mask, -math.inf).softmax(dim=-1) @ v).sum(dim=1)
            hidden = hidden + F.relu(hidden @ theta[f"{block}.mlp_in"].T + theta[f"{block}.mlp_b"]) @ theta[f"{block}.mlp_out"].T
        return hidden[:, pred_pos] @ theta["readout"].T

    def eval_margin_and_exact(ids: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
        with torch.no_grad():
            logits = forward_logits(ids)
            gold = logits.gather(-1, targets[..., None]).squeeze(-1)
            other = logits.scatter(-1, targets[..., None], float("-inf")).max(dim=-1).values
            min_margin = (gold - other).min().item()
            exact_acc = (logits.argmax(dim=-1) == targets).all(dim=1).double().mean().item()
            return min_margin, exact_acc

    log: list[dict[str, Any]] = []
    for step_idx in range(args.steps + 1):
        loss = F.cross_entropy(forward_logits(x_train).reshape(-1, num_states), y_train.reshape(-1))
        if step_idx % args.every == 0:
            min_margin, exact_acc = eval_margin_and_exact(x_eval, y_eval)
            log.append(
                {
                    "step": step_idx,
                    "loss": loss.item(),
                    "margin": min_margin,
                    "exact": exact_acc,
                    "rb": max(theta[f"{b}.rb"].abs().max().item() for b in range(2)),
                    "readout": theta["readout"].norm().item(),
                }
            )
            solved_step = next((r["step"] for r in log if r["margin"] > 0), None)
            if solved_step is not None and (args.armijo or step_idx >= solved_step + 5_000):
                break

        for param in trained_params:
            param.grad = None
        loss.backward()
        with torch.no_grad():
            for param in trained_params:
                assert param.grad is not None
                param -= args.lr * param.grad

    basin: dict[str, Any] = {}
    if args.armijo and any(r["margin"] > 0 for r in log):
        pi_min = 1.0 / (x_train.shape[0] * word_len)
        trace: list[dict[str, Any]] = []
        entered: int | None = None
        for upd in range(args.armijo):
            cur_loss = F.cross_entropy(forward_logits(x_train).reshape(-1, num_states), y_train.reshape(-1))
            for param in trained_params:
                param.grad = None
            cur_loss.backward()
            grad_sq = sum((p.grad ** 2).sum().item() for p in trained_params if p.grad is not None)
            ratio = 4.0 * cur_loss.item() / pi_min
            if entered is None and ratio < 1.0:
                entered = upd
            ev_margin, _ = eval_margin_and_exact(x_eval, y_eval)
            trace.append({"update": upd, "loss": cur_loss.item(), "ratio": ratio, "heldout_margin": ev_margin})
            if entered is not None and upd >= entered + 100:
                break

            eta = 100.0
            with torch.no_grad():
                saved_params = [p.detach().clone() for p in trained_params]
                while True:
                    for p, prev in zip(trained_params, saved_params):
                        assert p.grad is not None
                        p.copy_(prev - eta * p.grad)
                    new_loss = F.cross_entropy(forward_logits(x_train).reshape(-1, num_states), y_train.reshape(-1)).item()
                    if new_loss <= cur_loss.item() - 0.5 * eta * grad_sq or eta < 1e-8:
                        break
                    eta *= 0.5
        basin = {"entered": entered, "trace": trace}

    return log, basin


def cmd_reach(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="measure.py reach")
    parser.add_argument("--n", type=int, nargs="+", default=[6])
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--A", type=int, default=4)
    parser.add_argument("--sigma", type=float, nargs="+", default=[0.3])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=150_000)
    parser.add_argument("--batch", type=int, default=4_096)
    parser.add_argument("--every", type=int, default=500)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--format", choices=("step", "outcome"), default="step")
    parser.add_argument("--armijo", type=int, default=0)
    args = parser.parse_args(argv)
    torch.set_default_dtype(torch.float64)

    results: list[dict[str, Any]] = []
    for word_len in args.n:
        for sigma in args.sigma:
            for seed in args.seeds:
                t0 = time.monotonic()
                log, basin = _run_single_reach(word_len, args.K, args.A, sigma, seed, args)
                solved = next((r["step"] for r in log if r["margin"] > 0), None)
                last = log[-1]
                print(
                    f"n={word_len:3d} sigma={sigma:<6} seed={seed} solved_at={solved} "
                    f"final loss {last['loss']:.4f} margin {last['margin']:+.3f} "
                    f"exact {last['exact']:.3f} max|rb| {last['rb']:.2f} "
                    f"basin_entered={basin.get('entered')} ({time.monotonic() - t0:.0f}s)",
                    flush=True,
                )
                results.append(
                    {
                        "n": word_len,
                        "K": args.K,
                        "A": args.A,
                        "sigma": sigma,
                        "seed": seed,
                        "lr": args.lr,
                        "log": log,
                        "basin": basin,
                    }
                )
                if args.out:
                    out_path = Path(args.out)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    out_path.write_text(json.dumps(results, indent=2))


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------

COMMANDS: dict[str, Callable[[list[str]], None]] = {
    "spectrum": cmd_spectrum,
    "answers": cmd_answers,
    "exact": cmd_exact,
    "reach": cmd_reach,
}


def main(argv: list[str] | None = None) -> None:
    args = sys.argv[1:] if argv is None else argv
    if not args or args[0] not in COMMANDS:
        raise SystemExit(__doc__)
    COMMANDS[args[0]](args[1:])


if __name__ == "__main__":
    main()
