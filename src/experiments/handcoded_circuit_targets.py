"""Star QK/OV/MLP maps of `HandcodedProcessTransformer` (Paper 2 circuit match).

Buffers in code are `query`/`key`/`value` (not Q_buf/K_buf/V_buf). With
W_Q = query.T, W_K = key.T, W_V = value.T and W_O = I:

    C_QK = query.T @ key / sqrt(P)
    C_OV = value.T

Trust this file and `handcoded/models.py` over any mismatch in the prose
checklist; `algebra_audit()` records the deltas.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from handcoded.gates import phi
from handcoded.models import HandcodedProcessTransformer
from handcoded.tokenizer import make_tokenizer

STAR_HEAD_NAMES = ("S", "G", "F")
ALPHA = 80.0
MLP_BIAS = -1.5
UNEMBED_SCALE = 20.0
N_PROGRAMMED_HEADS = 3


@dataclass(frozen=True)
class ResidualLayout:
    """S ⊕ G ⊕ Pos ⊕ R ⊕ G_out ⊕ S_out residual partition."""

    K: int
    M: int
    D: int
    P: int
    d: int
    n_heads: int
    vocab: int
    n_bits: int
    S: slice
    G: slice
    Pos: slice
    R: slice
    G_out: slice
    S_out: slice
    gates: tuple[str, ...] = ()

    @property
    def KM(self) -> int:
        return self.K * self.M

    def as_dict(self) -> dict:
        return {
            "K": self.K, "M": self.M, "D": self.D, "P": self.P, "d": self.d,
            "n_heads": self.n_heads, "n_layers": 1, "vocab": self.vocab,
            "KM": self.KM, "alpha": ALPHA, "mlp_bias": MLP_BIAS,
            "unembed_scale": UNEMBED_SCALE,
            "slices": {
                name: [getattr(self, name).start, getattr(self, name).stop]
                for name in ("S", "G", "Pos", "R", "G_out", "S_out")
            },
        }


def residual_layout(tokenizer, depth: int) -> ResidualLayout:
    """Same slices as `HandcodedProcessTransformer.__init__`."""
    states, gates = tokenizer.n_states, len(tokenizer.gates)
    n_bits = int(math.log2(states))
    positions = 3 * depth + 4
    state = slice(0, states)
    gate = slice(state.stop, state.stop + gates)
    position = slice(gate.stop, gate.stop + positions)
    retrieved = slice(position.stop, position.stop + states)
    output_gate = slice(retrieved.stop, retrieved.stop + gates)
    output_state = slice(output_gate.stop, output_gate.stop + states)
    return ResidualLayout(
        K=states, M=gates, D=depth, P=positions, d=output_state.stop,
        n_heads=4, vocab=len(tokenizer.tokens), n_bits=n_bits,
        S=state, G=gate, Pos=position, R=retrieved, G_out=output_gate, S_out=output_state,
        gates=tuple(tokenizer.gates),
    )


def process_head_routes(depth: int) -> dict[str, dict[int, int]]:
    """Programmed dest←origin maps. Must match `HandcodedProcessTransformer`."""
    state_routes, gate_routes = {}, {}
    for step in range(depth):
        gate_position = depth + 2 + 2 * step
        state_routes[gate_position] = 0 if step == 0 else gate_position - 1
        gate_routes[gate_position - 1] = step + 1
    final_routes = {3 * depth + 2: 3 * depth + 1}
    return {"S": state_routes, "G": gate_routes, "F": final_routes}


def _as_matrix(module, name: str) -> torch.Tensor:
    tensor = getattr(module, name)
    if tensor is None:
        raise AttributeError(name)
    return tensor.detach().float()


def head_circuits(head) -> tuple[torch.Tensor, torch.Tensor]:
    """(C_QK, C_OV) in residual coordinates. Shapes (d, d)."""
    query, key, value = _as_matrix(head, "query"), _as_matrix(head, "key"), _as_matrix(head, "value")
    scale = math.sqrt(query.shape[0])
    c_qk = query.T @ key / scale
    c_ov = value.T
    return c_qk.cpu(), c_ov.cpu()


def mlp_weights(model) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """W_in (KM, d), b (KM,), W_out (d, KM) — same layout as the construction."""
    return _as_matrix(model, "mlp_in").cpu(), _as_matrix(model, "mlp_bias").cpu(), _as_matrix(model, "mlp_out").cpu()


@dataclass
class StarCircuits:
    layout: ResidualLayout
    routes: dict[str, dict[int, int]]
    c_qk: dict[str, torch.Tensor] = field(default_factory=dict)
    c_ov: dict[str, torch.Tensor] = field(default_factory=dict)
    w_in: torch.Tensor = None
    bias: torch.Tensor = None
    w_out: torch.Tensor = None
    n_unused_heads: int = 1

    def cpu(self) -> "StarCircuits":
        self.c_qk = {k: v.cpu() for k, v in self.c_qk.items()}
        self.c_ov = {k: v.cpu() for k, v in self.c_ov.items()}
        self.w_in = self.w_in.cpu()
        self.bias = self.bias.cpu()
        self.w_out = self.w_out.cpu()
        return self


def extract_star_circuits(model: HandcodedProcessTransformer | None = None, *, tokenizer=None, depth: int | None = None) -> StarCircuits:
    """Read C_QK*/C_OV*/MLP* off a constructed process executor."""
    if model is None:
        tokenizer = make_tokenizer() if tokenizer is None else tokenizer
        depth = 4 if depth is None else depth
        model = HandcodedProcessTransformer(tokenizer, depth)
    tokenizer = tokenizer or make_tokenizer()
    depth = int(model.max_length - 4) // 3 if depth is None else depth
    layout = residual_layout(tokenizer, depth)
    if tuple(model.heads[0].query.shape) != (layout.P, layout.d):
        raise ValueError(f"head query {tuple(model.heads[0].query.shape)} != {(layout.P, layout.d)}")
    star = StarCircuits(layout=layout, routes=process_head_routes(depth), n_unused_heads=len(model.heads) - N_PROGRAMMED_HEADS)
    for name, head in zip(STAR_HEAD_NAMES, model.heads):
        star.c_qk[name], star.c_ov[name] = head_circuits(head)
    star.w_in, star.bias, star.w_out = mlp_weights(model)
    return star.cpu()


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """<vec(A), vec(B)> / (||A||_F ||B||_F). Both-zero → 1."""
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    nx, ny = float(x.norm()), float(y.norm())
    if nx < 1e-12 and ny < 1e-12:
        return 1.0
    if nx < 1e-12 or ny < 1e-12:
        return 0.0
    return float(x @ y) / (nx * ny)


def centered_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.detach().float().reshape(-1), b.detach().float().reshape(-1)
    return cosine(x - x.mean(), y - y.mean())


def positional_qk(c_qk: torch.Tensor, layout: ResidualLayout) -> torch.Tensor:
    return c_qk[layout.Pos, layout.Pos]


def pointer_accuracy(c_qk: torch.Tensor, routes: dict[int, int], layout: ResidualLayout) -> float:
    """Fraction of programmed dest←origin routes whose Pos-block argmax is origin."""
    if not routes:
        return float("nan")
    block = positional_qk(c_qk, layout)
    hits = 0
    for dest, origin in routes.items():
        row = block[dest, : dest + 1]
        hits += int(int(row.argmax()) == origin)
    return hits / len(routes)


def ov_block(c_ov: torch.Tensor, src: slice, dst: slice) -> torch.Tensor:
    """C_OV[src, dst]: hidden_src @ this → output_dst."""
    return c_ov[src, dst]


def ov_block_scores(c_ov: torch.Tensor, src: slice, dst: slice) -> tuple[float, float]:
    """Cosine of the src→dst block vs I, and Frobenius mass in that block."""
    block = ov_block(c_ov, src, dst)
    eye = torch.eye(block.shape[0], block.shape[1], dtype=block.dtype)
    mass_num = float(block.square().sum())
    mass_den = float(c_ov.square().sum())
    mass = 0.0 if mass_den < 1e-12 else mass_num / mass_den
    return cosine(block, eye), mass


OV_BLOCK_SLICES = {
    "S": ("S", "R"),
    "G": ("G", "G_out"),
    "F": ("S", "S_out"),
}


def _transition_inputs(layout: ResidualLayout) -> tuple[torch.Tensor, torch.Tensor]:
    """Rows x_{s,g} = e_s^R + e_g^G, plus gold φ(s,g)."""
    index = torch.arange(layout.KM)
    s_ids = torch.div(index, layout.M, rounding_mode="floor")
    g_ids = index % layout.M
    x = torch.zeros(layout.KM, layout.d)
    x[index, layout.R.start + s_ids] = 1.0
    x[index, layout.G.start + g_ids] = 1.0
    gold = torch.tensor(
        [phi(int(s), layout.gates[int(g)], layout.n_bits) for s, g in zip(s_ids.tolist(), g_ids.tolist())],
        dtype=torch.long,
    )
    return x, gold


def _row_cosine_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a, dim=1, eps=1e-12)
    b_n = F.normalize(b, dim=1, eps=1e-12)
    return a_n @ b_n.T


def _hungarian_align(learned: torch.Tensor, star: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Permute learned rows to maximize cosine vs star rows. Returns (perm, aligned)."""
    from scipy.optimize import linear_sum_assignment

    sim = _row_cosine_matrix(learned, star).cpu().numpy()
    n = min(sim.shape)
    row_ind, col_ind = linear_sum_assignment(-sim[:n, :n])
    perm = torch.full((star.shape[0],), -1, dtype=torch.long)
    perm[torch.as_tensor(col_ind)] = torch.as_tensor(row_ind)
    aligned = learned[perm]
    return perm, aligned


def mlp_transition_scores(w_in, bias, w_out, star: StarCircuits) -> dict[str, float]:
    """Compact MLP scores vs W_MLP^* (no KM×d plots)."""
    layout = star.layout
    x, gold = _transition_inputs(layout)
    hidden = F.relu(x @ w_in.T + bias)
    out = hidden @ w_out.T
    pred = out[:, layout.S_out].argmax(dim=-1)
    readout_acc = float((pred == gold).float().mean())

    perm, aligned_in = _hungarian_align(w_in, star.w_in)
    aligned_out = w_out[:, perm]
    win_cos = cosine(aligned_in, star.w_in)
    wout_cos = cosine(aligned_out, star.w_out)
    peak = float((hidden.argmax(dim=-1) == perm).float().mean()) if (perm >= 0).all() else 0.0
    return {
        "mlp_readout_acc": readout_acc,
        "mlp_hidden_peak": peak,
        "mlp_win_cosine": win_cos,
        "mlp_wout_cosine": wout_cos,
        "mlp_sim": 0.5 * (win_cos + wout_cos),
    }


def match_star_heads(learned_qk: list[torch.Tensor], learned_ov: list[torch.Tensor], star: StarCircuits) -> dict[str, int]:
    """Assign each star head a unique learned head (Hungarian on QK+OV cosine)."""
    from scipy.optimize import linear_sum_assignment

    sim = torch.zeros(len(STAR_HEAD_NAMES), len(learned_qk))
    for i, name in enumerate(STAR_HEAD_NAMES):
        for j, (qk, ov) in enumerate(zip(learned_qk, learned_ov)):
            sim[i, j] = cosine(qk, star.c_qk[name]) + cosine(ov, star.c_ov[name])
    row_ind, col_ind = linear_sum_assignment(-sim.cpu().numpy())
    return {STAR_HEAD_NAMES[i]: int(j) for i, j in zip(row_ind.tolist(), col_ind.tolist())}


def circuit_match_metrics(model, star: StarCircuits) -> dict[str, float]:
    """Similarity of a process-architecture model to the constructed maps."""
    qk, ov = zip(*(head_circuits(head) for head in model.heads))
    assignment = match_star_heads(list(qk), list(ov), star)
    row: dict[str, float] = {}
    for name in STAR_HEAD_NAMES:
        h = assignment[name]
        src, dst = OV_BLOCK_SLICES[name]
        block_cos, block_mass = ov_block_scores(ov[h], getattr(star.layout, src), getattr(star.layout, dst))
        row[f"sim_qk_{name}"] = cosine(qk[h], star.c_qk[name])
        row[f"sim_ov_{name}"] = cosine(ov[h], star.c_ov[name])
        row[f"sim_qk_{name}_centered"] = centered_cosine(qk[h], star.c_qk[name])
        row[f"sim_ov_{name}_centered"] = centered_cosine(ov[h], star.c_ov[name])
        row[f"ptr_{name}"] = pointer_accuracy(qk[h], star.routes[name], star.layout)
        row[f"ov_block_{name}"] = block_cos
        row[f"ov_mass_{name}"] = block_mass
        row[f"head_{name}"] = h
    w_in, bias, w_out = mlp_weights(model)
    row.update(mlp_transition_scores(w_in, bias, w_out, star))
    return row


def algebra_audit(star: StarCircuits | None = None) -> list[dict]:
    """User checklist vs `handcoded/models.py`. Trust the code on disagreement."""
    star = star or extract_star_circuits()
    L = star.layout
    expected_d = 3 * L.K + 2 * L.M + L.P
    rows = [
        {"item": "K", "user": 16, "code": L.K, "ok": L.K == 16},
        {"item": "M", "user": 52, "code": L.M, "ok": L.M == 52},
        {"item": "D", "user": 4, "code": L.D, "ok": L.D == 4},
        {"item": "P=3D+4", "user": 16, "code": L.P, "ok": L.P == 3 * L.D + 4},
        {"item": "d=3K+2M+P", "user": 168, "code": L.d, "ok": L.d == expected_d},
        {"item": "KM", "user": 832, "code": L.KM, "ok": L.KM == L.K * L.M},
        {"item": "vocab V", "user": 72, "code": L.vocab, "ok": L.vocab == L.K + L.M + 4},
        {"item": "n_heads", "user": "3 (S,G,F)", "code": L.n_heads, "ok": False,
         "note": "code allocates 4 FixedAttentionHead; heads[:3] are S,G,F; head 3 is unused (Q=0,V=0, K[:,Pos]=I)"},
        {"item": "n_layers", "user": 1, "code": 1, "ok": True},
        {"item": "alpha", "user": 80, "code": ALPHA, "ok": ALPHA == 80.0},
        {"item": "mlp_bias", "user": -1.5, "code": MLP_BIAS, "ok": MLP_BIAS == -1.5},
        {"item": "unembed_scale", "user": 20, "code": UNEMBED_SCALE, "ok": UNEMBED_SCALE == 20.0},
        {"item": "Q/K/V names", "user": "Q_buf,K_buf,V_buf", "code": "query,key,value", "ok": False,
         "note": "same shapes P×d, P×d, d×d"},
        {"item": "C_QK", "user": "Q_buf.T @ K_buf / sqrt(P)", "code": "query.T @ key / sqrt(P)", "ok": True},
        {"item": "C_OV", "user": "V_buf.T", "code": "value.T (W_O=I)", "ok": True},
        {"item": "MLP W_in", "user": "e_s^R + e_g^G", "code": "mlp_in[u,R[s]]=1 and mlp_in[u,G[g]]=1 (token G, not G_out)", "ok": True},
        {"item": "default routes", "user": "only programmed dests", "code": "every dest in [SEP,P) with no route points at SEP", "ok": False,
         "note": "pointer accuracy uses programmed routes only; C_QK* includes the SEP fillers"},
        {"item": "unembed extra", "user": "20 I_K on S_out and 20 I_M on G_out", "code": "plus COLON/EOS from Pos slots", "ok": False,
         "note": "readout[colon, Pos(3D+1)]=20 and readout[eos, Pos(3D+3)]=20"},
    ]
    return rows


def verify_star_targets(star: StarCircuits | None = None) -> StarCircuits:
    """Construction self-check: programmed Pos(d),Pos(o) entries and MLP φ."""
    star = star or extract_star_circuits()
    L = star.layout
    scale = ALPHA / math.sqrt(L.P)
    for name in STAR_HEAD_NAMES:
        qk = star.c_qk[name]
        ptr = pointer_accuracy(qk, star.routes[name], L)
        if ptr != 1.0:
            raise AssertionError(f"star pointer accuracy {name}={ptr}, expected 1")
        for dest, origin in star.routes[name].items():
            entry = float(qk[L.Pos.start + dest, L.Pos.start + origin])
            if abs(entry - scale) > 1e-4:
                raise AssertionError(f"C_QK^{name}[Pos {dest}←{origin}]={entry}, expected {scale}")
        src, dst = OV_BLOCK_SLICES[name]
        block_cos, _ = ov_block_scores(star.c_ov[name], getattr(L, src), getattr(L, dst))
        if block_cos < 1.0 - 1e-5:
            raise AssertionError(f"star OV block {name} cosine={block_cos}, expected 1")
    mlp = mlp_transition_scores(star.w_in, star.bias, star.w_out, star)
    if mlp["mlp_readout_acc"] < 1.0 - 1e-6:
        raise AssertionError(f"star MLP readout acc={mlp['mlp_readout_acc']}")
    if mlp["mlp_hidden_peak"] < 1.0 - 1e-6:
        raise AssertionError(f"star MLP hidden peak={mlp['mlp_hidden_peak']}")
    if abs(float(star.bias.mean()) - MLP_BIAS) > 1e-6:
        raise AssertionError(f"star MLP bias mean={float(star.bias.mean())}")
    self = circuit_match_metrics(_star_model(star), star)
    for name in STAR_HEAD_NAMES:
        if self[f"sim_qk_{name}"] < 1.0 - 1e-5 or self[f"sim_ov_{name}"] < 1.0 - 1e-5:
            raise AssertionError(f"self-similarity failed for {name}: {self}")
    return star


def _star_model(star: StarCircuits) -> HandcodedProcessTransformer:
    tok = make_tokenizer()
    if star.layout.D != 4 or tok.n_states != star.layout.K:
        raise ValueError("verify path assumes D=4 / default tokenizer")
    return HandcodedProcessTransformer(tok, star.layout.D)


def star_qk_route_entries(star: StarCircuits) -> dict[str, dict[str, float]]:
    L = star.layout
    out = {}
    for name in STAR_HEAD_NAMES:
        out[name] = {
            f"{dest}<-{origin}": float(star.c_qk[name][L.Pos.start + dest, L.Pos.start + origin])
            for dest, origin in star.routes[name].items()
        }
    return out
