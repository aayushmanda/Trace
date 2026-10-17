"""Rebuild the two handcoded executors from the specification in the paper's executor appendix (app:architectures),
without calling any construction helper of lettertrace.py, and check that every tensor
equals handcoded_params() entry for entry.

    uv run python handcoded/theory_checks/executor_check.py
"""
import sys

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))  # handcoded/

import torch

import lettertrace as L


def build(task, n, A, K):
    d = 2 * n + 6                                   # positions = head dimension
    sizes = dict(tl=A, tv=K, pos=d, qry=A, match=1, prev=K, mt=1, cnt=1, s0=K, out=K)
    reg, o = {}, 0
    for name, s in sizes.items():
        reg[name] = list(range(o, o + s))
        o += s
    D = o
    F = max(A, 3 * K * (n + 1), 2 * n + 1)
    V = A + K + 3
    SEP, COLON, EOS = A + K, A + K + 1, A + K + 2
    gam = max(16.0, 3 * torch.log(torch.tensor(8.0 * (n + 3))).item())
    C = gam * d ** 0.5

    def zeros():
        p = {"wte": torch.zeros(V, D), "wpe": torch.zeros(d, D), "readout": torch.zeros(V, D)}
        for b in (0, 1):
            p[f"{b}.q"], p[f"{b}.k"], p[f"{b}.v"] = torch.zeros(3, d, D), torch.zeros(3, d, D), torch.zeros(3, D, D)
            p[f"{b}.mlp_in"], p[f"{b}.mlp_b"], p[f"{b}.mlp_out"] = torch.zeros(F, D), torch.zeros(F), torch.zeros(D, F)
        return p

    def route(p, b, h, src, fallback):
        """(W_K)_{j,pos[j]} = 1;  (W_Q)_{j,pos[i]} = C for j in src(i), else j = fallback."""
        for j in range(d):
            p[f"{b}.k"][h, j, reg["pos"][j]] = 1
        for i in range(d):
            for j in src.get(i, [fallback]):
                p[f"{b}.q"][h, j, reg["pos"][i]] = C

    def copy(p, b, h, dst, srcreg, scale=1.0):          # W_V = scale * E_{dst <- src}
        for a, c in zip(reg[dst], reg[srcreg]):
            p[f"{b}.v"][h, a, c] = scale

    def shared():
        p = zeros()
        for a in range(A):
            p["wte"][a, reg["tl"][a]] = 1
        for v in range(K):
            p["wte"][A + v, reg["tv"][v]] = 1
        for i in range(d):
            p["wpe"][i, reg["pos"][i]] = 1
        route(p, 0, 0, {1 + t: [1] for t in range(1, n + 1)}, fallback=0)
        copy(p, 0, 0, "qry", "tl")
        for a in range(A):                               # unit a
            p["0.mlp_in"][a, reg["tl"][a]] = 1
            p["0.mlp_in"][a, reg["qry"][a]] = 1
            p["0.mlp_b"][a] = -1.5
            p["0.mlp_out"][reg["match"][0], a] = 2
        return p

    def readout(p, i_colon, i_eos):
        for v in range(K):
            p["readout"][A + v, reg["out"][v]] = 20
        p["readout"][COLON, reg["pos"][i_colon]] = 20
        p["readout"][EOS, reg["pos"][i_eos]] = 20
        return p

    # step program: position n+1+t predicts s_t
    p = shared()
    route(p, 1, 0, {n + 2: [0], **{n + 1 + t: [n + 1 + t] for t in range(2, n + 1)}}, fallback=1)
    copy(p, 1, 0, "prev", "tv")
    route(p, 1, 1, {n + 1 + t: [1 + t] for t in range(1, n + 1)}, fallback=0)
    copy(p, 1, 1, "mt", "match")
    route(p, 1, 2, {2 * n + 3: [2 * n + 2]}, fallback=1)
    copy(p, 1, 2, "out", "tv")
    W_in, b_in, W_out = p["1.mlp_in"], p["1.mlp_b"], p["1.mlp_out"]
    if task == "count":
        for j in range(K):
            for beta in (0, 1):                          # unit u = 2j + beta
                u = 2 * j + beta
                W_in[u, reg["prev"][j]] = 1
                W_in[u, reg["mt"][0]] = 2 * beta - 1
                b_in[u] = -0.5 - beta
                W_out[reg["out"][(j + beta) % K], u] = 2
    else:                                                # first occurrence, K = n + 1
        u = 0
        for j in range(1, K):                            # keep a found index
            W_in[u, reg["prev"][j]] = 1; b_in[u] = -0.5; W_out[reg["out"][j], u] = 2; u += 1
        W_in[u, reg["prev"][0]] = 1; W_in[u, reg["mt"][0]] = -1; b_in[u] = -0.5   # not yet
        W_out[reg["out"][0], u] = 2; u += 1
        for t in range(1, n + 1):                        # found at step t
            W_in[u, reg["prev"][0]] = 1; W_in[u, reg["mt"][0]] = 1
            W_in[u, reg["pos"][n + 1 + t]] = 1; b_in[u] = -2.5
            W_out[reg["out"][t], u] = 2; u += 1
    step = readout(p, 2 * n + 2, 2 * n + 4)

    # outcome program: position n+3 (':') predicts s_n
    p = shared()
    ans = n + 3
    if task == "count":
        route(p, 1, 0, {ans: list(range(2, n + 2))}, fallback=0)
        copy(p, 1, 0, "cnt", "match", scale=n)
        route(p, 1, 1, {ans: [0]}, fallback=1)
        copy(p, 1, 1, "s0", "tv")
        g, kappa = n + 2, {1: 1, 0: -2, -1: 1}
        for j in range(K):
            for c in range(n + 1):
                for k, delta in enumerate((1, 0, -1)):   # unit u = 3((n+1)j + c) + k
                    u = 3 * ((n + 1) * j + c) + k
                    p["1.mlp_in"][u, reg["cnt"][0]] = 1
                    p["1.mlp_in"][u, reg["s0"][j]] = g
                    p["1.mlp_b"][u] = delta - c - g
                    p["1.mlp_out"][reg["out"][(j + c) % K], u] = kappa[delta]
    else:                                                # earliest matched position wins
        Wq, Wk = p["1.q"][0], p["1.k"][0]
        Wk[0, reg["match"][0]] = 1
        for j in range(d):
            Wk[1, reg["pos"][j]] = -j
        Wk[2, reg["pos"][1]] = 1
        Wq[0, reg["pos"][ans]] = C * (n + 3)
        Wq[1, reg["pos"][ans]] = C
        for i in range(d):
            if i != ans:
                Wq[2, reg["pos"][i]] = C
        for t in range(1, n + 1):
            p["1.v"][0, reg["out"][t], reg["pos"][1 + t]] = 1
        p["1.v"][0, reg["out"][0], reg["pos"][0]] = 1
    outcome = readout(p, n + 2, n + 4)
    return {"process": step, "outcome": outcome}


if __name__ == "__main__":
    ok = True
    for task, n, mod in (("count", 5, 3), ("count", 8, 2), ("count", 16, 3), ("first", 6, 2), ("first", 12, 2)):
        L.configure(task=task, word_len=n, mod=mod, device="cpu")
        ref = L.handcoded_params()
        mine = build(task, n, L.A, L.V)
        for mode in ("process", "outcome"):
            for key, t in ref[mode].items():
                same = torch.equal(t.cpu().float(), mine[mode][key].float())
                if not same:
                    ok = False
                    diff = (t.cpu().float() - mine[mode][key].float()).abs()
                    print(f"MISMATCH {task} n={n} {mode} {key}: max |diff| {diff.max().item():.3g} at {diff.argmax().item()}")
        print(f"{task} n={n} K={L.V}: {'identical' if ok else 'differs'}")
    sys.exit(0 if ok else 1)
