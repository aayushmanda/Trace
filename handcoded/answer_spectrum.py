"""Answer spectra of the validation tasks (Table 1 lambda and the validation-spectra appendix).

    python handcoded/answer_spectrum.py

Exact answer spectrum A_r = sum_{|S|=r} E||a_S||^2 for a reversible task with
uniform s0, by dynamic programming over second moments (Lemma B.2):
  t not in S: Sigma -> P^T Sigma P;  t in S: Sigma -> E[T^T Sigma T] - P^T Sigma P.
The answer may be a readout of the state (register machine reports x only)."""
import numpy as np, itertools
def mixing_rate(Ts, ws):
    """lambda of Eq. (2): largest singular value of the averaged move on mean-zero vectors."""
    K = Ts[0].shape[0]
    P = sum(w * T for w, T in zip(ws, Ts))
    C = np.eye(K) - np.ones((K, K)) / K
    return np.linalg.svd(C @ P @ C, compute_uv=False)[0]
def spectrum(Ts, ws, n, R):
    K = Ts[0].shape[0]
    P = sum(w * T for w, T in zip(ws, Ts))
    Sig = [np.zeros((K, K)) for _ in range(n + 1)]
    Sig[0] = (np.eye(K) - np.ones((K, K)) / K) / K          # E[(e_s0-u)(e_s0-u)^T]
    for t in range(n):
        new = [np.zeros((K, K)) for _ in range(n + 1)]
        for r in range(t + 1):
            S_ = Sig[r]
            avg = P.T @ S_ @ P
            new[r] += avg
            new[r + 1] += sum(w * T.T @ S_ @ T for w, T in zip(ws, Ts)) - avg
        Sig = new
    return np.array([np.trace(R.T @ S_ @ R) for S_ in Sig])   # answer = R^T(e_y - u)
# Boolean-8
NB = 4; K = 16
def bits(v): return [int(b) for b in f"{v:0{NB}b}"]
def val(b): return int("".join(map(str, b)), 2)
def apply(s, g):
    r = s.copy(); op = g[0]
    if op == "x": r[g[1]] ^= 1
    elif op == "c": r[g[2]] ^= r[g[1]]
    elif op == "s": r[g[1]], r[g[2]] = r[g[2]], r[g[1]]
    else: r[g[3]] ^= r[g[1]] & r[g[2]]
    return r
law = [(("x", i), .25 / 4) for i in range(NB)]
for op in "cs": law += [((op, a, b), .25 / 12) for a, b in itertools.permutations(range(NB), 2)]
law += [(("t", a, b, c), .25 / 24) for a, b, c in itertools.permutations(range(NB), 3)]
Ts, ws = [], []
for g, w in law:
    T = np.zeros((K, K))
    for s in range(K): T[s, val(apply(bits(s), g))] = 1
    Ts.append(T); ws.append(w)
print("boolean lambda %.4f" % mixing_rate(Ts, ws))
A = spectrum(Ts, ws, 8, np.eye(K))
print("boolean D=8  total %.4f (1-1/K=%.4f)" % (A.sum(), 1 - 1 / K), " A_r:", np.round(A, 4).tolist(),
      " share at r>=D/2: %.3f" % (A[4:].sum() / A.sum()), " share r<=1: %.4f" % (A[:2].sum() / A.sum()))
# register, answer = x register
M = 17; K = M * M
def reg(x, y, i): return {"a": ((x + y) % M, y), "b": (x, (x + y) % M), "c": (y, x), "d": ((x + 1) % M, y), "e": (x, (y + 1) % M)}[i]
Ts = []
for i in "abcde":
    T = np.zeros((K, K))
    for x in range(M):
        for y in range(M):
            a, b = reg(x, y, i); T[x * M + y, a * M + b] = 1
    Ts.append(T)
R = np.zeros((K, M))
for x in range(M):
    for y in range(M): R[x * M + y, x] = 1
print("register lambda %.4f" % mixing_rate(Ts, [.2] * 5))
A = spectrum(Ts, [.2] * 5, 16, R)
print("register D=16 x-readout total %.4f (1-1/17=%.4f)" % (A.sum(), 1 - 1 / 17), " A_r:", np.round(A, 4).tolist(),
      " share r>=8: %.3f" % (A[8:].sum() / A.sum()), " share r<=1: %.4f" % (A[:2].sum() / A.sum()),
      " mean order %.2f" % ((np.arange(17) * A).sum() / A.sum()))
