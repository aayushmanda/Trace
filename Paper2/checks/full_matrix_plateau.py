"""Full-matrix cross-entropy plateau (Theorem 3): exit time from the eps-ball around the structured initialization.

Every entry of the query, key, value, output, MLP (with bias) and readout matrices trains from exactly the matrices of
Section 2.1 (r1 = r2 = eps, zero scores). Exact population of the anti-copy law, binary cross-entropy, backtracking
gradient descent with fixed step 0.5 (flow time = 0.5 x steps). Records the first flow time with ||W - W0||_F >= eps, the loss decrease up to that
time, and the largest source-label correlation |E[Y z]| seen inside the ball, which Theorem 3 bounds by gamma C_B eps^3.

    python Paper2/checks/full_matrix_plateau.py              # residual width 7, as in the paper's network
    python Paper2/checks/full_matrix_plateau.py --dres 64    # embeddings and W0 zero-padded to width 64

With --dres D > 7 every matrix is D-dimensional and fully trainable, the setting of the theorem for any width.
From exactly W0 the padded coordinates receive zero gradient and stay zero, so the run reproduces width 7 exactly.
--perturb starts instead from W0 plus a Gaussian perturbation of Frobenius norm eps^{3/2} spread over every entry,
padded coordinates included (part (iii) of the theorem); the exit time is still measured from W0.
"""
import json
import math
import sys
from pathlib import Path

DRES = int(sys.argv[sys.argv.index("--dres") + 1]) if "--dres" in sys.argv else 7
PERTURB = "--perturb" in sys.argv
sys.argv = [sys.argv[0], "--device", "cuda" if __import__("torch").cuda.is_available() else "cpu"]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import untied_ladder as U

torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[1]
N, ETA, B = 16, .1, 4.
EPS = [.03, .02, .01, .005, .003, .002]
LAWS = {1.: 0, 0.: 1}                                   # kappa -> number of anti-copies
dev = U.dev


def structured(eps, H=4, D=DRES):
    th = dict(Wq=torch.zeros(D, D), Wk=torch.zeros(D, D), Wv=torch.zeros(D, D), Wo=torch.zeros(D, D),
              W1=torch.zeros(H, D), b1=torch.zeros(H), W2=torch.zeros(D, H), u=torch.zeros(D))
    th["Wq"][0, 2] = th["Wv"][0, 0] = th["Wo"][5, 0] = th["u"][6] = 1.
    th["W1"][:4, 5] = eps * torch.tensor([1., -1., 1., -1.])
    th["W1"][:4, 1] = B * torch.tensor([1., 1., -1., -1.])
    th["b1"][:4] = -B
    th["W2"][6, :4] = eps * torch.tensor([1., -1., -1., 1.])
    return th


def main():
    rows = []
    for kappa, m in LAWS.items():
        e, y, clean, _, _, P = U.law_rows("anti", m, N, ETA)
        e = torch.nn.functional.pad(e, (0, DRES - e.shape[-1]))      # zero-pad the embeddings to width DRES
        e, y, P = e.to(dev), y.to(dev), P.to(dev)
        R = len(EPS)
        th0 = {k: torch.stack([structured(eps)[k] for eps in EPS]).to(dev) for k in structured(.1)}
        th = {k: v.clone() for k, v in th0.items()}
        if PERTURB:                                        # Frobenius norm eps^{3/2} per run, over all entries
            g = torch.Generator().manual_seed(0)
            noise = {k: torch.randn(v.shape, generator=g).to(dev) for k, v in th0.items()}
            norm = torch.sqrt(sum((noise[k] ** 2).flatten(1).sum(1) for k in noise))
            scale = torch.tensor(EPS, device=dev) ** 1.5 / norm
            th = {k: th[k] + noise[k] * scale.view(-1, *[1] * (th[k].dim() - 1)) for k in th}
        th = {k: v.requires_grad_() for k, v in th.items()}
        eps_t = torch.tensor(EPS, device=dev)
        lr = torch.full((R,), .5, device=dev)
        clock = torch.zeros(R, device=dev)
        taken = torch.zeros(R, device=dev)
        prev = torch.full((R,), math.inf, device=dev)
        backup = {k: v.detach().clone() for k, v in th.items()}
        exit_t = torch.full((R,), math.nan, device=dev)
        loss0 = None
        corr_max = torch.zeros(R, device=dev)
        drop_at_exit = torch.full((R,), math.nan, device=dev)
        t_cap = (80. if PERTURB else 40.) / eps_t
        for step in range(3_000_000):
            z = U.forward(th, e, N)[0]
            loss = (P * torch.nn.functional.softplus(-y * z)).sum(1)
            grads = torch.autograd.grad(loss.sum(), list(th.values()))
            with torch.no_grad():
                if loss0 is None:
                    loss0 = loss.clone()
                bad = ~torch.isfinite(loss)
                good = ~bad
                for k, v in th.items():
                    v[bad] = backup[k][bad]
                    backup[k][good] = v[good]
                clock[bad] -= taken[bad]
                lr[bad] /= 2
                prev[good] = loss[good]
                dist = torch.sqrt(sum(((th[k] - th0[k]) ** 2).flatten(1).sum(1) for k in th))
                inside = good & (dist < eps_t) & exit_t.isnan()
                corr = (P * y * z).sum(1).abs()
                corr_max[inside] = torch.maximum(corr_max[inside], corr[inside])
                out = good & (dist >= eps_t) & exit_t.isnan()
                exit_t[out] = clock[out]
                drop_at_exit[out] = (loss0 - loss)[out]
                active = good & exit_t.isnan() & (clock < t_cap)
                for p_, g_ in zip(th.values(), grads):
                    p_[active] -= lr[active].view(-1, *[1] * (p_.dim() - 1)) * g_[active]
                taken = torch.where(active, lr, torch.zeros_like(lr))
                clock += taken
                if step % 20000 == 0:
                    print(f"kappa={kappa} step {step} clock {clock.tolist()} exited {exit_t.tolist()}", flush=True)
                if not active.any() and not bad.any():
                    break
        for i, eps in enumerate(EPS):
            row = dict(dres=DRES, perturbed=PERTURB, kappa=kappa, eps=eps, exit_time=None if math.isnan(exit_t[i]) else float(exit_t[i]),
                       censored_at=float(clock[i]), loss_drop_at_exit=float(drop_at_exit[i]),
                       max_corr_inside=float(corr_max[i]), corr_over_eps3=float(corr_max[i]) / eps ** 3)
            rows.append(row)
            print(row, flush=True)
    name = "full_matrix_plateau" + ("" if DRES == 7 else f"_d{DRES}") + ("_perturbed" if PERTURB else "") + ".json"
    (ROOT / "checks" / "results" / name).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
