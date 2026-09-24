"""Train the reduced network of Section 3.3 under process and outcome loss; measure escape time and credit.

p(. | s, a) = softmax(V h), h = ReLU(psi e_(s,a)), K states, M permutation rules. Losses are exact
population losses: process averages the transition cross-entropy over all (s, a); outcome enumerates every
start and every length-D operation sequence, so the last step is exactly balanced. Plain gradient descent.

Starts: `path` is Theorem 1's start moved by delta along the true-rule path (psi = I, V = delta T);
`random` draws psi ~ N(0, 2/H) and V ~ N(0, sigma^2).
Escape = first step with mean_{s0, a} p(y) >= 1/2, the same criterion for both losses.
For the path start the tied-path prediction integrates the exact one-parameter dynamics
  alpha <- alpha - eta / ((K - 1) M) * dL_out/dalpha,  L_out = -log((1 + (K - 1) beta^D) / K),
which is exact when psi is frozen. Outcome runs also log per-step credit E ||Pi f_{t-1}|| ||Pi b_t|| / p_y.

Run from the repository root:
  uv run --offline --no-project --with 'torch==2.7.1' --with numpy \
    python -m src.experiments.alignment.reduced_network_escape --depths 2 3 4 5 6 7 8 --output results/paper/reduced_escape/K4.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F


def prefix_products(P, depth):
    """out[t] has shape (M^t, K, K); index digits are a_1 ... a_t, most significant first."""
    K = P.shape[-1]
    out = [torch.eye(K, dtype=P.dtype)[None]]
    for _ in range(depth):
        out.append((out[-1][:, None] @ P[None]).reshape(-1, K, K))
    return out


def suffix_products(P, depth):
    """out[r] multiplies the last r steps; index digits are the suffix operations, first most significant."""
    K = P.shape[-1]
    out = [torch.eye(K, dtype=P.dtype)[None]]
    for _ in range(depth):
        out.append((P[:, None] @ out[-1][None]).reshape(-1, K, K))
    return out


class ReducedNetwork(torch.nn.Module):
    def __init__(self, K, M, start, delta, sigma, rules, generator):
        super().__init__()
        H = K * M
        self.K, self.M = K, M
        if start == "path":
            psi = torch.eye(H, dtype=torch.float64)
            V = torch.zeros(K, H, dtype=torch.float64)
            V[rules.reshape(-1), torch.arange(H)] = delta
        else:
            psi = torch.randn(H, H, generator=generator, dtype=torch.float64) * math.sqrt(2 / H)
            V = torch.randn(K, H, generator=generator, dtype=torch.float64) * sigma
        self.psi, self.V = torch.nn.Parameter(psi), torch.nn.Parameter(V)

    def logits(self):
        return (self.V @ F.relu(self.psi)).T.reshape(self.M, self.K, self.K)  # row (a, s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--K", type=int, default=4)
    parser.add_argument("--M", type=int, default=4)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 3, 4, 5, 6, 7, 8])
    parser.add_argument("--starts", nargs="+", default=["path:0.25", "path:0.5", "random:0.5"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--lr", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=300000)
    parser.add_argument("--rule-seed", type=int, default=0)
    parser.add_argument("--freeze-psi", action="store_true", help="train V only (the tied-path theory is exact)")
    parser.add_argument("--credit-every", type=int, default=0, help="log per-step credit every n steps (0: off)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    K, M = args.K, args.M
    g = torch.Generator().manual_seed(args.rule_seed)
    rules = torch.stack([torch.randperm(K, generator=g) for _ in range(M)])  # rules[a, s] = F(s, a)
    T = F.one_hot(rules, K).double()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    credit_path = args.output.with_name(args.output.stem + "_credit.csv")
    fields = ["K", "M", "freeze_psi", "depth", "start", "scale", "seed", "loss", "escape_step", "predicted_step", "final_py"]
    with args.output.open("w", newline="") as handle, credit_path.open("w", newline="") as credit_handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        credit_writer = csv.writer(credit_handle)
        credit_writer.writerow(["depth", "start", "scale", "seed", "step", "t", "credit", "pi_f", "pi_b", "py"])
        for depth in args.depths:
            answers = prefix_products(T, depth)[-1].argmax(-1)  # (M^D, K): y for each sequence and start
            for spec in args.starts:
                start, scale = spec.split(":")
                scale = float(scale)
                seeds = [0] if start == "path" else args.seeds
                for seed in seeds:
                    predicted = tied_path_steps(K, M, depth, scale, args.lr, args.max_steps) if start == "path" else None
                    for loss_name in ("process", "outcome"):
                        model = ReducedNetwork(K, M, start, scale, scale, rules, torch.Generator().manual_seed(seed))
                        model.psi.requires_grad_(not args.freeze_psi)
                        escape, py = None, 0.0
                        for step in range(args.max_steps + 1):
                            logits = model.logits()
                            P = logits.softmax(-1)
                            full = prefix_products(P, depth)[-1]
                            p_y = full.gather(2, answers[..., None]).squeeze(-1)
                            py = p_y.mean().item()
                            if loss_name == "outcome" and args.credit_every and step % args.credit_every == 0:
                                log_credit(credit_writer, P.detach(), answers, depth, spec, seed, step, start, scale)
                            if py >= 0.5:
                                escape = step
                                break
                            if step == args.max_steps:
                                break
                            if loss_name == "process":
                                loss = F.cross_entropy(logits.reshape(-1, K), rules.reshape(-1))
                            else:
                                loss = -p_y.log().mean()
                            model.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                            with torch.no_grad():
                                for parameter in model.parameters():
                                    if parameter.grad is not None:
                                        parameter -= args.lr * parameter.grad
                        row = dict(K=K, M=M, freeze_psi=args.freeze_psi, depth=depth, start=start, scale=scale, seed=seed, loss=loss_name,
                                   escape_step=escape, predicted_step=predicted, final_py=py)
                        writer.writerow(row)
                        handle.flush()
                        credit_handle.flush()
                        print(json.dumps(row), flush=True)


def tied_path_steps(K, M, depth, delta, lr, cap):
    alpha = delta
    for n in range(cap + 1):
        e = math.exp(alpha)
        beta, dbeta = (e - 1) / (e + K - 1), K * e / (e + K - 1) ** 2
        p = (1 + (K - 1) * beta ** depth) / K
        if p >= 0.5:
            return n
        alpha += lr / ((K - 1) * M) * (K - 1) * depth * beta ** (depth - 1) * dbeta / (K * p)
    return None


@torch.no_grad()
def log_credit(writer, P, answers, depth, spec, seed, step, start, scale):
    K, M = P.shape[-1], P.shape[0]
    prefix, suffix = prefix_products(P, depth), suffix_products(P, depth)
    n = torch.arange(M ** depth)
    starts = torch.arange(K)
    p_y = prefix[-1][n[:, None], starts[None], answers]  # (N, K)
    for t in range(1, depth + 1):
        f = prefix[t - 1][n // M ** (depth - t + 1)]  # (N, K_start, K)
        b = suffix[depth - t][n % M ** (depth - t)]  # (N, K, K)
        b = b.gather(2, answers[:, None, :].expand(-1, K, -1)).transpose(1, 2)  # (N, K_start, K): b_t for each start
        pi_f = (f - f.mean(-1, keepdim=True)).norm(dim=-1)
        pi_b = (b - b.mean(-1, keepdim=True)).norm(dim=-1)
        writer.writerow([depth, start, scale, seed, step, t, (pi_f * pi_b / p_y).mean().item(),
                         pi_f.mean().item(), pi_b.mean().item(), p_y.mean().item()])


if __name__ == "__main__":
    main()
