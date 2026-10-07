"""Training-set basin certificate at a long horizon (Theorem 4 on a finite item set).

Phase 1 trains the notebook's two-block model (lettertrace.py, AdamW, minibatch 256) on one
format. Phase 2 switches to exact float64 Armijo gradient descent on the training-set loss:
every training prompt and every supervised token is one item with weight 1/(N P), so
pi_min/4 corresponds to summed cross-entropy below 1/4. The first update with L < pi_min/4
certifies, by Theorem 4, convergence to a model that is exact on every training prompt;
the run then continues for --post-updates to check the Armijo inequality, positive margins
and nonzero ReLU pre-activations along the way. Held-out accuracy is reported separately.

    python handcoded/certificate_trainset.py --mode process --seed 42
    python handcoded/certificate_trainset.py --mode outcome --seed 42
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lettertrace as lt


def forward_diag(p, ids):
    """lettertrace.forward, also returning the smallest |ReLU pre-activation|."""
    T = ids.shape[1]
    x = p["wte"][ids] + p["wpe"][:T]
    future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
    min_pre = math.inf
    for b in range(lt.n_blocks):
        q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
        k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
        v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
        att = (q @ k.transpose(-1, -2) / math.sqrt(lt.P)).masked_fill(future, -math.inf)
        x = x + (att.softmax(-1) @ v).sum(1)
        pre = x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]
        min_pre = min(min_pre, float(pre.detach().abs().min()))
        x = x + F.relu(pre) @ p[f"{b}.mlp_out"].T
    return x @ p["readout"].T, min_pre


def make_loss(inputs, targets, chunk):
    n_items = int((targets != -100).sum())

    def loss_and_grad(p, grad):
        """Mean CE over all supervised tokens (= the weighted item loss); optional gradient."""
        params = list(p.values())
        total, grads = 0.0, None
        min_margin, min_pre = math.inf, math.inf
        for i in range(0, inputs.shape[0], chunk):
            with torch.set_grad_enabled(grad):
                logits, mp = forward_diag(p, inputs[i:i + chunk])
                tgt = targets[i:i + chunk]
                part = F.cross_entropy(logits.flatten(0, 1), tgt.flatten(), ignore_index=-100,
                                       reduction="sum") / n_items
                if grad:
                    g = torch.autograd.grad(part, params)
                    grads = list(g) if grads is None else [a + b for a, b in zip(grads, g)]
            total += float(part.detach())
            if not grad:
                with torch.no_grad():
                    mask = tgt != -100
                    lg = logits[mask]
                    gold = tgt[mask]
                    gl = lg.gather(1, gold[:, None]).squeeze(1)
                    other = lg.clone()
                    other.scatter_(1, gold[:, None], -math.inf)
                    min_margin = min(min_margin, float((gl - other.max(1).values).min()))
                    min_pre = min(min_pre, mp)
        return total, grads, dict(min_margin=min_margin, min_abs_preactivation=min_pre)

    return loss_and_grad, n_items


def run(args):
    lt.configure(task="count", alphabet=args.alphabet, word_len=args.n, mod=args.mod, n_blocks=2,
                 train_size=args.train_size, test_size=1000, steps=args.adam_steps, batch_size=256,
                 lr=1e-3, init_std=0.02, seed=args.seed, modes=args.mode, eval_every=args.adam_steps // 8)
    out = Path(args.output) / f"{args.mode}_n{args.n}_K{args.mod}_s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "summary.json").exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    t0 = time.monotonic()
    train, shown, test = lt.make_corpus()
    theta_0 = lt.init_theta()
    p32, history = lt.train_mode(args.mode, theta_0, train, shown, test)
    adam_test_exact, adam_test_answer, _ = lt.free_run(p32, test, args.mode)
    adam_train_exact, _, _ = lt.free_run(p32, train, args.mode)

    # Phase 2: exact float64 Armijo GD on the training-set loss.
    torch.set_default_dtype(torch.float64)
    p = {k: v.detach().double().clone().requires_grad_() for k, v in p32.items()}
    inputs, targets = lt.encode([lt.row(e, args.mode, s) for e, s in zip(train, shown)], args.mode)
    loss_and_grad, n_items = make_loss(inputs, targets, args.chunk)
    pi_min = 1.0 / n_items
    threshold = pi_min / 4
    L, _, diag = loss_and_grad(p, grad=False)
    handoff = dict(loss=L, ratio=L / threshold, train_exact=adam_train_exact, test_exact=adam_test_exact,
                   test_answer=adam_test_answer, **diag)
    print(f"{args.mode} handoff R={handoff['ratio']:.3g} train_exact={adam_train_exact:.4f} "
          f"test_exact={adam_test_exact:.4f} margin={diag['min_margin']:.3g}", flush=True)
    entry = 0 if handoff["ratio"] < 1 else None
    params = list(p.values())
    log, termination = [], "search_budget_reached"
    post = dict(min_slack=math.inf, min_margin=math.inf, min_pre=math.inf)
    for update in range(1, args.gd_updates + args.post_updates + 1):
        L_prev, grads, _ = loss_and_grad(p, grad=True)
        g2 = sum(float((g * g).sum()) for g in grads)
        old = [v.detach().clone() for v in params]
        eta = args.max_step
        for backtracks in range(args.max_backtracks):
            with torch.no_grad():
                for v, v_old, g in zip(params, old, grads):
                    v.copy_(v_old - eta * g)
            L_new, _, diag = loss_and_grad(p, grad=False)
            slack = L_prev - L_new - eta * g2 / 2
            if math.isfinite(L_new) and slack >= 0:
                break
            eta *= args.shrink
        else:
            with torch.no_grad():
                for v, v_old in zip(params, old):
                    v.copy_(v_old)
            termination = "backtracking_exhausted"
            break
        ratio = L_new / threshold
        if entry is None and ratio < 1:
            entry = update
        if entry is not None:
            post["min_slack"] = min(post["min_slack"], slack)
            post["min_margin"] = min(post["min_margin"], diag["min_margin"])
            post["min_pre"] = min(post["min_pre"], diag["min_abs_preactivation"])
        log.append(dict(update=update, loss=L_new, ratio=ratio, eta=eta, armijo_slack=slack,
                        backtracks=backtracks, **diag))
        if update % 100 == 0:
            print(f"{args.mode} gd j={update} R={ratio:.4g} margin={diag['min_margin']:.3g} entry={entry}", flush=True)
        if entry is None and update >= args.gd_updates:
            break
        if entry is not None and update >= entry + args.post_updates:
            termination = "post_entry_complete"
            break

    p_final = {k: v.detach() for k, v in p.items()}
    final_train_exact = lt.free_run(p_final, train, args.mode)[0]
    final_test_exact, final_test_answer, _ = lt.free_run(p_final, test, args.mode)
    summary = dict(
        mode=args.mode, n=args.n, K=args.mod, alphabet=args.alphabet, seed=args.seed,
        lambda_=abs(1 - 2 / args.alphabet) if args.mod == 2 else None,
        train_prompts=len(train), items=n_items, pi_min=pi_min, threshold=threshold,
        adam=dict(steps=args.adam_steps, final_minibatch_loss=history[-1]["loss"]),
        handoff=handoff,
        gd=dict(entry_update=entry, updates=len(log), termination=termination,
                final_ratio=log[-1]["ratio"] if log else handoff["ratio"],
                min_post_entry_armijo_slack=None if entry is None else post["min_slack"],
                min_post_entry_margin=None if entry is None else post["min_margin"],
                min_post_entry_abs_preactivation=None if entry is None else post["min_pre"]),
        final=dict(train_exact=final_train_exact, test_exact=final_test_exact, test_answer=final_test_answer),
        elapsed_seconds=time.monotonic() - t0,
        scope="Certifies exactness on the training prompts only; float64 numerical checks.")
    (out / "gd_log.json").write_text(json.dumps(log))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["process", "outcome"], required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--mod", type=int, default=3)
    ap.add_argument("--alphabet", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-size", type=int, default=20000)
    ap.add_argument("--adam-steps", type=int, default=32000)
    ap.add_argument("--gd-updates", type=int, default=5000)
    ap.add_argument("--post-updates", type=int, default=1000)
    ap.add_argument("--max-step", type=float, default=100.0)
    ap.add_argument("--max-backtracks", type=int, default=80)
    ap.add_argument("--shrink", type=float, default=0.5)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--output", default="results/paper/basin_certificate_trainset")
    run(ap.parse_args())
