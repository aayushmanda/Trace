"""Init probes and trace interventions.

    python handcoded/probes.py local_j seed=42 out=results/x.json
    python handcoded/probes.py residual
    python handcoded/probes.py dependence
    python handcoded/probes.py flip
    python handcoded/probes.py context --help
"""
import sys

_COMMANDS = (
    'local_j',
    'residual',
    'dependence',
    'flip',
    'context',
    'causal',
)

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in _COMMANDS:
        print(f"usage: python {sys.argv[0]} <command> [args...]")
        print("commands:", ", ".join(_COMMANDS))
        raise SystemExit(2)
    _cmd = sys.argv[1]
    sys.argv = [_cmd] + sys.argv[2:]
    if _cmd == 'local_j':
        """Init residual with full-context J vs J restricted to the local triple.

        The displayed triple (prev state, query, letter) is blind. Replacing every other
        displayed state with a dummy token makes J_t a function of that triple plus the
        Z-independent prompt. If the 0.21-0.25 gap is blindness, the local residual
        collapses and the full-context one does not.
        """
        import ast, json, random, sys
        from pathlib import Path

        import torch
        from torch.nn import functional as F

        word_len = 8
        alphabet = 4
        mod = 3
        rho = 0.5
        n_blocks = 2
        init_std = 0.02
        seed = 42
        train_size = 20_000
        grad_batch = 1024
        out = ""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for arg in sys.argv[1:]:
            key, value = arg.split("=", 1)
            assert key in globals(), key
            globals()[key] = value if key in ("device", "out") else ast.literal_eval(value)

        n, A, V = word_len, alphabet, mod
        SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
        P, H = 2 * n + 6, 3
        F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
        W = 2 * A + 4 * V + P + 3  # TL+QRY + TV+PREV+S0+OUT + POS + MATCH+MT+CNT
        mu = rho * (1 + 1 / (mod - 1)) - 1 / (mod - 1)
        random.seed(seed)
        torch.manual_seed(seed)


        def step(s, letter, q):
            return (s + (letter == q)) % mod


        def make_example(rng):
            q, word = rng.randrange(A), tuple(rng.randrange(A) for _ in range(n))
            s0, s, states = rng.randrange(mod), None, []
            s = s0
            for letter in word:
                s = step(s, letter, q)
                states.append(s)
            return s0, q, word, states


        def corrupt_trace(ex, rng):
            s0, q, word, _ = ex
            shown, s = [], s0
            for letter in word:
                right = step(s, letter, q)
                s = rng.choice([v for v in range(mod) if v != right])
                shown.append(s)
            return shown


        def row(ex, shown):
            s0, q, word, states = ex
            return [A + s0, q, *word, SEP, *[A + v for v in shown], COLON, A + states[-1], EOS]


        STATE_POS = list(range(n + 3, 2 * n + 3))
        ANS_POS = [2 * n + 4]
        ALL_POS = STATE_POS + ANS_POS
        DUMMY = A + 0


        def encode(rows, keep):
            ids = torch.tensor(rows, device=device)
            inputs, targets = ids[:, :-1], torch.full_like(ids[:, 1:], -100)
            idx = torch.tensor(keep, device=device)
            targets[:, idx - 1] = ids[:, idx]
            return inputs, targets


        def local_rows(rows, t):
            """Keep only the previous displayed state; dummy the rest."""
            out = [list(r) for r in rows]
            for r in out:
                for i, pos in enumerate(STATE_POS):
                    if i != t - 1:
                        r[pos] = DUMMY
            return out


        def zero_params():
            p = {"wte": torch.zeros(A + V + 3, W), "wpe": torch.zeros(P, W), "readout": torch.zeros(A + V + 3, W)}
            for b in range(n_blocks):
                p[f"{b}.q"], p[f"{b}.k"] = torch.zeros(H, P, W), torch.zeros(H, P, W)
                p[f"{b}.v"] = torch.zeros(H, W, W)
                p[f"{b}.mlp_in"], p[f"{b}.mlp_b"] = torch.zeros(F_WIDTH, W), torch.zeros(F_WIDTH)
                p[f"{b}.mlp_out"] = torch.zeros(W, F_WIDTH)
            return p


        def forward(p, ids):
            T = ids.shape[1]
            x = p["wte"][ids] + p["wpe"][:T]
            future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
            for b in range(n_blocks):
                q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
                k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
                v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
                att = (q @ k.transpose(-1, -2) / (P ** 0.5)).masked_fill(future, -float("inf"))
                x = x + (att.softmax(-1) @ v).sum(1)
                x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
            return x @ p["readout"].T


        def kway_targets(targets):
            out = targets.clone()
            mask = targets != -100
            out = out.masked_fill(~mask, 0) - A
            return out.clamp(min=0, max=V - 1).masked_fill(~mask, -100)


        def theory_grad(p, inputs, gold_ids):
            for t in p.values():
                if t.grad is not None:
                    t.grad = None
                t.requires_grad_(True)
            logits = forward(p, inputs)[:, :, A:A + V]
            mask = gold_ids != -100
            gold_v = kway_targets(gold_ids).clamp(min=0)
            e = torch.eye(V, device=device)[gold_v]
            u = torch.full_like(e, 1 / V)
            loss = (((e - u) * logits).sum(-1) * mask).sum() / inputs.shape[0]
            loss.backward()
            return torch.cat([t.grad.detach().flatten() for t in p.values()])


        def residual(g_rho, g_clean, g_ans):
            combo = mu * g_clean + g_ans
            return (g_rho - combo).norm().item() / (g_rho.norm().item() + 1e-12)


        rng = random.Random(seed)
        train = [make_example(rng) for _ in range(train_size)]
        uniforms = [rng.random() for _ in train]
        wrong = [corrupt_trace(ex, rng) for ex in train]
        shown = [st if u < rho else bad for (_, _, _, st), u, bad in zip(train, uniforms, wrong)]
        gold_rows = [row(e, e[3]) for e in train]
        mix_rows = [row(e, s) for e, s in zip(train, shown)]

        theta = {k: (torch.randn_like(t) * init_std).to(device).requires_grad_() for k, t in zero_params().items()}
        take = torch.randperm(len(train), device=device)[:grad_batch]
        take_list = take.tolist()
        mix_b = [mix_rows[i] for i in take_list]
        gold_b = [gold_rows[i] for i in take_list]

        inp_mix, tgt_rho = encode(mix_b, ALL_POS)
        inp_gold, tgt_clean = encode(gold_b, STATE_POS)
        _, tgt_ans = encode(mix_b, ANS_POS)
        g_rho = theory_grad(theta, inp_mix, tgt_rho)
        g_clean = theory_grad(theta, inp_gold, tgt_clean)
        g_ans = theory_grad(theta, inp_mix, tgt_ans)
        resid_full = residual(g_rho, g_clean, g_ans)

        g_rho_l = g_clean_l = g_ans_l = None
        for t in range(n):
            inp_m, tgt_m = encode(local_rows(mix_b, t), [STATE_POS[t]])
            inp_g, tgt_g = encode(local_rows(gold_b, t), [STATE_POS[t]])
            gr = theory_grad(theta, inp_m, tgt_m)
            gc = theory_grad(theta, inp_g, tgt_g)
            ga = theory_grad(theta, inp_m, tgt_ans) if t == 0 else None
            g_rho_l = gr if g_rho_l is None else g_rho_l + gr
            g_clean_l = gc if g_clean_l is None else g_clean_l + gc
            if ga is not None:
                g_ans_l = ga
        resid_local = residual(g_rho_l, g_clean_l, g_ans_l)

        # First-step only: context is (s0, q, w1), always independent of Z.
        inp_m1, tgt_m1 = encode(mix_b, [STATE_POS[0]])
        inp_g1, tgt_g1 = encode(gold_b, [STATE_POS[0]])
        resid_s1 = residual(theory_grad(theta, inp_m1, tgt_m1), theory_grad(theta, inp_g1, tgt_g1), g_ans)

        payload = dict(
            seed=seed, rho=rho, mu=mu, n=n, K=mod,
            residual_full=resid_full, residual_local=resid_local, residual_s1=resid_s1,
        )
        print(json.dumps(payload, indent=2), flush=True)
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(payload, indent=2))
    elif _cmd == 'residual':
        """Linear probe for Z, and the three-gradient residual of Thm. 3, along training.

        Trains count K=3, rho=0.5 scatter, block process, then at each checkpoint:
          residual = ||g_rho - mu g_clean - (1-mu) g_unif - g_ans|| / ||g_rho||
          probe    = linear readout of last-layer x at s1 vs s2 predicting Z (clean vs corrupt)
        """
        import ast, json, math, os, random, sys
        from pathlib import Path

        import torch
        from torch.nn import functional as F

        task = "count"
        alphabet = 4
        word_len = 8
        mod = 3
        rho = 0.5
        corrupt = "scatter"
        n_blocks = 2
        steps = 16000
        eval_every = 2000
        batch_size = 256
        lr = 1e-3
        init_std = 0.02
        seed = 42
        train_size = 20000
        test_size = 2000
        grad_batch = 1024
        save_dir = ""
        out = ""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for arg in sys.argv[1:]:
            key, value = arg.split("=", 1)
            assert key in globals(), key
            globals()[key] = value if key in ("device", "corrupt", "save_dir", "out") else ast.literal_eval(value)

        n, A, V = word_len, alphabet, mod
        SEP, COLON, EOS = A + V, A + V + 1, A + V + 2
        P, H = 2 * n + 6, 3
        F_WIDTH = max(A, 3 * V * (n + 1), 2 * n + 1)
        _sizes = dict(TL=A, TV=V, POS=P, QRY=A, MATCH=1, PREV=V, MT=1, CNT=1, S0=V, OUT=V)
        _start = 0
        for _name, _size in _sizes.items():
            globals()[_name] = slice(_start, _start + _size)
            _start += _size
        W = _start
        c = 1 / (mod - 1)
        mu = rho * (1 + c) - c
        random.seed(seed)
        torch.manual_seed(seed)


        def step(s, letter, q):
            return (s + (letter == q)) % mod


        def make_example(rng):
            q, word = rng.randrange(A), tuple(rng.randrange(A) for _ in range(n))
            s0, s, states = rng.randrange(mod), None, []
            s = s0
            for letter in word:
                s = step(s, letter, q)
                states.append(s)
            return s0, q, word, states


        def corrupt_trace(ex, rng):
            s0, q, word, _ = ex
            shown, s = [], s0
            for letter in word:
                right = step(s, letter, q)
                s = rng.choice([v for v in range(mod) if v != right])
                shown.append(s)
            return shown


        def row(ex, shown):
            s0, q, word, states = ex
            return [A + s0, q, *word, SEP, *[A + v for v in shown], COLON, A + states[-1], EOS]


        def encode(rows, keep):
            ids = torch.tensor(rows, device=device)
            inputs, targets = ids[:, :-1], torch.full_like(ids[:, 1:], -100)
            idx = torch.tensor(keep, device=device)
            targets[:, idx - 1] = ids[:, idx]
            return inputs, targets


        STATE_POS = list(range(n + 3, 2 * n + 3))          # s1..sn
        ANS_POS = [2 * n + 4]                               # answer value after colon
        ALL_POS = STATE_POS + ANS_POS                       # theorem: states + answer, no EOS


        def zero_params():
            p = {"wte": torch.zeros(A + V + 3, W), "wpe": torch.zeros(P, W), "readout": torch.zeros(A + V + 3, W)}
            for b in range(n_blocks):
                p[f"{b}.q"], p[f"{b}.k"] = torch.zeros(H, P, W), torch.zeros(H, P, W)
                p[f"{b}.v"] = torch.zeros(H, W, W)
                p[f"{b}.mlp_in"], p[f"{b}.mlp_b"] = torch.zeros(F_WIDTH, W), torch.zeros(F_WIDTH)
                p[f"{b}.mlp_out"] = torch.zeros(W, F_WIDTH)
            return p


        def hidden(p, ids):
            T = ids.shape[1]
            x = p["wte"][ids] + p["wpe"][:T]
            future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
            for b in range(n_blocks):
                q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
                k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
                v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
                att = (q @ k.transpose(-1, -2) / math.sqrt(P)).masked_fill(future, -math.inf)
                x = x + (att.softmax(-1) @ v).sum(1)
                x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
            return x


        def forward(p, ids):
            return hidden(p, ids) @ p["readout"].T


        def theory_grad(p, inputs, gold_ids):
            """Label-dependent gradient E[J^T(e-u)] in the K-way simplex (Thm. 1/2)."""
            for t in p.values():
                if t.grad is not None:
                    t.grad = None
                t.requires_grad_(True)
            logits = forward(p, inputs)[:, :, A:A + V]
            mask = gold_ids != -100
            gold_v = kway_targets(gold_ids).clamp(min=0)
            e = torch.eye(V, device=device)[gold_v]
            u = torch.full_like(e, 1 / V)
            loss = (((e - u) * logits).sum(-1) * mask).sum() / inputs.shape[0]
            loss.backward()
            return torch.cat([t.grad.detach().flatten() for t in p.values()])


        def kway_targets(targets):
            """Map token ids A+v to class v; keep -100 ignore."""
            out = targets.clone()
            mask = targets != -100
            out = out.masked_fill(~mask, 0) - A
            return out.clamp(min=0, max=V - 1).masked_fill(~mask, -100)


        def grad_vec(p, inputs, targets, uniform=False):
            """K-way state CE, mean over batch, sum over positions, so the three terms add."""
            for t in p.values():
                if t.grad is not None:
                    t.grad = None
                t.requires_grad_(True)
            logits = forward(p, inputs)[:, :, A:A + V]
            B = inputs.shape[0]
            if uniform:
                logp = F.log_softmax(logits, dim=-1)
                mask = targets != -100
                loss = -(logp.mean(-1) * mask).sum() / B
            else:
                loss = F.cross_entropy(logits.flatten(0, 1), kway_targets(targets).flatten(),
                                       ignore_index=-100, reduction="sum") / B
            loss.backward()
            return torch.cat([t.grad.detach().flatten() for t in p.values()])


        def probe_acc(p, rows, z, pos):
            """Linear probe of last-layer x[:, pos] for Z. Fit on first half, test on second."""
            with torch.no_grad():
                ids = torch.tensor(rows, device=device)
                feats, labs = [], []
                for i in range(0, len(rows), 256):
                    feats.append(hidden(p, ids[i:i + 256][:, :pos + 1])[:, -1].float())
                    labs.append(z[i:i + 256])
                X = torch.cat(feats)
                y = torch.cat(labs).float()
            n_tr = len(X) // 2
            Xtr, ytr, Xte, yte = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]
            Xtr = torch.cat([Xtr, torch.ones(len(Xtr), 1, device=device)], 1)
            Xte = torch.cat([Xte, torch.ones(len(Xte), 1, device=device)], 1)
            w = torch.linalg.lstsq(Xtr, ytr[:, None]).solution
            pred = (Xte @ w).squeeze(1) > 0.5
            return (pred == (yte > 0.5)).float().mean().item()


        rng_tr, rng_c, rng_te = random.Random(seed), random.Random(seed + 10_000), random.Random(seed + 20_000)
        train = [make_example(rng_tr) for _ in range(train_size)]
        uniforms = [rng_c.random() for _ in train]
        wrong = [corrupt_trace(ex, rng_c) for ex in train]
        shown = [st if u < rho else bad for (_, _, _, st), u, bad in zip(train, uniforms, wrong)]
        z_train = torch.tensor([float(u < rho) for u in uniforms], device=device)
        test = [make_example(rng_te) for _ in range(test_size)]
        u_te = [rng_te.random() for _ in test]
        wrong_te = [corrupt_trace(ex, rng_te) for ex in test]
        shown_te = [st if u < rho else bad for (_, _, _, st), u, bad in zip(test, u_te, wrong_te)]
        z_te = torch.tensor([float(u < rho) for u in u_te], device=device)

        gold_rows = [row(e, e[3]) for e in train]
        mix_rows = [row(e, s) for e, s in zip(train, shown)]
        test_rows = [row(e, s) for e, s in zip(test, shown_te)]
        inp_mix, tgt_rho = encode(mix_rows, ALL_POS)
        inp_gold, tgt_clean = encode(gold_rows, STATE_POS)
        _, tgt_unif = encode(gold_rows, STATE_POS)
        _, tgt_ans = encode(gold_rows, ANS_POS)

        theta = {k: (torch.randn_like(t) * init_std).to(device).requires_grad_() for k, t in zero_params().items()}
        opt = torch.optim.AdamW(theta.values(), lr=lr, weight_decay=0.0)
        g = torch.Generator().manual_seed(seed)
        records = []
        print(f"seed={seed} rho={rho} mu={mu:.3f} K={mod} n={n}", flush=True)
        last_loss = float("nan")


        def evaluate(it):
            take = torch.randperm(len(train), device=device)[:grad_batch]
            g_rho = grad_vec(theta, inp_mix[take], tgt_rho[take])
            g_clean = grad_vec(theta, inp_gold[take], tgt_clean[take])
            g_unif = grad_vec(theta, inp_gold[take], tgt_unif[take], uniform=True)
            g_ans = grad_vec(theta, inp_mix[take], tgt_ans[take])
            combo = mu * g_clean + (1 - mu) * g_unif + g_ans
            combo_thm = mu * g_clean + g_ans
            resid = (g_rho - combo).norm().item() / (g_rho.norm().item() + 1e-12)
            resid_thm = (g_rho - combo_thm).norm().item() / (g_rho.norm().item() + 1e-12)
            p1 = probe_acc(theta, test_rows, z_te, pos=n + 2)
            p2 = probe_acc(theta, test_rows, z_te, pos=n + 3)
            g_rho_lab = theory_grad(theta, inp_mix[take], tgt_rho[take])
            g_cl_lab = theory_grad(theta, inp_gold[take], tgt_clean[take])
            g_ans_lab = theory_grad(theta, inp_mix[take], tgt_ans[take])
            combo_lab = mu * g_cl_lab + g_ans_lab
            resid_lab = (g_rho_lab - combo_lab).norm().item() / (g_rho_lab.norm().item() + 1e-12)
            rec = dict(step=it, loss=last_loss, residual=resid, residual_thm=resid_thm,
                       residual_lab=resid_lab, probe_s1=p1, probe_s2=p2,
                       g_rho=g_rho.norm().item(), g_clean=g_clean.norm().item(),
                       g_unif=g_unif.norm().item(), g_ans=g_ans.norm().item(), mu=mu)
            records.append(rec)
            print(f"step {it:5d}  residual {resid:.3f}  thm {resid_thm:.3f}  "
                  f"lab {resid_lab:.3f}  probe_s1 {p1:.3f}  probe_s2 {p2:.3f}", flush=True)


        evaluate(0)
        for it in range(1, steps + 1):
            idx = torch.randint(len(train), (batch_size,), generator=g).to(device)
            loss = F.cross_entropy(forward(theta, inp_mix[idx]).flatten(0, 1), tgt_rho[idx].flatten(), ignore_index=-100)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(theta.values(), 1.0)
            opt.step()
            last_loss = float(loss.detach())
            if it % eval_every and it != steps:
                continue
            evaluate(it)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                torch.save({"config": dict(task=task, word_len=n, alphabet=A, mod=mod, n_blocks=n_blocks,
                                           layout="block", seed=seed),
                            "params": {k: t.detach().cpu() for k, t in theta.items()},
                            "mode": "process", "step": it},
                           os.path.join(save_dir, f"s{seed}_process_step{it}.pt"))

        payload = dict(seed=seed, rho=rho, mu=mu, records=records)
        print(json.dumps(payload, indent=2))
        if out:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(payload, indent=2))
    elif _cmd == 'dependence':
        """Does s_n depend on displayed s_{n-1} or on earlier letters?

        Block process prefix: s0 q w1..wn SEP s1..s_{n-1}. Predict s_n.
        Conditions keep the local pair (s_{n-1}, w_n) or break it.
        """
        import ast, json, math, random, sys
        from pathlib import Path

        import torch
        from torch.nn import functional as F

        ckpt = ""
        examples = 2048
        batch = 256
        out = ""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for arg in sys.argv[1:]:
            key, value = arg.split("=", 1)
            assert key in globals(), key
            globals()[key] = value if key in ("ckpt", "device", "out") else ast.literal_eval(value)
        assert ckpt, "ckpt= is required"

        saved = torch.load(ckpt, map_location=device)
        cfg = saved["config"]
        task, n, A, mod = cfg["task"], cfg["word_len"], cfg.get("alphabet", 4), cfg["mod"]
        n_blocks = cfg["n_blocks"]
        assert task == "count"
        V, SEP, H, P = mod, A + mod, 3, 2 * n + 6


        def forward(p, ids):
            T = ids.shape[1]
            x = p["wte"][ids] + p["wpe"][:T]
            future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
            for b in range(n_blocks):
                q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
                k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
                v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
                att = (q @ k.transpose(-1, -2) / math.sqrt(P)).masked_fill(future, -1e9)
                x = x + (att.softmax(-1) @ v).sum(1)
                x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
            return x @ p["readout"].T


        theta = {k: t.to(device).float() for k, t in saved["params"].items()}
        rng = random.Random(0)


        def gold_states(s0, q, word):
            s, out = s0, []
            for c in word:
                s = (s + (c == q)) % mod
                out.append(s)
            return out


        def prefix(s0, q, word, states, *, prev=None, letters=None, early=None):
            w = list(letters if letters is not None else word)
            st = list(states)
            if early is not None:
                st = list(early) + st[len(early):]
            if prev is not None:
                st = st[:-2] + [prev] + st[-1:]
            return [A + s0, q, *w, SEP, *[A + v for v in st[:-1]]]


        def score(prefs, targets):
            acc = []
            for i in range(0, len(prefs), batch):
                ids = torch.tensor(prefs[i:i + batch], device=device)
                gold = torch.tensor(targets[i:i + batch], device=device)
                pred = forward(theta, ids)[:, -1, A:A + V].argmax(-1)
                acc.append((pred == gold).float())
            return torch.cat(acc).mean().item()


        pairs = []
        for _ in range(examples):
            q = rng.randrange(A)
            word = [rng.randrange(A) for _ in range(n)]
            s0 = rng.randrange(mod)
            states = gold_states(s0, q, word)
            local = states[-1]
            pairs.append((s0, q, word, states, local))

        conds = {}
        # clean
        conds["clean"] = score([prefix(*p[:4]) for p in pairs], [p[4] for p in pairs])
        # random displayed s_{n-1}
        conds["rand_prev"] = score(
            [prefix(*p[:4], prev=rng.randrange(mod)) for p in pairs], [p[4] for p in pairs])
        # random earlier displayed states s_1..s_{n-2}, keep s_{n-1}
        conds["rand_early"] = score(
            [prefix(*p[:4], early=[rng.randrange(mod) for _ in range(n - 2)]) for p in pairs],
            [p[4] for p in pairs])
        # random letters except w_n
        conds["rand_early_letters"] = score(
            [prefix(*p[:4], letters=[rng.randrange(A) for _ in range(n - 1)] + [p[2][-1]]) for p in pairs],
            [p[4] for p in pairs])
        # random w_n, keep displayed s_{n-1}
        conds["rand_wn"] = score(
            [prefix(*p[:4], letters=p[2][:-1] + [rng.randrange(A)]) for p in pairs],
            [p[4] for p in pairs])
        # vs local rule after rand_prev (should follow displayed prev if local)
        local_after_flip = []
        prefs = []
        for p in pairs:
            prev = rng.randrange(mod)
            local_after_flip.append((prev + (p[2][-1] == p[1])) % mod)
            prefs.append(prefix(*p[:4], prev=prev))
        conds["rand_prev_vs_local"] = score(prefs, local_after_flip)

        rec = dict(ckpt=ckpt, step=saved.get("step"), mode=saved.get("mode"), n=n, K=mod, **conds)
        print(json.dumps(rec, indent=2))
        if out:
            Path(out).write_text(json.dumps(rec, indent=2))
    elif _cmd == 'flip':
        """Flip-the-first-step intervention (Proposition 3).

        Block layout: the letters are in the prompt, then SEP, then displayed states.
        We keep (q, word) fixed and only change the first displayed state s1 to be
        consistent or inconsistent with the true rule, then score the next-state
        prediction against the local rule on the displayed s1.
        """
        import ast
        import json
        import math
        import random
        import sys
        from pathlib import Path

        import torch
        from torch.nn import functional as F

        ckpt = ""
        examples = 2048
        batch = 256
        out = ""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for arg in sys.argv[1:]:
            key, value = arg.split("=", 1)
            assert key in globals(), f"unknown setting {key}"
            globals()[key] = value if key in ("ckpt", "device", "out") else ast.literal_eval(value)
        assert ckpt, "ckpt=... is required"

        saved = torch.load(ckpt, map_location=device)
        cfg = saved["config"]
        task, n, A, mod = cfg["task"], cfg["word_len"], cfg.get("alphabet", 4), cfg["mod"]
        n_blocks = cfg["n_blocks"]
        layout = cfg.get("layout", "block")
        assert task == "count"

        V = mod
        H = 3
        P = 2 * n + 6
        SEP = A + V


        def forward(p, ids):
            T = ids.shape[1]
            x = p["wte"][ids] + p["wpe"][:T]
            future = torch.ones(T, T, dtype=torch.bool, device=ids.device).triu(1)
            for b in range(n_blocks):
                q = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.q"])
                k = torch.einsum("btw,hdw->bhtd", x, p[f"{b}.k"])
                v = torch.einsum("btw,hvw->bhtv", x, p[f"{b}.v"])
                att = (q @ k.transpose(-1, -2) / math.sqrt(P)).masked_fill(future, -1e9)
                x = x + (att.softmax(-1) @ v).sum(1)
                x = x + F.relu(x @ p[f"{b}.mlp_in"].T + p[f"{b}.mlp_b"]) @ p[f"{b}.mlp_out"].T
            return x @ p["readout"].T


        theta = {k: t.to(device).float() for k, t in saved["params"].items()}
        rng = random.Random(0)


        def make_pair():
            q = rng.randrange(A)
            word = [rng.randrange(A) for _ in range(n)]
            s0 = rng.randrange(mod)
            s1_clean = (s0 + (word[0] == q)) % mod
            s1_flip = (s1_clean + 1) % mod
            local_clean = (s1_clean + (word[1] == q)) % mod
            local_flip = (s1_flip + (word[1] == q)) % mod
            if layout == "block":
                pref_clean = [A + s0, q, *word, SEP, A + s1_clean]
                pref_flip = [A + s0, q, *word, SEP, A + s1_flip]
            else:
                pref_clean = [A + s0, q, word[0], A + s1_clean, word[1]]
                pref_flip = [A + s0, q, word[0], A + s1_flip, word[1]]
            return pref_clean, pref_flip, local_clean, local_flip


        pairs = [make_pair() for _ in range(examples)]


        @torch.no_grad()
        def score(prefs, targets):
            ids = torch.tensor(prefs, device=device)
            tgt = torch.tensor(targets, device=device)
            acc, margin = [], []
            for i in range(0, len(prefs), batch):
                logits = forward(theta, ids[i:i + batch])[:, -1, A:A + V]
                pred = logits.argmax(-1)
                gold = tgt[i:i + batch]
                acc.append((pred == gold).float())
                top = logits.gather(1, gold[:, None]).squeeze(1)
                other = logits.clone()
                other[torch.arange(len(gold), device=device), gold] = -1e9
                margin.append(top - other.max(-1).values)
            return torch.cat(acc).mean().item(), torch.cat(margin).mean().item()


        clean_acc, clean_m = score([p[0] for p in pairs], [p[2] for p in pairs])
        flip_local_acc, flip_local_m = score([p[1] for p in pairs], [p[3] for p in pairs])
        flip_gold_acc, flip_gold_m = score([p[1] for p in pairs], [p[2] for p in pairs])
        rec = dict(
            ckpt=ckpt, step=saved.get("step"), mode=saved.get("mode"),
            n=n, K=mod, layout=layout,
            after_correct_first_local_acc=clean_acc, after_correct_first_margin=clean_m,
            after_flipped_first_local_acc=flip_local_acc, after_flipped_first_local_margin=flip_local_m,
            after_flipped_first_gold_acc=flip_gold_acc, after_flipped_first_gold_margin=flip_gold_m,
        )
        print(json.dumps(rec, indent=2))
        if out:
            Path(out).write_text(json.dumps(rec, indent=2))
    elif _cmd in ('context', 'causal'):
        #!/usr/bin/env python3
        """
        This script tests the paper's "context breaks blindness" claim on the count-mod-K task.
        It constructs *paired* process prefixes that have exactly the same

            q, w1, displayed s1, w2

        and therefore the same local step-2 target F(s1, w2), but differ in s0 so that:

            clean donor:   displayed s1 is consistent with the true rule at step 1
            corrupt recv.: displayed s1 is inconsistent with the true rule at step 1

        Thus the two prefixes differ in *trace consistency* while keeping the current displayed
        state and the next local-rule target fixed.

        We then use NNsight activation patching: for each non-final Transformer block and each
        prefix position, copy the donor block output into the corrupt receiver and measure how
        much the model's preference for the true local rule is restored.

        For a two-block model, the key causal test is block 0 -> block 1. Patching block 0 at
        positions such as s1 or w2 can change what block 1 predicts; patching the final block at
        an earlier position cannot causally affect the already-computed final-position logits, so
        final-block position patches are intentionally omitted.

        Example
        -------
            pip install nnsight

            python context_intervention_nnsight.py \
                --ckpt ckpt/rho05_shift/process_step8000.pt \
                --corrupt shift \
                --examples 2048 \
                --batch-size 256 \
                --out logs/context_patch.csv

        Recommended checkpoint
        ----------------------
        Use a PROCESS checkpoint trained on noisy traces, ideally on count mod 3 with coherent
        shift corruption around rho=0.5, because the theory predicts a particularly informative
        zero-margin regime there.
        """

        import argparse
        import csv
        import math
        import os
        from dataclasses import dataclass

        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        try:
            from nnsight import NNsight
        except ImportError as e:
            raise SystemExit(
                "NNsight is not installed. Run `pip install nnsight` and try again."
            ) from e


        # -----------------------------------------------------------------------------
        # Model: exact nn.Module re-expression of the functional model in lettertrace.py


        class LetterBlock(nn.Module):
            def __init__(self, q, k, v, mlp_in, mlp_b, mlp_out, pos_width):
                super().__init__()
                self.q = nn.Parameter(q, requires_grad=False)
                self.k = nn.Parameter(k, requires_grad=False)
                self.v = nn.Parameter(v, requires_grad=False)
                self.mlp_in = nn.Parameter(mlp_in, requires_grad=False)
                self.mlp_b = nn.Parameter(mlp_b, requires_grad=False)
                self.mlp_out = nn.Parameter(mlp_out, requires_grad=False)
                self.pos_width = pos_width

            def forward(self, x):
                T = x.shape[1]
                future = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)

                q = torch.einsum("btw,hdw->bhtd", x, self.q)
                k = torch.einsum("btw,hdw->bhtd", x, self.k)
                v = torch.einsum("btw,hvw->bhtv", x, self.v)

                att = (q @ k.transpose(-1, -2) / math.sqrt(self.pos_width)).masked_fill(
                    future, -math.inf
                )
                x = x + (att.softmax(-1) @ v).sum(1)
                x = x + F.relu(x @ self.mlp_in.T + self.mlp_b) @ self.mlp_out.T
                return x


        class LetterTransformer(nn.Module):
            def __init__(self, saved):
                super().__init__()
                cfg = saved["config"]
                p = saved["params"]

                self.task = cfg["task"]
                self.n = int(cfg["word_len"])
                self.A = int(cfg["alphabet"])
                self.K = int(cfg["mod"])
                self.n_blocks = int(cfg["n_blocks"])
                self.layout = cfg["layout"]

                if self.task != "count":
                    raise ValueError("This intervention is defined for task=count checkpoints.")
                if self.layout != "stream":
                    raise ValueError("This intervention assumes layout=stream.")
                if self.n_blocks < 2:
                    raise ValueError("Need at least two blocks for a nontrivial cross-layer intervention.")

                self.P = 2 * self.n + 6
                self.VOCAB = self.A + self.K + 3

                self.wte = nn.Parameter(p["wte"].clone(), requires_grad=False)
                self.wpe = nn.Parameter(p["wpe"].clone(), requires_grad=False)
                self.readout = nn.Parameter(p["readout"].clone(), requires_grad=False)

                blocks = []
                for b in range(self.n_blocks):
                    blocks.append(
                        LetterBlock(
                            p[f"{b}.q"].clone(),
                            p[f"{b}.k"].clone(),
                            p[f"{b}.v"].clone(),
                            p[f"{b}.mlp_in"].clone(),
                            p[f"{b}.mlp_b"].clone(),
                            p[f"{b}.mlp_out"].clone(),
                            self.P,
                        )
                    )
                self.blocks = nn.ModuleList(blocks)

            def forward(self, ids):
                T = ids.shape[1]
                x = self.wte[ids] + self.wpe[:T]
                for block in self.blocks:
                    x = block(x)
                return x @ self.readout.T


        # -----------------------------------------------------------------------------
        # Paired contexts


        @dataclass
        class PairBatch:
            clean_ids: torch.Tensor
            wrong_ids: torch.Tensor
            target_state: torch.Tensor
            wrong_offset: torch.Tensor


        def make_pair_batch(B, A, K, corrupt, generator, device):
            """Create paired prefixes for predicting s2.

            Prefix layout is exactly the stream process prefix at step 2:
                [s0, q, w1, displayed_s1, w2]

            Both contexts have the SAME q, w1, displayed_s1, w2.
            They differ only in s0.

            Clean donor:
                displayed_s1 = s0_clean + [w1=q]  (mod K)

            Corrupt receiver:
                displayed_s1 != s0_wrong + [w1=q] (mod K)
                and for shift it is exactly +1 from the correct step-1 successor.
            """
            q = torch.randint(A, (B,), generator=generator)
            w1 = torch.randint(A, (B,), generator=generator)
            w2 = torch.randint(A, (B,), generator=generator)
            s1 = torch.randint(K, (B,), generator=generator)

            b1 = (w1 == q).long()
            b2 = (w2 == q).long()

            if corrupt == "shift":
                offset = torch.ones(B, dtype=torch.long)
            else:
                # Any nonzero offset is supported by scatter. Sample one uniformly.
                offset = torch.randint(1, K, (B,), generator=generator)

            s0_clean = (s1 - b1) % K
            s0_wrong = (s1 - b1 - offset) % K

            # Because displayed s1 and w2 are identical across the pair, the *local* true-rule
            # target at step 2 is identical across donor and receiver.
            target = (s1 + b2) % K

            clean = torch.stack([A + s0_clean, q, w1, A + s1, w2], dim=1)
            wrong = torch.stack([A + s0_wrong, q, w1, A + s1, w2], dim=1)

            return PairBatch(
                clean.to(device), wrong.to(device), target.to(device), offset.to(device)
            )


        # -----------------------------------------------------------------------------
        # Metrics


        def rule_margin(logits_last, target_state, A, K, corrupt):
            """Preference for the true local rule over the corruption rule.

            shift:   z_true - z_{true+1}
            scatter: z_true - mean_{j != true} z_j
            """
            z = logits_last[:, A : A + K]
            rows = torch.arange(len(z), device=z.device)
            z_true = z[rows, target_state]

            if corrupt == "shift":
                competitor = (target_state + 1) % K
                z_bad = z[rows, competitor]
            else:
                z_bad = (z.sum(1) - z_true) / (K - 1)

            return z_true - z_bad


        def true_accuracy(logits_last, target_state, A):
            pred = logits_last.argmax(-1)
            return (pred == A + target_state).float()


        # -----------------------------------------------------------------------------
        # Main


        def load_checkpoint(path, device):
            try:
                return torch.load(path, map_location=device, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=device)


        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--ckpt", required=True, help="process checkpoint from lettertrace.py")
            ap.add_argument("--corrupt", choices=["shift", "scatter"], default="shift")
            ap.add_argument("--examples", type=int, default=2048)
            ap.add_argument("--batch-size", type=int, default=256)
            ap.add_argument("--seed", type=int, default=12345, help="pair-sampling seed")
            ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
            ap.add_argument("--out", default="context_patch.csv")
            args = ap.parse_args()

            saved = load_checkpoint(args.ckpt, args.device)
            if saved.get("mode") != "process":
                raise ValueError(
                    f"Expected a process checkpoint, got mode={saved.get('mode')!r}. "
                    "Train/save with modes=process."
                )

            net = LetterTransformer(saved).to(args.device).eval()
            traced = NNsight(net)

            A, K, L = net.A, net.K, net.n_blocks
            if K < 2:
                raise ValueError("mod/K must be >= 2")

            # Only positions available in the step-2 prefix.
            position_names = ["s0", "q", "w1", "s1", "w2"]
            patch_layers = list(range(L - 1))  # non-final blocks only

            # Aggregate sums across batches.
            base = dict(clean_margin=0.0, wrong_margin=0.0, clean_acc=0.0, wrong_acc=0.0, n=0)
            rows = {
                (layer, pos): dict(margin=0.0, acc=0.0, n=0)
                for layer in patch_layers
                for pos in list(range(5)) + ["all"]
            }

            gen = torch.Generator(device="cpu").manual_seed(args.seed)

            for start in range(0, args.examples, args.batch_size):
                B = min(args.batch_size, args.examples - start)
                pair = make_pair_batch(B, A, K, args.corrupt, gen, args.device)

                # Baselines.
                with traced.trace(pair.clean_ids):
                    clean_logits_saved = traced.output[:, -1, :].save()
                with traced.trace(pair.wrong_ids):
                    wrong_logits_saved = traced.output[:, -1, :].save()

                clean_logits = clean_logits_saved
                wrong_logits = wrong_logits_saved
                cm = rule_margin(clean_logits, pair.target_state, A, K, args.corrupt)
                wm = rule_margin(wrong_logits, pair.target_state, A, K, args.corrupt)
                ca = true_accuracy(clean_logits, pair.target_state, A)
                wa = true_accuracy(wrong_logits, pair.target_state, A)

                base["clean_margin"] += cm.sum().item()
                base["wrong_margin"] += wm.sum().item()
                base["clean_acc"] += ca.sum().item()
                base["wrong_acc"] += wa.sum().item()
                base["n"] += B

                # Patch one non-final block at a time.
                for layer in patch_layers:
                    # Capture the clean donor residual after this block.
                    with traced.trace(pair.clean_ids):
                        donor_saved = traced.blocks[layer].output.clone().save()
                    donor = donor_saved.to(args.device)

                    # Individual positions.
                    for pos in range(5):
                        with traced.trace(pair.wrong_ids):
                            traced.blocks[layer].output[:, pos, :] = donor[:, pos, :]
                            patched_saved = traced.output[:, -1, :].save()
                        pm = rule_margin(patched_saved, pair.target_state, A, K, args.corrupt)
                        pa = true_accuracy(patched_saved, pair.target_state, A)
                        rows[(layer, pos)]["margin"] += pm.sum().item()
                        rows[(layer, pos)]["acc"] += pa.sum().item()
                        rows[(layer, pos)]["n"] += B

                    # Sanity upper bound: patch the whole observed prefix at this layer.
                    with traced.trace(pair.wrong_ids):
                        traced.blocks[layer].output[:, :5, :] = donor[:, :5, :]
                        patched_all_saved = traced.output[:, -1, :].save()
                    pm = rule_margin(patched_all_saved, pair.target_state, A, K, args.corrupt)
                    pa = true_accuracy(patched_all_saved, pair.target_state, A)
                    rows[(layer, "all")]["margin"] += pm.sum().item()
                    rows[(layer, "all")]["acc"] += pa.sum().item()
                    rows[(layer, "all")]["n"] += B

            N = base["n"]
            clean_margin = base["clean_margin"] / N
            wrong_margin = base["wrong_margin"] / N
            clean_acc = base["clean_acc"] / N
            wrong_acc = base["wrong_acc"] / N
            gap = clean_margin - wrong_margin

            print("\n=== Paired context baseline ===")
            print(f"checkpoint: {args.ckpt}")
            print(f"task=count mod={K} blocks={L} corruption={args.corrupt} N={N}")
            print("pair differs only in s0; q,w1,displayed s1,w2 and the step-2 local target are fixed")
            print(f"clean-consistent context: margin={clean_margin:+.4f}  true-token acc={clean_acc:.3f}")
            print(f"wrong-consistent context: margin={wrong_margin:+.4f}  true-token acc={wrong_acc:.3f}")
            print(f"context effect (clean - wrong margin): {gap:+.4f}")

            os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
            with open(args.out, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "layer",
                        "position",
                        "patched_margin",
                        "patched_true_acc",
                        "wrong_margin",
                        "clean_margin",
                        "margin_gain_over_wrong",
                        "fraction_of_clean_context_gap_recovered",
                    ],
                )
                writer.writeheader()

                print("\n=== NNsight activation patching ===")
                print("layer position  patched_margin  gain_vs_wrong  gap_recovered  true_acc")
                for layer in patch_layers:
                    for pos in list(range(5)) + ["all"]:
                        r = rows[(layer, pos)]
                        pm = r["margin"] / r["n"]
                        pa = r["acc"] / r["n"]
                        gain = pm - wrong_margin
                        recovery = gain / gap if abs(gap) > 1e-12 else float("nan")
                        pname = "all" if pos == "all" else position_names[pos]
                        print(
                            f"{layer:5d} {pname:>8s}  {pm:+13.4f}  {gain:+13.4f}  "
                            f"{recovery:+12.3f}  {pa:.3f}"
                        )
                        writer.writerow(
                            dict(
                                layer=layer,
                                position=pname,
                                patched_margin=pm,
                                patched_true_acc=pa,
                                wrong_margin=wrong_margin,
                                clean_margin=clean_margin,
                                margin_gain_over_wrong=gain,
                                fraction_of_clean_context_gap_recovered=recovery,
                            )
                        )

            print(f"\nwrote {args.out}")
            print(
                "\nInterpretation: a positive gain means that inserting the clean-consistent activation "
                "into the corrupt-consistent context causally moves the prediction toward the true local rule. "
                "A recovery near 1 means that patch alone explains most of the clean-vs-corrupt context gap."
            )


        if __name__ == "__main__":
            main()
