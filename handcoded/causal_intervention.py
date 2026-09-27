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
    raise SystemExit("NNsight is not installed. Run `pip install nnsight` and try again.") from e


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

        att = (q @ k.transpose(-1, -2) / math.sqrt(self.pos_width)).masked_fill(future, -math.inf)
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
