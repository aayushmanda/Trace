"""Linear probes for intermediate states inside outcome-trained UnifiedExecutors.

Loads the models saved by unified_depth_dynamics.py --save-models. On fresh outcome-format circuits,
reads the residual stream at the answer-prediction position after the embedding (block 0) and after
every block, and fits one multinomial logistic probe per (block, step t) to predict s_t. The same
probes are fitted on the shared random initialization as a control. Accuracy is on held-out circuits.

Run from the repository root:
  uv run --offline --no-project --with 'torch==2.7.1' --with numpy \
    python -m src.experiments.alignment.unified_depth_probes \
    --models results/paper/unified_depth_models --output results/paper/unified_depth_probes.csv
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from handcoded.data import encode_dataset
from handcoded.gates import make_circuits
from handcoded.models import build_random_trainable_unified
from handcoded.tokenizer import make_tokenizer


@torch.no_grad()
def operation_residuals(model, batch, depth):
    """(blocks + 1, examples, depth, width) residuals at the position of operation a_t (token index t)."""
    hidden = model.token_features[batch.inputs] + model.position_features[: batch.inputs.shape[1]][None]
    out = [hidden[:, 1:depth + 1]]
    for block in model.blocks:
        hidden = block(hidden)
        out.append(hidden[:, 1:depth + 1])
    return torch.stack(out)


@torch.no_grad()
def residuals(model, batch, n_states):
    """(blocks + 1, examples, width) residuals at the position that predicts the answer."""
    is_state = (batch.targets >= 0) & (batch.targets < n_states)
    position = (is_state * torch.arange(batch.targets.shape[1], device=batch.targets.device)).argmax(1)
    rows = torch.arange(len(position), device=position.device)
    hidden = model.token_features[batch.inputs] + model.position_features[: batch.inputs.shape[1]][None]
    out = [hidden[rows, position]]
    for block in model.blocks:
        hidden = block(hidden)
        out.append(hidden[rows, position])
    return torch.stack(out)


def probe_accuracy(features, labels, n_train, classes, steps=300):
    mean, std = features[:n_train].mean(0), features[:n_train].std(0) + 1e-6
    x = (features - mean) / std
    weight = torch.zeros(x.shape[1], classes, device=x.device, requires_grad=True)
    bias = torch.zeros(classes, device=x.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(x[:n_train] @ weight + bias, labels[:n_train]) + 1e-3 * (weight ** 2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        return ((x[n_train:] @ weight + bias).argmax(1) == labels[n_train:]).float().mean().item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depths", nargs="+", type=int, default=[2, 4, 6, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-bits", type=int, default=2)
    parser.add_argument("--examples", type=int, default=4000)
    parser.add_argument("--train-fraction", type=float, default=0.75)
    parser.add_argument("--model-seed", type=int, default=42)
    parser.add_argument("--probe-data-seed", type=int, default=70000)
    parser.add_argument("--site", choices=["answer", "operation"], default="answer",
                        help="answer: position that predicts s_D; operation: position of a_t when probing s_t")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    device = torch.device(args.device)
    tokenizer = make_tokenizer(args.n_bits, None)
    fields = ["depth", "seed", "model", "block", "t", "accuracy", "majority"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for depth in args.depths:
            for seed in args.seeds:
                path = args.models / f"D{depth}_s{seed}_outcome.pt"
                if not path.exists():
                    print("missing", path, flush=True)
                    continue
                circuits = make_circuits(args.examples, args.probe_data_seed + seed, depth, n_bits=args.n_bits)
                batch = encode_dataset(circuits, tokenizer, "outcome").to(device)
                states = torch.tensor([c.states for c in circuits], device=device)
                n_train = int(args.train_fraction * len(circuits))
                init = build_random_trainable_unified(tokenizer, depth, seed=args.model_seed + seed, device=device)
                trained = build_random_trainable_unified(tokenizer, depth, seed=args.model_seed + seed, device=device)
                trained.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                for name, model in (("outcome", trained), ("init", init)):
                    model.eval()
                    if args.site == "answer":
                        stream = residuals(model, batch, tokenizer.n_states)
                    else:
                        stream = operation_residuals(model, batch, depth)
                    for block in range(depth + 1):
                        for t in range(1, depth + 1):
                            labels = states[:, t - 1]
                            majority = labels[n_train:].bincount(minlength=tokenizer.n_states).max().item() / (len(labels) - n_train)
                            features = stream[block] if args.site == "answer" else stream[block][:, t - 1]
                            accuracy = probe_accuracy(features, labels, n_train, tokenizer.n_states)
                            writer.writerow(dict(depth=depth, seed=seed, model=name, block=block, t=t,
                                                 accuracy=accuracy, majority=majority))
                    handle.flush()
                print("done", depth, seed, flush=True)


if __name__ == "__main__":
    main()
