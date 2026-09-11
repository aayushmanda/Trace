"""CPU runner for the theory-closure section of two_model_reachability.ipynb.

Kernel (Thm 2, Cor. 3, Thm 4) plus Thm 1 patches / Def 20 / C_t at the
notebook depth. Not mixed-format §7 transfer. Not a training job.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import nnsight  # noqa: F401  # import before other heavy modules; nnsight inspects the caller
import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")

import _paths  # noqa: F401

import closure_kernel as ck
import handcoded
from handcoded import (
    DATA_SEED,
    DEPTH,
    MODEL_SEED,
    HandcodedOutcomeTransformer,
    encode_dataset,
    make_circuits,
    make_tokenizer,
)
from handcoded.local_credit import credit_signals, gold_states_tensor
from handcoded.models import attach_local_heads, build_random_trainable_outcome_architecture
from handcoded.plotting import apply_style
from src.eval import executor_comparison as c
from src.eval.local_credit import evaluate_probes, measure_induced_credit, run_patches
from src.training.seed import set_seed

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "closure_handcoded"


def as_trainable(model, device):
    trainable = copy.deepcopy(model).cpu()

    def convert(module):
        for name, buffer in list(module._buffers.items()):
            if buffer is None:
                continue
            value = buffer.detach().clone()
            del module._buffers[name]
            module.register_parameter(name, torch.nn.Parameter(value))
        for child in module.children():
            convert(child)

    convert(trainable)
    return trainable.to(device)


def main() -> int:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    tokenizer = make_tokenizer()
    print(f"device={device} depth={DEPTH} nnsight={getattr(nnsight, '__version__', 'present')}", flush=True)

    rng = np.random.default_rng(2001)
    print("=== Theorem 4 ===", flush=True)
    thm4_rows = ck.thm4_at_mixing(rng, n=4000)
    ck.write_csv(OUT / "thm4_mixing.csv", thm4_rows)
    print("=== Theorem 2 ===", flush=True)
    residual_rows = ck.thm2_residuals(rng)
    ck.write_csv(OUT / "residuals.csv", residual_rows)
    print("=== Corollary 3 ===", flush=True)
    cor3_rows = ck.corollary3(rng, list(range(2, 13)), [0.1, 0.3, 0.5, 0.9], n=200)
    ck.write_csv(OUT / "corollary3.csv", cor3_rows)
    mixing_fits = ck.fit_exponents([r for r in cor3_rows if r["eps"] <= 0.5], list(range(2, 13)))
    ck.write_csv(OUT / "exponents_mixing_ball.csv", mixing_fits)
    ck.write_csv(OUT / "exponents.csv", ck.fit_exponents(cor3_rows, list(range(2, 13))))
    ck.draw({
        "corollary3": cor3_rows,
        "exponents": mixing_fits,
        "residuals": residual_rows,
        "thm4_mixing": thm4_rows,
    })

    print("=== Theorem 1 patches ===", flush=True)
    set_seed(42 + DEPTH)
    oracle = HandcodedOutcomeTransformer(tokenizer, DEPTH).to(device).eval()
    patch_train = c.unique_circuits(16, 123, DEPTH)
    patch_val = c.unique_circuits(8, 8000, DEPTH, c.circuit_keys(patch_train))
    probes, probe_train_acc, probe_eval_acc = evaluate_probes(
        oracle, patch_train, patch_val, tokenizer, device, tokenizer.n_states, 40, 0.05,
    )
    print(
        f"probe train={float(np.mean(probe_train_acc)):.3f} eval={float(np.mean(probe_eval_acc)):.3f}",
        flush=True,
    )
    patch_rows = run_patches(
        oracle, patch_val, tokenizer, device,
        probes=probes, model_name="oracle_outcome", donor_ids=[0, 5, 10, 15],
    )
    patch_frame = pd.DataFrame(patch_rows)
    patch_summary = {
        method: float(acc)
        for method, acc in patch_frame.groupby("method")["counterfactual_accuracy"].mean().items()
    }
    print("patch summary", patch_summary, flush=True)
    ck.write_csv(OUT / "notebook_oracle_patches.csv", patch_rows)

    print("=== Def 20 init ===", flush=True)
    credit_circuits = make_circuits(64, 8000, DEPTH)
    init_model = build_random_trainable_outcome_architecture(
        tokenizer, DEPTH, seed=MODEL_SEED, device=device,
    ).eval()
    measured = measure_induced_credit(init_model, tokenizer, credit_circuits, device, DEPTH)
    print("init Def 20", measured, flush=True)
    ck.write_csv(OUT / "notebook_init_induced.csv", [
        {"model": "init_outcome_architecture", "step": 0, **measured},
    ])

    print("=== C_t ===", flush=True)
    train_circuits = make_circuits(64, DATA_SEED, DEPTH)
    batch = encode_dataset(train_circuits, tokenizer, "outcome").to(device)
    gold_batch = gold_states_tensor(train_circuits, device)
    oracle_long = HandcodedOutcomeTransformer(tokenizer, DEPTH, positions=3 * DEPTH + 4).to(device).eval()
    ct_rows = []
    for name, model in [
        ("trainable_oracle", as_trainable(oracle_long, device)),
        ("init_outcome_architecture", init_model),
    ]:
        model = model.to(device)
        if getattr(model, "local_heads", None) is None:
            attach_local_heads(model, seed=0, device=device)
        for param in model.parameters():
            param.requires_grad_(True)
        rows = credit_signals(model, batch, gold_batch, probes=None)
        for row in rows:
            ct_rows.append({"model": name, **row})
        print(f"C_t {name}:", rows, flush=True)
    ck.write_csv(OUT / "notebook_ct.csv", ct_rows)

    summary = {
        "depth": DEPTH,
        "thm4_rho_half_process": next(
            r["process_coefficient"] for r in thm4_rows if abs(r["rho"] - 0.5) < 1e-9
        ),
        "thm4_rho_half_predicted": next(
            r["predicted_coefficient"] for r in thm4_rows if abs(r["rho"] - 0.5) < 1e-9
        ),
        "thm4_outcome_rule": thm4_rows[0]["outcome_rule_frobenius_mean"],
        "cor3_D4_slope_mixing_ball": next(r["slope"] for r in mixing_fits if int(r["depth"]) == 4),
        "oracle_slot": patch_summary.get("oracle_slot"),
        "random_subspace": patch_summary.get("random_subspace"),
        "probe_subspace": patch_summary.get("probe_subspace"),
        "init_mixing": measured["mixing_radius_mean"],
        "init_outcome_credit": measured["outcome_rule_credit_mean"],
        "init_process_floor": measured["process_floor_mean"],
        "probe_eval_acc": float(np.mean(probe_eval_acc)),
    }
    (OUT / "notebook_closure_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
