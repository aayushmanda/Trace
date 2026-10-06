"""Exact numerical check of the paper's fixed-feature lookup proposition.

Run from the repository root:
    python3 handcoded/lookup_convergence.py

The features are constructed, not measured from a trained Transformer.
The script checks nonorthogonal readout GD, the quantitative comparator bound,
the invariant loss barrier, and the orthogonal large-step corollary. It also
checks the within-class variation and norm bounds, gives a counterexample to
the class-only barrier for varying features, and enumerates counting words
to check the reduction to the 2K transition classes.
Results are written to results/lookup_convergence/check.json.
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np


def counting_probabilities(k, p, n, first_step=1):
    """Exact population sum over words, starts, and steps first_step..n."""
    if not 1 <= first_step <= n:
        raise ValueError("first_step must lie between 1 and n")
    probabilities = np.zeros(2 * k)
    for word in itertools.product((0, 1), repeat=n):
        weight = p ** sum(word) * (1 - p) ** (n - sum(word)) / k
        for start in range(k):
            state = start
            for position, bit in enumerate(word):
                if position >= first_step - 1:
                    probabilities[2 * state + bit] += weight / (n - first_step + 1)
                state = (state + bit) % k
    return probabilities


def evaluate(w, phi, labels, probabilities):
    logits = w @ phi
    centered = logits - logits.max(axis=0, keepdims=True)
    log_normalizer = np.log(np.exp(centered).sum(axis=0))
    log_probabilities = centered - log_normalizer
    columns = np.arange(len(labels))
    loss = float(-probabilities @ log_probabilities[labels, columns])
    q = np.exp(log_probabilities)
    errors = q.copy()
    errors[labels, columns] -= 1
    gradient = (errors * probabilities) @ phi.T
    margins = logits[labels, columns][None, :] - logits
    wrong = np.arange(w.shape[0])[:, None] != labels[None, :]
    return loss, gradient, margins[wrong]


def run_case(k, p, rho, steps, seed, eta=1.0, radius_ratio=0.0):
    classes = 2 * k
    gram = (1 - rho) * np.eye(classes) + rho * np.ones((classes, classes))
    values, vectors = np.linalg.eigh(gram)
    phi = (vectors * np.sqrt(values)) @ vectors.T
    labels = (np.repeat(np.arange(k), 2) + np.tile([0, 1], k)) % k
    probabilities = np.tile([(1 - p) / k, p / k], k)
    class_minimum = float(probabilities.min())
    kappa = float(values.min())
    epsilon = radius_ratio * math.sqrt(kappa / k)
    gamma = 1 - 2 * radius_ratio
    if gamma <= 0:
        raise ValueError("The stated separator bound requires positive gamma.")
    if radius_ratio > 0:
        directions = np.random.default_rng(seed + 1).normal(size=phi.shape)
        directions /= np.linalg.norm(directions, axis=0, keepdims=True)
        displacements = epsilon * directions
        phi = np.stack((phi + displacements, phi - displacements), axis=2).reshape(
            classes, 2 * classes
        )
        labels = np.repeat(labels, 2)
        probabilities = np.repeat(probabilities / 2, 2)
    beta = float(0.5 * probabilities @ np.square(phi).sum(axis=0))
    rng = np.random.default_rng(seed)
    w = rng.normal(scale=0.1, size=(k, classes))
    initial_norm = float(np.linalg.norm(w))
    barrier = class_minimum * math.log(2)
    previous_loss, _, previous_margins = evaluate(w, phi, labels, probabilities)
    barrier_update = None
    certified_update = None
    first_correct = None
    largest_loss_increase = 0.0
    lowest_margin_increment = math.inf
    largest_bound_ratio = 0.0
    largest_norm_to_bound_ratio = 0.0
    snapshots = {}
    for update in range(1, steps + 1):
        _, gradient, _ = evaluate(w, phi, labels, probabilities)
        w -= eta * gradient
        loss, _, margins = evaluate(w, phi, labels, probabilities)
        largest_loss_increase = max(largest_loss_increase, loss - previous_loss)
        lowest_margin_increment = min(
            lowest_margin_increment, float((margins - previous_margins).min())
        )
        if margins.min() > 0 and first_correct is None:
            first_correct = update
        if radius_ratio == 0 and barrier_update is None and loss < barrier:
            barrier_update = update
        if barrier_update is not None:
            if margins.min() <= 0 or loss >= barrier:
                raise AssertionError("The invariant loss barrier failed.")
        if eta <= 1 / beta and update >= 2:
            bound = (k - 1) / update + (
                initial_norm + math.sqrt(2 * k / kappa) / gamma * math.log(update)
            ) ** 2 / (2 * eta * update)
            largest_bound_ratio = max(largest_bound_ratio, loss / bound)
            if loss > bound + 1e-12:
                raise AssertionError("The comparator bound failed.")
            if loss > previous_loss + 1e-12:
                raise AssertionError("Loss increased under the permitted step size.")
            norm_bound = initial_norm + math.sqrt(2 * eta * (k - 1)) + (
                2 * math.sqrt(2 * k / kappa) / gamma * math.log(update)
            )
            norm_ratio = float(np.linalg.norm(w)) / norm_bound
            largest_norm_to_bound_ratio = max(largest_norm_to_bound_ratio, norm_ratio)
            if norm_ratio > 1 + 1e-12:
                raise AssertionError("The comparator norm bound failed.")
        threshold = class_minimum * np.logaddexp(
            0.0, -2 * math.sqrt(2) * epsilon * np.linalg.norm(w)
        )
        if loss < threshold:
            if margins.min() <= 0:
                raise AssertionError("The context-uniform correctness criterion failed.")
            if certified_update is None:
                certified_update = update
        if rho == 0 and radius_ratio == 0:
            if np.any(margins <= previous_margins):
                raise AssertionError("An orthogonal-feature margin failed to increase.")
        if update in (100, 1000, 10000, steps):
            snapshots[str(update)] = {
                "loss": loss,
                "minimum_margin": float(margins.min()),
            }
        previous_loss, previous_margins = loss, margins
    return {
        "rho": rho,
        "kappa": kappa,
        "epsilon": epsilon,
        "gamma": gamma,
        "alpha": 8 * radius_ratio / gamma,
        "beta": beta,
        "eta": eta,
        "steps": steps,
        "seed": seed,
        "first_correct_update": first_correct,
        "barrier_update": barrier_update,
        "first_uniform_certificate_update": certified_update,
        "loss_barrier": barrier,
        "largest_loss_increase": largest_loss_increase,
        "smallest_margin_increment": lowest_margin_increment,
        "largest_loss_to_bound_ratio": largest_bound_ratio,
        "largest_norm_to_bound_ratio": largest_norm_to_bound_ratio,
        "snapshots": snapshots,
    }


def rare_context_counterexample():
    """A well-conditioned, separable dataset violating the class-only barrier."""
    k, p, rare_probability, displacement = 2, 0.25, 1e-6, 1 / 64
    feature_basis = np.eye(5)
    h_good = feature_basis[:, 0] + displacement * feature_basis[:, 4]
    h_bad = feature_basis[:, 0] - displacement * feature_basis[:, 4]
    phi = np.column_stack((h_good, h_bad, feature_basis[:, 1:4]))
    probabilities = np.array([
        (1 - p) / k * (1 - rare_probability),
        (1 - p) / k * rare_probability,
        p / k, (1 - p) / k, p / k,
    ])
    labels = np.array([0, 0, 1, 1, 0])
    score = np.array([9.5, -20.0, -20.0, 20.0, 10.5 / displacement])
    w = np.stack((score / 2, -score / 2))
    loss, _, margins = evaluate(w, phi, labels, probabilities)
    prototypes = np.column_stack((
        (1 - rare_probability) * h_good + rare_probability * h_bad,
        feature_basis[:, 1:4],
    ))
    kappa = float(np.linalg.eigvalsh(prototypes.T @ prototypes).min())
    radius = 2 * (1 - rare_probability) * displacement
    class_barrier = min(p, 1 - p) / k * math.log(2)
    assert kappa >= 1 - 1e-12
    assert radius < math.sqrt(kappa / k) / 16
    assert loss < class_barrier and margins.min() < 0
    return {
        "loss": loss,
        "class_only_barrier": class_barrier,
        "minimum_margin": float(margins.min()),
        "prototype_kappa": kappa,
        "within_class_radius": radius,
        "rare_context_probability_given_class": rare_probability,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path,
                        default=Path("results/lookup_convergence/check.json"))
    args = parser.parse_args()
    if args.steps < 10000:
        parser.error("--steps must be at least 10000")
    k, p = 3, 0.25
    target_probabilities = np.tile([(1 - p) / k, p / k], k)
    horizons = {}
    for n in (2, 4, 8):
        horizons[str(n)] = {}
        for first_step in (1, 2):
            enumerated = counting_probabilities(k, p, n, first_step)
            error = float(np.abs(enumerated - target_probabilities).max())
            if error > 1e-12:
                raise AssertionError("The population reduction depends on the horizon.")
            horizons[str(n)][f"steps_{first_step}_through_n"] = error
    cases = [
        run_case(k, p, rho, args.steps, args.seed)
        for rho in (0.0, 0.5, 0.9, 0.99)
    ]
    large_step = run_case(k, p, 0.0, 10000, args.seed, eta=8.0)
    variation_cases = [
        run_case(k, p, rho, args.steps, args.seed, radius_ratio=1 / 32)
        for rho in (0.9, 0.99)
    ]
    report = {
        "scope": "constructed frozen features; no Transformer feature-formation claim",
        "K": k,
        "p": p,
        "population_reduction_max_errors": horizons,
        "cases": cases,
        "orthogonal_large_step": large_step,
        "within_class_variation": variation_cases,
        "class_barrier_counterexample": rare_context_counterexample(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for row in cases:
        print(
            f"kappa={row['kappa']:.2g} "
            f"L1000={row['snapshots']['1000']['loss']:.6g} "
            f"L10000={row['snapshots']['10000']['loss']:.6g} "
            f"barrier_update={row['barrier_update']} "
            f"max_bound_ratio={row['largest_loss_to_bound_ratio']:.6g}"
        )
    for row in variation_cases:
        print(
            f"variation kappa={row['kappa']:.2g} epsilon={row['epsilon']:.6g} "
            f"L10000={row['snapshots']['10000']['loss']:.6g} "
            f"uniform_certificate_update={row['first_uniform_certificate_update']} "
            f"max_bound_ratio={row['largest_loss_to_bound_ratio']:.6g}"
        )
    print(f"Saved exact lookup checks to {args.output}")


if __name__ == "__main__":
    main()
