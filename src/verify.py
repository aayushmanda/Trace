"""Pre-flight checks. Run before launching anything that takes hours.

Every check here has cost something in a previous run: a task whose examples
silently truncated, a declared chance floor that drifted after a sampler edit,
a parser that read the wrong number out of a generation, a cache that did not
round-trip, an answer-forcing context that did not line up with the training
string. They are cheap, they are deterministic, and they fail loudly.

    python main.py verify              # instant checks only
    python main.py verify --train      # + a short training smoke test
"""

from __future__ import annotations

import random
import tempfile
import time
from dataclasses import replace

import torch

from config import Config
from data import (
    CacheMiss,
    encode_row,
    expand,
    load_index_stream,
    build_eval_set,
    load_pools,
    mix_pools,
    teacher_forcing_rows,
)
from engine import evaluate, train
from tasks import TASKS, estimate_reference_levels, get_task

TOL = 0.01  # tolerance on measured vs. declared reference levels


def check_examples() -> bool:
    """Sampled examples, vocabulary and context budget for every task."""
    print("=" * 78)
    print(" 1. EXAMPLES, BUDGETS AND TRUNCATION")
    print("=" * 78)
    ok = True
    for task in TASKS.values():
        longest = task.max_example_len(n_probe=20_000)
        headroom = task.block_size - longest
        ok = ok and headroom >= 0
        print(f"\n--- {task.name} --- {task.description}")
        print(
            f"  vocab={task.tokenizer.vocab_size}  block_size={task.block_size}  "
            f"longest={longest}  headroom={headroom}  "
            f"[{'OK' if headroom >= 0 else 'TRUNCATES'}]"
        )
        inst = task.sample()
        for mode in ("correct_think", "wrong_think", "no_think"):
            p, a = task.render(inst, mode)
            print(f"  {mode:<14}: {p}{a}")
    return ok


def check_reference_levels(n: int) -> bool:
    """The declared chance floor and Bayes ceiling must still match the sampler."""
    print("\n" + "=" * 78)
    print(f" 2. REFERENCE LEVELS (Monte Carlo, n={n:,})")
    print("=" * 78)
    print(f"{'task':<14} | {'chance (measured vs declared)':>34} | "
          f"{'ceiling (measured vs declared)':>34} | verdict")
    print("-" * 78)
    ok = True
    for task in TASKS.values():
        chance, ceiling = estimate_reference_levels(task, n=n)
        good = abs(chance - task.chance_acc) < TOL and abs(ceiling - task.ceiling_acc) < TOL
        ok = ok and good
        print(
            f"{task.name:<14} | {chance:>16.5f} vs {task.chance_acc:<15.5f} | "
            f"{ceiling:>16.5f} vs {task.ceiling_acc:<15.5f} | {'ok' if good else 'DRIFT'}"
        )
    if not ok:
        print("\n[WARN] declared levels in tasks.py drift from the samplers; update "
              "chance_acc/ceiling_acc or every plot floor and every fit is wrong.")
    return ok


def check_parser() -> bool:
    """Answer parsing, in both the free and the answer-forced reading."""
    print("\n" + "=" * 78)
    print(" 3. ANSWER PARSER")
    print("=" * 78)
    ok = True

    # (tail, expected under free generation, expected under answer forcing)
    cases = {
        "word_index": [
            (" 0h 1w 2v 3u : 2\n", "2", "2"),
            (" : 12\n", "12", "12"),
            (" junk\n", None, None),
            ("3\n", "3", "3"),
            ("3 7\n", "7", "3"),  # overrun after the answer position
        ],
        "count_char": [(" a1 b1 c2 : 7\n", "7", "7"), ("0\n", "0", "0")],
        "multiply": [(" 5*48=240 90*48=4320 240+4320=4560 : 4560\n", "4560", "4560")],
        "sort_letters": [(" ab>a b>b : abz\n", "abz", "abz"), (" : \n", None, None)],
    }
    for name, rows in cases.items():
        task = get_task(name)
        for tail, free_expected, forced_expected in rows:
            got_free = task.extract_answer(tail)
            got_forced = task.extract_answer(tail, first=True)
            good = got_free == free_expected and got_forced == forced_expected
            ok = ok and good
            print(f"  {name:<13} {tail!r:<48} free={got_free!r:<8} "
                  f"forced={got_forced!r:<8} [{'ok' if good else 'FAIL'}]")
    return ok


def check_cache_roundtrip(n: int = 2000) -> bool:
    """The compact cache must rebuild x/y/mask exactly, and A must be denser than B.

    Under supervise="full" the loss mask starts right after the prompt; under
    "answer_only" right after the final ':'. Neither may ever cover padding.
    """
    print("\n" + "=" * 78)
    print(" 4. CACHE ROUND TRIP AND LOSS MASK")
    print("=" * 78)
    ok = True

    for task in TASKS.values():
        densities = {}
        for supervise, render_mode, cond in (
            ("full", "correct_think", "A full"),
            ("answer_only", "correct_think", "B answer_only"),
            ("full", "no_think", "C direct"),
        ):
            rows, bnds, expected = [], [], []
            random.seed(4321)
            for _ in range(n):
                inst = task.sample()
                prompt, answer = task.render(inst, render_mode)
                row, bnd, _ = encode_row(task, prompt, answer, supervise)
                rows.append(row)
                bnds.append(bnd)
                # Independent reference construction of the same mask.
                full = task.tokenizer.encode(prompt + answer + "\n")[: task.block_size]
                expected.append(
                    [1.0 if (i + 1) >= bnd else 0.0 for i in range(len(full) - 1)]
                    + [0.0] * (task.block_size - len(full))
                )

            _, _, mask = expand(
                torch.tensor(rows, dtype=torch.uint8),
                torch.tensor(bnds, dtype=torch.int16),
                task.tokenizer.pad_id,
            )
            same = bool(torch.equal(mask, torch.tensor(expected, dtype=torch.float)))
            ok = ok and same
            density = float(mask.sum()) / float(mask.numel())
            densities[cond] = density
            print(f"  {task.name:<13} {cond:<14} mask matches direct build: "
                  f"{'yes' if same else 'NO':<4} supervised={density * 100:5.1f}%")

        denser = densities["A full"] > densities["B answer_only"]
        ok = ok and denser
        if not denser:
            print(f"  {task.name:<13} [FAIL] condition A is not denser than B")

    with tempfile.TemporaryDirectory() as tmp:
        task = get_task("word_index")
        a = load_pools(task, "full", n, cache_dir=tmp, verbose=False)
        b = load_pools(task, "full", n, cache_dir=tmp, verbose=False)
        same = all(torch.equal(a[k].tokens, b[k].tokens) for k in a)
        half = load_pools(task, "full", n // 2, cache_dir=tmp, verbose=False)
        prefix = all(torch.equal(a[k].tokens[: n // 2], half[k].tokens) for k in a)
        ok = ok and same and prefix
        print(f"  cache reload identical: {'yes' if same else 'NO'}")
        print(f"  smaller request is a prefix of the cache: {'yes' if prefix else 'NO'}")

        # Strict mode is what every training path uses: a pool that is not on
        # disk must stop the run, not be sampled while a GPU waits for it.
        try:
            load_pools(get_task("count_char"), "full", n, cache_dir=tmp,
                       verbose=False, require_cache=True)
            refused = False
        except CacheMiss:
            refused = True
        cached_ok = torch.equal(
            load_pools(task, "full", n, cache_dir=tmp, verbose=False,
                       require_cache=True)["correct"].tokens,
            a["correct"].tokens,
        )
        ok = ok and refused and cached_ok
        print(f"  strict mode refuses to sample a missing pool: {'yes' if refused else 'NO'}")
        print(f"  strict mode still reads a cached pool: {'yes' if cached_ok else 'NO'}")

        # Ratio mixing must take exactly `ratio` of its rows from the correct pool.
        mixed = mix_pools(a["correct"], a["wrong"], 0.25)
        k = int(n * 0.25)
        exact = bool(
            torch.equal(mixed.tokens[:k], a["correct"].tokens[:k])
            and torch.equal(mixed.tokens[k:], a["wrong"].tokens[k:])
        )
        ok = ok and exact
        print(f"  ratio mix takes the right rows: {'yes' if exact else 'NO'}")
    return ok


def check_training_order(n: int = 4096) -> bool:
    """The saved training order must be reproducible, balanced and shareable.

    Reproducible: reloading gives the same stream. Balanced: within one pass
    over the pool every example appears exactly once, so no row is over- or
    under-sampled. Shareable: a shorter request is a prefix of a longer one,
    which is what lets two batch sizes consume the same examples in the same
    order, and two ratios see the same problems.
    """
    print("\n" + "=" * 78)
    print(" 6. SAVED TRAINING ORDER")
    print("=" * 78)
    ok = True

    with tempfile.TemporaryDirectory() as tmp:
        a = load_index_stream(n, 2 * n, order_seed=0, cache_dir=tmp, verbose=False)
        b = load_index_stream(n, 2 * n, order_seed=0, cache_dir=tmp, verbose=False)
        short = load_index_stream(n, n // 2, order_seed=0, cache_dir=tmp, verbose=False)
        other = load_index_stream(n, 2 * n, order_seed=1, cache_dir=tmp, verbose=False)

        same = bool(torch.equal(a, b))
        prefix = bool(torch.equal(a[: n // 2], short))
        balanced = all(
            bool(torch.equal(torch.sort(a[i * n : (i + 1) * n].long()).values,
                             torch.arange(n)))
            for i in range(2)
        )
        differs = not bool(torch.equal(a, other))
        ok = same and prefix and balanced and differs

        print(f"  reloading gives the same stream:            {'yes' if same else 'NO'}")
        print(f"  a shorter request is a prefix of a longer:  {'yes' if prefix else 'NO'}")
        print(f"  each pass is a permutation (no resampling): {'yes' if balanced else 'NO'}")
        print(f"  a different order_seed gives a different order: {'yes' if differs else 'NO'}")
    return ok


def check_answer_forcing(n: int = 500) -> bool:
    """The forced context must be a genuine prefix of the training string.

    If it is not, answer forcing is scoring the model at a position it was never
    trained on and every forced number is meaningless.
    """
    print("\n" + "=" * 78)
    print(" 5. ANSWER FORCING CONTEXTS")
    print("=" * 78)
    ok = True

    for task in TASKS.values():
        random.seed(99)
        instances = [task.sample() for _ in range(n)]
        for mode, render_mode in (
            ("forced_correct", "correct_think"),
            ("forced_wrong", "wrong_think"),
            ("direct", "no_think"),
        ):
            prefix_ok = True
            for inst in instances:
                context = task.context(inst, mode)
                prompt, answer = task.render(inst, render_mode)
                if prompt + answer + "\n" != context + inst.gold + "\n":
                    prefix_ok = False
                    break
            ok = ok and prefix_ok

            tokens, bounds = teacher_forcing_rows(task, instances, mode, device="cpu")
            _, y, mask = expand(tokens, bounds, task.tokenizer.pad_id)
            # The scored region must be exactly the gold answer plus its newline.
            want = torch.tensor([len(task.tokenizer.encode(i.gold)) + 1 for i in instances])
            lengths_ok = bool(torch.equal(mask.sum(dim=1).long(), want))
            ok = ok and lengths_ok
            print(
                f"  {task.name:<13} {mode:<15} context is a prefix: "
                f"{'yes' if prefix_ok else 'NO':<4} scored tokens = gold+newline: "
                f"{'yes' if lengths_ok else 'NO'}"
            )
    return ok


def check_training(cfg: Config, steps: int, samples: int, val_examples: int, seed: int) -> None:
    """A short run per task: the loop, the sizing and every metric must work.

    The accuracies are not meaningful at this budget -- only the fact that all
    of them come back finite and in range is.
    """
    print("\n" + "=" * 78)
    print(f" 6. TRAINING SMOKE TEST (steps={steps}, samples={samples:,})")
    print("=" * 78)

    # The smoke test deliberately works on small throw-away data it samples
    # itself, so it opts out of the "read everything from disk" rule that real
    # runs are held to.
    cfg = replace(cfg, data=replace(cfg.data, require_cache=False))

    for task in TASKS.values():
        instances = build_eval_set(task, val_examples, cfg.data.val_seed, cache_dir=None)
        pools = load_pools(task, "full", samples, cache_dir=None, use_cache=False,
                           device=cfg.device, verbose=False)
        pool = mix_pools(pools["correct"], pools["wrong"], 1.0)

        t0 = time.time()
        model, loss = train(pool, task, cfg, seed, steps=steps, batch_size=256)
        t_train = time.time() - t0

        t0 = time.time()
        m = evaluate(model, task, instances, cfg)
        print(
            f"  {task.name:<13} loss={loss:5.3f}  free={m.free_acc * 100:5.1f}%  "
            f"forced={m.forced_acc * 100:5.1f}%  forced_wrong={m.forced_wrong_acc * 100:5.1f}%  "
            f"tf={m.answer_tf_acc * 100:5.1f}%  nll={m.answer_nll:5.3f}  "
            f"trunc={m.truncation_rate * 100:4.1f}%  "
            f"({t_train:.0f}s train, {time.time() - t0:.0f}s eval)"
        )
        del model, pool, pools


def run_checks(cfg: Config, mc_samples: int = 200_000, do_train: bool = False,
               steps: int = 300, samples: int = 20_000, val_examples: int = 200,
               seed: int = 1337) -> bool:
    ok = check_examples()
    ok = check_reference_levels(mc_samples) and ok
    ok = check_parser() and ok
    ok = check_cache_roundtrip() and ok
    ok = check_answer_forcing() and ok
    ok = check_training_order() and ok
    if do_train:
        check_training(cfg, steps, samples, val_examples, seed)

    print("\n" + "=" * 78)
    print(" ALL PRE-FLIGHT CHECKS PASSED" if ok else " PRE-FLIGHT CHECKS FOUND ISSUES")
    print("=" * 78)
    return ok
