"""Summarize the gate-count (operation-diversity) sweep.

Two arms at depth 8, clean traces, three seeds. `natural` holds the dataset at
20k examples so fewer operations means more exposure per operation; `matched`
scales N with M so expected examples per (state, operation) cell are constant.
M in {5,10,20,34} are alias-free subsets (one name per distinct permutation);
M=52 is the canonical task, whose 52 strings realize only 34 permutations, so
the 34 -> 52 step varies surface forms at fixed function count.
"""
import glob, os, re
import pandas as pd

OUT = "results/paper/gate_count"


def load():
    rows = []
    for f in sorted(glob.glob(f"{OUT}/*/*.csv")):
        arm = f.split("/")[-2]
        stem = os.path.basename(f)[:-4]
        m = int(re.search(r"_m(\d+)$", stem).group(1)) if "_m" in stem else 52
        d = pd.read_csv(f)
        for cond, g in d.groupby("condition"):
            rows.append(dict(arm=arm, M=m,
                             cond="outcome" if cond == "outcome" else "process",
                             mean=100 * g.answer_accuracy.mean(),
                             sd=100 * g.answer_accuracy.std(), n=len(g)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = load()
    for cond in ("process", "outcome"):
        sub = d[d.cond == cond]
        t = sub.pivot_table(index="M", columns="arm", values=["mean", "sd"])
        print(f"\n=== {cond} (answer accuracy %, 3 seeds, rho=1.0, chance 6.25) ===")
        print(t.round(2).to_string())
    p = d[(d.cond == "process")]
    for arm in sorted(p.arm.unique()):
        a = p[p.arm == arm].sort_values("M")
        if len(a) > 1:
            mono = all(x >= y for x, y in zip(a["mean"][:-1], a["mean"][1:]))
            print(f"\n{arm}: monotone decreasing in M? {mono}   "
                  f"({', '.join(f'M{int(r.M)}={r['mean']:.1f}' for _, r in a.iterrows())})")
