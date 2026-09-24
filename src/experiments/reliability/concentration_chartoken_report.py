"""Paired analysis of the character-token D=4 concentration sweep."""
import sys, math, itertools
import pandas as pd

OUT = "results/paper/concentration_chartoken_d4"
LAMS = {0: 0.0, 50: 0.5, 100: 1.0}
SEEDS = [2001, 2002, 2003, 2004, 2005]

def load():
    frames = []
    for tag, lam in LAMS.items():
        df = pd.read_csv(f"{OUT}/boolean_circuit_4_conc{tag}.csv")
        df = df[df.step == df.step.max()]
        df["lam"] = lam
        frames.append(df)
    return pd.concat(frames)

def acc(df, lam, rho, seed):
    r = df[(df.lam == lam) & (df.rho.round(2) == rho) & (df.seed == seed)]
    return float(r.answer_accuracy.iloc[0]) * 100.0

def paired(df, rho, a, b):
    d = [acc(df, a, rho, s) - acc(df, b, rho, s) for s in SEEDS]
    n = len(d); m = sum(d)/n
    sd = math.sqrt(sum((x-m)**2 for x in d)/(n-1))
    se = sd/math.sqrt(n); t = m/se if se else float("inf")
    crit = 2.776  # t_.975, df=4
    return m, (m-crit*se, m+crit*se), t, d

if __name__ == "__main__":
    df = load()
    print(f"{'rho':>5} {'lambda':>7} {'mean':>7} {'SD':>6}   per-seed")
    for rho in (0.5, 0.7):
        for lam in (0.0, 0.5, 1.0):
            v = [acc(df, lam, rho, s) for s in SEEDS]
            m = sum(v)/len(v)
            sd = math.sqrt(sum((x-m)**2 for x in v)/(len(v)-1))
            print(f"{rho:>5} {lam:>7} {m:>7.2f} {sd:>6.2f}   {['%.1f'%x for x in v]}")
    print("\nPaired contrasts (more concentration should LOWER accuracy => positive diff):")
    for rho in (0.5, 0.7):
        for a, b in [(0.0, 0.5), (0.5, 1.0), (0.0, 1.0)]:
            m, ci, t, d = paired(df, rho, a, b)
            star = "  *" if ci[0] > 0 else ""
            print(f"  rho={rho}  lam {a} - lam {b}: {m:+6.2f} pts  95% CI [{ci[0]:+6.2f},{ci[1]:+6.2f}]  t={t:+5.2f}{star}")
    print("\nMonotone in seed-mean at each rho:")
    for rho in (0.5, 0.7):
        ms = [sum(acc(df, l, rho, s) for s in SEEDS)/5 for l in (0.0, 0.5, 1.0)]
        print(f"  rho={rho}: {ms[0]:.2f} -> {ms[1]:.2f} -> {ms[2]:.2f}  "
              f"{'monotone decreasing' if ms[0] > ms[1] > ms[2] else 'NOT monotone'}")
