import sys
sys.path.insert(0, "Paper2/checks")
from reduced_flow import run
eps, eps0, lr = map(float, sys.argv[1:4])
for N in (8, 16, 32, 64, 128):
    for law in ("sym", "stale"):
        r = run(N, eps, law, eps0=eps0, lr=lr, steps=2_000_000, check=10)
        print(f"N={N:4d} {law:5s} T={r['T_route']} minNa={r['min_alpha_s']*N:.3g}@{r['t_min']} "
              f"max_ap={r['max_alpha_p']:.3f} mc={r['mc']:.3f} mr={r['mr']:.3f}", flush=True)
