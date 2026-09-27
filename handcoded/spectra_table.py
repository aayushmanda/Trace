"""
Tabulate spectrum.py JSON files, e.g. one per checkpoint, in step order.

    python handcoded/spectra_table.py logs/spec_track/*.json
"""
import json
import re
import sys

rows = []
for path in sys.argv[1:]:
    with open(path) as f:
        r = json.load(f)
    m = re.search(r"step (\d+)", r.get("weights", ""))
    step = int(m.group(1)) if m else -1
    W = r["W_r"]
    word = r["W1_fit"] + r["Whi_fit"] if "W1_fit" in r else sum(W[1:])
    hi = (r["hi_share"], r["hi_share_se"]) if "hi_share" in r else (sum(W[2:]) / word if word > 0 else 0.0, float("nan"))
    rows.append((r["site"], r.get("weights", "init").split(" ")[0], step, W[0], word, hi,
                 r.get("grad_outcome", r.get("grad_state", float("nan"))), path))

print(f"{'site':7s} {'trained':8s} {'step':>6s}  {'W_0':>9s}  {'W_>=1':>9s}  {'order>=2 share':>16s}  {'task grad':>9s}")
for site, mode, step, w0, word, (hi, hise), g, path in sorted(rows):
    print(f"{site:7s} {mode:8s} {step:6d}  {w0:9.3e}  {word:9.3e}  {hi:+7.2f} +- {hise:5.2f}  {g:9.3e}")