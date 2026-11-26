"""One cell of the seeded-initialization sweep: step and outcome training from one shared theta_0.

    python handcoded/experiments/seeded_sweep.py out=results/seeded/x.json init=handcoded_dirs init_std=0.1 word_len=12 mod=2

Every other key=value is a lettertrace.py setting. Writes the held-out exact accuracy curve of both modes, the final
exact and answer accuracy, and the first evaluated update with exact accuracy >= 0.9.
"""
import ast
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lettertrace as L  # noqa: E402

over, out = {}, None
for arg in sys.argv[1:]:
    key, value = arg.split("=", 1)
    if key == "out":
        out = Path(value)
        continue
    over[key] = value if key in L._STRING_KEYS else ast.literal_eval(value)
assert out is not None, "pass out=path.json"
if out.exists():
    sys.exit(0)
L.configure(**over)
train, shown, test = L.make_corpus()
theta_0 = L.init_theta()
rec = dict(config={k: getattr(L, k) for k in L.DEFAULTS if k != "device"},
           seeded_entries=sum(int((t != 0).sum()) for t in theta_0.values()),
           total_entries=sum(t.numel() for t in theta_0.values()), modes={})
t0 = time.time()
for mode in ("process", "outcome"):
    _, hist = L.train_mode(mode, theta_0, train, shown, test)
    evals = [h for h in hist if "exact" in h]
    rec["modes"][mode] = dict(final_exact=evals[-1]["exact"], final_answer=evals[-1]["answer"],
                              solved_at=next((h["step"] for h in evals if h["exact"] >= 0.9), None),
                              curve=[(h["step"], h["exact"]) for h in evals])
rec["seconds"] = time.time() - t0
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rec, indent=1))
print(f"{out.name}: step {rec['modes']['process']['final_exact']:.3f} (solved {rec['modes']['process']['solved_at']}) "
      f"outcome {rec['modes']['outcome']['final_exact']:.3f}", flush=True)
