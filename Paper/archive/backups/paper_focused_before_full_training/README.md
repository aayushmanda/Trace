# Focused Transformer theory paper

`main.pdf` is the 28-page compiled manuscript. `main.tex` inputs `body.tex`, `appendix.tex`, and local section fragments. All required figures, bibliography and styles are local.

From this directory:

```bash
make                   # Tectonic build; local manuscript dependencies
make figures           # regenerate figure/table from existing parent-repository runs
make verify            # exhaustive finite executors and original gradient/algebra checks
make check             # page count, references, theorem count and evidence assertions
make audit-routing     # reconstruct the 17-pair empirical routing summary
```

The default Python is `/home/hariguru/aayus/.venv/bin/python`; override `PYTHON` if necessary. Verification uses PyTorch, NumPy, mpmath and tqdm; figure regeneration also uses Matplotlib. The original repository environment contains them. `TECTONIC` can also be overridden.

Review `proof_audit.md`, `claims_matrix.md`, and `CHANGELOG.md` for scope, repairs and evidence. `evidence/` records the checks actually run and the original per-seed measurements. Full model training was not rerun. The main theorem is an aligned supplied-routing result with fixed features and tied scalar/Frobenius training, not unrestricted Transformer convergence.
