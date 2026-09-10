# How to run Trace (paper experiments)

Audience: research engineer reproducing the Boolean-circuit results. From the repo root. Training prints tqdm progress bars; set `TRACE_TQDM=0` to silence them.

**Revision positioning (10 Sep 2026):** [PAPER_POSITIONING.md](PAPER_POSITIONING.md), [THEORY_EMPIRICAL_REVISION_ROADMAP.md](THEORY_EMPIRICAL_REVISION_ROADMAP.md). Theorems 1–3 are an exact **shared-executor** explanation; **T2 is unproved** ([Paper/T2_CANDIDATE.md](Paper/T2_CANDIDATE.md)). Empirical centerpiece is **reliability \(\rho\)** (local vs rollout), not induced-rule/pullback. New dump dir: `results/paper_revision_v2/`. Ledger: `Paper/theorem_ledger.md`. Frozen week-2 protocols: `results/paper_revision_v2/protocols/`.

## Environment

Python 3.12+. Either:

```bash
conda activate aayus
# or
uv sync
```

This box: 4× A100 80GB. Prefer **GPU 2 or 3** when 0/1 are occupied. Do not kill other jobs.

```bash
export CUDA_VISIBLE_DEVICES=2   # then --device cuda:0
# or
python -m src … --device cuda:2
```

If `CUDA_VISIBLE_DEVICES` is set, scripts treat `cuda:0` as that physical GPU. Default device selection already prefers `cuda:2` then `cuda:3`. Cap concurrent jobs (`MAX_JOBS=2` in `experiments/run_revision_bridge.sh`).

### Optional 2-GPU training (DataParallel, not DDP)

Default is **one GPU**. Nothing occupies all 4 A100s unless you ask. Training can opt into `torch.nn.DataParallel` (in-process; **not** `torchrun` DDP):

| Switch | Example |
|---|---|
| CLI | `--distributed` / `--no-distributed` |
| YAML | `distributed: true` |
| Env | `TRACE_DEVICES=2,3` (2+ ids also turns DP on; `--no-distributed` turns it off) |
| Env override | `TRACE_DISTRIBUTED=0` or `1` |

This box: prefer physical **2 and 3**. `batch_size` is the **global** batch (split across GPUs; 128 on 2 GPUs → 64 each). Use a global batch divisible by the GPU count.

Induced-rule probes, generate, pullback readouts, and executor diagnostics stay on the **eager** module on the primary GPU. Compile stays off by default; if you combine `--compile --distributed`, the replica is compiled **after** the DataParallel wrap — do not patch `.forward`.

```bash
# 1-GPU (default): cuda:2 if present
python -m src induced --config configs/experiments/induced_rule.yaml --device cuda:2 --no-distributed --no-compile

# 2-GPU train, probes still 1-GPU
TRACE_DEVICES=2,3 python -m src induced --config configs/experiments/induced_rule.yaml --distributed --no-compile

# equivalent
python -m src induced --config configs/experiments/induced_rule.yaml --distributed --device cuda:2 --no-compile
```

Do not use `torchrun --nproc_per_node=N` with this stack; there is no DDP process-group wrapper.

**Compile is off by default.** Training may opt in with YAML `compile: true`, `TRACE_COMPILE=1`, or `--compile`. That compiles a **wrapper** used only for the train loss; `generate`, probes, induced-rule, and pullback always call the eager module (same parameters). Do not replace `model.forward` with `torch.compile`. `--no-compile` forces eager. Tests never compile. Architecture-controls `run_one` stays eager (no `compile: true` in that YAML).

```bash
python -m src executor --config configs/experiments/executor_comparison.yaml --compile   # train compiled
python -m src executor --no-compile   # override YAML compile: true
```

---

## Two stacks (do not mix forwards)

**GPT** — character-token LayerNorm Transformer (`src.models.gpt.GPTModel`). Boolean circuits as character strings. Used for Table 1 / Figs 1–2, and for **optional extras** (induced-rule, length, \(m_{\min}\)) that are **not** paper Figure 4. Tiny synthetic-token models; you cannot drop Llama/Qwen in here.

**Handcoded** — semantic-token executors (`handcoded/` package): one token per 4-bit state / gate. **Paper §7 / Figs 4–5 / Table 7** are this stack (one-block width 96), not GPT and not the oracle-shaped Table 4 nets. Same limit: the alphabet is the task tokenizer, not a HuggingFace LM.

**Pretrained LoRA** — rank-8 Peft on `AutoModelForCausalLM` (`python -m src lora`). Same Boolean/trace serialization as GPT, but a real HF tokenizer. This **is** where Qwen/Llama can be swapped. See the section below.

Shared: seed, YAML under `configs/experiments/`, CSV writers, device, progress bars.

Entry point: `python -m src <command>`. Equivalent paper scripts still exist under `experiments/` if you need a one-off flag.

---

## Pretrained LoRA: what you can swap (Qwen / Llama)

The paper’s pretrained paragraph is **rank-8 LoRA** on **SmolLM2-135M**, not a from-scratch GPT with a different backbone.

**Can swap.** HuggingFace causal LMs via `AutoModelForCausalLM` + Peft (`target_modules="all-linear"`). Default remains `HuggingFaceTB/SmolLM2-135M`. Pass another id with `--model`:

```bash
python -m src lora --help
python -m src lora --model HuggingFaceTB/SmolLM2-135M
# other AutoModelForCausalLM ids, e.g.:
# python -m src lora --model Qwen/Qwen2.5-0.5B --config configs/experiments/lora_transfer.yaml
```

YAML example (same defaults as the script): `configs/experiments/lora_transfer.yaml`. CLI overrides the file.

This path trains outcome / answer-first / process-\(\rho\) completions and reports **free-generation accuracy** (answer, exact trace, exact state rollout). For `boolean_circuit_*` it also runs the **16-way local state** scorer on the **same HF tokenizer**, not the GPT `GLOBAL_TOKENIZER`. Startup aborts if `encode(prefix)+encode(state) != encode(prefix+state)` at the `>` boundary — common on some BPE/SentencePiece models unless you add/map special tokens. You may also need a larger `--max-length` (this is the sequence cap; it is not GPT `block_size`).

**Cannot swap.** Do not pass a Llama/Qwen id into GPT training, induced-rule, pullback, split-verdict, length, margins, reliability, or handcoded executors. Those models are tiny synthetic-token nets (`CharTokenizer` / semantic gate-state tokens). Induced-rule and pullback readouts are hardcoded to `GLOBAL_TOKENIZER` (character-level 16-way tables). Llama is not a drop-in there.

Do not download a 7B checkpoint unless you intend to; the default 135M run is the paper setting.

**Provenance.** Headline Table 1 (`python -m src supervision`) and LoRA (`python -m src lora --config configs/experiments/lora_transfer.yaml`) need the command, YAML, seed list, and output path recorded under `results/paper_revision_v2/provenance/` before they are cited. Do not treat an undocumented CSV as the paper table.

---

## Headline empirical: trace reliability (E5)

This is the **main empirical command**. \(\rho\) is the probability a training example gets a **valid trace**; the **terminal answer stays correct**. Metrics: `answer_accuracy` (rollout/answer), `exact_trace_accuracy` (full trace), `trace_step_accuracy` (local). Each run writes a CSV and a sibling `*_persist.json` (seeds, \(\rho\), assignment seeds, rows).

Do **not** launch the full seed\(\times\rho\) GPU grid from this file. One logged job at a time. Prefer `--device cuda:2` or `cuda:3`.

```bash
# Logged smoke (CPU; not a paper cell):
python -m src reliability --task boolean_circuit_2 --rhos 0.8 --seeds 2001 \
  --checkpoints 2 --train-size 32 --val-size 8 --batch-size 8 \
  --output results/paper_revision_v2/e5_reliability/smoke.csv --device cpu --no-compile

# Paper-scale (GPU hours; one task; persist JSON beside the CSV):
python -m src reliability \
  --task boolean_circuit_8 \
  --rhos 0.0 0.5 0.8 1.0 \
  --seeds 2001 2002 2003 \
  --checkpoints 1000 2000 4000 \
  --train-size 20000 --val-size 1000 \
  --batch-size 128 --include-outcome \
  --output results/reliability_sweeps/boolean_circuit_8.csv \
  --device cuda:2 --no-compile
```

Figs 1–2 / Table 3 in the PDF come from this family (`results/reliability_sweeps/`). Replication must keep `train_seed`, `val_seed`, `ratio_seed`, `batch_seed`.

Interchange interventions (**E1**, ~75 runs) are **not** the first empirical move and are not launched here.

---

## Paper §7: handcoded semantic-token readout (Figs 4–5, Table 7)

This is the completed five-seed study in the compiled PDF. Protocol: one causal block, width 96, 4 heads, 192-unit ReLU MLP, no LN/dropout, \(D\in\{2,4,6\}\), seeds 42–46, 10k train circuits, 2k updates, process / outcome / mixed. Builder refuses incomplete seed×depth coverage.

```bash
# Single comparison run (empty output dir required):
python -m src executor --config configs/experiments/executor_comparison.yaml \
  --output results/executor_comparison/smoke --steps 2 --train-size 64 --test-size 16 \
  --probe-size 8 --batch-size 16 --checkpoints 0 2 --device cpu --threads 1

# Depth replication (paper: D=2,4,6 × seeds 42–46). Default workers are CPU.
python -m src executor-depths --config configs/experiments/executor_depths.yaml

# Verified figures + Table 7 TeX (run after replication is complete):
python experiments/build_executor_results.py \
  --input results/executor_comparison/depth_replication --paper Paper
```

Writes:

- `Paper/figures/trained_executor_bridge.pdf` — Figure 4
- `Paper/figures/trained_executor_depths.pdf` — Figure 5
- `Paper/data/trained_executor_rows.tex` — Table 7 rows (`\input` from `trained_model_protocol.tex`)
- `Paper/data/trained_executor_numbers.tex` — §7 macros (`\input` from `trained_model_results.tex`)
- paste copy: `Paper/data/trained_executor_table.md`

Existing complete run: `results/executor_comparison/depth_replication/` (`metrics.csv`, per-seed `report.md`). Do not treat GPT `induced_rule.py` output as this figure.

The YAML may list depth 8; the **manuscript study is \(D\in\{2,4,6\}\)** as in `protocol.json`. Do not pool with character-token GPT readouts.

---

## Handcoded tutorial notebook

Package: `handcoded/` (`gates`, `tokenizer`, `models`, `data`, `generate`, `train`, `eval`, `animate`). Paper hyperparameters: `configs/handcoded.yaml`. Smoke: `configs/handcoded_smoke.yaml` (or `SMOKE = True` in the notebook).

```bash
conda activate aayus   # or: uv sync && source .venv/bin/activate
cd /path/to/trace
jupyter notebook handcoded/handcoded_executors.ipynb
```

Device: the first cell prefers `cuda:2` if ≥3 GPUs exist, else `cuda`, else CPU. Plots use `src.plot_style.apply_style()`. `handcoded_utils.py` is a deprecated import shim; new code should `import handcoded`.

---

## Optional transfer diagnostics (induced-rule / pullback)

These are **not** the empirical centerpiece and **not** paper Figure 4 / Table 7. They diagnose whether local tables transfer to \(\theta\) — including **failure** (e.g. TV \(\approx 0.833\), cosine \(\approx 0.028\)). Do **not** treat them as confirmation of an internal executor. Compile off. One `(depth, seed)` at a time; no full GPU grid from this file.

Discard GPT readouts with `state_on_set_mass < 0.5`. Mixed-format checkpoints: `condition: both` / `trace_fraction: 0.5`. Full-vocab LM CE; Jacobian is \(\sum_t J_t^\top W_t\) (all steps).

```bash
# Pullback smoke (skip 16×52 readout; exact LM grads still run):
python -m src pullback --config configs/experiments/pullback.yaml \
  --depth 2 --seed 2001 --train-size 64 --val-size 16 --probe-size 8 \
  --checkpoints 0 --n-layer 1 --n-embd 32 --batch-size 8 --device cpu \
  --skip-readout --out results/revision/smoke_pullback.csv --no-compile

python -m src induced --config configs/experiments/induced_rule.yaml \
  --depth 4 --condition both --seed 2001 --with-pullback --device cuda:2 --no-compile

python -m src split-verdict --config configs/experiments/split_verdict.yaml
```

Split-verdict is a **limit-of-transfer** table (exponent, \(\delta_{\mathrm{comp}}\), cosine), not the main figure.

---

## Architecture × supervision 2×2 (E2; still required)

PDF **Table 6** is still the historical **one-seed** notebook. Do not overwrite those cells. YAML has `compile: false`; `run_one` is eager. Devices default to GPU 2/3.

**Job counts (do not start the 80-job confirm in a plumbing pass):** calibrate = 2 architectures × 5 LRs × 2 clips × 2 formats = **40** jobs; confirm after `selected_rates.json` = 2 × 10 seeds × 2 clips × 2 formats = **80** jobs. Frozen YAML: `configs/experiments/e2_architecture.yaml`.

```bash
python -m src architecture plan --config configs/experiments/e2_architecture.yaml \
  --output results/paper_revision_v2/e2_architecture
python -m src architecture single --output /tmp/arch_smoke --architecture process --mode process \
  --seed 1 --lr 0.0005 --clip 1 --stage calibrate --steps 1 \
  --train-size 8 --val-size 4 --test-size 4 --batch-size 4 --device cpu --no-compile
python -m src architecture summarize --config configs/experiments/architecture_controls.yaml
# Success = complete AND test acc > 95%. Unstable runs are failures in the denominator.
```

Output: `results/architecture_controls_n10/`. Existing leftover plan: `results/architecture_controls/protocol.json`.

---

## Shared trainable kernel executor

One tensor \(A\in\mathbb{R}^{M\times K\times K}\), \(P_g=\mathrm{softmax}(A_g)\). Same parameters for outcome compose vs process local CE. Cheap CPU/GPU; this is a **row-softmax logit** parameterization, not the T2 projected-simplex candidate.

```bash
python -m src escape --smoke
python -m src escape --config configs/experiments/escape_time.yaml --device cpu
```

CSV: `results/revision/shared_kernel.csv`.

Theory-matched **projected** simplex flow (Euclidean projection, \(r(t)\), discrete Dini ratio). Does **not** prove T2.

```bash
python -m src projected --smoke
python -m src projected --config configs/experiments/e4_projected_kernel.yaml --device cpu
```

CSV: `results/paper_revision_v2/e4_projected_kernel/`.

---

## Mechanism test: outcome + local credit (answer-only)

This is the **mechanism** experiment. A depth-\(D\) outcome architecture generates **COLON / answer / EOS only**. Three conditions; 1–2 share architecture, init, and direct-answer format:

| Condition | Loss | Tokens |
|---|---|---|
| \(L_{\mathrm{out}}\) | ordinary outcome CE | answer only |
| \(L_{\mathrm{out}}+\mathrm{local}\) | \(L_{\mathrm{out}}+\lambda\frac{1}{D}\sum_t \mathrm{CE}(H_t(h_t),s_t)\) | **no trace tokens**; \(H_t\) is a linear residual-stream head after block \(t\) |
| \(L_{\mathrm{process}}\) | ordinary process | process tokens (comparison) |

If outcome \(\approx\) chance but outcome+local \(\approx\) process \(\approx 100\%\), that isolates **credit placement**. Mixed-format \(\widehat P_g\) remains the **negative transfer** readout — do not treat it as validating the shared-executor model. Report \(C_t=\langle g_t^{\mathrm{term}},g_t^{\mathrm{local}}\rangle/\|g_t^{\mathrm{local}}\|^2\) vs \(D\); **do not** claim \(\varepsilon^{D-1}\) unless mixing assumptions are checked.

```bash
# Tests (oracle patch must be ~100%; random patch must not):
TRACE_TQDM=0 python -m unittest tests.test_outcome_local

# Smoke: D=2, tiny data, ~30 steps, 1 seed. CPU default; use cuda:2 if that card is free.
python -m src outcome-local --smoke --device cpu --no-compile
# or
python experiments/outcome_local_rescue.py --smoke --device cuda:2 --no-compile
```

YAML: `configs/experiments/outcome_local.yaml` (\(\lambda=1.0\)). Outputs: `results/paper_revision_v2/outcome_local/` (`metrics.csv`, `probes.csv`, `credit.csv`, `patch.csv`, `table.csv`, `apply_style` PDFs). Confirmation depths 2/4/6/8 are recorded in the YAML and **not** launched from this file.

---

## Length + \(m_{\min}\) (E6; after E5 / E2)

Train \(D=8\), eval 8/10/12/16; gold-path \(m_{\min}\) at \(\rho=0.80\). No figure/table number in `main.pdf`. Do not run the full grid now.

```bash
python -m src length --config configs/experiments/length_generalization.yaml --device cuda:2 --no-compile
python -m src margins --config configs/experiments/margin_histograms.yaml --device cuda:2 --no-compile
python -m src analyze   # Paper/figures/revision_*.pdf if CSVs exist
```

---

## Extra mechanism tasks (not Boolean / FSM / register)

`modular_program_*`, `stack_machine_*`, and `word_index_len*` are already registered sequential-executor families: exact local traces, same-length corruptions, correct gold, known chance accuracy (\(1/17\), \(1/17\), \(1/L\)). Boolean / FSM / register stay in `TASKS`. Also registered on the same interface: `tape_machine_*`, `queue_machine_*`, `grid_walk_*`. Tiny outcome vs process smoke (not a paper cell; the gap may be small):

```bash
# GPU 2 or 3; documentary YAML: configs/experiments/mechanism_tasks.yaml
python -m src supervision --tasks modular_program_8 stack_machine_8 word_index_len16 \
  --modes outcome process --seeds 2001 \
  --train-size 512 --val-size 128 --steps 80 --batch-size 32 \
  --device cuda:2 --no-compile

python -m src reliability --task stack_machine_8 --rhos 0.8 --seeds 2001 \
  --checkpoints 2 --train-size 32 --val-size 8 --batch-size 8 \
  --output results/paper_revision_v2/e5_reliability/stack_machine_8_smoke.csv \
  --device cpu --no-compile
```

## Smoke

```bash
TRACE_TQDM=0 python -m unittest tests.test_plumbing tests.test_revision_bridge tests.test_mechanism_tasks tests.test_local_machine_tasks tests.test_outcome_local
python -m src escape --smoke
python -m src projected --smoke
python -m src architecture plan --config configs/experiments/e2_architecture.yaml --output results/paper_revision_v2/e2_architecture
python -m src smoke   # tiny mixed-format induced D=2, skip 16×52 readout, exact LM pullback grads
```

`tests.test_plumbing.PlumbingTests.test_dataparallel_two_step_or_skip` runs a 2-step DataParallel train if two CUDA devices are visible (preferring physical 2 and 3); otherwise it skips with a note.

Handcoded smoke is the short `python -m src executor …` command in the §7 section above.

---

## Outputs

| What | Path |
|---|---|
| Positioning / roadmap | `PAPER_POSITIONING.md`, `THEORY_EMPIRICAL_REVISION_ROADMAP.md` |
| Theorem ledger | `Paper/theorem_ledger.md`, `Paper/T2_CANDIDATE.md` |
| Revision dumps | `results/paper_revision_v2/` |
| Week-2 protocols | `results/paper_revision_v2/protocols/`, `configs/experiments/e{1,2,3,4}_*.yaml` |
| Reliability (E5) | `results/reliability_sweeps/`, `*_persist.json` |
| §7 depth replication | `results/executor_comparison/depth_replication/` |
| Fig 4 / 5 + Table 7 TeX | `Paper/figures/trained_executor_*.pdf`, `Paper/data/trained_executor_*.tex` |
| Architecture n=10 | `results/architecture_controls_n10/` |
| Optional pullback / induced | `results/revision/pullback.csv`, `induced_rule.csv` |
| Shared kernel (row-softmax) | `results/revision/shared_kernel.csv` |
| Projected kernel (not a T2 proof) | `results/paper_revision_v2/e4_projected_kernel/` |
| Outcome+local mechanism test | `results/paper_revision_v2/outcome_local/` |
| `Paper/` | often gitignored; local manuscript |

---

## Tests

```bash
TRACE_TQDM=0 python -m unittest tests.test_plumbing tests.test_handcoded tests.test_executor_comparison tests.test_revision_bridge tests.test_mechanism_tasks tests.test_local_machine_tasks tests.test_outcome_local
# or
TRACE_TQDM=0 python -m unittest discover -s tests -v
```

Claim map: [experiments/README.md](experiments/README.md). Roadmap: [THEORY_EMPIRICAL_REVISION_ROADMAP.md](THEORY_EMPIRICAL_REVISION_ROADMAP.md). Manuscript map: [Paper/EXPERIMENT_MAP.md](Paper/EXPERIMENT_MAP.md).

---

## Leftover GPU (after week-2 protocol freeze)

GPU 2/3 were packed (high util) during the freeze, so calibration/confirm grids were **not** started. When memory **and** util are free on physical 2/3, do **not** kill other jobs. Compile off. **Do not** launch 80-job confirm until `selected_rates.json` exists.

```bash
PY=/home/hariguru/aayus/.venv/bin/python
export CUDA_VISIBLE_DEVICES=2   # or 3; then --device cuda:0
# E2 calibration only (40 jobs). Stop if the cards fill:
$PY -m src architecture calibrate --config configs/experiments/e2_architecture.yaml \
  --output results/paper_revision_v2/e2_architecture --devices cuda:0 --no-compile

# After selected_rates.json only:
# $PY -m src architecture confirm --config configs/experiments/e2_architecture.yaml \
#   --output results/paper_revision_v2/e2_architecture --devices cuda:0 --no-compile

# E1 five-condition confirmation (not a Table 1 paste until complete):
# $PY -m src supervision --config configs/experiments/e1_five_condition.yaml \
#   --seeds 2001 2002 2003 2004 2005 --steps 8000 --train-size 20000 --val-size 1000 \
#   --batch-size 128 --device cuda:0 --no-compile \
#   --output results/paper_revision_v2/e1_five_condition/confirm.csv

# E3 confirmation readout (skip_readout false; one depth/seed at a time):
# $PY -m src induced --config configs/experiments/induced_rule.yaml --depth 8 \
#   --condition both --seed 2001 --with-pullback --device cuda:0 --no-compile \
#   --out results/paper_revision_v2/e3_mask_trace/confirm.csv
```

