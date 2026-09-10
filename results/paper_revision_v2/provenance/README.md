# Provenance (Table 1 / LoRA)

Headline numbers must not be cited from memory. Record command, YAML, seeds, and output path here before they headline the paper. **Week 2 does not regenerate Table 1.**

## Table 1 (five-condition GPT supervision)

- Command: `python -m src supervision --config configs/experiments/e1_five_condition.yaml`
- Confirmation protocol (do not launch from week 2): `confirmation:` block in that YAML; freeze copy `../protocols/e1_five_condition.yaml`
- Pilot output: `../e1_five_condition/` (not PDF Table 1)
- PDF Table 1 in `Paper/EXPERIMENT_MAP.md` is still a recovered manuscript aggregate; original logs were not located.

## LoRA (SmolLM2-135M, rank 8)

- YAML: `configs/experiments/lora_transfer.yaml`
- Command (when a `src.eval` LoRA CLI exists): `python -m src lora --config configs/experiments/lora_transfer.yaml`
- Output: `results/paper/smollm/`
- Do not download a 7B checkpoint on this box unless intended.

Seeds, YAML, and paths above are the provenance contract. Empty artifact dirs mean the confirmation run has not been executed.
