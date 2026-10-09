#!/usr/bin/env bash
# Lean-repo cleanup. Review, then run:  bash Paper/build/cleanup.sh
# Gitignored items (results/, logs/, ckpt/, run.log) are NOT recoverable after this.
# Kept: handcoded/{lettertrace,regime_both,gate_steering}.py, handcoded/theory_checks/ (local only),
#       src/, configs/, data/bbh/, Paper/ sources, LoRA adapters under results/paper/{qwen05,smollm}/artifacts*.
set -euo pipefail
shopt -s nullglob dotglob
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

del() {  # untrack (if tracked) and remove
  for p in "$@"; do
    git rm -r -q --cached --ignore-unmatch -- "$p" >/dev/null
    rm -rf -- "$p"
  done
}

echo "before: $(du -sh --exclude=.venv . | cut -f1)"

# --- handcoded/: keep the three paper scripts and the local theory checks -----
for p in handcoded/*; do
  case "$(basename "$p")" in
    lettertrace.py|regime_both.py|gate_steering.py|theory_checks) ;;
    *) del "$p" ;;
  esac
done

# --- now gitignored: stop tracking, keep the files on disk ---------------------
git rm -r -q --cached --ignore-unmatch -- Paper/build handcoded/theory_checks >/dev/null

# --- root, docs, paper leftovers ----------------------------------------------
del run.sh run.log .ruff_cache docs Paper/build_end_training Paper/tools
del Paper/figures/credit_locality.* Paper/figures/executors.* Paper/figures/lambda_grid.* \
    Paper/figures/overlap_spectra.* Paper/figures/regime_both.*
del configs/experiments/e1_five_condition.yaml

# --- results/ (local): studies not in the paper --------------------------------
cd results
del checkpoint_competence_trajectory clean_convergence credit_suppression credit_transmission_probe \
    fraction_vs_amount gate_steering_gf gradient_alignment gradient_alignment_archived_3seed \
    gradient_alignment_by_group handcoded_circuit_match handcoded_circuit_match_coherent_n20k \
    handcoded_circuit_match_n20k handcoded_circuit_match_probe_smoke handcoded_circuit_match_symmetric_n20k \
    identifiability_probe landscape landscape_init.log landscape_islands.log lookup_convergence \
    my_reliability_sweep* noise_threshold notebook novelty_extension prop8_frontier radius \
    reliability_sweeps review_validation route state_machine_16_phase_* state_sweep* \
    zero_init_softmax_table zero_init_softmax_table_random_init shuffled/pilot shuffled/spectrum.json
del step_reach/repro*.json step_reach/routing_* step_reach/scaled_* step_reach/sigma* \
    step_reach/sweep.sh step_reach/sweep_K*.json step_reach/twoblock_smoke.json
del basin/start_* basin/align.json
cd paper
del ARTIFACT_MANIFEST.md answer_ablation basin_certificate basin_certificate_adamw basin_certificate_trainset \
    boolean_reliability_* circuit_match_gradient_polarity concentration_* depth_reliability \
    depth_reliability_lowrho gate_count gpu2_progress.log gpu3_progress.log gradient_polarity_threshold \
    higher_order_credit_checks.json lambda_sweep letter_block_results.md matched_corruption probes \
    reduced_escape register_agreement.csv register_models row_coverage.json tclean unified_depth_* \
    wr_intervention zero_init_softmax_table qwen05/coherent smollm/coherent \
    exact_spectra/everyk2_* exact_spectra/plateau_seeds
cd ../..

# --- logs/ (local): keep only logs that are the sole record of paper numbers ---
cd logs
del finish qwen05 smollm_long score8 shuffled_bbh shuffled_pilot slip slip_rho05 lambda_sweep \
    wr_intervention fig2 bs_*.log bl_*.log bm_*.log basin_success.log gsgf_*.log \
    route_n*.log routing_*.log step_reach_g*.log *queue*.log qwen05_retry.log
del must_run/rho8k_* must_run/rho32k_* must_run/clean32k_* must_run/wrong_K3_* \
    must_run/oom_rerun.log must_run/queue.log must_run/CONTROLS_DONE
del remaining/alpha_* remaining/everyk_* remaining/lights* remaining/opt* remaining/shell_* \
    remaining/first_cup_* remaining/flip remaining/queue.log
cd ..

# --- ckpt/ (local): keep only n8_block_s42 --------------------------------------
del ckpt/n8 ckpt/n8_long ckpt/n8_block ckpt/n8_block_long ckpt/n8_block_k2 \
    ckpt/n8_block_s43 ckpt/n8_block_s44 ckpt/rho05_*

echo "after:  $(du -sh --exclude=.venv . | cut -f1)"
git status --short | head -100
