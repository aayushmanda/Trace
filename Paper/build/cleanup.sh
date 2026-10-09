#!/usr/bin/env bash
# Repo cleanup: delete artifacts and code not used by, or inconsistent with, the current paper.
# Review, then run from anywhere:  bash Paper/build/cleanup.sh
# Gitignored items (results/, logs/, ckpt/, run.log) are NOT recoverable after this.
# Kept on purpose: LoRA adapters (results/paper/qwen05/artifacts, results/paper/smollm/artifacts*).
set -euo pipefail
shopt -s nullglob
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"

del() {  # untrack (if tracked) and remove
  for p in "$@"; do
    git rm -r -q --cached --ignore-unmatch -- "$p" >/dev/null
    rm -rf -- "$p"
  done
}

echo "before: $(du -sh . | cut -f1)"

# --- results/: studies not in the paper ---------------------------------------
cd results
del checkpoint_competence_trajectory clean_convergence credit_suppression credit_transmission_probe \
    fraction_vs_amount gate_steering gate_steering_gf gradient_alignment gradient_alignment_archived_3seed \
    gradient_alignment_by_group handcoded_circuit_match handcoded_circuit_match_coherent_n20k \
    handcoded_circuit_match_n20k handcoded_circuit_match_probe_smoke handcoded_circuit_match_symmetric_n20k \
    identifiability_probe landscape landscape_init.log landscape_islands.log lookup_convergence \
    my_reliability_sweep* noise_threshold notebook novelty_extension prop8_frontier radius \
    reliability_sweeps review_validation route state_machine_16_phase_* state_sweep* \
    zero_init_softmax_table zero_init_softmax_table_random_init
del shuffled/pilot shuffled/spectrum.json
# step_reach: keep only the two-block runs (tab:reach)
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

# --- logs/: keep only logs that are the sole record of paper numbers -----------
# kept: must_run/{tab3_*,tab4_*,tab11_*,out8k_K3*,out32k_K3*,zeroreadout_*,run_controls.sh},
#       remaining/depth_L*, n16_K3, handcoded_large, step_reach_outcome_*.log
cd logs
del finish qwen05 smollm_long score8 shuffled_bbh shuffled_pilot slip slip_rho05 lambda_sweep \
    wr_intervention fig2 bs_*.log bl_*.log bm_*.log basin_success.log gate_steering_*.log gsgf_*.log \
    route_n*.log routing_*.log step_reach_g*.log *queue*.log qwen05_retry.log
del must_run/rho8k_* must_run/rho32k_* must_run/clean32k_* must_run/wrong_K3_* \
    must_run/oom_rerun.log must_run/queue.log must_run/CONTROLS_DONE
del remaining/alpha_* remaining/everyk_* remaining/lights* remaining/opt* remaining/shell_* \
    remaining/first_cup_* remaining/flip remaining/queue.log
cd ..

# --- ckpt/: keep only n8_block_s42 (tab:main-evidence) -------------------------
del ckpt/n8 ckpt/n8_long ckpt/n8_block ckpt/n8_block_long ckpt/n8_block_k2 \
    ckpt/n8_block_s43 ckpt/n8_block_s44 ckpt/rho05_*

# --- root, docs, paper leftovers ----------------------------------------------
del run.log .ruff_cache docs/assets Paper/build_end_training Paper/tools
del Paper/figures/credit_locality.* Paper/figures/executors.* Paper/figures/lambda_grid.* \
    Paper/figures/overlap_spectra.* Paper/figures/regime_both.*
del configs/experiments/e1_five_condition.yaml

# --- handcoded/: scripts with no result in the paper, or superseded -----------
cd handcoded
del basin_certificate.py certificate_adamw.py certificate_report.py certificate_trainset.py curvature.py \
    exactness_radius.py program_radius.py radius_figure.py executor_diagram.py figures.py \
    gate_steering.py gate_steering_report.py lambda_grid_figure.py lambda_sweep.py landscape.py \
    landscape_figure.py lookup_convergence.py one_layer_degrees.py one_layer_gradients.py \
    one_layer_spectrum.py probes.py progress_check.py regime_figure.py route_theorem_sim.py \
    routing_sim.py routing_horizon_check.py routing_stages_check.py rule_alignment.py size_delims.py \
    step_reach.py step_reach_report.py step_trap.py tabular_check.py theorem_regime.py \
    theory_checks.py two_block_spectrum.py update_alignment.py \
    .appendix_backup.tex .body_backup.tex __pycache__ \
    animations/matrices animations/lettertrace_curves.* \
    animations/lettertrace_train.mp4 animations/lettertrace_train.pdf animations/lettertrace_train.png
cd ..

echo "after:  $(du -sh . | cut -f1)"
git status --short | head -80
