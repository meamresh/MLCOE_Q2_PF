#!/bin/bash
# =============================================================================
# Part 1 — Run All Experiments
# =============================================================================
#
# This script runs the complete experimental pipeline for the project.
# Experiments are organized into three categories:
#
# Category 1: Linear Gaussian SSM (Baseline)
#   reports/1_LinearGaussianSSM/
#   ├── figures/
#   │   ├── lgssm_kf_fig.png
#   │   ├── lgssm_kf_results.csv
#   │   ├── kf_stability_states.png
#   │   └── kf_stability_metrics.png
#   └── stability_summary.csv
#
# Category 2: Nonlinear / Non-Gaussian SSM
#   reports/2_Nonlinear_NonGaussianSSM/
#   ├── EKF_UKF_PF_Comparison/         (runtime & memory profiling)
#   ├── EKF_UKF_Experiment/            (range-bearing EKF vs UKF)
#   ├── EKF_UKF_Tuning/                (parameter tuning)
#   ├── linearization_sigma_pt_failures/  (EKF/UKF failure modes)
#   └── particle_degeneracy/           (PF degeneracy analysis)
#
# Category 3: Deterministic Kernel Flow Filters
#   reports/3_Deterministic_Kernel_Flow/
#   ├── Hu(21)/                        (Hu & van Leeuwen particle flow)
#   ├── Li(17)/                        (multi-target acoustic tracking)
#   └── Filters_Comparison_Diagnostics/ (comprehensive filter comparison)
#
# Usage:
#   bash scripts/run_part1.sh
#
# =============================================================================

set -e  # Exit on error

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

# Add project to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=============================================="
echo "Running All Experiments"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Started at: $(date)"
echo ""

# =============================================================================
# Category 1: Linear Gaussian SSM (Baseline)
# =============================================================================
echo "=============================================="
echo "Category 1: Linear Gaussian SSM"
echo "=============================================="

echo ""
echo "[1/9] exp_part1_lgssm_kf.py"
echo "      Basic Kalman filter on LGSSM"
echo "      Output: reports/1_LinearGaussianSSM/figures/"
python -m src.experiments.exp_part1_lgssm_kf \
    --config configs/ssm_linear.yaml \
    --out_dir reports/1_LinearGaussianSSM/figures
echo "      Done."

echo ""
echo "[2/9] exp_part1_lgssm_kf_compare.py"
echo "      Riccati vs Joseph covariance update comparison"
echo "      Output: reports/1_LinearGaussianSSM/"
python -m src.experiments.exp_part1_lgssm_kf_compare
echo "      Done."

# =============================================================================
# Category 2: Nonlinear / Non-Gaussian SSM
# =============================================================================
echo ""
echo "================================================================================="
echo "Category 2: Nonlinear / Non-Gaussian SSM (These Runs Take a Long Time)"
echo "================================================================================="

echo ""
echo "[3/9] exp_range_bearing_ekf_ukf.py (experiment mode)"
echo "      EKF vs UKF on range-bearing localization"
echo "      Output: reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_Experiment/"
python -m src.experiments.exp_range_bearing_ekf_ukf \
    --mode experiment \
    --scenario strong \
    --num_steps 100
echo "      Done."

echo ""
echo "[4/9] exp_range_bearing_ekf_ukf.py (tuning mode)"
echo "      UKF parameter tuning (alpha, beta, kappa)"
echo "      Output: reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_Tuning/"
python -m src.experiments.exp_range_bearing_ekf_ukf \
    --mode tuning \
    --scenario strong \
    --num_steps 100
echo "      Done."

echo ""
echo "[5/9] exp_linearization_sigma_pt_failures.py"
echo "      EKF linearization and UKF sigma point failure analysis"
echo "      Output: reports/2_Nonlinear_NonGaussianSSM/linearization_sigma_pt_failures/"
python -m src.experiments.exp_linearization_sigma_pt_failures
echo "      Done."

echo ""
echo "[6/9] exp_particle_degeneracy.py"
echo "      Particle filter degeneracy diagnostics"
echo "      Output: reports/2_Nonlinear_NonGaussianSSM/particle_degeneracy/"
python -m src.experiments.exp_particle_degeneracy
echo "      Done."

echo ""
echo "[7/9] exp_runtime_memory.py"
echo "      Runtime and memory profiling (EKF, UKF, PF)"
echo "      Output: reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_PF_Comparison/"
python -m src.experiments.exp_runtime_memory \
    --comprehensive \
    --scaling \
    --num-steps 200 \
    --num-particles 10000 \
    --seed 42
echo "      Done."

# =============================================================================
# Category 3: Deterministic Kernel Flow Filters 
# =============================================================================
echo ""
echo "=============================================="
echo "Category 3: Deterministic Kernel Flow Filters"
echo "=============================================="

echo ""
echo "[8/9] exp_hu_vanleeuwen_fig2_fig3.py"
echo "      Hu & van Leeuwen (2021) particle flow figures"
echo "      Output: reports/3_Deterministic_Kernel_Flow/Hu(21)/"
python -m src.experiments.exp_hu_vanleeuwen_fig2_fig3
echo "      Done."

echo ""
echo "[9/9] exp_Li(17)_multitarget_acoustic.py"
echo "      Li (2017) multi-target acoustic tracking"
echo "      Output: reports/3_Deterministic_Kernel_Flow/Li(17)/"
python -m "src.experiments.exp_Li(17)_multitarget_acoustic" \
    --filters ekf ukf pf pfpf_ledh pfpf_edh ledh edh \
    --output_dir reports/3_Deterministic_Kernel_Flow/Li\(17\) \
    --n_trajectories 50 \
    --n_runs 5 \
    --n_steps 20 \
    --seed 42
echo "      Done."

echo ""
echo "[BONUS] exp_filters_comparison_diagnostics.py"
echo "        Comprehensive filter comparison with diagnostics"
echo "        Output: reports/3_Deterministic_Kernel_Flow/Filters_Comparison_Diagnostics/"
python -m src.experiments.exp_filters_comparison_diagnostics \
    --filters all \
    --scenarios all \
    --output_dir reports/3_Deterministic_Kernel_Flow/Filters_Comparison_Diagnostics \
    --seed 42
echo "        Done."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "All Experiments Complete"
echo "=============================================="
echo "Finished at: $(date)"
echo ""
echo "Output structure:"
echo "  reports/"
echo "  ├── 1_LinearGaussianSSM/"
echo "  │   ├── figures/                          [exp_part1_lgssm_kf]"
echo "  │   └── stability_summary.csv             [exp_part1_lgssm_kf_compare]"
echo "  ├── 2_Nonlinear_NonGaussianSSM/"
echo "  │   ├── EKF_UKF_Experiment/               [exp_range_bearing_ekf_ukf]"
echo "  │   ├── EKF_UKF_Tuning/                   [exp_range_bearing_ekf_ukf tuning]"
echo "  │   ├── EKF_UKF_PF_Comparison/            [exp_runtime_memory]"
echo "  │   ├── linearization_sigma_pt_failures/ [exp_linearization_sigma_pt_failures]"
echo "  │   └── particle_degeneracy/              [exp_particle_degeneracy]"
echo "  └── 3_Deterministic_Kernel_Flow/"
echo "      ├── Hu(21)/                           [exp_hu_vanleeuwen_fig2_fig3]"
echo "      ├── Li(17)/                           [exp_Li(17)_multitarget_acoustic]"
echo "      └── Filters_Comparison_Diagnostics/   [exp_filters_comparison_diagnostics]"
