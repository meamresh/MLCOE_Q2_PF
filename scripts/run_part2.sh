#!/bin/bash
# =============================================================================
# Part 2 — Run All Experiments
# =============================================================================
#
# This script runs the experimental pipeline for Part 2 of the project:
#
# Category 4: Stochastic Particle Flow
#   reports/4_Stochastic_Particle_Flow/
#   └── Dai_Daum/                          (SPF with homotopy & stiffness analysis)
#
# Category 5: Differentiable PF & OT Resampling
#   reports/5_Differential_PF_OT_Resampling/
#   ├── bias_variance_speed/               (Corenflos Table 1 + bias/variance tradeoff)
#   ├── bias_variance_speed/               (DPF bias-variance-speed tradeoff grid)
#   └── dpf_comparison/                    (DPF accuracy, differentiability, SNR)
#
# Usage:
#   bash scripts/run_part2.sh
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
echo "Part 2 — Running All Experiments"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Started at: $(date)"
echo ""

# =============================================================================
# Category 4: Stochastic Particle Flow
# =============================================================================
echo "=============================================="
echo "Category 4: Stochastic Particle Flow"
echo "=============================================="

echo ""
echo "[1/4] exp_part2_1a_spf_dai_daum.py"
echo "      Stochastic Particle Flow (Dai & Daum) with homotopy & stiffness"
echo "      Output: reports/4_Stochastic_Particle_Flow/Dai_Daum/"
python -m src.experiments.exp_part2_1a_spf_dai_daum
echo "      Done."

echo ""
echo "[2/5] exp_part1_2b_Li(17)_multitarget_acoustic.py (PFPF comparison)"
echo "      PFPF Dai-Daum vs PFPF-LEDH comparison"
echo "      Output: reports/4_Stochastic_Particle_Flow/pfpf_comparison/"
python -m "src.experiments.exp_part1_2b_Li(17)_multitarget_acoustic" \
    --n_trajectories 5 \
    --show_inner_progress \
    --output_dir 'reports/4_Stochastic_Particle_Flow/pfpf_comparison/' \
    --filters pfpf_dai_daum pfpf_ledh \
    --n_runs 1 \
    --n_steps 30 \
    --plot_std
echo "      Done."

# =============================================================================
# Category 5: Differentiable PF & OT Resampling
# =============================================================================
echo ""
echo "================================================================================="
echo "Category 5: Differentiable PF & OT Resampling (These Runs Take a Long Time)"
echo "================================================================================="

echo ""
echo "[3/5] exp_part2_2a_reproduce_corenflos_table1.py"
echo "      Reproduce Corenflos et al. Table 1 + bias/variance analysis"
echo "      Output: reports/5_Differential_PF_OT_Resampling/bias_variance_speed/"
python -m src.experiments.exp_part2_2a_reproduce_corenflos_table1 \
    --table1 \
    --bias_variance
echo "      Done."

echo ""
echo "[4/5] exp_part2_2b_dpf_bias_variance_speed_tradeoff.py"
echo "      DPF bias-variance-speed tradeoff grid"
echo "      Output: reports/5_Differential_PF_OT_Resampling/bias_variance_speed/"
python -m src.experiments.exp_part2_2b_dpf_bias_variance_speed_tradeoff \
    --table1 \
    --bias_variance
echo "      Done."

echo ""
echo "[5/5] exp_part2_2c_DPF_comparison.py"
echo "      DPF comparison: accuracy, differentiability, SNR"
echo "      Output: reports/5_Differential_PF_OT_Resampling/dpf_comparison/"
python -m src.experiments.exp_part2_2c_DPF_comparison
echo "      Done."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Part 2 — All Experiments Complete"
echo "=============================================="
echo "Finished at: $(date)"
echo ""
echo "Output structure:"
echo "  reports/"
echo "  ├── 4_Stochastic_Particle_Flow/"
echo "  │   ├── Dai_Daum/                           [exp_part2_1a_spf_dai_daum]"
echo "  │   └── pfpf_comparison/                    [PFPF Dai-Daum vs PFPF-LEDH]"
echo "  └── 5_Differential_PF_OT_Resampling/"
echo "      ├── bias_variance_speed/                [exp_part2_2a + exp_part2_2b]"
echo "      └── dpf_comparison/                     [exp_part2_2c_DPF_comparison]"
