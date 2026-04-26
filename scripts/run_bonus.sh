#!/bin/bash
# =============================================================================
# Part 3 (Bonus) — Run All Experiments
# =============================================================================
#
# This script runs the experimental pipeline for the Bonus questions:
#
# Bonus Q1: HMC & Invertible Flows
#   reports/6_BonusQ1_HMC_Invertible_Flows/
#   ├── PFPF_LEDH/                         (PFPF-LEDH on Kitagawa model)
#   ├── HMC_vs_PMMH/                      (Standard HMC, L-HNN HMC, PMMH comparison)
#   │   ├── comparison/                    (traces, ACF, ESS/grad plots)
#   │   └── ablation/                      (Standard HMC + L-HNN ablation studies)
#   └── LHNN_HMC_vs_PMMH/                 (L-HNN specific results)
#
# Bonus Q2: Neural Optimal Transport
#   reports/7_BonusQ2_NeuralOT/
#   ├── question1/                         (Neural OT vs Sinkhorn comparison)
#   ├── question2/                         (Architecture ablation study)
#   ├── scaling/                           (Particle-count scaling study)
#   └── DeepONet/                          (Hyper-DeepONet neural operator)
#
# Bonus Q3: SSL Comparison
#   reports/8_BonusQ3_SSL_Comparison/       (PG vs DPF-HMC vs PMMH on SSL)
#
# Usage:
#   bash scripts/run_bonus.sh
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
echo "Part 3 (Bonus) — Running All Experiments"
echo "=============================================="
echo "Project root: $PROJECT_ROOT"
echo "Started at: $(date)"
echo ""

# =============================================================================
# Bonus Q1: HMC & Invertible Flows
# =============================================================================
echo "=============================================="
echo "Bonus Q1: HMC & Invertible Flows"
echo "=============================================="

echo ""
echo "[1/8] exp_part3_bonus1a_pfpf_ledh_kitagawa.py"
echo "      PFPF-LEDH filter on Kitagawa nonlinear model"
echo "      Output: reports/6_BonusQ1_HMC_Invertible_Flows/PFPF_LEDH/"
python -m src.experiments.exp_part3_bonus1a_pfpf_ledh_kitagawa
echo "      Done."

echo ""
echo "[2/8] exp_part3_bonus1b_hmc_vs_pmmh.py (comparison)"
echo "      Standard HMC vs L-HNN HMC vs PMMH — three-way sampler comparison"
echo "      Output: reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/"
python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh \
    --first_part
echo "      Done."

echo ""
echo "[3/8] exp_part3_bonus1b_hmc_vs_pmmh.py (Standard HMC ablation)"
echo "      Step-size and leapfrog-length sensitivity for Standard HMC"
echo "      Output: reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/ablation/"
python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh \
    --second_part \
    --standard
echo "      Done."

echo ""
echo "[4/8] exp_part3_bonus1b_hmc_vs_pmmh.py (L-HNN HMC ablation)"
echo "      L-HNN architecture and training-budget sensitivity with retraining"
echo "      Output: reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/ablation/"
python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh \
    --second_part \
    --hnn \
    --retrain_ablation
echo "      Done."

# =============================================================================
# Bonus Q2: Neural Optimal Transport
# =============================================================================
echo ""
echo "================================================================================="
echo "Bonus Q2: Neural Optimal Transport (These Runs Take a Long Time)"
echo "================================================================================="

echo ""
echo "[5/8] exp_part3_bonus2a_neural_ot.py"
echo "      Neural OT vs Sinkhorn comparison + architecture ablation"
echo "      Output: reports/7_BonusQ2_NeuralOT/question{1,2}/"
python -m src.experiments.exp_part3_bonus2a_neural_ot
echo "      Done."

echo ""
echo "[6/8] exp_part3_bonus2b_neural_ot_scaling.py"
echo "      Particle-count scaling: runtime and accuracy vs number of particles"
echo "      Output: reports/7_BonusQ2_NeuralOT/scaling/"
python -m src.experiments.exp_part3_bonus2b_neural_ot_scaling
echo "      Done."

echo ""
echo "[7/8] exp_part3_bonus2c_hyper_deeponet.py"
echo "      Hyper-DeepONet neural operator viability for OT resampling"
echo "      Output: reports/7_BonusQ2_NeuralOT/DeepONet/"
python -m src.experiments.exp_part3_bonus2c_hyper_deeponet
echo "      Done."

# =============================================================================
# Bonus Q3: SSL Comparison
# =============================================================================
echo ""
echo "=============================================="
echo "Bonus Q3: SSL Comparison"
echo "=============================================="

echo ""
echo "[8/8] exp_part3_bonus3_ssl_comparison.py"
echo "      Particle Gibbs vs DPF-HMC vs PMMH on Gaussian SSL"
echo "      Output: reports/8_BonusQ3_SSL_Comparison/"
python -m src.experiments.exp_part3_bonus3_ssl_comparison
echo "      Done."

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Part 3 (Bonus) — All Experiments Complete"
echo "=============================================="
echo "Finished at: $(date)"
echo ""
echo "Output structure:"
echo "  reports/"
echo "  ├── 6_BonusQ1_HMC_Invertible_Flows/"
echo "  │   ├── PFPF_LEDH/                         [exp_part3_bonus1a]"
echo "  │   └── HMC_vs_PMMH/"
echo "  │       ├── comparison/                     [--first_part]"
echo "  │       └── ablation/                       [--second_part --standard / --hnn]"
echo "  ├── 7_BonusQ2_NeuralOT/"
echo "  │   ├── question1/                          [exp_part3_bonus2a Q1]"
echo "  │   ├── question2/                          [exp_part3_bonus2a Q2]"
echo "  │   ├── scaling/                            [exp_part3_bonus2b]"
echo "  │   └── DeepONet/                           [exp_part3_bonus2c]"
echo "  └── 8_BonusQ3_SSL_Comparison/               [exp_part3_bonus3]"
