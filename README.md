# Particle Flow Filters & Differentiable Particle Filtering (DPF)

By **Amresh Verma**_

> This project implements and compares classical state-space filtering, particle filters, **particle flow** methods (EDH/LEDH, invertible PF-PF), **kernel-embedded particle flow** for higher dimensions, **differentiable particle filtering** with entropy-regularized optimal transport (Sinkhorn), **HMC-based inference** (standard & L-HNN accelerated), and **neural optimal transport** for learned resampling.

---

## Goals & Deliverables

### Part 1
- Literature review & rationale for method choices  
- Implement:
  - KF / EKF / UKF  
  - Particle Filter (ESS, resampling)  
  - EDH / LEDH particle flows  
  - Invertible PF-PF  
  - Kernel particle flow filter (scalar vs matrix kernels)

### Part 2
- Stochastic particle flows (stiffness mitigation)  
- Differentiable PF with **entropy-regularized OT (Sinkhorn)**  
- Soft resampling  
- Consolidated comparisons, gradient-stability analysis

### Part 3 (Bonus)
- **HMC & Invertible Flows**: Standard HMC, L-HNN accelerated HMC, PMMH comparison
- **Neural Optimal Transport**: Learned OT resampling via mGradNet, DeepONet, Hyper-DeepONet
- **SSL Comparison**: Particle Gibbs vs DPF-HMC vs PMMH on Gaussian state-space LSTM

---

## Repository Structure

```text
MLCOE_Q2_PF/
├── configs/                        # Model & experiment configs
├── scripts/
│   ├── run_part1.sh                # Part 1 experiments (9 tasks)
│   ├── run_part2.sh                # Part 2 experiments (5 tasks)
│   ├── run_bonus.sh                # Bonus experiments (8 tasks)
│   └── run_tests.sh                # Run all tests
├── src/
│   ├── data/                       # Synthetic data generators
│   ├── models/                     # SSM definitions (LGSSM, range-bearing, Lorenz-96,
│   │                               #   multi-target acoustic, Dai-Daum, Kitagawa, SSL)
│   ├── filters/                    # Filter implementations
│   │   ├── kalman / ekf / ukf      # Classical filters
│   │   ├── edh / ledh              # Exact & Local Daum-Huang flows
│   │   ├── pfpf_filter             # Particle Flow Particle Filter (LEDH/EDH)
│   │   ├── pff_kernel              # Kernel-embedded PFF
│   │   ├── spf_dai_daum            # Stochastic Particle Flow (Dai & Daum)
│   │   ├── dpf/                    # Differentiable PF (Sinkhorn OT, transformer)
│   │   └── bonus/                  # HMC, L-HNN, PMMH, Neural OT, SSL inference
│   ├── metrics/                    # RMSE, NEES, ESS, stability checks
│   ├── experiments/                # 19 experiment runners (exp_part{1,2,3}_*.py)
│   └── utils/                      # Linear algebra, logging, experiment helpers
├── tests/                          # 28 unit & integration tests (452 tests total)
└── reports/                        # Generated outputs
    ├── 1_LinearGaussianSSM/
    ├── 2_Nonlinear_NonGaussianSSM/
    ├── 3_Deterministic_Kernel_Flow/
    ├── 4_Stochastic_Particle_Flow/
    ├── 5_Differential_PF_OT_Resampling/
    ├── 6_BonusQ1_HMC_Invertible_Flows/
    ├── 7_BonusQ2_NeuralOT/
    └── 8_BonusQ3_SSL_Comparison/
```

---

## Quickstart

### Run Tests

```bash
bash scripts/run_tests.sh           # Run all tests
bash scripts/run_tests.sh -v        # Verbose output
bash scripts/run_tests.sh -f        # Stop on first failure
```

### Run All Experiments

```bash
bash scripts/run_part1.sh           # Part 1: KF/EKF/UKF/PF/EDH/LEDH/PFPF (9 tasks)
bash scripts/run_part2.sh           # Part 2: SPF, DPF, Sinkhorn OT (5 tasks)
bash scripts/run_bonus.sh           # Bonus: HMC, Neural OT, SSL comparison (8 tasks)
```

### Run Individual Experiments

```bash
# Linear Gaussian SSM baseline
python -m src.experiments.exp_part1_1a_lgssm_kf --config configs/ssm_linear.yaml

# EKF vs UKF on range-bearing
python -m src.experiments.exp_part1_1c_range_bearing_ekf_ukf --mode experiment

# Particle flow (Hu & van Leeuwen figures)
python -m src.experiments.exp_part1_2a_hu_vanleeuwen_fig2_fig3

# Full filter comparison with diagnostics
python -m src.experiments.exp_part1_2c_filters_comparison_diagnostics --filters all

# Stochastic Particle Flow (Dai & Daum)
python -m src.experiments.exp_part2_1a_spf_dai_daum

# DPF comparison
python -m src.experiments.exp_part2_2c_DPF_comparison

# HMC vs PMMH comparison
python -m src.experiments.exp_part3_bonus1b_hmc_vs_pmmh --first_part

# Neural OT vs Sinkhorn
python -m src.experiments.exp_part3_bonus2a_neural_ot

# SSL comparison (PG vs DPF-HMC vs PMMH)
python -m src.experiments.exp_part3_bonus3_ssl_comparison
```

### Run Individual Tests

```bash
# Specific test file
python -m unittest tests.test_filters -v

# Specific test class
python -m unittest tests.test_advanced_filters.TestEDH -v
```

---

## Experiments & Outputs

### Part 1

| Experiment | Description | Output Directory |
|------------|-------------|------------------|
| `exp_part1_1a_lgssm_kf.py` | Baseline KF on LGSSM | `reports/1_LinearGaussianSSM/figures/` |
| `exp_part1_1b_lgssm_kf_compare.py` | Riccati vs Joseph stability | `reports/1_LinearGaussianSSM/` |
| `exp_part1_1c_range_bearing_ekf_ukf.py` | EKF vs UKF comparison | `reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_*/` |
| `exp_part1_1d_linearization_sigma_pt_failures.py` | EKF/UKF failure modes | `reports/2_Nonlinear_NonGaussianSSM/linearization_sigma_pt_failures/` |
| `exp_part1_1e_particle_degeneracy.py` | PF degeneracy diagnostics | `reports/2_Nonlinear_NonGaussianSSM/particle_degeneracy/` |
| `exp_part1_1f_runtime_memory.py` | Runtime & memory profiling | `reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_PF_Comparison/` |
| `exp_part1_2a_hu_vanleeuwen_fig2_fig3.py` | Hu & van Leeuwen (2021) figures | `reports/3_Deterministic_Kernel_Flow/Hu(21)/` |
| `exp_part1_2b_Li(17)_multitarget_acoustic.py` | Li (2017) multi-target tracking | `reports/3_Deterministic_Kernel_Flow/Li(17)/` |
| `exp_part1_2c_filters_comparison_diagnostics.py` | Comprehensive filter comparison | `reports/3_Deterministic_Kernel_Flow/Filters_Comparison_Diagnostics/` |

### Part 2

| Experiment | Description | Output Directory |
|------------|-------------|------------------|
| `exp_part2_1a_spf_dai_daum.py` | Stochastic Particle Flow (Dai & Daum) | `reports/4_Stochastic_Particle_Flow/Dai_Daum/` |
| `exp_part1_2b_Li(17)_multitarget_acoustic.py` | PFPF Dai-Daum vs PFPF-LEDH comparison | `reports/4_Stochastic_Particle_Flow/pfpf_comparison/` |
| `exp_part2_2a_reproduce_corenflos_table1.py` | Reproduce Corenflos et al. Table 1 | `reports/5_Differential_PF_OT_Resampling/bias_variance_speed/` |
| `exp_part2_2b_dpf_bias_variance_speed_tradeoff.py` | DPF bias-variance-speed tradeoff grid | `reports/5_Differential_PF_OT_Resampling/bias_variance_speed/` |
| `exp_part2_2c_DPF_comparison.py` | DPF accuracy, differentiability, SNR | `reports/5_Differential_PF_OT_Resampling/dpf_comparison/` |

### Part 3 (Bonus)

| Experiment | Description | Output Directory |
|------------|-------------|------------------|
| `exp_part3_bonus1a_pfpf_ledh_kitagawa.py` | PFPF-LEDH on Kitagawa model | `reports/6_BonusQ1_HMC_Invertible_Flows/PFPF_LEDH/` |
| `exp_part3_bonus1b_hmc_vs_pmmh.py --first_part` | Standard HMC vs L-HNN HMC vs PMMH comparison | `reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/` |
| `exp_part3_bonus1b_hmc_vs_pmmh.py --second_part` | HMC / L-HNN ablation studies | `reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/ablation/` |
| `exp_part3_bonus2a_neural_ot.py` | Neural OT vs Sinkhorn + ablation | `reports/7_BonusQ2_NeuralOT/question{1,2}/` |
| `exp_part3_bonus2b_neural_ot_scaling.py` | Particle-count scaling study | `reports/7_BonusQ2_NeuralOT/scaling/` |
| `exp_part3_bonus2c_hyper_deeponet.py` | Hyper-DeepONet neural operator viability | `reports/7_BonusQ2_NeuralOT/DeepONet/` |
| `exp_part3_bonus3_ssl_comparison.py` | PG vs DPF-HMC vs PMMH on Gaussian SSL | `reports/8_BonusQ3_SSL_Comparison/` |

---

## Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/meamresh/MLCOE_Q2_PF.git
cd MLCOE_Q2_PF/

# Install dependencies
pip install -r requirements.txt

# Or install as a package (editable mode)
pip install -e .
```

### Development Install
```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linting
black src/ tests/
isort src/ tests/
flake8 src/ tests/
```

### GPU Support (Mac)
```bash
pip install tensorflow-metal
```

## Requirements

- **Python >= 3.9**
- **TensorFlow >= 2.16** (includes NumPy as dependency)
- **TensorFlow Probability >= 0.24**
- **Matplotlib, tqdm, PyYAML**

> **Note:** All core models and filtering algorithms are implemented exclusively in TensorFlow and TensorFlow Probability. NumPy is used selectively in experiments and tests for analysis, visualization, and validation, and is not used in any core computational paths.

GPU is optional; CPU runs are sufficient for all experiments.

---

## Key References

* **PF & SSM fundamentals**  
  Doucet & Johansen, *A tutorial on particle filtering and smoothing*

* **Exact / Local particle flows**  
  Daum & Huang (2010, 2011)

* **Invertible PF-PF**  
  Li & Coates (2017)

* **Kernel-embedded PFF (high-dim)**  
  Hu & van Leeuwen (2021)

* **Stochastic particle flows (stiffness)**  
  Dai & Daum (2022)

* **Differentiable PF via OT (Sinkhorn)**  
  Corenflos et al., ICML 2021

* **HMC & MCMC for particle filters**  
  Neal (2011), Andrieu et al. (2010)

* **Neural OT / monotone networks**  
  Kovachki et al. (2023), Jha et al. (2025)

* **LSTM / Gibbs**  
  Zheng et al. (2025)

---

## Reproducibility

* Fixed **random seeds** (configurable via `--seed`)
* All experiments use **TensorFlow only** (no NumPy in core computations)
* Deterministic CPU mode via `tf.config.experimental.enable_op_determinism()`
* Logged **configs** per run
* All figures generated via scripted runners

---

## Continuous Integration

GitHub Actions runs on every push/PR:
- **Tests** with coverage on Ubuntu and macOS (Python 3.9, 3.10, 3.11)
- **Linting** with flake8, black, isort

See `.github/workflows/ci.yml` for details.

---

## Contact

**Amresh Verma**  
amreshverma702@gmail.com

Feel free to open issues or PRs for bugs, clarifications, or reproducibility notes.
