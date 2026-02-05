# Particle Flow Filters & Differentiable Particle Filtering (DPF)

_Work-in-progress repo by **Amresh Verma**_

> This project implements and compares classical state-space filtering, particle filters, **particle flow** methods (EDH/LEDH, invertible PF-PF), **kernel-embedded particle flow** for higher dimensions, and **differentiable particle filtering** with entropy-regularized optimal transport (Sinkhorn).  

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

---

## Repository Structure

```text
MLCOE_Q2_PF/
├── README.md
├── pyproject.toml                  # Modern Python packaging config
├── requirements.txt                # Dependencies
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD
├── configs/                        # Model & experiment configs
│   └── ssm_linear.yaml
├── scripts/                        # Shell scripts
│   ├── run_part1.sh                # Run all experiments
│   └── run_tests.sh                # Run all tests
├── src/
│   ├── data/
│   │   └── generators.py           # Synthetic data generators (LGSSM from YAML)
│   ├── models/                     # State-space model definitions
│   │   ├── ssm_lgssm.py            # Linear Gaussian SSM
│   │   ├── ssm_range_bearing.py    # Range-bearing localization
│   │   ├── ssm_lorenz96.py         # Lorenz-96 chaotic system
│   │   └── ssm_multi_target_acoustic.py  # Multi-target acoustic tracking
│   ├── filters/                    # Filter implementations
│   │   ├── base.py                 # Abstract base classes (BaseFilter, BaseParticleFilter)
│   │   ├── kalman.py               # Kalman Filter (Riccati & Joseph)
│   │   ├── ekf.py                  # Extended Kalman Filter
│   │   ├── ukf.py                  # Unscented Kalman Filter
│   │   ├── particle_filter.py      # Bootstrap Particle Filter
│   │   ├── resampling.py           # Systematic, multinomial, stratified, residual
│   │   ├── edh.py                  # Exact Daum-Huang flow
│   │   ├── ledh.py                 # Localized EDH flow
│   │   ├── pfpf_filter.py          # Particle Flow Particle Filter (PFPF)
│   │   └── pff_kernel.py           # Kernel-embedded PFF (scalar & matrix)
│   ├── metrics/                    # Evaluation metrics
│   │   ├── accuracy.py             # RMSE, MAE, NEES, NLL
│   │   ├── stability.py            # Condition numbers, symmetry, PD checks
│   │   └── particle_filter_metrics.py  # ESS, weight statistics
│   ├── experiments/                # Experiment runners
│   │   ├── exp_part1_lgssm_kf.py
│   │   ├── exp_part1_lgssm_kf_compare.py
│   │   ├── exp_range_bearing_ekf_ukf.py
│   │   ├── exp_linearization_sigma_pt_failures.py
│   │   ├── exp_particle_degeneracy.py
│   │   ├── exp_runtime_memory.py
│   │   ├── exp_hu_vanleeuwen_fig2_fig3.py
│   │   ├── exp_Li(17)_multitarget_acoustic.py
│   │   └── exp_filters_comparison_diagnostics.py
│   └── utils/
│       ├── linalg.py               # TF linear algebra utilities
│       ├── logging_config.py       # Centralized logging configuration
│       └── experiment.py           # Experiment helpers
├── tests/                          # Unit & integration tests
│   ├── test_filters.py
│   ├── test_advanced_filters.py
│   ├── test_pfpf_components.py     # Granular PFPF component tests
│   ├── test_ukf_components.py      # Granular UKF component tests
│   ├── test_edh_components.py      # Granular EDH/LEDH component tests
│   ├── test_models.py
│   ├── test_models_advanced.py
│   ├── test_resampling.py
│   ├── test_linalg.py
│   ├── test_metrics.py
│   ├── test_full_pipeline_integration.py
│   └── ...
└── reports/                        # Generated outputs
    ├── 1_LinearGaussianSSM/
    ├── 2_Nonlinear_NonGaussianSSM/
    ├── 3_Deterministic_Kernel_Flow/
    └── Report_I_AmreshVerma.pdf    # First Report
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
bash scripts/run_part1.sh
```

This runs all 9 experiments and generates outputs in `reports/`.

### Run Individual Experiments

```bash
# Linear Gaussian SSM baseline
python -m src.experiments.exp_part1_lgssm_kf --config configs/ssm_linear.yaml

# EKF vs UKF on range-bearing
python -m src.experiments.exp_range_bearing_ekf_ukf --mode experiment

# Particle flow (Hu & van Leeuwen figures)
python -m src.experiments.exp_hu_vanleeuwen_fig2_fig3

# Full filter comparison with diagnostics
python -m src.experiments.exp_filters_comparison_diagnostics --filters all
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

| Experiment | Description | Output Directory |
|------------|-------------|------------------|
| `exp_part1_lgssm_kf.py` | Baseline KF on LGSSM | `reports/1_LinearGaussianSSM/figures/` |
| `exp_part1_lgssm_kf_compare.py` | Riccati vs Joseph stability | `reports/1_LinearGaussianSSM/` |
| `exp_range_bearing_ekf_ukf.py` | EKF vs UKF comparison | `reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_*/` |
| `exp_linearization_sigma_pt_failures.py` | EKF/UKF failure modes | `reports/2_Nonlinear_NonGaussianSSM/linearization_sigma_pt_failures/` |
| `exp_particle_degeneracy.py` | PF degeneracy diagnostics | `reports/2_Nonlinear_NonGaussianSSM/particle_degeneracy/` |
| `exp_runtime_memory.py` | Runtime & memory profiling | `reports/2_Nonlinear_NonGaussianSSM/EKF_UKF_PF_Comparison/` |
| `exp_hu_vanleeuwen_fig2_fig3.py` | Hu & van Leeuwen (2021) figures | `reports/3_Deterministic_Kernel_Flow/Hu(21)/` |
| `exp_Li(17)_multitarget_acoustic.py` | Li (2017) multi-target tracking | `reports/3_Deterministic_Kernel_Flow/Li(17)/` |
| `exp_filters_comparison_diagnostics.py` | Comprehensive filter comparison | `reports/3_Deterministic_Kernel_Flow/Filters_Comparison_Diagnostics/` |

---

## Installation

### Quick Install
```bash
# Clone the repository
git clone https://github.com/amreshverma/particle-flow-filters.git
cd particle-flow-filters

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

> **Note:** All core computations use TensorFlow only (no NumPy/SciPy in filter implementations).

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

---

## Reproducibility

* Fixed **random seeds** (configurable via `--seed`)
* All experiments use **TensorFlow only** (no NumPy in core computations)
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
