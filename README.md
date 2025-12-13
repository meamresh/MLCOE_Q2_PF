```markdown

**Particle Flow Filters & Differentiable Particle Filtering (DPF)**  
_Work-in-progress repo by **Amresh Verma**_

> This project implements and compares classical state-space filtering, particle filters, **particle flow** methods (EDH/LEDH, invertible PF‑PF), **kernel‑embedded particle flow** for higher dimensions, and **differentiable particle filtering** with entropy‑regularized optimal transport (Sinkhorn). The work targets the internship assignment’s **Part 1 (filters & flows)** and **Part 2 (stochastic flows & DPF)** deliverables.

---

## 🎯 Goals & Deliverables
- **Part 1**  
  - Literature review & rationale for method choices  
  - Implement: KF/EKF/UKF, PF (with ESS/resampling), EDH/LEDH flows, invertible PF‑PF, kernel PFF (scalar vs matrix kernels)  
  - Clear answers to assignment items, **testing plans & results** (accuracy, ESS, runtime/memory, stability)
- **Part 2**  
  - Stochastic particle flows (stiffness mitigation)  
  - Differentiable PF with **entropy‑regularized OT (Sinkhorn)** and **soft resampling**  
  - Consolidated comparisons, gradient‑stability analysis, and final report

---

## 🗂️ Repository Structure (initial)
```

mlcoe-q2/
├─ README.md
├─ environment.yml
├─ configs/                  # model & experiment configs (seeded)
├─ src/
│  ├─ data/                  # synthetic generators (LGSSM, SV, range–bearing, mini L96)
│  ├─ models/                # SSM definitions
│  ├─ filters/               # KF/EKF/UKF/PF, resampling, EDH/LEDH, PF-PF, kernel PFF
│  ├─ dpf/                   # soft-resampling & OT (Sinkhorn) modules
│  ├─ metrics/               # RMSE/NLL, ESS, runtime/memory, stability/grad diagnostics
│  ├─ experiments/           # runners for Part 1 & Part 2
│  └─ utils/                 # seeding, logging, plotting
├─ tests/                    # unit & integration tests
└─ reports/
├─ part1/                 # short report (PDF) + figures
└─ final/                 # final report (PDF) + figures

````

> Uses Python ≥3.10, TensorFlow, TensorFlow Probability, NumPy/SciPy, and Matplotlib. GPU is optional; CPU runs are sufficient for the baseline experiments.

***

## 🚀 Quickstart (placeholders)

```bash
# Part 1 — baseline & flows
bash scripts/run_part1.sh

# Part 2 — differentiable PF (soft → OT/Sinkhorn)
bash scripts/run_part2.sh
```

Each script prints metrics and writes plots to `reports/*/figures/`.  
Config files (e.g., `configs/ssm_sv.yaml`) control seeds, noise levels, particle counts, and flow/OT hyperparameters.


***

## 📚 Key References

*   **PF & SSM fundamentals**: A. Doucet & A. Johansen, *A tutorial on particle filtering and smoothing*.
*   **Exact/Local particle flows**: F. Daum & J. Huang (2010, 2011), *Exact particle flow for nonlinear filters*; *Particle degeneracy: root cause and solution*.
*   **Invertible particle flow PF‑PF**: Y. Li & M. Coates (2017), *Particle filtering with invertible particle flow*.
*   **Kernel‑embedded PFF (high‑dim)**: C.-C. Hu & P. J. van Leeuwen (2021), *A particle flow filter for high‑dimensional system applications*.
*   **Stochastic particle flows (stiffness)**: L. Dai & F. Daum (2022), *Stiffness mitigation in stochastic particle flow filters*.
*   **Differentiable PF via OT (Sinkhorn)**: A. Corenflos et al. (ICML 2021), *Differentiable particle filtering via entropy‑regularized optimal transport*.
*   **PMCMC baseline (optional)**: C. Andrieu, A. Doucet, R. Holenstein (2010), *Particle Markov chain Monte Carlo methods*.

> These align with the assignment’s reference list provided by the MLCOE TSRL team.

***

## 🔒 Reproducibility

*   Fixed **random seeds**, version‑pinned `environment.yml`, and logged **configs** per run
*   All figures are generated via scripted runners (`scripts/run_part1.sh`, `scripts/run_part2.sh`)

***

## 🙋‍♂️ Contact

**Amresh Verma** · `amreshverma702@gmail.com` 
Feel free to open issues or PRs for bugs, clarifications, or reproducibility notes.

```
```
