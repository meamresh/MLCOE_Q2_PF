# Particle Flow Filters & Differentiable Particle Filtering (DPF)

_Work-in-progress repo by **Amresh Verma**_

> This project implements and compares classical state-space filtering, particle filters, **particle flow** methods (EDH/LEDH, invertible PF-PF), **kernel-embedded particle flow** for higher dimensions, and **differentiable particle filtering** with entropy-regularized optimal transport (Sinkhorn).  

---

## 🎯 Goals & Deliverables

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

## 🗂️ Repository Structure (initial)

```text
mlcoe-q2/
├─ README.md
├─ environment.yml
├─ configs/                  # model & experiment configs (seeded)
├─ src/
│  ├─ data/                  # synthetic generators (LGSSM, SV, range–bearing, mini L96)
│  ├─ models/                # SSM definitions
│  ├─ filters/               # KF/EKF/UKF/PF, resampling, EDH/LEDH, PF-PF, kernel PFF
│  ├─ dpf/                   # soft-resampling & OT (Sinkhorn)
│  ├─ metrics/               # RMSE/NLL, ESS, runtime/memory, stability diagnostics
│  ├─ experiments/           # runners for Part 1 & Part 2
│  └─ utils/                 # seeding, logging, plotting
├─ tests/                    # unit & integration tests
└─ reports/
   ├─ part1/                 # short report (PDF) + figures
   └─ final/                 # final report (PDF) + figures
```

> Uses **Python ≥3.10**, TensorFlow, TensorFlow Probability, and Matplotlib.
> GPU is optional; CPU runs are sufficient for baseline experiments.

---

## 🚀 Quickstart (placeholders)

```bash
# Part 1 — baseline & flows
bash scripts/run_part1.sh

# Part 2 — differentiable PF (soft → OT/Sinkhorn)
bash scripts/run_part2.sh
```

Each script prints metrics and writes plots to `reports/*/figures/`.

Configuration files (e.g. `configs/ssm_sv.yaml`) control seeds, noise levels, particle counts, and flow / OT hyperparameters.

---

## 📚 Key References

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

* **PMCMC baseline (optional)**
  Andrieu, Doucet & Holenstein (2010)


---

## 🔒 Reproducibility

* Fixed **random seeds**
* Version-pinned `environment.yml`
* Logged **configs** per run
* All figures generated via scripted runners

---

## 🙋‍♂️ Contact

**Amresh Verma**
📧 `amreshverma702@gmail.com`

Feel free to open issues or PRs for bugs, clarifications, or reproducibility notes.