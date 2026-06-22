# Results Traceability for `interview_answers.tex`

Every reported number, table entry, and experimental outcome in
`interview_answers.tex` mapped to the code, configuration, data directory,
and command that produces it.

- **Repository root:** `/Users/amreshverma/Documents/Random Work/MLCOE_Q2_PF`
- **Reports root (`$R`):** `/Users/amreshverma/Documents/Random Work/MLCOE_Q2_PF/reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH`
- **Run prelude for every command below:**
  ```bash
  cd "/Users/amreshverma/Documents/Random Work/MLCOE_Q2_PF"
  export PYTHONPATH="$PWD:${PYTHONPATH:-}"
  ```
- **Environment of record:** TensorFlow 2.16.2, TFP 0.24.0, CPU. Absolute
  wall-times are platform-dependent; ratios and posterior summaries are not.
- **Determinism:** all particle-filter calls use fixed common-random-number
  seeds (`tf.random.set_seed`), so log-likelihoods and gradients reproduce
  up to XLA/platform numerics. HMC posteriors reproduce given the same
  `--data_seed` / `--base_seed`.

---

# Code Index

## Core filter / model modules (`src/filters/`)

| Path (absolute) | Role |
|---|---|
| `src/filters/bonus/extra_bonus/differentiable_ledh_svssm.py` | **Univariate SVSSM DPF likelihood** `DifferentiableLEDHLogLikelihoodSVSSM` — the 1-D HMC target. LEDH particle-flow + Sinkhorn OT + HRS quasi-likelihood. JIT kernel `_timestep_1d_xla`. (§2, §3, §4) |
| `src/filters/bonus/extra_bonus/differentiable_ledh_svssm_multivariate.py` | **Multivariate SVSSM DPF** `DifferentiableLEDHLogLikelihoodSVSSMmulti` — diagonal-Φ XLA path (`_jit_step`), full-Σ_η path `call_full`, matrix-Φ path `call_mat_phi`. (§2 d=2, §3 d=2, §8) |
| `src/filters/bonus/extra_bonus/differentiable_ledh_neural_ot_svssm.py` | SVSSM filter with **NN-OT** resample in place of Sinkhorn; `build_svssm_context_scalars` (the 7-D θ-context). (§3, §9, §10) |
| `src/filters/bonus/extra_bonus/svssm_neural_ot_training.py` | **Operator training** harness (supervised MSE to Sinkhorn map, plateau early-stop, best-weight restore). (§10) |
| `src/filters/bonus/deeponet_ot.py` | **`DeepONetMonotoneOT`** branch–trunk operator: softplus-gated affine + ridge terms ⇒ PSD Jacobian; `log_det_jacobian` (diag-plus-rank-K). (§9, §10) |
| `src/filters/bonus/mgradnet_ot.py` | `ConditionalMGradNet` (monotone-gradient operator, used in the wiring sanity test). (§9) |
| `src/filters/bonus/lhnn_nuts.py` | **L-HNN NUTS** sampler: surrogate-gradient leapfrog + H-error monitor + Conrad final-state MH; `kinetic_mh`. (§11) |
| `src/filters/bonus/lhnn_hmc_pf.py` | L-HNN pilot/train + fixed-L HMC engine. (§11) |
| `src/filters/dpf/sinkhorn.py` | `sinkhorn_potentials(a,b,x,y,eps,n_iters)` — entropic OT, `@tf.function(jit_compile=True)`. (§2 Sinkhorn N-scaling, baseline OT) |

## Profiling / verification scripts (`scripts/`)

| Path | Produces | Section |
|---|---|---|
| `scripts/profile_section1_svssm.py` | `$R/profile_section1_svssm/profile_svssm.json` | §2 (univariate JIT/timing/N-scaling/HMC-step) |
| `scripts/profile_section1_svssm_multi.py` | `$R/profile_section1_svssm_multi/profile_svssm_multi.json` | §2 (d=2 profiling) |
| `scripts/verify_no_retracing_svssm.py` | `$R/profile_section1/retracing_verification_svssm.json` | §2 (no-retracing, 3 init types, HMC) |
| `scripts/sanity_test_svssm_ledh.py` | stdout (warm timings, retracing, likelihood-peak) | §2 (corroboration) |
| `scripts/sanity_test_svssm_neural_ot.py` | stdout (NN-OT wiring, 7-D context check) | §9 (context vector) |
| `scripts/compare_nnot_sinkhorn_posteriors.py` | `…/d1_lhnn_nnot_moderate_T100/nnot_vs_sinkhorn_posteriors.{txt,json}` | §3 (NN-OT vs Sinkhorn KS / Wasserstein table, R3.3) |
| `scripts/profile_section1.py` *(OLD module)* | `$R/profile_section1/results.json` | **NOT used** — profiles old generic `differentiable_pfpf_ledh`; retained for history |
| `scripts/profile_hmc_step_xla.py` *(OLD module)* | `$R/profile_section1/profile_hmc_step_summary.json` | **NOT used** — old `differentiable_ledh` |

## Experiment drivers (`scripts/exp/`)

| Path | Role | Section |
|---|---|---|
| `scripts/exp/exp_hmc_svssm.py` | 1-D SVSSM HMC driver (`--T --out_dir --prior_* --num_chains --num_results --init_type`). | §3, §4 |
| `scripts/exp/slurm.slurm` | Launches the 1-D **wide-shifted** T-sweep (`SWEEP_TAG=svssm_hmc_sweep_wide`, one subdir per T). | §3 (1-D table) |
| `scripts/exp/ablate_init_h0.py` | `h_0` init ablation (stationary / fixed_mu / diffuse). | §4 |
| `scripts/exp/exp_hmc_svssm_multivariate_full_phi_lhnn_nuts.py` | Multivariate full-Φ L-HNN-NUTS HMC driver. | §3 d=2, §8 |
| `scripts/exp/launch_lhnn_nuts_parallel.sh` | Train-once → N parallel chains → stitch wrapper for the multivariate driver. | §3 d=2 |
| `scripts/exp/run_lhnn_nnot_d2_T200.sh` / `_T500.sh` / `_T1000.sh` | d=2 sweep launchers (train operator + L-HNN-NUTS + plots). | §3 d=2, §8 |
| `scripts/exp/rerun_lhnn_nnot_tsweep_2500.sh` | d=2 sweep re-run at 3000 draws → `reports/d2_lhnn_nnot_B_T*`. | §3 d=2, §8 |
| `scripts/exp/run_lhnn_nnot_d1_moderate.sh` | d=1 moderate-prior L-HNN+NN-OT (φ≈0.808). | §3, §9 |
| `scripts/exp/run_lhnn_sinkhorn_d1_moderate.sh` | d=1 moderate-prior L-HNN+**Sinkhorn** (φ≈0.796). | §3, §9 |
| `scripts/exp/run_lhnn_nnot_d1_widephi.sh` | d=1 operator φ-grid-width control (0.05→0.20). | §3, §9 |
| `scripts/exp/exp_v2_identifiability_demo.py` | V2 1-D scale-ridge (free-A vs A=1). | §5 |
| `scripts/exp/exp_v2_three_way_fixes.py` | V2 three equivalent restrictions (A=1 / σ_η=1 / μ=1). | §6 |
| `scripts/exp/exp_v2_multivariate_demo.py` | V2 d=2 FREE-A vs FIXED A=I. | §5, §7 |
| `scripts/exp/analyze_v2_mv_vehtari.py` | Post-hoc Vehtari (split-R̂, bulk/tail-ESS) for the V2 d=2 run. | §7 |
| `scripts/exp/phase16_train_multi_nnot_nd.py` | n-D DeepONet operator trainer (supervised MSE, early-stop). | §3 d=2, §10 |
| `scripts/exp/phase2_train_svssm_neural_ot.py` | d=1 SVSSM operator trainer. | §10 |
| `scripts/exp/phase4_loss_modes.py` | Loss-mode comparison (supervised / monge_ampere / mixed) → `section2_phase4/`. | §10 |
| `scripts/plot_trace_stationary_cov.py` | Derives Σ_h (Smith-doubling Lyapunov) per draw for the stationary-combination table. | §8 |
| `scripts/exp/exp_hmc_svssm_neural_ot.py` | NN-OT HMC driver (eager vs XLA). | §10 |
| `scripts/exp/probe_nnot_jit.py` | Isolates the `.numpy()`→XLA-block fix. | §10 |
| `scripts/exp/exp_hmc_svssm_lhnn.py` | L-HNN fixed-L HMC (T=200 benchmark row 2). | §11 |
| `scripts/exp/exp_hmc_svssm_lhnn_disperse.py` | L-HNN disperse-pilot (row 3). | §11 |
| `scripts/exp/exp_hmc_svssm_lhnn_nuts.py` | L-HNN-NUTS (rows 4–5, depth sweep; `--weights_cache`). | §11 |
| `scripts/exp/d4_orchestrate_nnot.sh` | d=4 dimension-ceiling orchestration. | §11 |
| `scripts/exp/compare_svssm_hmc_methods.py` | Vehtari rank-R̂ / ESS comparison utility (`rank_rhat`). | §3, §11 |

## Key output directories

| Path | Contents | Section |
|---|---|---|
| `$R/profile_section1_svssm/` | `profile_svssm.json` | §2 (d=1) |
| `$R/profile_section1_svssm_multi/` | `profile_svssm_multi.json` | §2 (d=2) |
| `$R/profile_section1/retracing_verification_svssm.json` | trace-count deltas | §2 |
| `$R/new/svssm_hmc_sweep_wide_T{50,100,200}/` | 1-D wide-prior recovery; also vanilla T=200 ref | §3, §11 |
| `reports/d2_lhnn_nnot_B_T50_2500/`, `…_T200_2500/`, `…_T500_2500/`, `…_T1000/` | d=2 recovery sweep — **quoted** table source (samples `.npz`, summaries). Plain `…_T200/…_T500/` (no suffix) are earlier cache-only runs, not quoted. | §3, §8 |
| `reports/d1_lhnn_nnot_moderate_T100/`, `reports/d1_lhnn_sinkhorn_moderate_T100/`, `reports/d1_lhnn_nnot_widephi_T100/` | NN-OT vs Sinkhorn φ; grid-width control | §3, §9 |
| `$R/h0_ablation/h0_ablation_results.json` | h_0 ablation | §4 |
| `$R/v2_identifiability_demo/` | `v2_id_demo_result.json`, `v2_samples.npz` | §5 |
| `$R/v2_three_way_fixes/v2_three_way_result.json` | three-way fixes | §6 |
| `$R/v2_multivariate_demo_long/` + `v2_mv_vehtari.json` | d=2 FREE/FIXED Vehtari | §5, §7 |
| `$R/section2_phase4/` | supervised/monge_ampere/mixed weights + `phase4_summary.json` | §10 |
| `$R/svssm_hmc_lhnn_T200_{wide,disperse,nuts,nuts_depth3,nuts_depth5,nuts_depth7}/` | 6-run L-HNN benchmark | §11 |
| `reports/d4_T200_lhnn_nuts/` (+ `d4_T200_{training,nnot,sinkhorn}/`) | d=4 ceiling (logs; run killed) | §11 |

---

# Result Traceability

## §1 — Model and estimation pipeline

The HRS constants — `μ_z = −1.2704`, `Var = π²/2 ≈ 4.93` — are **analytic**
(mean/variance of the `log χ²₁` distribution), not measured. They are coded
in the filter's `log(y²)` transform.

* **Source:** `src/filters/bonus/extra_bonus/differentiable_ledh_svssm.py` (the
  `z_t = log(y_t²+δ)` linearisation and `μ_z`, `σ_z²` constants).
* **Reproduction:** none required (closed form). The *consequences* of the
  HRS bias (σ_h² coverage loss) are measured in §6/§8 below.

---

## §2 — Efficient differentiation (JIT, retracing, Sinkhorn cost)

### R2.1 No retracing: trace-count Δ = 0 across forward/grad/HMC (all init types); finite grads on all 10 evals
* **Directory:** `$R/profile_section1/`
* **Source code:** `src/filters/bonus/extra_bonus/differentiable_ledh_svssm.py`; `src/filters/dpf/sinkhorn.py`
* **Driver:** `scripts/verify_no_retracing_svssm.py` (config T=20, N=64, n_lambda=10, K=10 hard-coded in `make_ll`/`main`)
* **Command:** `python3 scripts/verify_no_retracing_svssm.py`
* **Artifact:** `$R/profile_section1/retracing_verification_svssm.json` (`results[].fwd_delta`/`grad_delta` = 0; `finite_grads:10`; `*_pass:true`)
* **Notes:** Also re-confirmed by `scripts/profile_section1_svssm.py` (`retracing.trace_delta_over_10_evals = 0`).

### R2.2 JIT table — fwd 1.69 s→14.7 ms (115×); fwd+grad 5.48 s→105 ms (52×); backward/forward 7.1×
* **Directory:** `$R/profile_section1_svssm/`
* **Source code:** `differentiable_ledh_svssm.py`, `sinkhorn.py`
* **Driver:** `scripts/profile_section1_svssm.py` (T=20, N=64, n_lambda=10, K=10, repeats=5, CPU)
* **Command:** `python3 scripts/profile_section1_svssm.py`
* **Artifact:** `$R/profile_section1_svssm/profile_svssm.json` → key `timing_N64` (`cold_fwd_s`, `warm_fwd_ms`, `cold_fwd_grad_s`, `warm_fwd_grad_ms`, `backward_over_forward`)
* **Notes:** "un-jitted ~5 s/gradient" = `cold_fwd_grad_s`. Corroborated at T=50 by `scripts/sanity_test_svssm_ledh.py` (warm fwd 31.4 ms, fwd+grad 251.7 ms, 8.0×). Absolute ms are CPU/TF-2.16.2-specific.

### R2.3 Sinkhorn vs full-filter N-scaling (N=64→512): Sinkhorn fwd 0.50→6.40 ms; filter fwd 14.7→127.5 ms; "Sinkhorn = 3–5 % of filter cost"
* **Directory:** `$R/profile_section1_svssm/`
* **Source code:** `sinkhorn.py` (standalone), `differentiable_ledh_svssm.py` (filter)
* **Driver:** `scripts/profile_section1_svssm.py`
* **Command:** `python3 scripts/profile_section1_svssm.py`
* **Artifact:** `profile_svssm.json` → keys `sinkhorn_n_scaling[]` and `filter_n_scaling[]`
* **Notes:** the 3–5 % ratio = `sinkhorn fwd_ms / filter fwd_ms` per N (0.50/14.7, 6.40/127.5).

### R2.4 Profiling: warm HMC step 0.64 s (5 grads × ~105 ms)
* **Directory:** `$R/profile_section1_svssm/`
* **Driver:** `scripts/profile_section1_svssm.py` (`hmc_step` block: TFP HMC, L=5)
* **Command:** `python3 scripts/profile_section1_svssm.py`
* **Artifact:** `profile_svssm.json` → key `hmc_step.warm_step_s`

### R2.5 Multivariate d=2: Δ=0 retracing; JIT 49–106×; fwd 16.6 ms / fwd+grad 109.5 ms (6.6×); warm HMC step 0.65 s; filter fwd 17.2→190.1 ms
* **Directory:** `$R/profile_section1_svssm_multi/`
* **Source code:** `differentiable_ledh_svssm_multivariate.py` (diagonal-Φ `_jit_step` path), `sinkhorn.py`
* **Driver:** `scripts/profile_section1_svssm_multi.py` (d=2, T=20, N=64, n_lambda=10, K=10; truth μ=(1,−1), φ=(0.95,0.80), σ_η=(0.312,0.6))
* **Command:** `python3 scripts/profile_section1_svssm_multi.py`
* **Artifact:** `$R/profile_section1_svssm_multi/profile_svssm_multi.json`
* **Notes:** "d=2 ≈ d=1, not faster" — compare to `profile_svssm.json`. This *corrects* the older `project_section2_phase14` "0.08 s faster" claim (different driver/config).

---

## §3 — Recovering the parameters

### R3.1 1-D wide-prior T-sweep table (μ/φ/σ²_η medians at T=50/100/200; 95% CI; P(φ<0) 13.6→8.0→4.9 %; rank-R̂ ≤ 1.044)
* **Directory:** `$R/new/svssm_hmc_sweep_wide_T{50,100,200}/`
* **Source code:** `src/filters/bonus/extra_bonus/differentiable_ledh_svssm.py`
* **Driver / config:** `scripts/exp/exp_hmc_svssm.py`, launched by `scripts/exp/slurm.slurm` with the wide-shifted prior (`--prior_mu_loc 2.0 --prior_mu_scale 3.0 --prior_phi_raw_loc 0.0 --prior_phi_raw_scale 2.0 --prior_log_sigma_eta_sq_loc 1.5 --prior_log_sigma_eta_sq_scale 3.0`), 4 chains × 2500, truth μ=0/φ=0.95/σ_η=0.3.
* **Command (per T, laptop equivalent of the Slurm body):**
  ```bash
  python3 scripts/exp/exp_hmc_svssm.py --T 200 --num_chains 4 --num_results 2500 \
    --init_type stationary \
    --prior_mu_loc 2.0 --prior_mu_scale 3.0 \
    --prior_phi_raw_loc 0.0 --prior_phi_raw_scale 2.0 \
    --prior_log_sigma_eta_sq_loc 1.5 --prior_log_sigma_eta_sq_scale 3.0 \
    --out_dir reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/new/svssm_hmc_sweep_wide_T200
  ```
  (repeat with `--T 50` / `--T 100`)
* **Artifacts:** per-dir `svssm_hmc_samples.npz`, `svssm_hmc_summary.json`, `svssm_hmc_results.txt`. `P(φ<0)` and medians are computed from `svssm_hmc_samples.npz` (`phi` array).
* **Notes:** Originally a Slurm HPC run; reproducible on a laptop but slow (hours/T). Posterior is reproducible given fixed seeds in `exp_hmc_svssm.py`.

### R3.2 d=2 recovery table (truth μ=(1,−1), φ_diag=(0.95,0.80), φ_off=0.05, σ_η=(0.312,0.6); medians at T=50/200/500/1000; coverage 7/7 except φ_11 at T=1000; max rank-R̂ 1.02/1.07/1.06/1.16)
* **Directory (source of the quoted table — the 3000-draw `_2500` dirs):**
  - T=50  → `reports/d2_lhnn_nnot_B_T50_2500/`
  - T=200 → `reports/d2_lhnn_nnot_B_T200_2500/`
  - T=500 → `reports/d2_lhnn_nnot_B_T500_2500/`
  - T=1000 → `reports/d2_lhnn_nnot_B_T1000/`
  - **NB:** the plain dirs `reports/d2_lhnn_nnot_B_T200/` and `…_T500/` are the *earlier, shorter* runs (slightly different medians, e.g. T=200 μ₀ 0.51 vs the quoted 0.58); they are **not** the table source — they only supply the operator/L-HNN *cache* that the `_2500` re-run reads. There is no plain `…_T50` dir.
* **Source code:** `differentiable_ledh_svssm_multivariate.py`, `src/filters/bonus/lhnn_nuts.py`, `src/filters/bonus/deeponet_ot.py`
* **Driver:** the quoted `_2500` dirs (T=50,200,500, 4 chains × 3000 draws) are written by `scripts/exp/rerun_lhnn_nnot_tsweep_2500.sh`, which **reads caches** produced by the plain per-T launchers; T=1000 by `scripts/exp/run_lhnn_nnot_d2_T1000.sh`. Each launcher calls operator trainer `phase16_train_multi_nnot_nd.py` then `launch_lhnn_nuts_parallel.sh` → `exp_hmc_svssm_multivariate_full_phi_lhnn_nuts.py`.
* **Command (full chain, in order):**
  ```bash
  # 1. build operator + L-HNN cache (writes the plain dirs reports/d2_lhnn_nnot_B[, _T200, _T500])
  bash scripts/exp/run_lhnn_nnot_d2_T200.sh
  bash scripts/exp/run_lhnn_nnot_d2_T500.sh
  # 2. converged 3000-draw re-run → the QUOTED reports/d2_lhnn_nnot_B_T{50,200,500}_2500
  bash scripts/exp/rerun_lhnn_nnot_tsweep_2500.sh
  # 3. T=1000 (its own launcher → reports/d2_lhnn_nnot_B_T1000)
  bash scripts/exp/run_lhnn_nnot_d2_T1000.sh
  ```
  (T=50's cache source is `reports/d2_lhnn_nnot_B`, built by the d=2 base launcher.)
* **Artifacts:** `svssm_hmc_multi_full_phi_samples.npz`, `..._summary.json`, trace plots per dir.
* **Notes:** rank-R̂ recomputed from `.npz` via `compare_svssm_hmc_methods.rank_rhat` (do NOT trust stored summary — see memory `feedback_rhat_recompute_from_npz`).

### R3.3 "Should θ be an operator input?" — NN-OT vs Sinkhorn posteriors (the §3 "Does it work?" table)
The §3 table (controlled comparison: two matched d=1 moderate-prior runs differing
ONLY in the OT resample — trained NN-OT operator vs exact Sinkhorn). Per param:
NN-OT/Sinkhorn medians μ −0.31/−0.29, φ **0.805/0.796**, σ²η 0.122/0.131; KS p
0.164/0.0016/0.0017; Wasserstein 0.029/0.023/0.007; |Δ median| ≤ 0.018.
* **Directory:** `reports/d1_lhnn_nnot_moderate_T100/`, `reports/d1_lhnn_sinkhorn_moderate_T100/`
* **Source code:** `differentiable_ledh_neural_ot_svssm.py` (NN-OT), `differentiable_ledh_svssm.py` (Sinkhorn), `deeponet_ot.py`, `lhnn_nuts.py`
* **Drivers (produce the two posteriors):** `scripts/exp/run_lhnn_nnot_d1_moderate.sh` (NN-OT) and `scripts/exp/run_lhnn_sinkhorn_d1_moderate.sh` (Sinkhorn), identical moderate prior / same sampler / same data+CRN (verified config-identical except `nnot_weights`).
* **Comparison script (produces the KS / Wasserstein table):** `scripts/compare_nnot_sinkhorn_posteriors.py`
* **Command:**
  ```bash
  # 1. the two matched runs (skip if already present)
  bash scripts/exp/run_lhnn_nnot_d1_moderate.sh
  bash scripts/exp/run_lhnn_sinkhorn_d1_moderate.sh
  # 2. the comparison (per-param summaries + KS stat/p + Wasserstein + verdict)
  python3 scripts/compare_nnot_sinkhorn_posteriors.py \
    --nnot_dir reports/d1_lhnn_nnot_moderate_T100 \
    --sinkhorn_dir reports/d1_lhnn_sinkhorn_moderate_T100
  ```
* **Artifacts:** `svssm_hmc_multi_full_phi_samples.npz` in each run dir; the comparison writes `reports/d1_lhnn_nnot_moderate_T100/nnot_vs_sinkhorn_posteriors.{txt,json}` (self-documenting: source dirs, truth, draw counts, per-param metrics).
* **Reading:** medians/CIs agree to MC noise (point inference identical); KS flags a *significant but tiny* φ/σ²η shape difference (KS stat 0.042, Wasserstein ≤0.023) — the operator-approximation signature, not a material shift.
* **Context-vector dimension `ctx(d)=3d+d(d−1)/2+d+3`:** verified by `python3 scripts/sanity_test_svssm_neural_ot.py` (Check 1: 7-D at d=1; `build_svssm_context_scalars` in `differentiable_ledh_neural_ot_svssm.py`).

### R3.4 Operator φ-grid-width control: spread 0.05→0.20 leaves φ unchanged (0.805→0.808); +23 % wall (642→791 s/chain)
* **Directory:** `reports/d1_lhnn_nnot_widephi_T100/` (vs `reports/d1_lhnn_nnot_moderate_T100/`)
* **Driver:** `scripts/exp/run_lhnn_nnot_d1_widephi.sh` (trains operator with `--phi_spread 0.20`)
* **Command:** `bash scripts/exp/run_lhnn_nnot_d1_widephi.sh`
* **Artifacts:** `svssm_hmc_multi_full_phi_samples.npz` (φ median); `chain_*/...summary.json` (`sampling_wall_s`)

### R3.5 Prior-only (no-likelihood) baseline — induced φ prior is the symmetric U (P(φ<0)≈49.5%, median≈0, equal ±1 lobes); σ²_η prior median 4.3, ~38% of draws > 10, q99 ≈ 4.7×10³
* **Directories (two independent seeds, both loc=0/scale=2 — confirms reproducibility):**
  - `$R/phase19_attractor/no_data_baseline/` (seed 300): P(φ<0)=**49.5%**, median φ=0.025, median σ²_η=4.33, P(σ²_η>10)=38.9%, q99=4525
  - `$R/prior_only_1d_confirm/` (seed 777): P(φ<0)=**49.9%**, median φ=0.006, median σ²_η=4.24, P(σ²_η>10)=38.4%, q99=4853 — **+ `phi_prior_histogram.png`** (the symmetric U)
* **Source code:** `src/filters/bonus/extra_bonus/differentiable_ledh_svssm.py` (target reduces to the pure prior when likelihood is off; `exp_hmc_svssm.py:build_target(..., no_likelihood=True)`)
* **Driver / exact 1-D prior:** `scripts/exp/exp_hmc_svssm.py --no_likelihood`
* **Command (the confirm run; prior-only, ~1 min, no filter):**
  ```bash
  python3 scripts/exp/exp_hmc_svssm.py --no_likelihood --T 200 --num_chains 4 \
    --num_results 3000 --num_burnin 500 --base_seed 777 \
    --prior_mu_loc 2.0 --prior_mu_scale 3.0 \
    --prior_phi_raw_loc 0.0 --prior_phi_raw_scale 2.0 \
    --prior_log_sigma_eta_sq_loc 1.5 --prior_log_sigma_eta_sq_scale 3.0 \
    --out_dir reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/prior_only_1d_confirm
  ```
* **Artifact:** `svssm_hmc_samples.npz` → `samples_constrained` (φ = col 1, σ²_η = col 2); `phi_prior_histogram.png`.
* **Stable vs unstable numbers:** φ P(<0)≈49.5–49.9% and median≈0 are **symmetry-pinned and robust**; σ²_η median 4.3, P(>10)≈38%, q99≈4.7×10³ are robust. The σ²_η **max draw is NOT reproducible** (438k seed-300 vs 1.6M seed-777 — an extreme tail order-statistic); the report quotes the quantile descriptors, never the max.
* **Role:** calibration baseline for §3's honesty correction — induced φ prior is symmetric/edge-piling (equal mass at ±1, trough at 0; loc=0, so the prior does NOT pre-favour +1), so φ's data evidence is sign-resolution + CI width, not the point landing on 0.954; the σ²_η prior heavy tail vs posterior 0.101 is the clean data-driven exhibit.

---

## §4 — The initial condition `h_0`

### R4.1 h_0 ablation table (T=50): stationary (μ 0.13/φ 0.947/σ² 0.133, all covered); fixed_mu (μ −0.51); diffuse (μ 0.96 N, φ 0.485 N, best R̂)
* **Directory:** `$R/h0_ablation/`
* **Source code:** `differentiable_ledh_svssm.py` (its `init_type` ∈ {stationary, fixed_mu, diffuse})
* **Driver:** `scripts/exp/ablate_init_h0.py` (truth μ=0/φ=0.95/σ_η=0.3, T=50, 2 chains, wide priors μ~N(0,5²), φ_raw~N(0,2²), logσ²~N(−2,2²))
* **Command:** `python3 scripts/exp/ablate_init_h0.py`
* **Artifact:** `$R/h0_ablation/h0_ablation_results.json` (per-init medians, coverage, rank-R̂)

---

## §5 — Estimating `A` (V2 additive model)

### R5.1 1-D scale-ridge table (truth μ=1,φ=0.9,σ_η=0.5,A=2,σ_ε=0.3; free-A vs A=1 sd ratios: σ²_η 27.6×, μ 4.3×, φ/σ²_ε 1.0×; products A·μ 1.73, A²σ²_η 0.81; free-A CI σ²_η [0.02,13])
* **Directory:** `$R/v2_identifiability_demo/`
* **Source code:** `scripts/exp/exp_v2_identifiability_demo.py` (self-contained Kalman-filter V2 likelihood + TFP HMC)
* **Driver:** `scripts/exp/exp_v2_identifiability_demo.py` (T=200, free-A vs A=1)
* **Command:** `python3 scripts/exp/exp_v2_identifiability_demo.py`
* **Artifacts:** `$R/v2_identifiability_demo/v2_id_demo_result.json`, `v2_samples.npz`, plots `v2_free_ridge.png` / `v2_marginals_compare.png` / `v2_identified_products.png`

### R5.2 Multivariate FREE-A Vehtari summary (A-entry bulk-ESS 58–79/12000, max rank-R̂ 1.16, 1/12 pass) and A=I (min bulk-ESS 1958, R̂≤1.006, 8/8 pass)
* See **R7.1** (same run; §5 quotes the summary, §7 the full tables).

---

## §6 — Restrictions on `A` / "meaningful"

### R6.1 Three-way-fix table (truth μ=2,φ=0.9,σ_η=0.4,A=3,σ_ε=0.3): A=1→μ5.62[6.0],σ²η1.19[1.44]; σ_η=1→μ5.22[5.0],A1.09[1.2]; μ=1→A5.60[6.0],σ²η0.039[0.04]; φ≈0.84 & σ²_ε≈0.09 invariant across all three
* **Directory:** `$R/v2_three_way_fixes/`
* **Source code:** `scripts/exp/exp_v2_three_way_fixes.py`
* **Driver:** `scripts/exp/exp_v2_three_way_fixes.py` (T=200, 2 chains × (400 burn + 1000))
* **Command:** `python3 scripts/exp/exp_v2_three_way_fixes.py`
* **Artifact:** `$R/v2_three_way_fixes/v2_three_way_result.json` (keys `ridge_predictions`, `A1_summary`, `sig1_summary`, `mu1_summary`, `invariance`)

---

## §7 — Which parameters need restricting

### R7.1 d=2 FREE-A (12-param) vs FIXED A=I (8-param) Vehtari tables (truth μ=(1,1), Φ=diag(0.9,0.85), Σ_η=diag(0.25,0.36), A=diag(2,2), Σ_ε=0.09I; A-entries R̂ 1.126–1.159 / bulk-ESS 58–79; σ²_ε passes in FREE R̂≤1.007; A=I all pass R̂≤1.006, bulk-ESS≥1958)
* **Directory:** `$R/v2_multivariate_demo_long/` (+ `v2_mv_vehtari.json`)
* **Source code:** `scripts/exp/exp_v2_multivariate_demo.py` (exact multivariate Kalman filter via `tf.while_loop`), `scripts/exp/analyze_v2_mv_vehtari.py` (post-hoc Vehtari)
* **Driver/command:** (the quoted *long* run used an explicit `--out_dir`; the script default is the shorter `.../v2_multivariate_demo`)
  ```bash
  python3 scripts/exp/exp_v2_multivariate_demo.py \
    --out_dir reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/v2_multivariate_demo_long   # 4 chains × 3000, FREE + FIXED
  python3 scripts/exp/analyze_v2_mv_vehtari.py \
    --out_dir reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/v2_multivariate_demo_long   # → v2_mv_vehtari.json
  ```
* **Artifacts:** samples `.npz` + `v2_mv_vehtari.json` (per-param median/sd/R̂/bulk-ESS/tail-ESS). Story figures via `scripts/exp/plot_v2_mv_story.py`.

---

## §8 — Identification (different parameters, same data)

### R8.1 V2 exact-ridge invariants `(Aμ, A²σ²_η, φ, σ_ε)`; "FREE-A chain walks the orbit (A sd ~3)"
* Same source as **R5.1 / R7.1** (`exp_v2_identifiability_demo.py` for the 1-D orbit; `exp_v2_multivariate_demo.py` for the d=2 wandering A-entries, sd ~3 in `v2_mv_vehtari.json`).

### R8.2 Exact-vs-weak regime table; P(φ<0) 13.6→4.9 %
* Same source as **R3.1** (`new/svssm_hmc_sweep_wide_T*`).

### R8.3 Stationary-combination table (derived σ²_h,0/σ²_h,1/ρ_h and structural φ_11/σ²_η,1 at T=50/200/500/1000; ρ_h error −102→−8 %; σ²_h coverage fails by T=200)
* **Directory:** the quoted `_2500` dirs `reports/d2_lhnn_nnot_B_T{50,200,500}_2500/` + `reports/d2_lhnn_nnot_B_T1000/` (same posterior draws as **R3.2**)
* **Source code:** stationary quantities derived per draw by solving the discrete Lyapunov equation `Σ_h = Φ Σ_h Φᵀ + Σ_η` (Smith doubling) — implemented in `scripts/plot_trace_stationary_cov.py` (called by the d=2 launchers).
* **Command:** (after R3.2 runs) `python3 scripts/plot_trace_stationary_cov.py --out_dir reports/d2_lhnn_nnot_B_T500_2500` (repeat per T-dir; T=1000 dir has no suffix)
* **Artifacts:** stationary-cov trace/summary alongside the `.npz` in each d=2 dir.
* **Notes:** the σ²_h upward bias is the HRS structural bias (§1), measured here as coverage degrading with T.

---

## §9 — Design of the neural operator

The §9 table (monotone/Brenier, differentiable-in-θ, smooth) describes
**architecture guarantees**, not measured numbers.

* **Source code:** `src/filters/bonus/deeponet_ot.py` — the map
  `T(x|θ,c)=a⊙x+offset+Σ_k softplus(β_k) σ(w_kᵀx+b_k) w_k`, Jacobian
  `J_T = diag(a)+Σ_k softplus(β_k) σ′ w_k w_kᵀ ⪰ 0` (class docstring +
  `call` + `log_det_jacobian`). Activations GELU/sigmoid/softplus (no ReLU).
* **Reproduction:** inspection of `deeponet_ot.py`. Differentiability/finiteness
  exercised by `python3 scripts/sanity_test_svssm_neural_ot.py` (Checks 3–4:
  finite forward + finite θ-gradient).
* **NN-OT≈Sinkhorn (0.808/0.796):** see **R3.3**.
* **Paper arXiv:2408.02697 (RePU/MRePU):** external reference (Kim & Kang 2024);
  no repo artifact.

---

## §10 — Training the operator

### R10.1 Supervised loss; best val-MSE 0.0038 at d=1 (tens of epochs); |Δ log p| ≤ 0.1 vs Sinkhorn
* **Directory:** training outputs under `$R/section2_phase*/` and `reports/.../phase16_*`
* **Source code:** `src/filters/bonus/extra_bonus/svssm_neural_ot_training.py`, `src/filters/bonus/deeponet_ot.py`
* **Driver:** `scripts/exp/phase16_train_multi_nnot_nd.py` (d=1) — supervised MSE, plateau early-stop, best-weight restore; prints `val_mse`, `best val_mse`.
* **Command:**
  ```bash
  python3 -m scripts.exp.phase16_train_multi_nnot_nd --d 1 --T 100 --N 64 \
    --n_theta 60 --seeds_per_theta 2 --max_epochs 60 --patience 10 \
    --out_dir reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase16_d1
  ```
* **Artifacts:** operator weights `*.weights.h5` + summary JSON with `best_val_mse`.
* **`|Δ log p| ≤ 0.1` downstream check:** drop trained weights into `exp_hmc_svssm_neural_ot.py` / the filter and compare to Sinkhorn baseline log p (phase 2/3 outputs).

### R10.2 Monge–Ampère loss fails: |Δ log p| ~ 10² vs Sinkhorn (supervised/mixed/MA comparison)
* **Directory:** `$R/section2_phase4/`
* **Source code:** `svssm_neural_ot_training.py` + `deeponet_ot.py` (loss-mode switch: `supervised` / `monge_ampere` / `mixed`)
* **Driver:** `scripts/exp/phase4_loss_modes.py` (trains all three modes; writes `supervised.weights.h5`, `monge_ampere.weights.h5`, `mixed.weights.h5`)
* **Command:** `python3 scripts/exp/phase4_loss_modes.py --out_dir reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/section2_phase4`
* **Artifacts:** `$R/section2_phase4/phase4_summary.json`, `phase4_report.txt`, `phase4_loss_curves.png`
* **Notes:** the |Δ log p|~10² for MA is recorded in `phase4_summary.json` / `phase4_report.txt`.

### R10.3 jit-inside-HMC: removing 3 `.numpy()` calls turns eager 3038 s → 327 s under XLA (9.3×)
* **Directory:** the **full-HMC walls (3038 s / 327 s)** are recorded in `section3_kernel_upgrade.tex` (Phase 6), **not** in a JSON artifact here. `$R/section2_phase4/nnot_jit_probe.json` holds a *related but different* **per-call microbench** (eager fwd 248 ms / fwd+grad 690 ms; graph/XLA variants), which demonstrates the same eager-vs-graph effect at the single-call level.
* **Source code:** `differentiable_ledh_neural_ot_svssm.py` (the `_neural_ot_resample_1d` `.numpy()`→tensor fix), `deeponet_ot.py`
* **Driver:** `scripts/exp/probe_nnot_jit.py` (writes `nnot_jit_probe.json`); full-HMC walls via `scripts/exp/exp_hmc_svssm_neural_ot.py` (`--jit_compile` on vs off).
* **Command:** `python3 scripts/exp/probe_nnot_jit.py` (cheap; reproduces the eager-vs-graph per-call gap). The 3038 s/327 s figures need the full HMC run (`exp_hmc_svssm_neural_ot.py`) and are platform-dependent.
* **Notes:** the *mechanism* (the 3 `.numpy()` calls breaking the XLA cluster) is cheaply reproducible; the headline walls are not (long HMC). The PSD-Jacobian determinant-lemma claim is architectural (`deeponet_ot.py:log_det_jacobian`).

---

## §11 — Accelerating HMC with L-HNN

### R11.1 Six-run T=200 benchmark (vanilla ref 39 842 s, R̂ 1.008; L-HNN fixed-L 5.97×/R̂1.746; disperse 5.67×/1.850; NUTS d=5 9.13×/1.016; NUTS d=7 4.75×/1.012)
* **Directories / drivers:**
  | Row | Directory | Driver |
  |---|---|---|
  | vanilla ref | `$R/new/svssm_hmc_sweep_wide_T200/` | `exp_hmc_svssm.py` (windowed, dense mass) |
  | L-HNN fixed-L | `$R/svssm_hmc_lhnn_T200_wide/` | `scripts/exp/exp_hmc_svssm_lhnn.py` |
  | disperse pilot | `$R/svssm_hmc_lhnn_T200_disperse/` | `scripts/exp/exp_hmc_svssm_lhnn_disperse.py` |
  | NUTS d=5 | `$R/svssm_hmc_lhnn_T200_nuts_depth5/` | `scripts/exp/exp_hmc_svssm_lhnn_nuts.py --max_treedepth 5` |
  | NUTS d=7 | `$R/svssm_hmc_lhnn_T200_nuts_depth7/` | `exp_hmc_svssm_lhnn_nuts.py --max_treedepth 7` |
  | (NUTS d=6) | `$R/svssm_hmc_lhnn_T200_nuts/` | `exp_hmc_svssm_lhnn_nuts.py` |
* **Source code:** `src/filters/bonus/lhnn_nuts.py`, `src/filters/bonus/lhnn_hmc_pf.py`, `differentiable_ledh_svssm.py`
* **Commands:**
  ```bash
  python3 scripts/exp/exp_hmc_svssm_lhnn.py          # fixed-L, T=200 wide
  python3 scripts/exp/exp_hmc_svssm_lhnn_disperse.py # disperse pilot
  python3 scripts/exp/exp_hmc_svssm_lhnn_nuts.py --max_treedepth 5 --weights_cache <path>
  python3 scripts/exp/exp_hmc_svssm_lhnn_nuts.py --max_treedepth 7 --weights_cache <path>
  ```
* **Artifacts:** per-dir samples + `*_summary.json` (`nuts_diagnostics.error_triggers_per_chain`, `sampling_wall_s`, `accept_rate_overall`). R̂ recomputed via `compare_svssm_hmc_methods.py`.
* **Notes:** these are multi-hour runs (vanilla ≈ 11 h). The `--weights_cache` flag shares pilot/train across the depth sweep. Speedups are vs the vanilla `39 842 s` wall.

### R11.2 Comfort-zone contrast: moderate prior err_trigs 0, ~2 grads/iter; wide prior err_trigs 81, ~8.5 grads/iter
* **Directory:** `reports/d1_lhnn_nnot_moderate_T100/` (moderate) vs `reports/d1_lhnn_nnot_wide_T100/` (wide)
* **Driver:** `scripts/exp/run_lhnn_nnot_d1_moderate.sh` and `scripts/exp/run_lhnn_nnot_d1_wide_reproduce.sh`
* **Artifacts:** `chain_*/...summary.json` → `nuts_diagnostics.error_triggers_per_chain`. (Wide run was killed mid-way; the 81/8.5 figures come from its progress log.)

### R11.3 err_trigs = 0 across 1.35 M leapfrog steps; sublinear T-cost 5.9→10.3 ks/chain (T:50→1000)
* **Directory:** the depth-sweep dirs (`$R/svssm_hmc_lhnn_T200_nuts_depth{3,5,7}/`) and the d=2 sweep dirs (`reports/d2_lhnn_nnot_B_T*` for the T-cost)
* **Artifacts:** `*_summary.json` (`error_triggers_per_chain`, `sampling_wall_s`)

### R11.4 d=4 dimension ceiling — FAILED (killed @6 h): surrogate underfits 18-D, chains froze, err_trigs ~300/chain
* **Directory:** `reports/d4_T200_lhnn_nuts/` (+ `reports/d4_T200_{training,nnot,sinkhorn}/`)
* **Driver:** `scripts/exp/d4_orchestrate_nnot.sh`
* **Notes:** **Cannot be reproduced as a completed result** — the run was deliberately killed at 6 h. Only the failure signature survives (training-loss plateau, frozen-chain logs, err_trigs ~300). Closest reproduction: re-launch `d4_orchestrate_nnot.sh` and observe the same non-convergence (train loss plateau ~9.5, α→0).

---

# Results that cannot be fully reproduced from the repo

| Result | Why | Closest path |
|---|---|---|
| §11 vanilla T=200 ref (39 842 s) and the long L-HNN benchmark walls | Multi-hour (≈11 h vanilla) runs; some originally on HPC/Slurm | Re-run the listed drivers; walls are platform-dependent (use ratios) |
| §3 R3.1 1-D sweep | Generated on HPC via `slurm.slurm` | Laptop reproduction with the per-T `exp_hmc_svssm.py` command above (slow) |
| §10 R10.3 3038 s→327 s | Long HMC walls; only the mechanism is cheap | `probe_nnot_jit.py` reproduces the `.numpy()`→XLA-block fix; full walls need `exp_hmc_svssm_neural_ot.py` |
| §11 R11.4 d=4 ceiling | Run was killed before completion (no converged artifact) | Re-launch `d4_orchestrate_nnot.sh`; observe identical divergence |
| §2 absolute wall-times | CPU/TF-2.16.2-specific | Re-run profilers on target hardware; speedup ratios are stable |
| §1 HRS constants (−1.2704, 4.93) | Analytic (log χ²₁ moments), not a run | Closed-form; no reproduction needed |

---

# Reproduction quick-start (fast, self-contained results)

These reproduce on a laptop in seconds–minutes and cover the §2, §5, §6, §4
numbers without HPC:

```bash
cd "/Users/amreshverma/Documents/Random Work/MLCOE_Q2_PF"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# §2 — JIT / retracing / N-scaling (d=1 and d=2)
python3 scripts/profile_section1_svssm.py
python3 scripts/profile_section1_svssm_multi.py
python3 scripts/verify_no_retracing_svssm.py

# §4 — h_0 ablation
python3 scripts/exp/ablate_init_h0.py

# §5 / §6 / §7 — V2 identifiability (1-D ridge, three-way fixes, d=2 FREE/FIXED)
python3 scripts/exp/exp_v2_identifiability_demo.py
python3 scripts/exp/exp_v2_three_way_fixes.py
python3 scripts/exp/exp_v2_multivariate_demo.py && python3 scripts/exp/analyze_v2_mv_vehtari.py
```
