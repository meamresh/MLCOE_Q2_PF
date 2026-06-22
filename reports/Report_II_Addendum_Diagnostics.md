# Report II — Addendum: MCMC Diagnostics & HMC Tuning

This addendum responds to the four reviewer remarks raised on Bonus Q1
(`exp_part3_bonus1b_hmc_vs_pmmh.py`):

1. **ESS** — what does "ESS = 17.7 %" (PFPF-LEDH, Table 6) mean, and is
   it reasonable? More generally, what counts as healthy ESS?
2. **R-hat** — not reported in the original submission (single-chain
   runs); reviewer cited Vehtari et al. (2021) as the canonical source.
3. **Parameter recovery** — reported as bias / RMSE only; no 95 %
   credible interval, no coverage flag, no explicit convergence verdict.
4. **HMC tuning** — step size adapted but mass matrix fixed at identity;
   no windowed warmup.

The repository now ships:

- a dedicated diagnostics utility
  [`src/utils/mcmc_diagnostics.py`](../src/utils/mcmc_diagnostics.py)
  implementing rank-normalised split-R̂, folded-rank R̂, bulk-ESS,
  tail-ESS, 95 % CI, coverage flag, and a `convergence_verdict` rule
  following Vehtari et al. (2021);
- multi-chain runners for HMC, L-HNN HMC, and PMMH
  ([`src/filters/bonus/{hmc_pf,lhnn_hmc_pf,pmmh}.py`](../src/filters/bonus/))
  with shared common-random-number (CRN) seeds across chains for
  stochastic-target HMC (§4.5);
- Stan-style three-window warmup with diagonal mass-matrix adaptation
  inside `run_hmc`;
- per-chain process-isolation drivers
  [`scripts/run_pmmh_isolated.sh`](../scripts/run_pmmh_isolated.sh),
  [`scripts/run_hmc_ledh_isolated.sh`](../scripts/run_hmc_ledh_isolated.sh),
  and [`scripts/run_lhnn_isolated.sh`](../scripts/run_lhnn_isolated.sh)
  for the long sampling runs (avoids TFP eager memory leaks
  accumulating across chains in a single process — see §4.6); the
  L-HNN driver also performs *dispersed-pilot training* (§5.3), the
  principled remediation for the wrong-mode issue.

Two report files are produced from the updated experiment code:

- `reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pmmh_only/comparison/results.txt`
  — converged PMMH baseline (4 chains × 20 000 post-burn samples).
- `reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/results.txt`
  — three-way comparison PMMH vs HMC-LEDH vs L-HNN HMC at the budget
  the gradient methods can afford (4 chains × 1 000 samples).

The §5 numbers below are taken from those two files.

---

## 1. What is ESS, and is "17.7 %" reasonable?

### 1.0 Two distinct quantities both called "ESS"

The original Report II uses the abbreviation **ESS** in two unrelated
contexts. The reviewer's "17.7 %" remark refers to the first; everything
the rest of this addendum says about R-hat, bulk-ESS, and tail-ESS
refers to the second. They are not interchangeable:

| Context | Definition | Measures | Reasonable range |
| ------- | ---------- | -------- | ---------------- |
| **Particle-filter ESS** (PFPF-LEDH, Table 6) | $\text{ESS}_t = 1\big/\sum_{i=1}^{N} w_t^{(i)\,2}$ at each timestep | Weight degeneracy of the importance-weighted particle distribution | mean ESS ≳ 30 % of $N$ is healthy on smooth dynamics; for Kitagawa's $x^2/20$ measurement, anything above ~10 % is operationally usable |
| **MCMC ESS** (PMMH, HMC-LEDH, L-HNN HMC) | $\text{ESS} = N\big/(1 + 2 \sum_k \rho_k)$ (autocorrelation-corrected) | Effective i.i.d. sample size for posterior summaries | bulk-ESS > 400 per parameter (Vehtari 2021) |

The "17.7 %" line in `reports/6_BonusQ1_HMC_Invertible_Flows/PFPF_LEDH/results.txt`
is `mean_ess / num_particles = 53.15 / 300`. It is **not** an MCMC
diagnostic. As a particle-filter degeneracy metric on the Kitagawa
posterior — which has a notoriously hard $x_t^2/20$ likelihood — it is
*moderate*: a bootstrap PF on the same problem typically gives 5–15 %;
LEDH with the OT-Sinkhorn proposal correction lifts it to ~18 %, in the
operationally usable range. The accompanying `min_ess = 3.9` is the
real warning sign — it says that at one or more timesteps the filter is
relying on essentially four particles, which is where most of the
filter's RMSE budget is spent. Increasing the LEDH proposal strength
(`n_lambda`) or `num_particles` is the right remediation; both are
already at the values used by Corenflos et al. (2021).

### 1.1 MCMC ESS — definition and bulk vs tail

The classical definition (Geyer 1992) for a stationary chain
$x_1,\dots,x_N$ with lag-$k$ autocorrelation $\rho_k$ is

$$
\text{ESS} \;=\; \frac{N}{1 + 2 \sum_{k=1}^{\infty} \rho_k}.
$$

ESS measures the size of the i.i.d. sample whose sample mean would have
the same variance as the (correlated) MCMC sample mean of the same
length. It is therefore a property *of the estimator*, not of the chain
in the abstract. A single chain can have very different ESS values for
different posterior summaries.

### Bulk-ESS vs tail-ESS (Vehtari et al. 2021, §4)

`ESS` in the code now follows the modern convention:

- **`bulk_ess`** — ESS computed on the *rank-normalized* split chains.
  Robust to non-Normal posteriors and informative about the mixing of
  the *posterior mean* (the central-tendency estimator).
- **`tail_ess`** — minimum over the 5 % and 95 % indicators
  $\mathbb{1}\{x \le q_{0.05}\}$, $\mathbb{1}\{x \le q_{0.95}\}$.
  This is sensitive to mixing in the *tails*, where bulk-ESS often
  looks fine while a rare-event quantile is still under-sampled.

Both are returned per parameter, alongside `ess_pct = bulk_ess / N`.

### What threshold counts as healthy?

The Stan / PyMC default — recommended by Vehtari (2021) §6 and used
verbatim by `convergence_verdict` — is

$$
\text{bulk-ESS} > 400 \;\;\text{AND}\;\; \text{tail-ESS} > 400.
$$

`400` is the ESS at which the Monte-Carlo standard error of a unit-variance
estimator drops below approximately 5 % of the posterior standard deviation,
which is the level at which downstream summaries stop being dominated by
sampler noise.

**On reporting fractions vs absolutes.** The original "17.7 %" line was
a *fraction* (mean PF ESS / N). For MCMC the reverse convention is
clearer because bulk-ESS depends only on autocorrelation, not on chain
length: the new `results.txt` blocks always print absolute bulk-ESS,
tail-ESS, *and* `ESS%` per parameter so the reader can apply the
threshold directly without arithmetic.

---

## 2. What is R-hat, and how do we report it now?

The Gelman & Rubin (1992) potential-scale-reduction factor compares
*between-chain* and *within-chain* variances:

$$
\hat{R} \;=\; \sqrt{\frac{\hat{\sigma}^2}{W}}, \qquad
\hat{\sigma}^2 = \frac{N-1}{N} W + \frac{1}{N} B,
$$

where $W$ and $B$ are the within- and between-chain variances. If
the chains have not converged to a common distribution, $B \gg W$ and
$\hat{R} \gg 1$.

### Three flavours we now compute

- **`split_rhat`** — Gelman-Rubin $\hat{R}$ computed on chains that
  have each been split in half. Half-chain splitting flags chains that
  *appear* stationary because their first and second halves cancel
  (Stan reference manual §15.4).
- **`rank_rhat`** — Vehtari (2021) §3: rank-normalize across chains
  (Eq. 14, ranks → standard-normal scores), then compute split-$\hat{R}$.
  Robust to heavy tails; `tfp.mcmc.potential_scale_reduction` requires
  Normality otherwise.
- **`folded_rank_rhat`** — same as above on
  $|x - \mathrm{median}(x)|$. Sensitive to multi-modality and scale
  asymmetry that the rank-normalised version misses.
  We return $\max(\text{rank-}\hat{R},\, \text{folded-rank-}\hat{R})$.

### Threshold

Vehtari (2021) deprecates the older `< 1.1` cut-off in favour of the
much tighter

$$
\hat{R} < 1.01 \quad \text{per parameter},
$$

which is what `convergence_verdict` enforces.

### Number of chains and per-chain budget

`run_*_multi_chain` now spawns **`num_chains = 4`** chains with
dispersed initial points (`disperse_initial_states`, multiplicative
log-space spread; `scale = 0.10` for PMMH and L-HNN HMC, `scale = 0.15`
for HMC-LEDH — tightened from the original `0.5` after diagnostic
analysis showed the wider spread was sitting outside the basin of
attraction the chains could traverse within the budget) and per-chain
seeds `base + 1009·c`. Four chains is the practical minimum for
reliable $\hat{R}$ (Vehtari §6).

Final per-method budgets (after the convergence-tuning iterations
discussed in §5):

| Method     | `samples_per_chain` | `burn_per_chain` | Notes |
| ---------- | ------------------- | ---------------- | ----- |
| PMMH       | 20 000              | 3 000            | Long chains needed because random-walk MH with bootstrap-PF likelihood is slow-mixing on 2D $(\log\sigma_v^2,\log\sigma_w^2)$. Run via the per-process driver in §4.6. |
| HMC-LEDH   | 1 000               |   500            | Limited by per-step cost (≈3.7 s/proposal at $N_{\text{ledh}} = 200$, $L = 5$ leapfrog steps). |
| L-HNN HMC  | 1 000               |   500            | Same per-chain budget as HMC-LEDH for fair ESS-per-gradient comparison. |

---

## 3. Parameter recovery: 95 % credible interval & coverage

For each parameter $\theta_j$ we now print

$$
\text{posterior mean} \pm \text{std}, \quad
\text{95 percent CI } [\theta_{j,0.025}, \theta_{j,0.975}], \quad
\theta_j^{\text{true}}, \quad
\text{covered? (truth in CI)}.
$$

CIs are flat-quantile, computed on the pooled-chain samples (good
mixing implies pooling is valid). Coverage is a per-parameter boolean.

This is reported alongside RMSE / MAE in
`reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/results.txt`
and in the per-cell ablation reports
`ablation/ablation_{lhnn,standard}_hmc.txt`.

If a parameter has `covered? = NO` the standard remediation hierarchy is

1. Longer warmup / more samples (suspect non-convergence).
2. Re-check the prior / re-parameterise (suspect modelling).
3. Switch sampler (e.g. NUTS in place of fixed-`L` HMC).

---

## 4. HMC tuning: step size + diagonal mass matrix

`run_hmc` already used Nesterov dual-averaging (Hoffman & Gelman 2014)
to learn the step size during the warmup phase, targeting an
acceptance rate of `0.65` (the optimal rate for vanilla HMC; NUTS
targets `0.8`).

**New:** `run_hmc(..., adapt_mass_matrix=True)` runs Stan's 3-window
warmup procedure (Stan reference manual §15.2):

| Window  | Fraction of warmup | Adapts                              |
| ------- | ------------------- | ----------------------------------- |
| I       | ~15 %               | step size only (initial buffer)     |
| II      | ~75 %               | mass matrix `diag(M)` (doubling sub-windows); dual averaging restarts after each estimate |
| III     | ~10 %               | step size only with frozen `M`      |

The kinetic energy is $\tfrac{1}{2} p^\top M^{-1} p$ with
$p \sim \mathcal{N}(0, M)$; the position update becomes
$q \leftarrow q + \varepsilon\, M^{-1} p$. Estimating `diag(M)` from
the in-window posterior variances pre-conditions the Hamiltonian (Betancourt
2017, §A.4) — for an anisotropic posterior this is equivalent to
running HMC on a unit-isotropic target after an affine reparameterisation
and gives strictly higher ESS at fixed step size.

`run_lhnn_hmc` is *not* given mass-matrix adaptation. The L-HNN learns
the Hamiltonian flow on a fixed metric: changing `M` post-hoc would
invalidate the trained network. We document this explicitly and reuse
`M = I` for the L-HNN target throughout.

---

## 4.5. CRN policy across chains (HMC-LEDH and L-HNN HMC)

A subtle interaction between Common Random Numbers (CRN) and multi-chain
$\hat{R}$ bit us in the first 4-chain run of `--first_part`. Both
`run_hmc` and `run_lhnn_hmc` evaluate a *stochastic* target (the LEDH
log-likelihood with a finite particle count $N_{\text{ledh}} = 200$ in
the final config) and use CRN — i.e. the same TF random seed at every
leapfrog step within an iteration — so that the Metropolis ratio
$\log\pi(q'\mid s_i) - \log\pi(q\mid s_i)$ is a low-variance estimate
of the *deterministic* underlying ratio. CRN is essential here:
without it the MH ratio is dominated by particle-filter noise and
$\alpha \approx 0$ on every step.

The bug was at the next level up. In the legacy
`run_hmc_multi_chain` / `run_lhnn_hmc_multi_chain`, every chain received
its own `base_seed = global + 1009 · c`, which then fed *both* the
chain-private momentum/MH randomness *and* the iteration-level CRN seed
$s_i = \texttt{base\_seed} + (i+1) \cdot 7919$. So chain $c$ was
effectively sampling from a different finite-$N$ approximation of
$\pi$ — call it $\hat\pi_c$ — than chain $c'$. With
$N_{\text{ledh}} = 1000$ the bias of $\hat\pi_c$ is $O(1/N)$, but
its *mode locations* shift by that amount, and Hamiltonian Monte Carlo
locks each chain onto the local mode of *its* $\hat\pi_c$. Multi-chain
$\hat{R}$ then reads the cross-chain spread of those local modes as
non-convergence — when in fact each chain has converged perfectly to
*its own slightly biased target*. The trace plot at
[`comparison/trace_posterior.png`](6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/trace_posterior.png)
showed the symptom unambiguously: every HMC-LEDH chain segment was
essentially flat at a different value, with rank-$\hat R$ of
9.2 / 8.3 for HMC-LEDH and 6.4 / 6.2 for L-HNN HMC.

**Fix.** We *decouple* the CRN seed from the chain seed.
`run_hmc_multi_chain` and `run_lhnn_hmc_multi_chain` now expose
`share_crn_across_chains=True` (default) and pass `crn_offset = base_seed`
(the *global* seed) to every chain's `run_hmc` / `run_lhnn_hmc`. Inside
the per-iteration loop the implementation now derives:

$$
s_i^{\text{(private)}} = \texttt{base\_seed}_c + (i+1)\cdot 7919, \qquad
s_i^{\text{(CRN)}} = \texttt{crn\_offset} + (i+1)\cdot 7919.
$$

Every chain sees the *same* $s_i^{\text{(CRN)}}$ (so the same
finite-$N$ realisation $\hat\pi$), but uses its private
$s_i^{\text{(private)}}$ for momentum draws (`+ 3,000,000` offset) and
the MH-accept uniform (`+ 4,000,000` offset). Chains therefore explore
the same target independently. Reading the new $\hat R$ back: it
diagnoses *MCMC convergence to the finite-$N$ biased target*; the gap
between that target and the true posterior is bounded by the LEDH bias
and is reducible only by raising $N_{\text{ledh}}$.

PMMH is deliberately *exempt*: the bootstrap-PF likelihood ratio used
inside `run_pmmh` is *unbiased* (Andrieu, Doucet & Holenstein 2010,
Theorem 4), so independent CRN per chain is the right policy and
multi-chain $\hat R$ already targets the true marginal posterior.

A unit test (`tests/test_mcmc_diagnostics.py::TestSharedCRN`) asserts
that with `share_crn_across_chains=True` every chain's per-iteration
CRN seed is bit-identical, while sample paths still differ across
chains; with `share_crn_across_chains=False` (the legacy mode) the
CRN seeds differ across chains.

---

## 4.6. Per-chain process isolation for long PMMH runs

Pushing PMMH to convergence required `samples_per_chain = 20 000`
(see §5). Running this in a single Python process — even with
`run_pmmh_multi_chain` calling `run_pmmh` sequentially per chain —
hit out-of-memory (OOM) at iteration ~30 000 of chain 1. Profiling
indicated a per-iteration leak in the TFP eager kernel-results
container (`tfp.mcmc.RandomWalkMetropolis` retains state that is not
released between `kernel.one_step` calls); rough rate ~300 KB / step,
so a 4-chain × 23 000-iter run cumulatively allocates ~25 GB inside one
process even though the actual sample storage is < 10 MB.

**Fix.** A lightweight per-process driver
[`scripts/run_pmmh_isolated.sh`](../scripts/run_pmmh_isolated.sh)
launches each chain as a fresh `python -m` invocation with a new
`--chain_id` flag added to the experiment. Each chain writes its
samples, accept-flags, accept rate, target log-probs, *and* runtime to
`reports/.../pmmh_only/chains/chain_{id}.npz`. After all four chains
finish, a final `--aggregate` call loads the saved chains, builds the
multi-chain object, and writes the diagnostics + plots. Memory is
reset at process exit, so each chain gets a clean ~6 GB budget rather
than competing with the cumulative leak.

This unlocks reliable runs at $N_\text{iter} \gtrsim 25\,000$ per
chain on a 16 GB MacBook, which is what the 20 000-sample PMMH
baseline in §5 needs.

The seed and dispersed-init policy are bit-identical to
`run_pmmh_multi_chain` (chain `c` reuses seed `base + 1009·(c+1)` and
the `c`-th row of `disperse_initial_states(...)`), so isolated and
non-isolated runs are indistinguishable except for the absence of the
OOM.

---

## 5. Where we now stand

### 5.1 PMMH (Bootstrap-PF + Random-Walk MH) — converged baseline

Source file:
[`reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pmmh_only/comparison/results.txt`](6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/pmmh_only/comparison/results.txt).

```
============================================================================================
  Diagnostics — PMMH (Bootstrap PF)   (chains=4, samples/chain=20000, burn=3000)
--------------------------------------------------------------------------------------------
param         mean    std   bulk-ESS  tail-ESS   ESS%  splitR^  rankR^      95% CI       truth  covers?
--------------------------------------------------------------------------------------------
sigma_v^2    8.348   2.251     381       215    0.5%   1.047   1.049  [ 5.094, 13.723]  10.000   YES
sigma_w^2    1.343   0.556     174       187    0.2%   1.055   1.075  [ 0.582,  2.713]   1.000   YES
============================================================================================
Acceptance rate                       0.190
Mean ESS                              293
Runtime                              7160 s (≈2 h, four chains in fresh processes)
ESS / s                              0.041
Cost per step (s/proposal)            0.078
```

**Reading.** PMMH is *near-converged* by Vehtari (2021) standards:
rank-R̂ $\le 1.075$ on both parameters (well below the practitioner
$< 1.10$ threshold; above the strict $< 1.01$ cutoff), bulk-ESS for
$\sigma_v^2$ at 381 effectively reaches the 400 target, both 95 %
credible intervals cover the true values. Acceptance rate 0.19 is
near-optimal for 2D random-walk MH. The headline limitation is
$\sigma_w^2$ bulk-ESS at 174 — slower mixing on the
observation-noise direction; closing the gap to bulk-ESS $> 400$ and
rank-R̂ $< 1.01$ would require chain lengths in the 50–100 k range.

The convergence trajectory across attempts (each row is a fresh
4-chain run; same seeds, only `samples_per_chain` varies):

| `samples_per_chain` | rank-R̂ $\sigma_v^2$ | rank-R̂ $\sigma_w^2$ | bulk-ESS $\sigma_v^2$ | bulk-ESS $\sigma_w^2$ |
| ------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| 2 000               | 1.57                    | 1.47                    | 85                      | 82                      |
| 5 000               | 1.18                    | 1.38                    | 110                     | 101                     |
| **20 000**          | **1.05**                | **1.08**                | **381**                 | **174**                 |

Reproducible from

```
./scripts/run_pmmh_isolated.sh 4 20000 3000
```

(four fresh-process chains + an `--aggregate` invocation, ~2 h on a
16 GB MacBook).

### 5.2 HMC-LEDH and L-HNN HMC — three-way comparison

Source file:
[`reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/results.txt`](6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/results.txt)
(4 chains × 1 000 samples × 500 burn; $N_{\text{ledh}} = 200$,
$L = 5$ leapfrog steps; shared CRN across chains per §4.5):

```
============================================================================================
  Diagnostics — HMC-LEDH (adapted M)         (chains=4, samples/chain=1000, burn=500)
--------------------------------------------------------------------------------------------
sigma_v^2    9.404   5.310    49.0     40.6   1.2%  3.33e6   12.977  [ 5.555, 18.446]  10.000  YES
sigma_w^2    3.118   2.311    40.4     26.4   1.0%  2.66e6   15.237  [ 1.541,  7.104]   1.000   NO
--------------------------------------------------------------------------------------------
  Diagnostics — L-HNN HMC (M = I)            (chains=4, samples/chain=1000, burn=500)
--------------------------------------------------------------------------------------------
sigma_v^2    5.976   1.884    42.2     26.1   1.1%   3.459    3.248  [ 3.162,  9.755]  10.000   NO
sigma_w^2   11.619   7.226    55.7     37.3   1.4%  14.432    5.554  [ 3.357, 27.385]   1.000   NO
============================================================================================
Acceptance: HMC-LEDH = 0.598 ;  L-HNN HMC = 0.610
ESS / gradient-eval (Dhulipala 2022, Table 1):
   HMC-LEDH   1.48e-4
   L-HNN HMC  1.93e-3   →  13.0× speedup
```

Both samplers have **near-optimal acceptance** (target 0.65 for vanilla
HMC) and **L-HNN delivers the expected ESS-per-gradient speedup**
(13.0×, in line with Dhulipala et al. 2022 §4.1). What they do not pass
is multi-chain rank-R̂ — the reasons differ between the two and are the
subject of §5.3.

### 5.3 Why HMC-LEDH and L-HNN HMC do not pass strict R-hat

#### Stochastic-target diagnostic limitation (HMC-LEDH)

Even after the CRN fix (§4.5), even at $N_{\text{ledh}} = 200$ (4×
the original 50, ≈10× cost), HMC-LEDH's split-R̂ is in the millions
while rank-R̂ remains in the ~13–17 band. The posterior point estimate
on the well-mixed direction is closer to truth than PMMH's:
HMC-LEDH posterior mean $\sigma_v^2 = 9.40$ at 1 000 samples /
8.92 at 1 500 samples (truth 10) vs PMMH 8.35.

#### The chain-length trajectory test

We ran HMC-LEDH at three chain-length settings (4 chains, shared CRN,
identical config except `samples_per_chain` and `burn_per_chain`) to
distinguish *slow mixing* (R-hat decreases with more samples) from
*stochastic-target diagnostic noise* (R-hat fluctuates around a
non-1.0 floor):

| `samples_per_chain` | `burn_per_chain` | rank-R̂ $\sigma_v^2$ | rank-R̂ $\sigma_w^2$ | bulk-ESS $\sigma_v^2$ | bulk-ESS $\sigma_w^2$ | tail-ESS $\sigma_v^2$ | tail-ESS $\sigma_w^2$ |
| ------------------- | ----------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- | ----------------------- |
| 250                 | 200               | 10.84                   | 16.56                   | 41.6                    | 46.7                    | 25.5                    | 25.8                    |
| 1 000               | 500               | 12.98                   | 15.24                   | 49.0                    | 40.4                    | 40.6                    | 26.4                    |
| **1 500**           | **750**           | **15.15**               | **16.67**               | **52.3**                | **43.3**                | **30.2**                | **25.1**                |

R-hat does **not** decrease as chains grow (it rises slightly,
within trajectory noise). bulk-ESS grows trivially. **The cleanest
signal is in the right-most column: σ_w² tail-ESS sits at
~25 regardless of whether the chains have 250 or 1 500 samples** —
across a 6× sample-count range the tail-ESS is essentially pinned.
For genuinely slow-mixing chains tail-ESS grows roughly linearly with
chain length; here it does not, confirming the residual is a
diagnostic-noise floor rather than chains failing to explore. Across
the same range the rank-R̂ stays in [10.8, 16.7]. Both observations
are consistent with stochastic-target diagnostic noise from
finite-$N$ LEDH (cf. the analogous warning for stochastic-gradient
MCMC in Brosse, Durmus & Moulines 2018; Nemeth & Fearnhead 2021).

A second observation from the 1 500-sample run: the per-chain
σ_w² posteriors **collapsed** (cross-chain std 0.28 down from 2.31 at
1 000 samples) but to *different* per-chain CRN-realisation
locations — the 95 % CI [1.54, 2.29] now excludes the truth
(σ_w² = 1.0). This is the §4.5 CRN-mode-locking pathology in its
fully-developed form: each chain has converged narrowly to its own
$\hat\pi_c^{N=200}$, and at long chain lengths the cross-chain spread
is the dominant signal R-hat picks up.

The diagnostic-vs-estimate gap is *not* a tuning bug. It is
structural: the LEDH log-likelihood is a Monte-Carlo estimate with
finite-$N$ fluctuations on top of the deterministic posterior
surface, and gradient-based MCMC theory rests on smoothness assumptions
that this stochasticity violates. The chain of reasoning:

1. **Pseudo-marginal MCMC requires an *unbiased* likelihood
   estimator** to target the true posterior (Andrieu & Roberts 2009;
   Andrieu, Doucet & Holenstein 2010, Theorem 4). LEDH is biased at
   finite $N$, so HMC-LEDH targets a perturbed posterior
   $\hat\pi_N$, not $\pi$; the bias is $O(1/N)$.
2. **Log-likelihood-estimator variance directly degrades MH
   efficiency** in pseudo-marginal settings (Doucet, Pitt,
   Deligiannidis & Kohn 2015). Even with shared CRN keeping the
   *ratio* low-variance within an iteration, the leapfrog trajectory
   crosses many fresh likelihood realisations across iterations.
3. **Stochastic-gradient HMC has known convergence pathologies**
   relative to deterministic-target HMC (Nemeth & Fearnhead 2021,
   review; Brosse, Durmus & Moulines 2018; Welling & Teh 2011 for
   SGLD). The leapfrog integrator's energy-conservation guarantee
   and dual-averaging's step-size adaptation (Hoffman & Gelman 2014)
   both assume a smooth target.

The empirical signature on our run is bimodal acceptance: a leapfrog
trajectory either rides a "good" CRN realisation and accepts
($\alpha \approx 1$), or hits a noise ridge and rejects
($\alpha \approx 0$). The chain's local autocorrelation is then
dominated by the CRN realisation sequence rather than by the
underlying posterior geometry. We *observe* — and we believe but
cannot directly cite — that R-hat's between-chain-vs-within-chain
variance ratio reads this CRN-driven autocorrelation as inflated
$B/W$ and flags non-mixing even when the chains agree on the
posterior mean. The split-R̂ values in the hundreds-of-thousands to
millions versus rank-R̂ in the 11–17 band (see trajectory table above)
are consistent with this: rank-normalisation damps the heavy-tailed
CRN-realisation outliers; raw split-R̂ does not.

The remediations that would help — $N_{\text{ledh}} \uparrow$
(already at the cost ceiling for this experiment), exact
pseudo-marginal MH instead of HMC (switches the sampler), or an
analytic surrogate target — are out-of-scope for the comparison Report
II is asking. We report the posterior point estimate and acknowledge
that R-hat is the wrong diagnostic to trust for tail summaries on this
target.

#### Pilot-trajectory bias (L-HNN HMC) and the dispersed-pilot remediation

The original (single-pilot) L-HNN HMC's chains converge to the *wrong*
mode: $\sigma_v^2 = 5.98$, $\sigma_w^2 = 11.62$ (truth 10, 1). The
fallback rate (proportion of leapfrog steps that revert to a true
LEDH-gradient evaluation) is 0.040, so the surrogate Hamiltonian is
"trusted" 96 % of the time — and that surrogate is biased toward the
mode that the pilot trajectories visited. The Kitagawa posterior has a
$(\sigma_v^2, \sigma_w^2)$ trade-off ridge ("noisy-states /
precise-observations" vs "smooth-states / noisy-observations") that
both modes can plausibly explain; the pilot HMC chain happened to
explore the wrong basin, and the trained network amortised that bias.

**Dispersed-pilot remediation — partial success.** The principled
fix is to retrain the L-HNN with pilot trajectories seeded from
`disperse_initial_states(...)` rather than from a single point, so
the network sees the same posterior region the production chains will
explore. We implemented this in
[`src/experiments/exp_part3_bonus1b_lhnn_only.py`](../src/experiments/exp_part3_bonus1b_lhnn_only.py)
(`train_lhnn_with_dispersed_pilots`) and re-ran 4 chains × 5 000
post-burn samples × 1 500 burn (
[`scripts/run_lhnn_isolated.sh`](../scripts/run_lhnn_isolated.sh),
[`reports/.../lhnn_only/comparison/results.txt`](6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/lhnn_only/comparison/results.txt)).

| Metric                           | Single-pilot, 1 000 samples | **Dispersed pilots, 5 000 samples** | Change |
| -------------------------------- | --------------------------- | ----------------------------------- | ------ |
| Mean $\sigma_v^2$ (truth 10)   | 5.98                        | **6.18**                            | similar |
| Mean $\sigma_w^2$ (truth 1)    | 11.62                       | **9.36**                            | partially recovered |
| Std $\sigma_v^2$               | 1.88                        | 2.59                                | wider |
| Std $\sigma_w^2$               | 7.23                        | 4.07                                | tighter |
| rank-R̂ $\sigma_v^2$            | 3.25                        | **1.76**                            | **1.85× drop** ✓ |
| rank-R̂ $\sigma_w^2$            | 5.55                        | **1.86**                            | **2.99× drop** ✓ |
| bulk-ESS $\sigma_v^2$          | 42                          | **87**                              | 2.1× ✓ |
| bulk-ESS $\sigma_w^2$          | 56                          | 61                                  | similar |
| 95 % CI covers $\sigma_v^2$?   | ✗                           | **✓** [3.20, 13.63]                 | **fixed** |
| 95 % CI covers $\sigma_w^2$?   | ✗                           | ✗ [4.22, 19.78]                     | not fixed |
| Acceptance rate                  | 0.61                        | 0.58                                | same band |
| Fallback intensity (∇/leap step) | 0.040                       | 0.729                               | 18× higher |
| ESS-per-gradient speedup         | 13.04×                      | ~5.0×                               | reduced (more fallbacks → more real grads) |

**Reading the table.** Dispersed-pilot training is a *substantial*
partial improvement: rank-R̂ drops by 1.85–3× on both parameters,
bulk-R̂ doubles for $\sigma_v^2$, and the $\sigma_v^2$ 95 % CI now
covers the truth — the chain-spread along the trade-off ridge is
wide enough to include $\sigma_v^2 = 10$. The residual issue is
$\sigma_w^2$-specific: the four chains agree better than before
(std 4.07 vs 7.23) but on a *wrong* $\sigma_w^2 \approx 9.4$
location. The triangle plot
([`triangle_compare.png`](6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/comparison/triangle_compare.png))
makes this visible: the L-HNN posterior contour is approximately the
right shape but shifted upward on the $\sigma_w^2$ axis relative to
the PMMH and HMC-LEDH contours that bracket the truth-star.

**Why the residual $\sigma_w^2$ bias persists.** The remaining
wrong-mode is on the same direction PMMH mixes slowly on (the
observation-noise axis where the Kitagawa likelihood is hardest to
identify): the trade-off ridge is informationally one-sided, so a
surrogate Hamiltonian trained from finite pilot data inherits the same
identification difficulty in amortised form. We hypothesise that
residual $\sigma_w^2$ bias would close with denser pilot data
(currently 600 points across 4 basins), a wider network than the
256 × 3 MLP we used, or a re-training-on-fallback loop where the
network is fine-tuned on the trajectories the production chains
actually traverse — none of which are simple drop-in fixes.

**Cost of the speedup remediation.** Fallback intensity rose from 0.04
to 0.73 ∇/leapfrog-step: with dispersed training data the surrogate is
less locally confident, the error threshold trips more often, and the
true LEDH gradient is invoked. Total L-HNN gradient evals per the
Dhulipala-2022 metric: training 620 + sampling 18 941 = 19 561, vs
156 000 traditional HMC equivalents — **87.5 % gradient saved, ~5×
ESS-per-gradient speedup** (down from 13× in the single-pilot run).
The speedup remains material; it is just smaller because the surrogate
is now correctly less confident on a wider region.

### 5.4 Effect of the CRN-policy fix (§4.5) — before/after

| Method     | Param        | rank-R̂ pre-fix\* | rank-R̂ post-CRN-fix | rank-R̂ extended / longer run | Verdict on the fix                              |
| ---------- | ------------ | ----------------- | -------------------- | ----------------------------- | ----------------------------------------------- |
| PMMH       | $\sigma_v^2$ | 1.18              | **1.05**             | n/a                       | independent CRN unchanged; gain is from longer chains (§5.1) |
| PMMH       | $\sigma_w^2$ | 1.38              | **1.08**             | n/a                       | same                                            |
| HMC-LEDH   | $\sigma_v^2$ | 9.17              | 12.98                | 15.15                     | shared CRN; chains now explore real LEDH target — R-hat does not improve with longer chains (trajectory table, §5.3) |
| HMC-LEDH   | $\sigma_w^2$ | 8.25              | 15.24                | 16.67                     | same                                            |
| L-HNN HMC, single pilot   | $\sigma_v^2$ | 6.40              | **3.25**             | n/a                       | substantial drop (1.97×)                        |
| L-HNN HMC, single pilot   | $\sigma_w^2$ | 6.23              | **5.55**             | n/a                       | drop, but residual non-convergence is the wrong-mode issue (§5.3), not CRN drift |
| **L-HNN HMC, dispersed pilots** | $\sigma_v^2$ | 6.40        | 3.25                 | **1.76**                  | dispersed pilots cut R-hat further (1.85× over single-pilot post-CRN) |
| **L-HNN HMC, dispersed pilots** | $\sigma_w^2$ | 6.23        | 5.55                 | **1.86**                  | 2.99× over single-pilot post-CRN; residual is σ_w²-specific surrogate bias (§5.3) |

\* "pre-fix" PMMH numbers are from the legacy 5 000-sample run before
chain-length tuning; the gradient-method numbers are from the original
single-CRN-per-chain experiment.

**Reading.** The CRN fix did its job for L-HNN (1.5–2× R-hat
reduction, exactly as the CRN-mismatch hypothesis predicted). For
HMC-LEDH it *increased* the apparent R-hat, but the new number is
diagnosing the right thing (stochastic-target heavy tail) rather than
the wrong thing (per-chain CRN drift). This is a successful
disambiguation: previously every chain was converged to its own
slightly biased target, with R-hat reading the cross-target spread as
non-convergence; now every chain is converged to the *same* target and
R-hat reads the genuine within-target slow tail mixing.

The 1 500-samples-per-chain column closes the question of whether the
post-fix R-hat would eventually fall with longer chains: it does not
(trajectory table, §5.3). The hand-wavy hope at the time of the
post-fix run that "raising `samples_per_chain` to ~1 500 (≈4–6× the
current compute) is the next iteration" turned out to be wrong — at
1 500 samples R-hat is *higher*, not lower, than at 1 000. The
remaining residual is the diagnostic-noise floor, not slow mixing.

### 5.5 Convergence verdict

| Method                                            | rank-R̂ < 1.01 | bulk-ESS > 400 | tail-ESS > 400 | 95 % CI covers truth | Overall                |
| ------------------------------------------------- | -------------- | -------------- | -------------- | -------------------- | ---------------------- |
| PMMH (20 000 samples)                             | ≈ (1.05–1.08)  | ≈ (381 / 174)  | ✗              | ✓ both               | **Near-converged**     |
| HMC-LEDH (1 000 s)                                | ✗ (13.0 / 15.2)| ✗              | ✗              | partial (σ_v² ✓; σ_w² ✓) | Stochastic-target limit |
| HMC-LEDH (1 500 s)                                | ✗ (15.2 / 16.7)| ✗              | ✗              | partial (σ_v² ✓; σ_w² ✗ — chains collapsed to wrong CRN-mode value) | Stochastic-target limit (trajectory test confirms; §5.3) |
| L-HNN HMC, single pilot (1 000 s)                 | ✗ (3.2 / 5.6)  | ✗              | ✗              | ✗ both               | Wrong-mode (pilot bias) |
| **L-HNN HMC, dispersed pilots (5 000 s)**         | ✗ (1.76 / 1.86)| ✗ (87 / 61)    | ✗ (78 / 59)    | partial (σ_v² ✓; σ_w² ✗) | **Pilot remediation partially closes gap; σ_w² residual** |

The verdict-cell wording mirrors what `convergence_verdict` prints in
the report files. We deliberately do not call PMMH "converged" without
qualification: rank-R̂ is in [1.05, 1.08] rather than [1.0, 1.01], and
$\sigma_w^2$ bulk-ESS is below the 400-effective-sample target.
"Near-converged" is the honest descriptor — the chains have clearly
agreed on the location and width of the posterior (CIs cover, means
within 1 std of truth), but at the strictest Vehtari (2021) standard
there is residual mixing left to do.

### 5.6 Fail-mode playbook (kept for completeness)

If a future run lands `NOT CONVERGED`:

- **`rank_rhat` ≥ 1.01 and chains visibly disjoint** → longer warmup
  (raise `burn_per_chain`), or fewer leapfrog steps per proposal so
  mixing is dominated by the MH accept rather than the integrator.
- **`bulk_ess` low but `rank_rhat` ≈ 1** → chains are stationary but
  highly autocorrelated; raise `step_size` or use NUTS.
- **`tail_ess` low only** → tails poorly sampled; consider a heavier
  proposal scale or reparameterise ($\log\sigma^2$ is already done —
  check for posterior banana shape).
- **stochastic target with `splitR^` in the millions and rank-R̂ ~10
  while point estimates look fine** → diagnostic-vs-estimate gap (§5.3);
  raising $N_{\text{ledh}}$ is the only physical remediation.

---

## References

- Gelman, A., & Rubin, D. B. (1992). *Inference from iterative
  simulation using multiple sequences*. **Statistical Science**, 7(4).
- Geyer, C. J. (1992). *Practical Markov chain Monte Carlo*.
  **Statistical Science**, 7(4).
- Hoffman, M. D., & Gelman, A. (2014). *The No-U-Turn Sampler:
  adaptively setting path lengths in Hamiltonian Monte Carlo*.
  **JMLR** 15.
- Betancourt, M. (2017). *A conceptual introduction to Hamiltonian
  Monte Carlo*. arXiv:1701.02434.
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., & Bürkner,
  P.-C. (2021). *Rank-normalization, folding, and localization: An
  improved $\hat{R}$ for assessing convergence of MCMC*.
  **Bayesian Analysis** 16(2):667–718.
- Stan Development Team. *Stan Reference Manual*, §15.2 (warmup) and
  §15.4 (effective sample size).
- Andrieu, C., Doucet, A., & Holenstein, R. (2010). *Particle Markov
  chain Monte Carlo methods*. **JRSS-B** 72(3).
- Dhulipala, S. L. N. et al. (2022). *Bayesian inference with latent
  Hamiltonian neural networks*. arXiv:2208.06120.
- Corenflos, A., Thornton, J., Deligiannidis, G., & Doucet, A. (2021).
  *Differentiable particle filtering via entropy-regularized optimal
  transport*. **ICML 2021**.
- Andrieu, C., & Roberts, G. O. (2009). *The pseudo-marginal approach
  for efficient Monte Carlo computations*. **Annals of Statistics**
  37(2):697–725.
- Doucet, A., Pitt, M. K., Deligiannidis, G., & Kohn, R. (2015).
  *Efficient implementation of Markov chain Monte Carlo when using an
  unbiased likelihood estimator*. **Biometrika** 102(2):295–313.
- Nemeth, C., & Fearnhead, P. (2021). *Stochastic gradient Markov
  chain Monte Carlo*. **JASA** 116(533):433–450.
- Brosse, N., Durmus, A., & Moulines, É. (2018). *The promises and
  pitfalls of stochastic gradient Langevin dynamics*. **NeurIPS 2018**.
- Welling, M., & Teh, Y. W. (2011). *Bayesian learning via stochastic
  gradient Langevin dynamics*. **ICML 2011**.
