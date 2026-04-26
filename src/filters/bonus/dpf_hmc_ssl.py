"""
DPF-HMC inference for Gaussian State-Space LSTM models.

Modular implementation of the full DPF-HMC pipeline for SSL inference:

1. **EmissionParameterizer** — packs/unpacks (C, b, log_R_diag) as a flat
   vector for HMC sampling.
2. **DifferentiableSSLAdapter** — SSM adapter where emission parameters
   (C, b, R) remain differentiable tensors so gradients flow from the
   LEDH filter log-likelihood back to the HMC state.
3. **DPFHMCTarget** — builds the log-posterior closure for HMC:
   log p(x_{1:T} | theta_emission, omega_LSTM_fixed) + log p(theta).
4. **warmup_em** — EM warm-up that trains the full SSL (LSTM + emission)
   and refines the reference trajectory via forward messages.
5. **dpf_hmc_inference** — complete pipeline: warm-up -> freeze LSTM ->
   HMC over emission parameters.
6. **pmmh_ssl_inference** — PMMH baseline for comparison.

The key insight: after warm-up trains the LSTM, we fix it and run HMC
over the low-dimensional emission parameters (C, b, log_R_diag) using
the differentiable LEDH filter to compute grad log p(x | theta).

References
----------
- Li & Coates (2017), "Particle Filtering with Invertible Particle Flow"
- Corenflos et al. (2021), "Differentiable Particle Filtering via
  Entropy-Regularized Optimal Transport"
- Zheng et al. (2017), "State Space LSTM Models with Particle MCMC
  Inference"
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import tensorflow as tf

from src.models.gaussian_ssl import GaussianSSL, GaussianSSLasSSM
from src.filters.bonus.differentiable_ledh import DifferentiableLEDHLogLikelihood
from src.filters.bonus.hmc_pf import run_hmc
from src.filters.bonus.pmmh import bootstrap_pf_log_likelihood


# =========================================================================
# Emission Parameterizer
# =========================================================================

class EmissionParameterizer:
    """Pack/unpack SSL emission parameters as a flat tensor for HMC.

    Supports two modes:
      - fix_C=False: sample (C, b, log_R_diag), dim = obs*state + obs + obs
      - fix_C=True:  sample (b, log_R_diag) only, dim = obs + obs
        C is fixed at the value provided via set_fixed_C().

    For a 2D SSL with fix_C=True, total dimension = 2 + 2 = 4.
    """

    def __init__(self, obs_dim: int, state_dim: int, fix_C: bool = False):
        """Initialise emission parameterizer for given obs/state dimensions."""
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.fix_C = fix_C
        self.c_size = obs_dim * state_dim
        self.b_size = obs_dim
        self.r_size = obs_dim
        if fix_C:
            self.total_dim = self.b_size + self.r_size
            self._fixed_C: Optional[tf.Tensor] = None
        else:
            self.total_dim = self.c_size + self.b_size + self.r_size

    def set_fixed_C(self, C: tf.Tensor):
        """Store the fixed C matrix (call after warm-up)."""
        self._fixed_C = tf.stop_gradient(tf.identity(C))

    def pack(self, C: tf.Tensor, b: tf.Tensor, log_R_diag: tf.Tensor) -> tf.Tensor:
        """Flatten emission params into a single vector."""
        if self.fix_C:
            return tf.concat([b, log_R_diag], axis=0)
        return tf.concat([tf.reshape(C, [-1]), b, log_R_diag], axis=0)

    def unpack(self, theta: tf.Tensor):
        """Unpack flat vector into (C, b, log_R_diag).

        Returns tensors derived from theta so gradients flow through.
        When fix_C=True, C is returned as a constant (no gradient).
        """
        if self.fix_C:
            b = theta[:self.b_size]
            log_R_diag = theta[self.b_size:self.b_size + self.r_size]
            return self._fixed_C, b, log_R_diag
        idx = 0
        C = tf.reshape(theta[idx:idx + self.c_size],
                        [self.obs_dim, self.state_dim])
        idx += self.c_size
        b = theta[idx:idx + self.b_size]
        idx += self.b_size
        log_R_diag = theta[idx:idx + self.r_size]
        return C, b, log_R_diag

    def extract_from_ssl(self, ssl: GaussianSSL) -> tf.Tensor:
        """Read current emission params from an SSL model."""
        if self.fix_C:
            self.set_fixed_C(ssl.C)
        return self.pack(ssl.C, ssl.b, ssl.log_R_diag)

    def apply_to_ssl(self, ssl: GaussianSSL, theta: tf.Tensor):
        """Write emission params to an SSL model (breaks gradients — EM only)."""
        C, b, log_R_diag = self.unpack(theta)
        ssl.C.assign(C)
        ssl.b.assign(b)
        ssl.log_R_diag.assign(log_R_diag)


# =========================================================================
# Differentiable SSM Adapter
# =========================================================================

class DifferentiableSSLAdapter:
    """SSM adapter where emission params are differentiable tensors.

    Unlike GaussianSSLasSSM (which eagerly computes R at construction),
    this adapter keeps C, b, log_R_diag as live tensors so that
    tf.GradientTape can differentiate the LEDH filter's output w.r.t.
    the emission parameters.

    The LSTM transition parameters (mus, sigmas) are pre-computed
    constants from the warm-up phase.

    Parameters
    ----------
    mus : tf.Tensor
        (T, state_dim) — pre-computed LSTM transition means.
    sigmas : tf.Tensor
        (T, state_dim) — pre-computed LSTM transition stds.
    C : tf.Tensor
        (obs_dim, state_dim) — emission matrix (differentiable).
    b : tf.Tensor
        (obs_dim,) — emission bias (differentiable).
    log_R_diag : tf.Tensor
        (obs_dim,) — log-diagonal of emission covariance (differentiable).
    state_dim : int
    obs_dim : int
    """

    def __init__(
        self,
        mus: tf.Tensor,
        sigmas: tf.Tensor,
        C: tf.Tensor,
        b: tf.Tensor,
        log_R_diag: tf.Tensor,
        state_dim: int,
        obs_dim: int,
    ):
        """Initialise adapter from pre-computed LSTM transitions and emission params."""
        self.state_dim = state_dim
        self.meas_per_landmark = obs_dim
        self._obs_dim = obs_dim
        self._mus = mus
        self._C = C
        self._b = b

        # Q from transition variance (constant, no gradient needed)
        mean_var = tf.reduce_mean(sigmas ** 2, axis=0)
        self.Q = tf.linalg.diag(tf.stop_gradient(mean_var))

        # R is differentiable w.r.t. log_R_diag
        self.R = tf.linalg.diag(tf.exp(log_R_diag))
        self.initial_var = 5.0

    def full_measurement_cov(self, num_landmarks=1):
        """Return the observation noise covariance R."""
        return self.R

    def motion_model(self, state, control):
        """Pre-computed LSTM transition means, indexed by timestep."""
        state = tf.cast(state, tf.float32)
        control = tf.cast(control, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        if len(control.shape) == 1:
            control = control[tf.newaxis, :]

        t_idx = tf.cast(tf.round(control[:, 0]), tf.int32)
        t_idx = tf.clip_by_value(t_idx, 0, tf.shape(self._mus)[0] - 1)
        return tf.gather(self._mus, t_idx)

    def measurement_model(self, state, landmarks=None):
        """Linear emission: x = C z + b. Differentiable w.r.t. C, b."""
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        return tf.matmul(state, self._C, transpose_b=True) + self._b

    def motion_jacobian(self, state, control):
        """Zero — transition mean is pre-computed, not a function of state."""
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        B = tf.shape(state)[0]
        return tf.zeros([B, self.state_dim, self.state_dim])

    def measurement_jacobian(self, state, landmarks=None):
        """d(Cz+b)/dz = C. Differentiable w.r.t. C."""
        state = tf.cast(state, tf.float32)
        if len(state.shape) == 1:
            state = state[tf.newaxis, :]
        B = tf.shape(state)[0]
        return tf.broadcast_to(
            self._C[tf.newaxis, :, :],
            [B, self._obs_dim, self.state_dim],
        )


# =========================================================================
# Pre-compute LSTM transition parameters
# =========================================================================

def precompute_lstm_params(
    ssl: GaussianSSL,
    z_ref: tf.Tensor,
) -> tuple:
    """Run LSTM forward along z_ref and store per-timestep (mu, sigma).

    These are constants once LSTM weights are frozen.

    Returns
    -------
    mus    : (T, state_dim) transition means
    sigmas : (T, state_dim) transition stds
    """
    T = z_ref.shape[0]
    d_z = ssl.state_dim_val
    lstm_state = ssl.get_initial_lstm_state(batch_size=1)
    z_prev = tf.zeros([1, d_z])

    mus = tf.TensorArray(dtype=tf.float32, size=T)
    sigmas = tf.TensorArray(dtype=tf.float32, size=T)

    for t in range(T):
        mu, sigma, lstm_state = ssl.transition_params(lstm_state, z_prev)
        mus = mus.write(t, mu[0])
        sigmas = sigmas.write(t, sigma[0])
        z_prev = z_ref[t:t + 1]

    return mus.stack(), sigmas.stack()


# =========================================================================
# HMC Target
# =========================================================================

class DPFHMCTarget:
    """Differentiable log-posterior for HMC over emission parameters.

    log pi(theta) = log p_hat(x_{1:T} | theta, omega_fixed) + log p(theta)

    where theta = (C, b, log_R_diag) and omega = LSTM weights (frozen).
    The LEDH filter provides the differentiable log-likelihood estimate.

    Parameters
    ----------
    mus, sigmas : tf.Tensor
        Pre-computed LSTM transition params (from precompute_lstm_params).
    x_obs : tf.Tensor
        (T, obs_dim) observations.
    state_dim, obs_dim : int
        Dimensions.
    param : EmissionParameterizer
        For unpacking theta.
    n_particles : int
        Number of particles for LEDH filter.
    n_lambda : int
        Number of pseudo-time flow steps.
    sinkhorn_epsilon : float
        Entropy regularization for OT resampling.
    sinkhorn_iters : int
        Sinkhorn iterations.
    grad_window : int
        Stop-gradient window size.
    prior_scale : float
        Std of Gaussian prior on emission params.
    """

    def __init__(
        self,
        mus: tf.Tensor,
        sigmas: tf.Tensor,
        x_obs: tf.Tensor,
        state_dim: int,
        obs_dim: int,
        param: EmissionParameterizer,
        theta_init: Optional[tf.Tensor] = None,
        n_particles: int = 50,
        n_lambda: int = 3,
        sinkhorn_epsilon: float = 2.0,
        sinkhorn_iters: int = 20,
        grad_window: int = 5,
        prior_scale: float = 2.0,
    ):
        """Initialise HMC target; see class docstring for parameter details."""
        self.mus = tf.stop_gradient(mus)
        self.sigmas = tf.stop_gradient(sigmas)
        self.x_obs = x_obs
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.param = param
        self.prior_scale = prior_scale
        self.theta_init = tf.stop_gradient(theta_init) if theta_init is not None else tf.zeros(param.total_dim)

        self.filt = DifferentiableLEDHLogLikelihood(
            num_particles=n_particles,
            n_lambda=n_lambda,
            sinkhorn_epsilon=sinkhorn_epsilon,
            sinkhorn_iters=sinkhorn_iters,
            grad_window=grad_window,
            jit_compile=False,
        )

    def log_prior(self, theta: tf.Tensor) -> tf.Tensor:
        """Informative Gaussian prior centered at warm-up initialization.

        When fix_C=True (default), C is not sampled so only b and log_R
        get priors. When fix_C=False, C gets a tight prior (scale=0.3)
        to prevent non-identifiability drift with the LSTM.
        """
        C, b, log_R_diag = self.param.unpack(theta)
        _, b_init, logR_init = self.param.unpack(self.theta_init)

        lp = tf.constant(0.0)
        # C prior (only when C is being sampled)
        if not self.param.fix_C:
            C_init, _, _ = self.param.unpack(self.theta_init)
            c_scale = 0.3
            lp = lp - 0.5 * tf.reduce_sum((C - C_init) ** 2) / (c_scale ** 2)
        # b prior: centered at warm-up b
        lp = lp - 0.5 * tf.reduce_sum((b - b_init) ** 2) / (self.prior_scale ** 2)
        # log_R prior: same scale as b so the chain can escape a poor warm-up R
        lp = lp - 0.5 * tf.reduce_sum((log_R_diag - logR_init) ** 2) / (self.prior_scale ** 2)
        return lp

    def __call__(self, theta: tf.Tensor) -> tf.Tensor:
        """Compute log p_hat(x | theta) + log p(theta).

        Gradients flow from the LEDH filter output through the
        DifferentiableSSLAdapter back to theta.

        Includes a soft boundary: if any parameter is too far from
        initialization, returns -inf to prevent the chain from
        wandering into numerically unstable filter regions.
        """
        C, b, log_R_diag = self.param.unpack(theta)

        # Soft boundary: penalize extreme values that destabilize LEDH
        boundary_penalty = tf.constant(0.0)
        if not self.param.fix_C:
            max_C = tf.reduce_max(tf.abs(C))
            boundary_penalty = boundary_penalty + tf.maximum(max_C - 2.0, 0.0) ** 2 * 200.0
        max_logR = tf.reduce_max(tf.abs(log_R_diag))
        boundary_penalty = boundary_penalty + tf.maximum(max_logR - 3.0, 0.0) ** 2 * 100.0

        adapter = DifferentiableSSLAdapter(
            mus=self.mus,
            sigmas=self.sigmas,
            C=C,
            b=b,
            log_R_diag=log_R_diag,
            state_dim=self.state_dim,
            obs_dim=self.obs_dim,
        )

        ll = self.filt(adapter, self.x_obs)

        # Guard against filter numerical failures
        ll = tf.where(tf.math.is_finite(ll), ll, tf.constant(-1e6, tf.float32))

        return ll + self.log_prior(theta) - boundary_penalty


# =========================================================================
# Filtering utilities (pure TF)
# =========================================================================

def compute_filtered_trajectory(
    ssl: GaussianSSL,
    x_obs: tf.Tensor,
    z_ref: tf.Tensor,
    n_particles: int = 50,
) -> tf.Tensor:
    """Compute filtered z trajectory using particle-based forward filtering.

    Uses multiple particles with the optimal proposal (gamma) and resampling
    to avoid error compounding that occurs with single-chain roll-outs.

    All P particles are batched through a single forward_messages call per
    timestep (no per-particle Python loop).

    Parameters
    ----------
    ssl : GaussianSSL
    x_obs : (T, obs_dim)
    z_ref : (T, state_dim) — reference trajectory (unused, kept for API compat)
    n_particles : int — number of filtering particles

    Returns
    -------
    z_filtered : (T, state_dim) — weighted mean of particles at each step
    """
    T = x_obs.shape[0]
    d_z = ssl.state_dim_val
    P = n_particles

    # Batched LSTM state: (P, lstm_units) for both h and c
    lstm_state = ssl.get_initial_lstm_state(batch_size=P)  # tuple of (P, H)
    z_prev_all = tf.zeros([P, d_z])

    z_out = tf.TensorArray(dtype=tf.float32, size=T)

    for t in range(T):
        # Broadcast x_t to (P, obs_dim)
        x_t = tf.broadcast_to(x_obs[t:t + 1], [P, ssl.obs_dim_val])

        # Single batched call for all P particles
        alphas, gamma_mus, gamma_vars, new_lstm_state = ssl.forward_messages(
            lstm_state, z_prev_all, x_t
        )
        # alphas: (P,), gamma_mus: (P, d_z), gamma_vars: (P, d_z, d_z)

        # Batched sampling from optimal proposal: z ~ N(gamma_mu, gamma_var)
        L = tf.linalg.cholesky(gamma_vars + tf.eye(d_z) * 1e-6)  # (P, d_z, d_z)
        eps = tf.random.normal([P, d_z, 1])
        z_new = gamma_mus + tf.squeeze(tf.matmul(L, eps), axis=-1)  # (P, d_z)

        # Weighted mean as filtered estimate
        log_w = alphas - tf.reduce_logsumexp(alphas)
        weights = tf.exp(log_w)  # (P,)
        z_mean = tf.reduce_sum(weights[:, tf.newaxis] * z_new, axis=0)
        z_out = z_out.write(t, z_mean)

        # Resample particles (multinomial) — all tensor ops, no .numpy()
        if t < T - 1:
            indices = tf.cast(
                tf.random.categorical(log_w[tf.newaxis, :], P)[0],
                tf.int32,
            )
            z_prev_all = tf.gather(z_new, indices)
            # Resample LSTM states via tf.gather (batched tensors)
            h, c = new_lstm_state
            lstm_state = (tf.gather(h, indices), tf.gather(c, indices))
        else:
            z_prev_all = z_new
            lstm_state = new_lstm_state

    return z_out.stack()


# =========================================================================
# Warm-up (EM)
# =========================================================================

def warmup_em(
    ssl: GaussianSSL,
    x_obs: tf.Tensor,
    *,
    n_outer: int = 10,
    n_m_steps: int = 5,
    m_step_lr: float = 1e-3,
    n_particles: int = 50,
    sinkhorn_epsilon: float = 2.0,
    verbose: bool = True,
) -> tuple:
    """EM warm-up: train full SSL (LSTM + emission), refine z_ref.

    Alternates:
      - M-step: Adam on full joint log-likelihood (LSTM + emission params)
      - S-step: forward messages to refine z_ref (MAP estimate)

    Parameters
    ----------
    ssl : GaussianSSL
        Model to train (modified in-place).
    x_obs : tf.Tensor
        (T, obs_dim) observations.
    n_outer : int
        Number of EM iterations.
    n_m_steps : int
        Gradient steps per M-step.
    m_step_lr : float
        Adam learning rate.
    n_particles : int
        Particles for monitoring filter LL (not used in M-step).
    sinkhorn_epsilon : float
        Sinkhorn epsilon for filter LL evaluation.
    verbose : bool

    Returns
    -------
    z_ref : tf.Tensor
        (T, state_dim) refined reference trajectory.
    log_lls : list[float]
        Filter log-likelihoods per outer iteration.
    """
    T = x_obs.shape[0]
    d_z = ssl.state_dim_val
    d_x = int(x_obs.shape[1]) if len(x_obs.shape) > 1 else 1
    optimizer = tf.keras.optimizers.Adam(learning_rate=m_step_lr)

    # Initialize z_ref from observations.  With C≈I, b≈0 at init time,
    # x ≈ z + noise, so this gives the LSTM a real signal from the first
    # M-step.  Random-noise init lets R absorb the full dynamics variance
    # (e.g. var(sin)≈0.5), trapping the EM in a bad equilibrium.
    if d_x == d_z:
        z_ref = tf.identity(x_obs)
    elif d_x > d_z:
        z_ref = x_obs[:, :d_z]
    else:
        z_ref = tf.concat(
            [x_obs, tf.zeros([T, d_z - d_x])], axis=-1,
        )
    log_lls = []

    filt = DifferentiableLEDHLogLikelihood(
        num_particles=n_particles,
        n_lambda=3,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=20,
        grad_window=5,
        jit_compile=False,
    )

    for outer in range(n_outer):
        # ---- Monitor filter LL ----
        adapter = GaussianSSLasSSM(ssl, z_ref)
        try:
            ll_val = float(filt(adapter, x_obs))
        except Exception:
            ll_val = float("-inf")
        log_lls.append(ll_val)

        # ---- M-step: train all SSL parameters ----
        for _ in range(n_m_steps):
            with tf.GradientTape() as tape:
                lstm_st = ssl.get_initial_lstm_state(batch_size=1)
                z_prev = tf.zeros([1, d_z])
                total_ll = tf.constant(0.0)

                for t in range(T):
                    mu_t, sigma_t, lstm_st = ssl.transition_params(
                        lstm_st, z_prev
                    )
                    total_ll = total_ll + ssl.transition_log_prob(
                        z_ref[t:t + 1], mu_t, sigma_t
                    )[0]
                    total_ll = total_ll + ssl.emission_log_prob(
                        z_ref[t:t + 1], x_obs[t:t + 1]
                    )[0]
                    z_prev = z_ref[t:t + 1]

                loss = -total_ll

            grads = tape.gradient(loss, ssl.trainable_variables)
            clipped = [
                tf.clip_by_norm(g, 5.0) if g is not None else g
                for g in grads
            ]
            optimizer.apply_gradients(
                [(g, v) for g, v in zip(clipped, ssl.trainable_variables)
                 if g is not None]
            )

        # ---- S-step: refine z_ref via particle-based forward filtering ----
        z_ref = compute_filtered_trajectory(ssl, x_obs, z_ref,
                                            n_particles=min(n_particles, 30))

        if verbose and (outer + 1) % max(1, n_outer // 5) == 0:
            tf.print(
                f"  EM warm-up {outer + 1:3d}/{n_outer}",
                "  filter_LL=", ll_val,
                "  M-loss=", loss,
            )

    return z_ref, log_lls


# =========================================================================
# Result containers
# =========================================================================

class DPFHMCResult(NamedTuple):
    """Result from dpf_hmc_inference."""
    hmc_samples: tf.Tensor        # (n_samples, emission_dim)
    hmc_accept_rate: tf.Tensor    # scalar
    hmc_log_probs: tf.Tensor      # (n_samples,)
    posterior_mean_C: tf.Tensor   # (obs_dim, state_dim)
    posterior_mean_b: tf.Tensor   # (obs_dim,)
    posterior_mean_R: tf.Tensor   # (obs_dim,) — diagonal of R
    warmup_log_lls: list          # EM warm-up filter LLs
    z_ref: tf.Tensor              # (T, state_dim) reference trajectory


class PMMHSSLResult(NamedTuple):
    """Result from pmmh_ssl_inference."""
    z_samples: tf.Tensor          # (n_outer, T, state_dim)
    log_lls: list                 # bootstrap LL per iteration
    warmup_log_lls: list          # EM warm-up filter LLs
    z_ref: tf.Tensor              # final reference trajectory


# =========================================================================
# DPF-HMC Inference (full pipeline)
# =========================================================================

def dpf_hmc_inference(
    ssl: GaussianSSL,
    x_obs: tf.Tensor,
    *,
    # Warm-up params
    n_warmup: int = 10,
    n_m_steps: int = 5,
    m_step_lr: float = 1e-3,
    # HMC params
    n_hmc_samples: int = 200,
    n_hmc_burnin: int = 100,
    hmc_step_size: float = 0.02,
    n_leapfrog: int = 5,
    adapt_step_size: bool = False,
    target_accept_prob: float = 0.90,
    # Filter params
    n_particles: int = 50,
    n_lambda: int = 3,
    sinkhorn_epsilon: float = 2.0,
    sinkhorn_iters: int = 20,
    grad_window: int = 5,
    # Prior
    prior_scale: float = 2.0,
    fix_C: bool = True,
    verbose: bool = True,
) -> DPFHMCResult:
    """Full DPF-HMC pipeline for Gaussian SSL.

    Phase 1 (warm-up): EM iterations training all SSL parameters.
    Phase 2 (HMC): Fix LSTM weights, sample emission params using HMC
                    with the differentiable LEDH filter.

    By default fix_C=True: C is fixed at its warm-up value and only
    (b, log_R_diag) are sampled (4D). This avoids the C/LSTM
    non-identifiability that causes poor test log-likelihood.

    Parameters
    ----------
    ssl : GaussianSSL
        Model (modified in-place during warm-up, then LSTM frozen).
    x_obs : tf.Tensor
        (T, obs_dim) observations.
    n_warmup : int
        EM warm-up iterations.
    n_hmc_samples : int
        Post-burn-in HMC samples.
    n_hmc_burnin : int
        HMC burn-in iterations.
    hmc_step_size : float
        Initial leapfrog step size.
    n_leapfrog : int
        Leapfrog steps per HMC proposal.
    n_particles : int
        LEDH filter particles.
    prior_scale : float
        Gaussian prior std on emission params.
    fix_C : bool
        If True (default), fix C at warm-up value. Only sample b, R.
    verbose : bool

    Returns
    -------
    DPFHMCResult
    """
    d_z = ssl.state_dim_val
    d_x = ssl.obs_dim_val
    param = EmissionParameterizer(obs_dim=d_x, state_dim=d_z, fix_C=fix_C)

    # ---- Phase 1: EM warm-up ----
    if verbose:
        tf.print("=" * 60)
        tf.print("  Phase 1: EM warm-up (training LSTM + emission)")
        tf.print("=" * 60)

    # Initialize SSL weights with a forward pass
    _ = ssl.transition_params(
        ssl.get_initial_lstm_state(1), tf.zeros([1, d_z])
    )

    # Seed R at a small fraction of observation variance so the S-step
    # filtering trusts observations from iteration 1.  Default R=exp(0)=1
    # is far too large: the filter ignores data, z_ref stays flat, and R
    # absorbs the full dynamics variance (e.g. var(sin)≈0.5).
    obs_var = tf.math.reduce_variance(x_obs, axis=0)  # per-dim
    ssl.log_R_diag.assign(
        tf.math.log(tf.maximum(obs_var * 0.2, tf.constant(1e-4)))
    )
    if verbose:
        tf.print("  R_diag init →", tf.exp(ssl.log_R_diag).numpy())

    z_ref, warmup_lls = warmup_em(
        ssl, x_obs,
        n_outer=n_warmup,
        n_m_steps=n_m_steps,
        m_step_lr=m_step_lr,
        n_particles=n_particles,
        sinkhorn_epsilon=sinkhorn_epsilon,
        verbose=verbose,
    )

    # ---- Freeze LSTM, pre-compute transition params ----
    if verbose:
        tf.print("\n  Freezing LSTM weights, pre-computing transitions...")

    mus, sigmas = precompute_lstm_params(ssl, z_ref)

    # ---- Phase 2: HMC over emission parameters ----
    if verbose:
        tf.print("=" * 60)
        tf.print(
            "  Phase 2: HMC over emission params",
            f"(dim={param.total_dim})"
        )
        tf.print("=" * 60)

    # Initial HMC state = current emission params from warm-up
    theta_init = param.extract_from_ssl(ssl)

    # Build HMC target (prior centered at warm-up params)
    target = DPFHMCTarget(
        mus=mus,
        sigmas=sigmas,
        x_obs=x_obs,
        state_dim=d_z,
        obs_dim=d_x,
        param=param,
        theta_init=theta_init,
        n_particles=n_particles,
        n_lambda=n_lambda,
        sinkhorn_epsilon=sinkhorn_epsilon,
        sinkhorn_iters=sinkhorn_iters,
        grad_window=grad_window,
        prior_scale=prior_scale,
    )

    if verbose:
        tf.print("  Initial emission params:", theta_init)
        tf.print(
            f"  HMC: {n_hmc_samples} samples, {n_hmc_burnin} burn-in,",
            f"init_step={hmc_step_size}, L={n_leapfrog},",
            f"adapt={adapt_step_size}",
        )

    # Run HMC
    hmc_result = run_hmc(
        target_log_prob_fn=target,
        initial_state=theta_init,
        num_results=n_hmc_samples,
        num_burnin=n_hmc_burnin,
        step_size=hmc_step_size,
        num_leapfrog_steps=n_leapfrog,
        adapt_step_size=adapt_step_size,
        target_accept_prob=target_accept_prob,
        seed=42,
        verbose=verbose,
    )

    # Report final adapted step size
    final_step = float(hmc_result.step_sizes[-1])
    if verbose:
        tf.print(
            "\n  HMC done. Accept rate:",
            hmc_result.accept_rate,
        )
        tf.print("  Final adapted step size:", final_step)

    # ---- Compute posterior means ----
    samples = hmc_result.samples  # (n_samples, emission_dim)
    mean_theta = tf.reduce_mean(samples, axis=0)
    C_mean, b_mean, logR_mean = param.unpack(mean_theta)
    R_mean = tf.exp(logR_mean)

    # Apply posterior mean to SSL for downstream use
    param.apply_to_ssl(ssl, mean_theta)

    # Final filtering pass with warm-up LSTM + HMC emission params.
    # The warm-up LSTM already captures the latent dynamics; retraining
    # it after changing emission params risks a degeneration loop where
    # a poor filtering pass produces a flat z_ref that the LSTM then
    # memorises, collapsing to a constant prediction.
    if verbose:
        tf.print("  Computing final filtered trajectory ...")
    z_ref = compute_filtered_trajectory(
        ssl, x_obs, z_ref, n_particles=n_particles,
    )

    return DPFHMCResult(
        hmc_samples=samples,
        hmc_accept_rate=hmc_result.accept_rate,
        hmc_log_probs=hmc_result.target_log_probs,
        posterior_mean_C=C_mean,
        posterior_mean_b=b_mean,
        posterior_mean_R=R_mean,
        warmup_log_lls=warmup_lls,
        z_ref=z_ref,
    )


# =========================================================================
# PMMH-SSL Inference
# =========================================================================

def pmmh_ssl_inference(
    ssl: GaussianSSL,
    x_obs: tf.Tensor,
    *,
    n_warmup: int = 10,
    n_m_steps: int = 5,
    m_step_lr: float = 1e-3,
    n_outer: int = 50,
    n_particles: int = 100,
    verbose: bool = True,
) -> PMMHSSLResult:
    """PMMH-style inference: EM warm-up then bootstrap PF + M-step loop.

    Parameters
    ----------
    ssl : GaussianSSL
        Model (modified in-place).
    x_obs : tf.Tensor
        (T, obs_dim) observations.
    n_warmup : int
        EM warm-up iterations (shared with DPF-HMC for fair comparison).
    n_outer : int
        Post-warmup PMMH outer iterations.
    n_particles : int
        Bootstrap PF particles.
    verbose : bool

    Returns
    -------
    PMMHSSLResult
    """
    T = x_obs.shape[0]
    d_z = ssl.state_dim_val
    optimizer = tf.keras.optimizers.Adam(learning_rate=m_step_lr)

    # ---- Phase 1: EM warm-up (same as DPF-HMC) ----
    if verbose:
        tf.print("=" * 60)
        tf.print("  PMMH: EM warm-up")
        tf.print("=" * 60)

    _ = ssl.transition_params(
        ssl.get_initial_lstm_state(1), tf.zeros([1, d_z])
    )

    z_ref, warmup_lls = warmup_em(
        ssl, x_obs,
        n_outer=n_warmup,
        n_m_steps=n_m_steps,
        m_step_lr=m_step_lr,
        n_particles=min(n_particles, 30),
        verbose=verbose,
    )

    # ---- Phase 2: PMMH outer loop ----
    if verbose:
        tf.print("=" * 60)
        tf.print("  PMMH: Bootstrap PF + M-step")
        tf.print("=" * 60)

    z_samples_list = []
    log_lls = []

    for outer in range(n_outer):
        adapter = GaussianSSLasSSM(ssl, z_ref)

        # Bootstrap PF log-likelihood
        try:
            ll_val = float(bootstrap_pf_log_likelihood(
                adapter, x_obs, num_particles=n_particles
            ))
        except Exception:
            ll_val = float("-inf")
        log_lls.append(ll_val)

        # M-step
        for _ in range(n_m_steps):
            with tf.GradientTape() as tape:
                lstm_st = ssl.get_initial_lstm_state(batch_size=1)
                z_prev = tf.zeros([1, d_z])
                total_ll = tf.constant(0.0)
                for t in range(T):
                    mu_t, sigma_t, lstm_st = ssl.transition_params(
                        lstm_st, z_prev
                    )
                    total_ll = total_ll + ssl.transition_log_prob(
                        z_ref[t:t + 1], mu_t, sigma_t
                    )[0]
                    total_ll = total_ll + ssl.emission_log_prob(
                        z_ref[t:t + 1], x_obs[t:t + 1]
                    )[0]
                    z_prev = z_ref[t:t + 1]
                loss = -total_ll

            grads = tape.gradient(loss, ssl.trainable_variables)
            clipped = [
                tf.clip_by_norm(g, 5.0) if g is not None else g
                for g in grads
            ]
            optimizer.apply_gradients(
                [(g, v) for g, v in zip(clipped, ssl.trainable_variables)
                 if g is not None]
            )

        # S-step: particle-based forward filtering
        z_ref = compute_filtered_trajectory(ssl, x_obs, z_ref,
                                            n_particles=min(n_particles, 30))
        z_samples_list.append(z_ref)

        if verbose and (outer + 1) % max(1, n_outer // 10) == 0:
            tf.print(
                f"  PMMH {outer + 1:3d}/{n_outer}",
                "  bootstrap_LL=", ll_val,
                "  M-loss=", loss,
            )

    return PMMHSSLResult(
        z_samples=tf.stack(z_samples_list),
        log_lls=log_lls,
        warmup_log_lls=warmup_lls,
        z_ref=z_ref,
    )


def test_log_likelihood(
    ssl: GaussianSSL,
    x_test: tf.Tensor,
    z_ref: tf.Tensor,
    n_particles: int = 100,
) -> tf.Tensor:
    """Estimate test log-likelihood using SSL forward messages.

    Runs a batched particle-based forward pass over test data, computing
    marginal alpha weights and resampling via categorical.

    Returns scalar test LL (tf.Tensor).
    """
    T = x_test.shape[0]
    d_z = ssl.state_dim_val
    P = n_particles
    P_f = tf.cast(P, tf.float32)

    lstm_state = ssl.get_initial_lstm_state(batch_size=P)  # (P, H) each
    z_prev_all = tf.zeros([P, d_z])
    log_ml = tf.constant(0.0)

    for t in range(T):
        x_t = tf.broadcast_to(x_test[t:t + 1], [P, ssl.obs_dim_val])

        # Single batched call
        alphas, gamma_mus, gamma_vars, new_lstm_state = ssl.forward_messages(
            lstm_state, z_prev_all, x_t
        )

        log_ml = log_ml + (
            tf.reduce_logsumexp(alphas) - tf.math.log(P_f)
        )

        # Resample then sample z from resampled gamma
        log_w = alphas - tf.reduce_logsumexp(alphas)
        indices = tf.cast(
            tf.random.categorical(log_w[tf.newaxis, :], P)[0],
            tf.int32,
        )

        # Gather resampled gamma params
        gmu_r = tf.gather(gamma_mus, indices)    # (P, d_z)
        gvar_r = tf.gather(gamma_vars, indices)  # (P, d_z, d_z)

        # Batched sampling
        L = tf.linalg.cholesky(gvar_r + tf.eye(d_z) * 1e-6)
        eps = tf.random.normal([P, d_z, 1])
        z_prev_all = gmu_r + tf.squeeze(tf.matmul(L, eps), axis=-1)

        # Resample LSTM states
        h, c = new_lstm_state
        lstm_state = (tf.gather(h, indices), tf.gather(c, indices))

    return log_ml


def filtering_rmse(z_est: tf.Tensor, z_true: tf.Tensor) -> tf.Tensor:
    """Root mean squared error of state estimates."""
    return tf.sqrt(tf.reduce_mean((z_est - z_true) ** 2))
