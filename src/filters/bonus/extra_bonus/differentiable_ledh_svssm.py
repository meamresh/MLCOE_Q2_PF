"""
Differentiable LEDH log-likelihood for the canonical stochastic-volatility SSM.

Model
~~~~~
State (latent log-volatility):
    h_t = mu + phi * (h_{t-1} - mu) + sigma_eta * eta_t,    eta_t ~ N(0, 1).

Observation (multiplicative Gaussian noise on returns):
    y_t = exp(h_t / 2) * eps_t,    eps_t ~ N(0, 1).

Inference works in the **log-squared** observation space
    z_t = log(y_t^2 + delta)
so the observation becomes additive:
    z_t = h_t + mu_z + e_t,
where
    mu_z   = E[log chi^2_1] = psi(1/2) + log(2) ~= -1.2704,
    Var[e_t] = psi'(1/2) = pi^2 / 2 ~= 4.9348.

We use the **Gaussian quasi-likelihood** of Harvey-Ruiz-Shephard (1994):
e_t is approximated by N(0, pi^2/2). The resulting LEDH flow is the
standard linear-Gaussian LEDH for an additive-noise observation with the
trivial (constant) measurement Jacobian H = 1. The quasi-likelihood has a
small theta-independent bias --- it largely cancels in the HMC acceptance
ratio --- and gives a smooth, finite, gradient-stable target. For exact
inference, replace the Gaussian quasi-likelihood with the 7-component
mixture-of-normals approximation of Kim-Shephard-Chib (1998); that swap
is local to the weight increment, the LEDH flow machinery is unchanged.

LEDH simplifications under the SVSSM transformed model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Because h(x) = x + mu_z is linear with unit slope:
    H(x) = 1  (constant)
    e    = h(eta) - H * eta = mu_z  (constant)
    S    = lam * P + R       (R = pi^2/2 is constant)
    A    = -0.5 * P / S
    b    = I_2lam_A * (I_lam_A * P / R * (z - mu_z) + A * eta)
The motion model also simplifies: F = phi is constant, so the per-particle
variance evolves as P_new = phi^2 * P + sigma_eta^2.

Initial condition
~~~~~~~~~~~~~~~~~
Assumes the stationary distribution
    h_0 ~ N(mu, sigma_eta^2 / (1 - phi^2)),
provided |phi| < 1 (caller's responsibility, e.g. parametrise phi via
phi = tanh(phi_raw)). This is the natural choice and answers Section 3's
"what should h_0 be" question.

Public API
~~~~~~~~~~
    ll = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=200, n_lambda=10, sinkhorn_iters=10,
        sinkhorn_epsilon=1.0, resample_threshold=0.5, grad_window=4,
        jit_compile=True, integrator="exp",
    )
    log_p_z = ll(mu, phi, sigma_eta_sq, y_obs)

The returned value is log p(z_{1:T} | theta) up to a theta-independent
constant (the Jacobian |dz/dy| = 2|y| / (y^2 + delta) is dropped because
it cancels in the HMC acceptance ratio).

References
----------
- Harvey, Ruiz, Shephard (1994), "Multivariate Stochastic Variance Models",
  RES 61. Source for the log-y^2 + Gaussian-quasi-likelihood transform.
- Kim, Shephard, Chib (1998), "Stochastic Volatility: Likelihood Inference
  and Comparison with ARCH Models", RES 65. Source for the 7-component
  Gaussian-mixture refinement.
- Li & Coates (2017), "Particle Filtering with Invertible Particle Flow".
- Corenflos et al. (2021), "Differentiable Particle Filtering via
  Entropy-Regularized Optimal Transport".
"""

from __future__ import annotations

import math

import tensorflow as tf

from src.filters.dpf.resampling import det_resample

_EPS = 1e-6
_CLAMP = 1e4

# Moments of log(chi^2_1) used in the Gaussian quasi-likelihood transform.
# mu_z = digamma(1/2) + log(2) = -gamma - log(2) + log(2) = -gamma... actually
# the closed form is digamma(1/2) + log(2); numerical value below is exact to
# float64. Var = trigamma(1/2) = pi^2/2.
LOG_CHI2_MEAN = -1.270362845461478
LOG_CHI2_VAR = math.pi ** 2 / 2.0


def _safe_scalar(x: tf.Tensor) -> tf.Tensor:
    """Replace NaN/Inf with 0, clamp to [-_CLAMP, _CLAMP]."""
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    return tf.clip_by_value(x, -_CLAMP, _CLAMP)


def _as_real_log_scalar(x: tf.Tensor) -> tf.Tensor:
    """Return a finite float32 scalar log-density."""
    x = tf.convert_to_tensor(x)
    x = tf.cast(tf.math.real(x), tf.float32)
    return tf.where(tf.math.is_finite(x), x, tf.constant(-1e6, tf.float32))


class DifferentiableLEDHLogLikelihoodSVSSM:
    """LEDH log-quasi-likelihood for the canonical SVSSM.

    Parameters
    ----------
    num_particles : int
        Number of particles N.
    n_lambda : int
        Number of LEDH pseudo-time substeps per timestep. Default 10
        (matches the exp-integrator recommendation for the Kitagawa filter).
    sinkhorn_epsilon : float
        Entropic regularisation strength for OT resampling.
    sinkhorn_iters : int
        Number of Sinkhorn iterations.
    resample_threshold : float
        ESS / N threshold. (Currently unused: resampling is unconditional
        every step, matching DifferentiableLEDHLogLikelihood. Kept for
        future ESS-triggered support.)
    grad_window : int
        Insert tf.stop_gradient on the particle, P, and log_w state every
        `grad_window` timesteps, to truncate the backward chain.
    jit_compile : bool
        If True, compile the per-timestep kernel with XLA.
    integrator : {"exp", "euler"}
        Discretisation of the per-substep linear flow ODE
        dx/dlam = A x + b.
        "exp" (default): exact-local exponential integrator
        (machine precision per substep; log|J| = A*eps).
        "euler" (legacy): forward Euler
        (O(eps^2) per substep; log|J| = log|1 + eps*A|).
    log_y_sq_offset : float
        Delta added inside log(y^2 + delta) to avoid log(0) when y is zero.
    """

    LOG_CHI2_MEAN = LOG_CHI2_MEAN
    LOG_CHI2_VAR = LOG_CHI2_VAR

    def __init__(
        self,
        num_particles: int = 200,
        n_lambda: int = 10,
        sinkhorn_epsilon: float = 1.0,
        sinkhorn_iters: int = 10,
        resample_threshold: float = 0.5,
        grad_window: int = 4,
        jit_compile: bool = True,
        integrator: str = "exp",
        log_y_sq_offset: float = 1e-8,
        init_type: str = "stationary",
        diffuse_var: float = 100.0,
    ):
        if integrator not in {"euler", "exp"}:
            raise ValueError(f"integrator must be 'euler' or 'exp', got {integrator!r}")
        if init_type not in {"stationary", "fixed_mu", "diffuse"}:
            raise ValueError(
                f"init_type must be 'stationary', 'fixed_mu', or 'diffuse', "
                f"got {init_type!r}"
            )
        self.num_particles = int(num_particles)
        self.n_lambda = int(n_lambda)
        self.sinkhorn_epsilon = float(sinkhorn_epsilon)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.resample_threshold = float(resample_threshold)
        self.grad_window = int(grad_window)
        self.jit_compile = bool(jit_compile)
        self.integrator = integrator
        self.log_y_sq_offset = float(log_y_sq_offset)
        self.init_type = init_type
        self.diffuse_var = float(diffuse_var)

        # Geometric pseudo-time sizes: epsilon_j = epsilon_1 * q^j (sum = 1).
        q = 1.2
        self.epsilon_1 = (1.0 - q) / (1.0 - q ** self.n_lambda)
        self.epsilons = [self.epsilon_1 * q ** j for j in range(self.n_lambda)]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
        y_obs: tf.Tensor,
    ) -> tf.Tensor:
        """Run the differentiable LEDH-OT filter on transformed SVSSM data.

        Parameters
        ----------
        mu, phi, sigma_eta_sq : tf.Tensor
            SVSSM parameters (scalar tensors). Caller is responsible for
            ensuring `|phi| < 1`.
        y_obs : tf.Tensor
            Observed returns, shape (T,) or (T, 1).

        Returns
        -------
        tf.Tensor
            log p(z_{1:T} | theta) where z_t = log(y_t^2 + delta).
            The Jacobian of the y -> z transform is omitted because it is
            theta-independent and cancels in the HMC acceptance ratio.
        """
        y_obs = tf.reshape(tf.cast(y_obs, tf.float32), [-1])
        z_obs = tf.math.log(tf.square(y_obs) + tf.constant(self.log_y_sq_offset, tf.float32))

        mu_t = tf.cast(mu, tf.float32)
        phi_t = tf.cast(phi, tf.float32)
        sigma_eta_sq_t = tf.maximum(tf.cast(sigma_eta_sq, tf.float32), _EPS)

        # Initial-state distribution: three choices for the §7.4 ablation.
        # stationary: h_0 ~ N(mu, sigma_eta^2 / (1 - phi^2))   <- ties h_0 to (phi, sigma_eta)
        # fixed_mu:   h_0 = mu (tiny variance for particle diversity)
        # diffuse:    h_0 ~ N(0, diffuse_var)                  <- independent of params
        if self.init_type == "stationary":
            one_minus_phi_sq = tf.maximum(1.0 - phi_t ** 2, _EPS)
            init_var = sigma_eta_sq_t / one_minus_phi_sq
            init_mean = mu_t
        elif self.init_type == "fixed_mu":
            init_var = sigma_eta_sq_t  # one-step-ahead variance, particles tight around mu
            init_mean = mu_t
        else:  # "diffuse"
            init_var = tf.constant(self.diffuse_var, tf.float32)
            init_mean = tf.constant(0.0, tf.float32)

        if self.jit_compile:
            log_ml = self._run_1d_xla(mu_t, phi_t, sigma_eta_sq_t, z_obs, init_var, init_mean)
        else:
            log_ml = self._run_1d_eager(mu_t, phi_t, sigma_eta_sq_t, z_obs, init_var, init_mean)
        return _as_real_log_scalar(log_ml)

    # ------------------------------------------------------------------
    # 1-D per-timestep kernel (XLA-compiled)
    # ------------------------------------------------------------------

    @tf.function(jit_compile=True)
    def _timestep_1d_xla(
        self,
        particles: tf.Tensor,
        P: tf.Tensor,
        log_w: tf.Tensor,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
        R_val: tf.Tensor,
        z_val: tf.Tensor,
        do_predict: tf.Tensor,
    ):
        """XLA-compiled single-timestep kernel."""
        return self._timestep_1d_body(
            particles, P, log_w, mu, phi, sigma_eta_sq, R_val, z_val, do_predict
        )

    def _timestep_1d_body(
        self,
        particles: tf.Tensor,
        P: tf.Tensor,
        log_w: tf.Tensor,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
        R_val: tf.Tensor,
        z_val: tf.Tensor,
        do_predict: tf.Tensor,
    ):
        """Core per-timestep logic (predict -> LEDH flow -> weight -> OT-resample)."""
        N = self.num_particles
        R_inv_val = 1.0 / R_val
        log_det_R = tf.math.log(R_val)
        log_norm_const = -0.5 * (log_det_R + tf.math.log(2.0 * 3.141592653589793))
        mu_z = tf.constant(self.LOG_CHI2_MEAN, tf.float32)

        # --- Predict step: h_t = mu + phi(h_{t-1} - mu) + sigma_eta * eta ---
        def _predict():
            x_old = particles
            x_det = _safe_scalar(mu + phi * (x_old - mu))
            new_parts = x_det + tf.random.normal([N]) * tf.sqrt(sigma_eta_sq)
            new_parts = _safe_scalar(new_parts)
            # Motion Jacobian F = phi (constant in SVSSM).
            new_P = tf.clip_by_value(phi ** 2 * P + sigma_eta_sq, _EPS, _CLAMP)
            return new_parts, new_P

        def _no_predict():
            return particles, P

        particles_t, P_t = tf.cond(do_predict, _predict, _no_predict)

        # --- LEDH flow ---
        # Linearised observation: h(x) = x + mu_z, H = 1, e = mu_z (constant).
        eta = tf.identity(particles_t)
        log_det_jac = tf.zeros([N])
        lam_cum = 0.0

        use_exp = self.integrator == "exp"

        for j in range(self.n_lambda):
            eps_j = self.epsilons[j]
            lam_k = lam_cum + eps_j / 2.0
            lam_cum += eps_j

            # H = 1 (constant), so H^2 = 1.
            S = tf.maximum(lam_k * P_t + R_val, _EPS)
            A = tf.clip_by_value(-0.5 * P_t / S, -10.0, 0.0)

            lam_A = lam_k * A
            I_lam_A = 1.0 + lam_A
            I_2lam_A = 1.0 + 2.0 * lam_A

            # innov = z - e = z - mu_z (e = mu_z when H = 1 and h(x) = x + mu_z).
            innov = tf.clip_by_value(z_val - mu_z, -100.0, 100.0)
            # b = I_2lam_A * (I_lam_A * P * H * R^-1 * innov + A * eta),  H = 1.
            b_vec = I_2lam_A * (I_lam_A * P_t * R_inv_val * innov + A * eta)
            b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)

            if use_exp:
                # Exact-local solution of dx/dlam = A x + b for constant (A, b).
                Az = A * eps_j
                exp_Az = tf.exp(Az)
                expm1_Az = tf.math.expm1(Az)
                A_safe = tf.where(tf.abs(A) > 1e-8, A, tf.fill(tf.shape(A), 1e-8))
                phi_A = expm1_Az / A_safe
                particles_t = _safe_scalar(particles_t * exp_Az + b_vec * phi_A)
                eta = _safe_scalar(eta * exp_Az + b_vec * phi_A)
                # log|J| = log(exp(A eps)) = A eps exactly (always finite).
                log_det_jac = log_det_jac + Az
            else:
                vel = tf.clip_by_value(A * particles_t + b_vec, -50.0, 50.0)
                particles_t = _safe_scalar(particles_t + eps_j * vel)
                vel_eta = tf.clip_by_value(A * eta + b_vec, -50.0, 50.0)
                eta = _safe_scalar(eta + eps_j * vel_eta)
                J_val = tf.maximum(tf.abs(1.0 + eps_j * A), _EPS)
                log_det_jac = log_det_jac + tf.math.log(J_val)

        # --- Weight increment ---
        # Gaussian quasi-likelihood: log p(z | h) = log N(z; h + mu_z, R).
        resid = z_val - (particles_t + mu_z)
        log_lik = -0.5 * R_inv_val * resid ** 2 + log_norm_const
        log_lik = _safe_scalar(log_lik)

        log_w_incr = log_lik + log_det_jac
        log_w_incr = tf.where(
            tf.math.is_finite(log_w_incr),
            log_w_incr,
            tf.constant(-100.0, dtype=tf.float32),
        )

        log_w_t = log_w + log_w_incr
        log_ev = tf.reduce_logsumexp(log_w_t)
        log_w_t = log_w_t - log_ev

        # --- Unconditional OT resampling ---
        particles_t, log_w_t = self._ot_resample_1d(particles_t, log_w_t)
        P_mean = tf.reduce_mean(P_t)
        P_t = tf.fill([N], P_mean)

        return particles_t, P_t, log_w_t, log_ev

    # ------------------------------------------------------------------
    # 1-D outer loop variants
    # ------------------------------------------------------------------

    def _run_1d_xla(
        self,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
        z_obs: tf.Tensor,
        init_var: tf.Tensor,
        init_mean: tf.Tensor,
    ):
        """Eager outer loop calling the XLA-compiled per-timestep kernel."""
        N = self.num_particles
        T = int(z_obs.shape[0])
        N_f = tf.cast(N, tf.float32)
        R_val = tf.constant(self.LOG_CHI2_VAR, tf.float32)

        particles = init_mean + tf.random.normal([N]) * tf.sqrt(init_var)
        P = tf.fill([N], init_var)
        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0)

        for t_int in range(1, T + 1):
            if self.grad_window > 0 and t_int > 1 and (t_int - 1) % self.grad_window == 0:
                particles = tf.stop_gradient(particles)
                P = tf.stop_gradient(P)
                log_w = tf.stop_gradient(log_w)

            particles, P, log_w, log_ev = self._timestep_1d_xla(
                particles, P, log_w,
                mu, phi, sigma_eta_sq, R_val,
                z_obs[t_int - 1],
                tf.constant(t_int >= 2),
            )
            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0)
            )

        return log_ml

    def _run_1d_eager(
        self,
        mu: tf.Tensor,
        phi: tf.Tensor,
        sigma_eta_sq: tf.Tensor,
        z_obs: tf.Tensor,
        init_var: tf.Tensor,
        init_mean: tf.Tensor,
    ):
        """Pure eager execution (useful for debugging)."""
        N = self.num_particles
        T = int(z_obs.shape[0])
        N_f = tf.cast(N, tf.float32)
        R_val = tf.constant(self.LOG_CHI2_VAR, tf.float32)

        particles = init_mean + tf.random.normal([N]) * tf.sqrt(init_var)
        P = tf.fill([N], init_var)
        log_w = tf.fill([N], -tf.math.log(N_f))
        log_ml = tf.constant(0.0)

        for t_int in range(1, T + 1):
            if self.grad_window > 0 and t_int > 1 and (t_int - 1) % self.grad_window == 0:
                particles = tf.stop_gradient(particles)
                P = tf.stop_gradient(P)
                log_w = tf.stop_gradient(log_w)

            particles, P, log_w, log_ev = self._timestep_1d_body(
                particles, P, log_w,
                mu, phi, sigma_eta_sq, R_val,
                z_obs[t_int - 1],
                tf.constant(t_int >= 2),
            )
            log_ml = log_ml + tf.where(
                tf.math.is_finite(log_ev), log_ev, tf.constant(-10.0)
            )

        return log_ml

    # ------------------------------------------------------------------
    # OT resampling helper (shared with DifferentiableLEDHLogLikelihood)
    # ------------------------------------------------------------------

    def _ot_resample_1d(self, particles_1d: tf.Tensor, log_w: tf.Tensor):
        """OT resampling for 1-D particles (N,) -> (N,)."""
        p2d = particles_1d[:, tf.newaxis]
        p_mean = tf.reduce_mean(p2d, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(p2d, axis=0, keepdims=True) + _EPS
        p_norm = (p2d - p_mean) / p_std

        p_resampled_norm, w_uniform = det_resample(
            p_norm,
            log_w,
            epsilon=self.sinkhorn_epsilon,
            n_iters=self.sinkhorn_iters,
        )

        # det_resample can return complex64 intermediates under XLA when the
        # Sinkhorn log-sum-exp hits near-zero or negative values in float32.
        w_uniform = tf.cast(tf.math.real(tf.cast(w_uniform, tf.complex64)), tf.float32)
        w_uniform = tf.maximum(w_uniform, 0.0)

        p_resampled = p_resampled_norm * p_std + p_mean
        return p_resampled[:, 0], tf.math.log(w_uniform + 1e-20)
