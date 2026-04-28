"""
lhnn_hmc_pf.py
==============
Latent Hamiltonian Neural Network (L-HNN) accelerated HMC for
differentiable stochastic targets (e.g. the LEDH particle filter).

Background
----------
Standard HMC calls the target *L+1* times per iteration (once per leapfrog
step plus the final accept/reject evaluation), each requiring a full
backward pass through the particle filter.  For a LEDH filter with N
particles over T time steps this is extremely costly.

After a one-off offline training phase, L-HNNs replace every leapfrog
gradient with a cheap neural-network backward pass.  The particle filter
is then called only **once per HMC iteration** — value only, no gradient —
for the Metropolis accept/reject step.

Expected gradient-evaluation savings (per sampling iteration)
-------------------------------------------------------------
  Traditional HMC : L + 1  particle-filter gradient evaluations
  L-HNN HMC       : ~0     (pure L-HNN leapfrog)
                  + ~1     (accept/reject, value only ≈ half cost of gradient)
                  + occasional fallback steps (rare, controlled by Δmax_hnn)

Architecture (Dhulipala et al. 2022, §3.1)
-------------------------------------------
  Input  : z = [q, p]  ∈ R^{2d}
  Hidden : P layers of width H, tanh activation
  Output : λ  ∈ R^d   (d latent variables)
  H_θ    = Σ_i λ_i    (scalar, learned Hamiltonian)

  Gradients ∂H_θ/∂q and ∂H_θ/∂p are obtained via GradientTape.

Training loss (Eq. 3.1)
------------------------
  L(θ) = ‖ ∂H_θ/∂p − dq/dt ‖ + ‖ −∂H_θ/∂q − dp/dt ‖

  where dq/dt = p  (exact kinetic energy K = ½‖p‖²)
  and   dp/dt = ∇_q log π(q)  (computed from the real target during training).

Synchronized leapfrog (Algorithm 3.2, m_i = 1)
-----------------------------------------------
  Cache g = ∂H_θ/∂q(q_0, p_0) before the loop, then for l = 0…L-1:

    q_{l+1} = q_l + ε·p_l − (ε²/2)·g_l          # position (uses exact p)
    g_{l+1}  = ∂H_θ/∂q(q_{l+1}, p_l)              # one L-HNN call, cached
    p_{l+1}  = p_l − (ε/2)·(g_l + g_{l+1})        # momentum

  Cost: 1 L-HNN forward+backward pass per step (g_l is reused).

Online error monitoring (§4.2)
-------------------------------
  At each leapfrog step the L-HNN Hamiltonian H_θ is evaluated.
  If |H_θ(t+ε) − H_θ(0)| > Δmax_hnn the step is replaced by a real-
  gradient leapfrog step and the sampler enters a cooldown of N_lf
  real-gradient steps before returning to L-HNN mode.

  Proposition 4.1 (convergence): as Δmax_hnn → −∞, the algorithm
  converges to traditional NUTS/HMC.

Drop-in replacement
-------------------
  `run_lhnn_hmc` has the same signature as `run_hmc` in hmc_pf.py with
  additional keyword-only arguments for L-HNN configuration.

References
----------
- Dhulipala, Che & Shields (2022), "Bayesian Inference with Latent
  Hamiltonian Neural Networks", arXiv:2208.06120.
- Neal (2011), "MCMC using Hamiltonian dynamics", Handbook of MCMC.
- Hoffman & Gelman (2014), "The No-U-Turn Sampler", JMLR.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, List, NamedTuple, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.hmc_pf import HMCResult, _eval_target_and_grad, _safe_grad

# ---------------------------------------------------------------------------
# Public configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class LHNNConfig:
    """
    Hyper-parameters for the Latent HNN.

    Parameters
    ----------
    hidden_units : int
        Width of each hidden layer.
    num_hidden : int
        Number of hidden layers (depth).
    epochs : int
        Training epochs.
    lr : float
        Adam learning rate during training.
    batch_size : int
        Mini-batch size for training.
    num_pilot_trajectories : int
        Number of Hamiltonian trajectories used to generate training data.
        Each trajectory contributes `pilot_steps_per_trajectory` data points.
        Rule of thumb: 20–50 for low-d, 50–100 for high-d problems.
    pilot_steps_per_trajectory : int
        Leapfrog steps per pilot trajectory.
    error_threshold : float
        Δmax_hnn: maximum allowed change in H_θ before triggering fallback.
        The paper recommends ~10 (much tighter than the leapfrog threshold
        of 1 000 used in traditional NUTS).
    cooldown_steps : int
        N_lf: real-gradient steps after a fallback trigger before resuming
        L-HNN mode.  5–20 is typically sufficient.
    retrain_on_fallback : bool
        If True, augment training data with fallback trajectory data and
        retrain the L-HNN when the fallback rate exceeds
        `fallback_rate_threshold`.  Disabled by default for simplicity.
    fallback_rate_threshold : float
        Fraction of iterations triggering fallback above which retraining
        is considered (only relevant when `retrain_on_fallback=True`).
    """

    hidden_units: int = 256
    num_hidden: int = 3
    epochs: int = 3_000
    lr: float = 1e-3
    batch_size: int = 64
    num_pilot_trajectories: int = 30
    pilot_steps_per_trajectory: int = 50
    error_threshold: float = 10.0
    cooldown_steps: int = 10
    retrain_on_fallback: bool = False
    fallback_rate_threshold: float = 0.20


@dataclass(frozen=True)
class LHNNHMCDiagnostics:
    """
    Real target-gradient accounting for :func:`run_lhnn_hmc`.

    Used for ESS/gradient and fallback metrics (Dhulipala et al. 2022, Table 1).

    Attributes
    ----------
    training_gradient_evals : int
        Real ∇ log π evaluations during pilot data generation.
        0 when ``pretrained_lhnn`` is supplied.
    sampling_real_gradient_evals : int
        Real ∇ log π evaluations during MCMC (leapfrog fallback / cooldown steps).
    total_mcmc_iterations : int
        ``num_burnin + num_results``.
    leapfrog_steps_per_iter : int
        ``num_leapfrog_steps`` (L).
    """

    training_gradient_evals: int
    sampling_real_gradient_evals: int
    total_mcmc_iterations: int
    leapfrog_steps_per_iter: int

    @property
    def total_leapfrog_steps(self) -> int:
        """Total leapfrog steps across all MCMC iterations."""
        return self.total_mcmc_iterations * self.leapfrog_steps_per_iter

    @property
    def total_real_gradient_evals(self) -> int:
        """Training + sampling real gradient evaluations."""
        return self.training_gradient_evals + self.sampling_real_gradient_evals

    @property
    def sampling_fallback_intensity(self) -> float:
        """
        Mean real gradient evaluations per leapfrog step during MCMC.

        Can exceed 1.0 when a cooldown window spans multiple steps —
        ``N_lf`` real-gradient sub-steps are taken consecutively.
        """
        return self.sampling_real_gradient_evals / max(self.total_leapfrog_steps, 1)


# ---------------------------------------------------------------------------
# Latent HNN model
# ---------------------------------------------------------------------------

class LatentHNN(tf.keras.Model):
    """
    Latent Hamiltonian Neural Network (L-HNN).

    Takes z = [q, p] ∈ R^{2d} as input and outputs latent variables λ ∈ R^d.
    The learned Hamiltonian is H_θ = Σᵢ λᵢ(q, p).

    Both kinetic and potential gradients are trained simultaneously:
      loss = ‖∂H_θ/∂p − p‖₁  +  ‖−∂H_θ/∂q − ∇ log π‖₁

    The kinetic term (‖∂H/∂p − p‖₁) is mathematically trivial but
    provides first-order gradient signal into the network weights.
    This "warm-up" is what allows the harder potential term to converge.
    Removing the kinetic term leaves only a second-order gradient signal
    for the potential, which empirically fails to drive learning on this
    target (loss stays flat at ~23 throughout training).

    The output bias is disabled: a constant shift on H cancels in all
    gradient computations and is never updated by the loss, which would
    trigger a spurious Keras warning.
    """

    def __init__(self, d: int, hidden_units: int = 256, num_hidden: int = 3):
        """Initialise L-HNN with *d* latent outputs and *num_hidden* layers."""
        super().__init__()
        self.d = d
        self.dense_layers: List[tf.keras.layers.Dense] = [
            tf.keras.layers.Dense(
                hidden_units,
                activation="tanh",
                kernel_initializer="glorot_normal",
                bias_initializer="zeros",
            )
            for _ in range(num_hidden)
        ]
        # No output bias: constant shift on H has zero gradient in the loss.
        self.output_layer = tf.keras.layers.Dense(
            d,
            kernel_initializer="glorot_normal",
            use_bias=False,
        )

    # ------------------------------------------------------------------
    def call(self, z: tf.Tensor) -> tf.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        z : tf.Tensor, shape (batch, 2d)
            Concatenated [q, p] vectors.

        Returns
        -------
        λ : tf.Tensor, shape (batch, d)
        """
        x = z
        for layer in self.dense_layers:
            x = layer(x)
        return self.output_layer(x)

    # ------------------------------------------------------------------
    def potential_grad_and_hamiltonian(
        self, q: tf.Tensor, p: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Compute ∂H_θ/∂q and the scalar H_θ for a single (q, p) pair.

        Returns
        -------
        dH_dq : tf.Tensor, shape (d,)
        H_val  : tf.Tensor, scalar
        """
        q = tf.cast(q, tf.float32)
        p = tf.cast(p, tf.float32)
        z = tf.concat([q, p], axis=0)[tf.newaxis, :]  # (1, 2d)

        with tf.GradientTape() as tape:
            tape.watch(z)
            lam = self(z)                  # (1, d)
            H_val = tf.reduce_sum(lam)     # scalar

        dH_dz = tape.gradient(H_val, z)    # (1, 2d)
        if dH_dz is None:
            dH_dq = tf.zeros_like(q)
        else:
            dH_dq = dH_dz[0, : self.d]    # (d,) — q-slice only

        return dH_dq, H_val

    # ------------------------------------------------------------------
    def hamiltonian(self, q: tf.Tensor, p: tf.Tensor) -> tf.Tensor:
        """Evaluate H_θ(q, p) without computing gradients."""
        q = tf.cast(q, tf.float32)
        p = tf.cast(p, tf.float32)
        z = tf.concat([q, p], axis=0)[tf.newaxis, :]
        lam = self(z)
        return tf.reduce_sum(lam)


# ---------------------------------------------------------------------------
# Training-data generation: pilot HMC with real target gradients
# ---------------------------------------------------------------------------

def generate_training_data(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_state: tf.Tensor,
    num_trajectories: int,
    steps_per_trajectory: int,
    step_size: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, int]:
    """
    Simulate Hamiltonian trajectories with the real target gradient to
    build an L-HNN training dataset.

    For the standard Euclidean HMC Hamiltonian H = U(q) + ½‖p‖²:
      • dq/dt = ∂H/∂p = p          (exact — no target call needed)
      • dp/dt = −∂H/∂q = ∇ log π   (requires one gradient evaluation per step)

    The pilot uses the synchronized leapfrog integrator (Alg. 3.2) so that
    training-data quality matches the integration scheme used at sample time.

    Parameters
    ----------
    target_log_prob_fn : callable
        Differentiable log-posterior.
    initial_state : tf.Tensor, shape (d,)
        Starting point.  Subsequent trajectories chain from the final
        position of the previous one to cover a broader posterior region.
    num_trajectories : int
        Number of independent trajectories.
    steps_per_trajectory : int
        Leapfrog steps per trajectory.
    step_size : float
        Leapfrog step size ε (should match the HMC step size).
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    q_data    : tf.Tensor, (N, d)  — positions
    p_data    : tf.Tensor, (N, d)  — momenta at the same phase-space point
    dq_data   : tf.Tensor, (N, d)  — time derivative of q  (= p)
    dp_data   : tf.Tensor, (N, d)  — time derivative of p  (= grad log π)
    total_grad_evals : int
        Exact pilot ∇ log π count (see module docstring).
    """
    d = int(initial_state.shape[0])
    eps = step_size

    q_list: list = []
    p_list: list = []
    dq_list: list = []
    dp_list: list = []

    q = tf.cast(tf.identity(initial_state), tf.float32)
    total_grad_evals = 0
    t0 = time.time()

    for traj_i in range(num_trajectories):
        traj_seed = seed + traj_i * 7_919

        # Sample momentum
        tf.random.set_seed(traj_seed)
        p = tf.random.normal([d], dtype=tf.float32)

        # Initial gradient at start of trajectory
        lp, grad = _eval_target_and_grad(target_log_prob_fn, q, traj_seed)
        total_grad_evals += 1

        # Synchronized leapfrog, recording one data point per step
        for _ in range(steps_per_trajectory):
            # Record (q, p, dq/dt=p, dp/dt=grad_log_pi)
            q_list.append(tf.identity(q))
            p_list.append(tf.identity(p))
            dq_list.append(tf.identity(p))       # dq/dt = ∂K/∂p = p  (exact)
            dp_list.append(tf.identity(grad))  # dp/dt = ∇_q log π

            # --- Synchronized leapfrog step (Eq. 2.7–2.8) ---------------
            # ∂H/∂q = −∇ log π, so the position update is:
            #   q_new = q + ε·p − (ε²/2)·∂H/∂q = q + ε·p + (ε²/2)·grad
            q_new = q + eps * p + (eps**2 / 2.0) * grad

            lp_new, grad_new = _eval_target_and_grad(
                target_log_prob_fn, q_new, traj_seed
            )
            total_grad_evals += 1

            # p_new = p − (ε/2)·(∂H/∂q_old + ∂H/∂q_new)
            #       = p + (ε/2)·(grad_old + grad_new)
            p = p + (eps / 2.0) * (grad + grad_new)

            q = q_new
            grad = grad_new
            lp = lp_new

        if verbose:
            elapsed = time.time() - t0
            print(
                f"  Pilot traj {traj_i + 1}/{num_trajectories}"
                f"  grad evals so far: {total_grad_evals}"
                f"  ({elapsed:.1f}s)"
            )

    q_arr = tf.stack(q_list, axis=0)
    p_arr = tf.stack(p_list, axis=0)
    dq_arr = tf.stack(dq_list, axis=0)
    dp_arr = tf.stack(dp_list, axis=0)

    if verbose:
        print(
            f"  Training data: {int(q_arr.shape[0])} points from {num_trajectories}"
            f" trajectories  ({total_grad_evals} gradient evaluations total)"
            f"  (= {num_trajectories} × ({steps_per_trajectory} steps + 1 init) )"
        )
    return q_arr, p_arr, dq_arr, dp_arr, total_grad_evals


# ---------------------------------------------------------------------------
# L-HNN training
# ---------------------------------------------------------------------------

def train_lhnn(
    lhnn: LatentHNN,
    q_data: Any,
    p_data: Any,
    dq_dt: Any,
    dp_dt: Any,
    epochs: int = 3_000,
    lr: float = 1e-3,
    batch_size: int = 64,
    verbose: bool = True,
) -> LatentHNN:
    """
    Train the L-HNN by minimising the Hamilton's-equations residual.

    Loss (Eq. 3.1):
        L(θ) = ‖ ∂H_θ/∂p − dq/dt ‖_1 + ‖ −∂H_θ/∂q − dp/dt ‖_1

    where the mean L1-norm over the batch is used for robustness.

    Parameters
    ----------
    lhnn     : LatentHNN
    q_data   : (N, d)  positions from pilot trajectories
    p_data   : (N, d)  momenta
    dq_dt    : (N, d)  = p_data (exact, but kept separate for API clarity)
    dp_dt    : (N, d)  = ∇_q log π at each q
    epochs   : training epochs
    lr       : Adam learning rate
    batch_size : mini-batch size
    verbose  : print loss every 200 epochs

    Returns
    -------
    lhnn : LatentHNN (trained in-place, also returned for chaining)
    """
    q_data = tf.convert_to_tensor(q_data, dtype=tf.float32)
    p_data = tf.convert_to_tensor(p_data, dtype=tf.float32)
    dq_dt = tf.convert_to_tensor(dq_dt, dtype=tf.float32)
    dp_dt = tf.convert_to_tensor(dp_dt, dtype=tf.float32)
    d = int(q_data.shape[1])
    N = int(q_data.shape[0])
    # Cosine decay: starts at lr, decays to lr/10 over the full training run.
    # Prevents the oscillation (epoch 200→400 loss spike) caused by a fixed
    # lr=1e-3 that is too large once the network is near a minimum.
    steps_per_epoch = max(1, N // batch_size)
    total_steps = epochs * steps_per_epoch
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=total_steps,
        alpha=0.1,   # final lr = lr * alpha = lr / 10
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

    q_tf = q_data
    p_tf = p_data
    dq_tf = dq_dt
    dp_tf = dp_dt

    @tf.function
    def train_step(
        q_b: tf.Tensor,
        p_b: tf.Tensor,
        dq_b: tf.Tensor,
        dp_b: tf.Tensor,
    ) -> tf.Tensor:
        """One mini-batch: Hamilton-residual loss and Adam update on L-HNN weights."""
        z_b = tf.concat([q_b, p_b], axis=1)  # (B, 2d)
        with tf.GradientTape() as outer:
            # persistent=True so the outer tape can differentiate through
            # the inner gradient operation (second-order gradients into θ).
            with tf.GradientTape(persistent=True) as inner:
                inner.watch(z_b)
                lam = lhnn(z_b)                    # (B, d)
                H_b = tf.reduce_sum(lam, axis=1)   # (B,)
            dH_dz = inner.gradient(H_b, z_b)       # (B, 2d)
            del inner

            dH_dp_pred = dH_dz[:, d:]              # ∂H_θ/∂p  ≈  dq/dt = p
            dH_dq_pred = dH_dz[:, :d]              # ∂H_θ/∂q  ≈  −dp/dt = −∇ log π

            # Kinetic term: easy first-order signal that warms up the optimizer.
            # Potential term: the hard regression target we actually need.
            # Both are necessary — removing kinetic leaves only a second-order
            # gradient signal that empirically fails to drive learning.
            loss_kinetic = tf.reduce_mean(
                tf.reduce_sum(tf.abs(dH_dp_pred - dq_b), axis=1)
            )
            loss_potential = tf.reduce_mean(
                tf.reduce_sum(tf.abs(-dH_dq_pred - dp_b), axis=1)
            )
            loss = loss_kinetic + loss_potential

        grads = outer.gradient(loss, lhnn.trainable_variables)
        grads = [
            tf.clip_by_norm(g, 1.0) if g is not None else g
            for g in grads
        ]
        optimizer.apply_gradients(zip(grads, lhnn.trainable_variables))
        return loss

    t0 = time.time()
    for epoch in range(epochs):
        # Shuffle each epoch
        idx = tf.random.shuffle(tf.range(N))
        q_s = tf.gather(q_tf, idx)
        p_s = tf.gather(p_tf, idx)
        dq_s = tf.gather(dq_tf, idx)
        dp_s = tf.gather(dp_tf, idx)

        epoch_loss = 0.0
        num_batches = 0
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            loss = train_step(q_s[start:end], p_s[start:end],
                              dq_s[start:end], dp_s[start:end])
            epoch_loss += float(loss.numpy())
            num_batches += 1

        if verbose and (epoch + 1) % 200 == 0:
            elapsed = time.time() - t0
            current_lr = float(schedule((epoch + 1) * steps_per_epoch))
            print(
                f"  L-HNN epoch {epoch + 1:>5d}/{epochs}"
                f"  loss = {epoch_loss / num_batches:.5f}"
                f"  lr = {current_lr:.2e}"
                f"  ({elapsed:.1f}s)"
            )

    if verbose:
        print(f"  L-HNN training complete in {time.time() - t0:.1f}s.")
    return lhnn


# ---------------------------------------------------------------------------
# L-HNN leapfrog with online error monitoring
# ---------------------------------------------------------------------------

class _LeapfrogResult(NamedTuple):
    """Outputs of one L-HNN leapfrog trajectory with fallback accounting."""

    q_prop: tf.Tensor
    p_prop: tf.Tensor
    lhnn_fallback_steps: int   # real-gradient steps consumed inside leapfrog


def _leapfrog_lhnn(
    lhnn: LatentHNN,
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    q: tf.Tensor,
    p: tf.Tensor,
    eps: float,
    num_steps: int,
    crn_seed: int,
    H0_lhnn: tf.Tensor,         # H_θ at the start of the trajectory (for drift monitoring)
    error_threshold: float,
    cooldown_steps: int,
    fallback_remaining: int,    # inherited cooldown from previous iteration
) -> Tuple[tf.Tensor, tf.Tensor, int, int]:
    """
    Synchronized leapfrog using L-HNN gradients with online error monitoring.

    At each step:
      • If `fallback_remaining > 0` (cooldown active): use real gradient.
      • Else: use L-HNN gradient; if |H_θ drift| > error_threshold trigger
        cooldown and switch this step to real gradient.

    Returns
    -------
    q_prop           : proposed position
    p_prop           : proposed (negated) momentum
    fallback_remaining : updated cooldown counter (carry-over to next iteration)
    real_grad_steps  : number of real gradient evaluations consumed here
    """
    q_cur = tf.cast(q, tf.float32)
    p_cur = tf.cast(p, tf.float32)

    real_grad_steps = 0

    # Pre-compute gradient for step 0 (L-HNN or real depending on state)
    if fallback_remaining > 0:
        _, grad_cur = _eval_target_and_grad(target_log_prob_fn, q_cur, crn_seed)
        real_grad_steps += 1
        fallback_remaining -= 1
        using_lhnn = False
    else:
        grad_cur, _ = lhnn.potential_grad_and_hamiltonian(q_cur, p_cur)
        # ∂H_θ/∂q = −∇ log π, so grad_log_pi = −∂H_θ/∂q
        # We need grad_log_pi for the leapfrog position update sign convention.
        # Flip sign: we store "grad" as ∇ log π (positive = go up the posterior).
        grad_cur = -grad_cur   # convert ∂H/∂q → ∇ log π
        using_lhnn = True

    for _ in range(num_steps):
        # --- Position update (Eq. 2.7, m_i = 1) ---
        # q_new = q + ε·p + (ε²/2)·grad_log_pi
        #       = q + ε·p − (ε²/2)·∂H/∂q
        q_new = q_cur + eps * p_cur + (eps**2 / 2.0) * grad_cur

        # --- Gradient at new position ---
        if fallback_remaining > 0:
            _, grad_new = _eval_target_and_grad(
                target_log_prob_fn, q_new, crn_seed
            )
            real_grad_steps += 1
            fallback_remaining -= 1
            using_lhnn = False
        else:
            # Try L-HNN gradient
            dH_dq_new, H_new = lhnn.potential_grad_and_hamiltonian(q_new, p_cur)
            grad_lhnn_new = -dH_dq_new   # ∂H/∂q → ∇ log π

            # --- Online error monitoring (§4.2) ---
            H_drift = tf.abs(H_new - H0_lhnn)
            if float(H_drift.numpy()) > error_threshold and using_lhnn:
                # Trigger fallback: recompute with real gradient
                _, grad_new = _eval_target_and_grad(
                    target_log_prob_fn, q_new, crn_seed
                )
                real_grad_steps += 1
                fallback_remaining = cooldown_steps - 1   # count this step
                using_lhnn = False
            else:
                grad_new = grad_lhnn_new
                using_lhnn = True

        # --- Momentum update (Eq. 2.8) ---
        # p_new = p + (ε/2)·(grad_old + grad_new)   [∇ log π sign convention]
        p_cur = p_cur + (eps / 2.0) * (grad_cur + grad_new)

        q_cur = q_new
        grad_cur = grad_new

    # Negate momentum for time-reversibility
    p_prop = -p_cur

    return q_cur, p_prop, fallback_remaining, real_grad_steps


# ---------------------------------------------------------------------------
# Main sampler: run_lhnn_hmc
# ---------------------------------------------------------------------------

def run_lhnn_hmc(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_state: tf.Tensor,
    # — Standard HMC parameters (same defaults as run_hmc) —
    num_results: int = 1_000,
    num_burnin: int = 500,
    step_size: float = 0.001,
    num_leapfrog_steps: int = 10,
    target_accept_prob: float = 0.45,
    # Why 0.45, not the deterministic-HMC optimum of 0.65:
    # For stochastic targets (particle filters) dual averaging with target=0.65
    # drives eps toward the floor because the L-HNN cannot simultaneously
    # achieve high acceptance AND large-enough eps for mixing.  The theoretical
    # optimum for noisy-gradient HMC is lower (0.40–0.50).  0.45 leaves room
    # for the chain to take meaningful steps while staying stable.
    seed: Optional[int] = None,
    verbose: bool = True,
    adapt_step_size: bool = False,
    # — L-HNN configuration —
    lhnn_config: Optional[LHNNConfig] = None,
    # — Optional: pass a pre-trained L-HNN to skip training —
    pretrained_lhnn: Optional[LatentHNN] = None,
    crn_offset: Optional[int] = None,
    **kwargs,
) -> Tuple[HMCResult, LatentHNN, LHNNHMCDiagnostics]:
    """
    L-HNN accelerated HMC — drop-in replacement for ``run_hmc``.

    Phases
    ------
    1. **Pilot run** (if `pretrained_lhnn` is None):
       Generates training trajectories with real target gradients.
       Cost: ``cfg.num_pilot_trajectories × cfg.pilot_steps_per_trajectory``
       real gradient evaluations (one-time, amortised over all samples).

    2. **L-HNN training** (if `pretrained_lhnn` is None):
       Trains the neural network on the pilot data.  Pure GPU/CPU compute,
       no target evaluations.

    3. **HMC sampling** (post-burn-in samples):
       Each iteration:
         a. Sample p ~ N(0, I).
         b. L-HNN leapfrog (L steps) — no target gradient evaluations.
         c. Online error monitoring — occasional real-gradient fallback.
         d. Accept/reject — **one** target value evaluation (no gradient).

    Parameters
    ----------
    target_log_prob_fn : callable
        Differentiable log-posterior log π(θ).
    initial_state : tf.Tensor, shape (d,)
        Starting point in parameter space.
    num_results : int
        Post-burn-in samples to collect.
    num_burnin : int
        Burn-in iterations (step-size adaptation occurs here if enabled).
    step_size : float
        Leapfrog step size ε.
    num_leapfrog_steps : int
        L: leapfrog steps per proposal.
    target_accept_prob : float
        Target acceptance for dual averaging (if ``adapt_step_size=True``).
    seed : int, optional
        Base random seed.
    verbose : bool
        Print progress during all phases.
    adapt_step_size : bool
        Enable Nesterov dual averaging during burn-in.
    lhnn_config : LHNNConfig, optional
        L-HNN hyper-parameters.  Defaults to ``LHNNConfig()``.
    pretrained_lhnn : LatentHNN, optional
        Skip training and use this pre-trained model directly.
    crn_offset : int, optional
        Optional override for the CRN seed used for *target* evaluations
        (both the standalone ``target_log_prob_fn`` calls in this loop and
        the fallback ``_eval_target_and_grad`` calls inside
        :func:`_leapfrog_lhnn`).  When ``None`` the CRN seed equals the
        chain-private ``iter_seed`` derived from ``seed`` — bit-exact
        identical to the previous behaviour.  When set explicitly the
        per-iteration CRN seed is ``crn_offset + (i+1)*7919``.  Used by
        :func:`run_lhnn_hmc_multi_chain` to share the stochastic-target
        realisation across chains; see that function's docstring.

    Returns
    -------
    result : HMCResult
        Same named-tuple as ``run_hmc``.
    lhnn : LatentHNN
        The trained network (can be reused as `pretrained_lhnn` for
        subsequent calls without re-training).
    diagnostics : LHNNHMCDiagnostics
        Exact training vs sampling real gradient counts, and derived
        ESS/gradient metrics (Dhulipala et al. 2022, Table 1).
    """
    _ = kwargs
    cfg = lhnn_config if lhnn_config is not None else LHNNConfig()
    base_seed = seed if seed is not None else 42
    d = int(initial_state.shape[0])
    total = num_burnin + num_results
    L = num_leapfrog_steps
    training_cost = 0  # exact pilot gradient evals; updated below if we train
    q_data: Optional[tf.Tensor] = None
    p_data: Optional[tf.Tensor] = None

    # ------------------------------------------------------------------ #
    # Phase 1 + 2: Build and train L-HNN (unless pre-trained model given) #
    # ------------------------------------------------------------------ #

    if pretrained_lhnn is not None:
        lhnn = pretrained_lhnn
        if verbose:
            print("[L-HNN] Using pre-trained L-HNN — skipping training.")
    else:
        if verbose:
            print(
                f"[L-HNN] Generating training data: {cfg.num_pilot_trajectories}"
                f" trajectories × {cfg.pilot_steps_per_trajectory} steps …"
            )
        q_data, p_data, dq_data, dp_data, training_cost = generate_training_data(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=tf.cast(initial_state, tf.float32),
            num_trajectories=cfg.num_pilot_trajectories,
            steps_per_trajectory=cfg.pilot_steps_per_trajectory,
            step_size=step_size,
            seed=base_seed,
            verbose=verbose,
        )
        # training_cost is the EXACT gradient-eval count returned by
        # generate_training_data: num_trajectories × (steps + 1).
        # (Do NOT use q_data.shape[0] here — that equals num_traj × steps,
        # missing the one initial-gradient eval at the start of each trajectory.)

        if verbose:
            print(
                f"[L-HNN] Training on {q_data.shape[0]} data points"
                f" for {cfg.epochs} epochs …"
            )
        lhnn = LatentHNN(d, cfg.hidden_units, cfg.num_hidden)
        # Trigger weight creation with a dummy pass
        _ = lhnn(tf.zeros([1, 2 * d]))

        train_lhnn(
            lhnn=lhnn,
            q_data=q_data,
            p_data=p_data,
            dq_dt=dq_data,
            dp_dt=dp_data,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
            verbose=verbose,
        )
        if verbose:
            print(
                f"[L-HNN] Training complete.  "
                f"Upfront gradient cost: {training_cost} evals "
                f"(vs {(L + 1) * total} for traditional HMC over the same run)."
            )

    # Compute the effective error threshold from the trained network.
    # cfg.error_threshold is the *user-requested* threshold, but the
    # L-HNN's Hamiltonian scale varies with N (larger N → sharper likelihood
    # → larger |H_θ| values).  We evaluate H_θ on a sample of the training
    # positions and set the actual threshold = max(cfg.error_threshold,
    # 3 × std(H_θ)).  This prevents the N=250 over-triggering problem where
    # |H_θ| drifts by ~20–30 on legitimate steps because the network's
    # absolute output scale is larger than the threshold.
    if q_data is not None and int(q_data.shape[0]) > 0:
        n = int(q_data.shape[0])
        k = min(50, n)
        tf.random.set_seed(base_seed + 999_999)
        idx = tf.random.shuffle(tf.range(n, dtype=tf.int32))[:k]
        h_vals = []
        for j in range(k):
            ij = idx[j]
            q_s = tf.gather(q_data, ij)
            p_s = tf.gather(p_data, ij)
            h_vals.append(lhnn.hamiltonian(q_s, p_s))
        h_tensor = tf.stack(h_vals)
        h_std = float(tf.math.reduce_std(h_tensor).numpy())
        adaptive_threshold = max(cfg.error_threshold, 3.0 * h_std)
        if verbose:
            print(
                f"[L-HNN] H_θ std on training data: {h_std:.2f}  "
                f"→ error threshold: {adaptive_threshold:.2f} "
                f"(cfg={cfg.error_threshold})"
            )
    else:
        adaptive_threshold = cfg.error_threshold

    # ------------------------------------------------------------------ #
    # Phase 3: HMC sampling with L-HNN leapfrog                          #
    # ------------------------------------------------------------------ #

    # Dual-averaging state (Hoffman & Gelman 2014, Alg. 5)
    mu = tf.math.log(10.0 * step_size)
    log_eps = tf.math.log(tf.constant(step_size, tf.float32))
    log_eps_bar = tf.constant(0.0)
    H_bar = tf.constant(0.0)
    gamma, t0_da, kappa = 0.05, 10.0, 0.75
    delta = target_accept_prob

    q = tf.cast(initial_state, tf.float32)

    samples_list: list = []
    accepted_list: list = []
    log_prob_list: list = []
    step_size_list: list = []
    real_grad_count_list: list = []

    fallback_remaining = 0   # carry-over cooldown counter
    total_real_grads = 0     # diagnostic counter
    consecutive_full_fallback = 0  # iterations where every leapfrog step fell back

    crn_base = crn_offset if crn_offset is not None else base_seed

    for i in range(total):
        iter_seed = base_seed + (i + 1) * 7_919   # chain-private (momentum, MH)
        crn_seed = crn_base + (i + 1) * 7_919     # shared across chains when crn_offset is set
        eps = float(tf.exp(log_eps).numpy())

        # Re-evaluate log π(q) with THIS iteration's CRN seed before doing
        # anything else.  The particle filter is stochastic, so
        #   log π(q | seed_i)  ≠  log π(q | seed_j)
        # lp_prop (below) also uses crn_seed, making the Metropolis ratio
        #   log α = log π(q_prop | seed_i) − log π(q | seed_i)
        # a consistent estimate of the true log acceptance ratio.
        # Without this re-evaluation the comparison is apples-vs-oranges:
        # lp_cur was set with a different seed (iter_{i-1} or init_seed),
        # so log_alpha is dominated by particle-filter noise → alpha ≈ 0
        # → dual averaging drives eps to the floor on the first iteration.
        tf.random.set_seed(crn_seed)
        lp_cur_raw = target_log_prob_fn(q)
        lp_cur = tf.cast(tf.math.real(tf.cast(lp_cur_raw, tf.complex64)), tf.float32)
        lp_cur = tf.where(tf.math.is_finite(lp_cur), lp_cur, tf.constant(-1e6, tf.float32))

        # 1. Sample momentum p ~ N(0, I)
        tf.random.set_seed(iter_seed + 3_000_000)
        p = tf.random.normal([d], dtype=tf.float32)
        K_cur = 0.5 * float(tf.reduce_sum(p**2).numpy())

        # 2. L-HNN Hamiltonian at start of trajectory (for drift monitoring)
        H0_lhnn = lhnn.hamiltonian(q, p)

        # 3. L-HNN leapfrog with online error monitoring
        q_prop, p_prop, fallback_remaining, real_grads_this_iter = _leapfrog_lhnn(
            lhnn=lhnn,
            target_log_prob_fn=target_log_prob_fn,
            q=q,
            p=p,
            eps=eps,
            num_steps=L,
            crn_seed=crn_seed,
            H0_lhnn=H0_lhnn,
            error_threshold=adaptive_threshold,
            cooldown_steps=cfg.cooldown_steps,
            fallback_remaining=fallback_remaining,
        )
        total_real_grads += real_grads_this_iter

        # 4. Evaluate target at proposed position (value only, no gradient)
        #    This is the only mandatory target evaluation per iteration.
        tf.random.set_seed(crn_seed)
        lp_prop_raw = target_log_prob_fn(q_prop)
        lp_prop = tf.cast(
            tf.math.real(tf.cast(lp_prop_raw, tf.complex64)), tf.float32
        )
        lp_prop = tf.where(tf.math.is_finite(lp_prop), lp_prop, tf.constant(-1e6))

        K_prop = 0.5 * float(tf.reduce_sum(p_prop**2).numpy())

        # 5. Hamiltonian accept/reject
        #    H = −log π(q) + ½‖p‖²  →  log α = (log π_prop − log π_cur) + (K_cur − K_prop)
        log_alpha_val = float(lp_prop.numpy()) - float(lp_cur.numpy()) + K_cur - K_prop
        log_alpha_val = log_alpha_val if math.isfinite(log_alpha_val) else -1e6
        alpha = min(math.exp(min(log_alpha_val, 0.0)), 1.0)

        tf.random.set_seed(iter_seed + 4_000_000)
        accept = float(tf.random.uniform([]).numpy()) < alpha

        if accept:
            q = tf.identity(q_prop)
            consecutive_full_fallback = 0
        else:
            # Track how many consecutive iterations used maximum fallback.
            # If real_grads_this_iter == L, every leapfrog step fell back,
            # meaning the chain is in a region completely outside the L-HNN's
            # training coverage.  After 3 such iterations the chain is stuck
            # in the tail — force-reject and reset to break the cycle.
            if real_grads_this_iter >= L:
                consecutive_full_fallback += 1
            else:
                consecutive_full_fallback = 0

            if consecutive_full_fallback >= 3:
                # Hard reset: discard proposal, clear cooldown.
                # The chain stays at q (already unchanged on reject), but
                # resetting fallback_remaining gives the L-HNN a clean start
                # from the current position next iteration.
                fallback_remaining = 0
                consecutive_full_fallback = 0
                if verbose:
                    print(f"  [L-HNN] Tail-escape reset at iter {i+1} — clearing cooldown.")
        # lp_cur is re-evaluated from scratch at the top of the next iteration
        # with that iteration's CRN seed — no cache to maintain here.

        # 6. Step-size adaptation (burn-in, if enabled)
        if adapt_step_size and i < num_burnin:
            m = float(i + 1)
            w = 1.0 / (m + t0_da)
            H_bar = (1.0 - w) * H_bar + w * (delta - alpha)
            log_eps = mu - (m**0.5 / gamma) * H_bar
            # Ceiling: 1.2× initial step_size.
            # At 2× (eps=0.010), the L-HNN trajectory (L=10 steps × 0.010 = 0.10)
            # exceeds the region where the network was trained, causing alpha≈0
            # and sending eps straight to the floor.  1.2× (eps=0.006) keeps
            # the trajectory within the trained region while still allowing
            # meaningful exploration.
            log_eps = tf.minimum(log_eps, tf.math.log(tf.constant(step_size * 1.2, tf.float32)))
            # Floor: 50% of initial step_size (see earlier comment).
            log_eps = tf.maximum(log_eps, tf.math.log(tf.constant(step_size * 0.5, tf.float32)))
            log_eps_bar = (
                m**(-kappa) * log_eps + (1.0 - m**(-kappa)) * log_eps_bar
            )
        elif adapt_step_size and i >= num_burnin:
            log_eps = log_eps_bar

        # 7. Bookkeeping
        step_size_list.append(eps)
        real_grad_count_list.append(real_grads_this_iter)

        if i >= num_burnin:
            samples_list.append(tf.identity(q))
            accepted_list.append(accept)
            log_prob_list.append(float(lp_cur.numpy()))

        if verbose and (i + 1) % 10 == 0:
            phase = "burn" if i < num_burnin else "sample"
            print(
                f"  L-HNN HMC {i + 1}/{total} ({phase})"
                f"  eps={eps:.4f}"
                f"  alpha={alpha:.3f}"
                f"  real_grads={real_grads_this_iter}"
                f"  fallback_cool={fallback_remaining}"
            )

    # ------------------------------------------------------------------ #
    # Diagnostics                                                          #
    # ------------------------------------------------------------------ #

    samples = tf.stack(samples_list)
    is_accepted = tf.constant(accepted_list)
    accept_rate = tf.reduce_mean(tf.cast(is_accepted, tf.float32))
    target_log_probs = tf.constant(log_prob_list, dtype=tf.float32)
    step_sizes = tf.constant(step_size_list, dtype=tf.float32)

    trad_hmc_grad_cost = (L + 1) * total
    fallback_cost = total_real_grads
    # Use training_cost captured from generate_training_data (exact count).
    # When pretrained_lhnn is supplied, no training was performed → 0.
    upfront_cost = training_cost if pretrained_lhnn is None else 0

    if verbose:
        print("\n[L-HNN HMC] Sampling complete.")
        print(f"  Acceptance rate        : {float(accept_rate.numpy()):.3f}")
        print(f"  Real gradient evals    : {total_real_grads} (sampling) "
              f"+ {upfront_cost} (training) = {total_real_grads + upfront_cost} total")
        print(f"  Traditional HMC would  : {trad_hmc_grad_cost} gradient evals")
        if trad_hmc_grad_cost > 0:
            saved = trad_hmc_grad_cost - (total_real_grads + upfront_cost)
            pct = 100.0 * saved / trad_hmc_grad_cost
            print(f"  Gradient evals saved   : {saved} ({pct:.1f}%)")

    result = HMCResult(
        samples=samples,
        is_accepted=is_accepted,
        accept_rate=accept_rate,
        target_log_probs=target_log_probs,
        step_sizes=step_sizes,
    )
    diagnostics = LHNNHMCDiagnostics(
        training_gradient_evals=int(training_cost),
        sampling_real_gradient_evals=int(total_real_grads),
        total_mcmc_iterations=int(total),
        leapfrog_steps_per_iter=int(L),
    )
    return result, lhnn, diagnostics


# ---------------------------------------------------------------------------
# Convenience: compute Effective Sample Size per gradient evaluation
# ---------------------------------------------------------------------------

def ess_per_gradient(
    samples: tf.Tensor,
    total_gradient_evals: int,
) -> tf.Tensor:
    """
    Estimate ESS / gradient_eval for each dimension, averaged over dimensions.

    Parameters
    ----------
    samples : tf.Tensor, shape (N, d)
    total_gradient_evals : int
        Total number of real target gradient evaluations (training + sampling).

    Returns
    -------
    mean_ess_per_grad : tf.Tensor, scalar
    """
    ess = tfp.mcmc.effective_sample_size(samples)
    mean_ess = tf.reduce_mean(ess)
    denom = tf.maximum(
        tf.cast(total_gradient_evals, tf.float32), tf.constant(1e-12, tf.float32)
    )
    return tf.cast(mean_ess / denom, tf.float32)


# ---------------------------------------------------------------------------
# Multi-chain wrapper for L-HNN HMC
# ---------------------------------------------------------------------------

class MultiChainLHNNHMCResult(NamedTuple):
    """Container for multi-chain L-HNN HMC output."""

    samples: tf.Tensor          # (num_chains, num_results, d)
    is_accepted: tf.Tensor      # (num_chains, num_results)
    accept_rate: tf.Tensor      # (num_chains,)
    target_log_probs: tf.Tensor # (num_chains, num_results)
    step_sizes: tf.Tensor       # (num_chains, num_burnin + num_results)
    per_chain_diagnostics: list # length num_chains, each LHNNHMCDiagnostics


def run_lhnn_hmc_multi_chain(
    target_log_prob_fn: Callable[[tf.Tensor], tf.Tensor],
    initial_states: tf.Tensor,
    num_results: int = 1_000,
    num_burnin: int = 500,
    step_size: float = 0.001,
    num_leapfrog_steps: int = 10,
    target_accept_prob: float = 0.45,
    seed: Optional[int] = None,
    verbose: bool = True,
    adapt_step_size: bool = False,
    lhnn_config: Optional[LHNNConfig] = None,
    pretrained_lhnn: Optional[LatentHNN] = None,
    share_crn_across_chains: bool = True,
    **kwargs,
) -> Tuple[MultiChainLHNNHMCResult, LatentHNN]:
    """Run independent L-HNN HMC chains with dispersed initial points.

    L-HNN training (the expensive one-time step) is done once on the
    *first* chain's data; the trained network is reused as
    ``pretrained_lhnn`` for chains 2..num_chains.  This is the same
    "share the surrogate, vary the chain" pattern Stan uses for adapted
    metrics across chains.

    CRN policy across chains
    ------------------------
    The L-HNN forward pass itself is deterministic given the trained
    weights, but the *fallback* path (real-gradient calls inside
    :func:`_leapfrog_lhnn`) and the per-iteration accept/reject value
    evaluations both go through the *stochastic* LEDH likelihood.  When
    ``share_crn_across_chains=True`` (default) every chain shares the
    same CRN seed sequence ``crn_offset = base_seed + (i+1)*7919`` so all
    chains see the same stochastic-target realisation at iteration ``i``.
    Each chain still has a private ``base_seed = global + 1009*(c+1)``
    for momentum draws and MH acceptance, so chains explore
    independently.  Without this, finite-N noise in the LEDH likelihood
    creates per-chain local maxima that look like non-convergence to
    R-hat.  See Report_II_Addendum_Diagnostics §4.5.

    Setting ``share_crn_across_chains=False`` reverts to the legacy
    behaviour (independent CRN per chain) — appropriate only when the
    target is deterministic.

    Parameters
    ----------
    initial_states : tf.Tensor, shape ``(num_chains, d)``
        Dispersed starting points.
    share_crn_across_chains : bool
        See discussion above.  Default ``True``.
    Other args forwarded to :func:`run_lhnn_hmc`.

    Returns
    -------
    MultiChainLHNNHMCResult
        Stacked per-chain outputs plus the diagnostics list.
    LatentHNN
        The trained (or pre-supplied) L-HNN network.
    """
    base_seed = seed if seed is not None else 42
    num_chains = int(initial_states.shape[0])
    shared_crn = base_seed if share_crn_across_chains else None

    samples_list: list = []
    accepted_list: list = []
    accept_rate_list: list = []
    lp_list: list = []
    eps_list: list = []
    diag_list: list = []

    trained = pretrained_lhnn

    for c in range(num_chains):
        chain_seed = base_seed + 1009 * (c + 1)
        if verbose:
            crn_tag = f"shared({base_seed})" if share_crn_across_chains else "private"
            print(f"\n[L-HNN HMC chain {c + 1}/{num_chains}]  seed={chain_seed}"
                  f"  crn={crn_tag}"
                  f"  init={initial_states[c].numpy().tolist()}")
        result, trained, diag = run_lhnn_hmc(
            target_log_prob_fn=target_log_prob_fn,
            initial_state=initial_states[c],
            num_results=num_results,
            num_burnin=num_burnin,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            target_accept_prob=target_accept_prob,
            seed=chain_seed,
            verbose=verbose,
            adapt_step_size=adapt_step_size,
            lhnn_config=lhnn_config,
            pretrained_lhnn=trained,  # train once on chain 0; reuse afterwards
            crn_offset=shared_crn,
            **kwargs,
        )
        samples_list.append(result.samples)
        accepted_list.append(result.is_accepted)
        accept_rate_list.append(result.accept_rate)
        lp_list.append(result.target_log_probs)
        eps_list.append(result.step_sizes)
        diag_list.append(diag)

    return MultiChainLHNNHMCResult(
        samples=tf.stack(samples_list, axis=0),
        is_accepted=tf.stack(accepted_list, axis=0),
        accept_rate=tf.stack(accept_rate_list, axis=0),
        target_log_probs=tf.stack(lp_list, axis=0),
        step_sizes=tf.stack(eps_list, axis=0),
        per_chain_diagnostics=diag_list,
    ), trained