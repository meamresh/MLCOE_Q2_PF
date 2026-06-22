"""
SVSSM neural-OT training infrastructure (Phase 2).

Two components:

1. :func:`generate_svssm_training_data` -- runs the SVSSM filter logic at a
   grid of (mu, phi, sigma_eta) values and harvests
   (particles_normalised, weights, ctx_7, sinkhorn_target_normalised)
   tuples at every timestep.  These are the supervised training pairs.

2. :class:`SVSSMNeuralOTTrainer` -- trains a ConditionalMGradNet (or
   any model with the same call signature) with:
     - supervised MSE loss against Sinkhorn targets
     - 80/20 train/val split
     - Adam optimiser
     - **plateau-based early stop** on val loss
     - **best-checkpoint save** so we never lose the best model
     - training-curve logging

This is intentionally a focused trainer, not a fork of
``neural_ot_resampling.NeuralOTTrainer``: the SVSSM context dimension
is different (7-D vs 6-D), the filter dynamics are different, and we
want the convergence-detection logic written from scratch so it can be
discussed cleanly in the writeup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm import (
    build_svssm_context_scalars,
    _compute_ess,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    LOG_CHI2_MEAN,
    LOG_CHI2_VAR,
    _EPS,
    _CLAMP,
)
from src.filters.dpf.resampling import det_resample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_scalar(x):
    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
    return tf.clip_by_value(x, -_CLAMP, _CLAMP)


def gen_svssm_observations(T: int, mu: float, phi: float, sigma_eta: float,
                            seed: int) -> tf.Tensor:
    """Sample T observations from the canonical SVSSM model."""
    tf.random.set_seed(seed)
    sigma_eta_t = tf.constant(sigma_eta, tf.float32)
    h = tf.constant(float(mu), tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta_t * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------


@dataclass
class SVSSMTrainingDataset:
    """Container for (input, target) pairs across many theta and timesteps."""

    particles_norm: np.ndarray  # (M, N) -- normalised input particles
    weights: np.ndarray          # (M, N) -- input weights
    ctx: np.ndarray              # (M, 7) -- 7-D context vector
    targets_norm: np.ndarray     # (M, N) -- normalised Sinkhorn target particles
    # Metadata
    theta_idx: np.ndarray        # (M,)   -- which theta each sample came from
    t_idx: np.ndarray            # (M,)   -- which timestep each sample is

    def __len__(self) -> int:
        return self.particles_norm.shape[0]

    def split_train_val(self, val_frac: float = 0.2, seed: int = 0):
        """Random 80/20 split over the M samples. Returns (train, val)."""
        rng = np.random.default_rng(seed)
        M = len(self)
        perm = rng.permutation(M)
        n_val = int(round(val_frac * M))
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        def take(idx):
            return SVSSMTrainingDataset(
                particles_norm=self.particles_norm[idx],
                weights=self.weights[idx],
                ctx=self.ctx[idx],
                targets_norm=self.targets_norm[idx],
                theta_idx=self.theta_idx[idx],
                t_idx=self.t_idx[idx],
            )
        return take(train_idx), take(val_idx)


def _run_one_filter_capture(
    mu: float, phi: float, sigma_eta_sq: float, y_obs: tf.Tensor,
    N: int, n_lambda: int, sinkhorn_epsilon: float, sinkhorn_iters: int,
    integrator: str = "exp",
) -> list[dict]:
    """Run the SVSSM filter logic on one observation series and capture
    (particles_norm, weights, ctx, target_norm) tuples at every timestep.

    This mirrors ``DifferentiableLEDHLogLikelihoodSVSSM`` per-step body
    but records intermediates at the resample boundary. The filter
    continues with the Sinkhorn output (so subsequent timesteps see
    realistic state).
    """
    T = int(y_obs.shape[0])
    R_val = tf.constant(LOG_CHI2_VAR, tf.float32)
    R_inv_val = 1.0 / R_val
    log_det_R = tf.math.log(R_val)
    log_norm_const = -0.5 * (log_det_R + tf.math.log(2.0 * 3.141592653589793))
    mu_z = tf.constant(LOG_CHI2_MEAN, tf.float32)

    # Transform observations
    y_flat = tf.reshape(tf.cast(y_obs, tf.float32), [-1])
    z_obs = tf.math.log(tf.square(y_flat) + 1e-8)

    mu_t = tf.cast(mu, tf.float32)
    phi_t = tf.cast(phi, tf.float32)
    sigma_eta_sq_t = tf.maximum(tf.cast(sigma_eta_sq, tf.float32), _EPS)

    # Stationary init
    one_minus_phi_sq = tf.maximum(1.0 - phi_t ** 2, _EPS)
    init_var = sigma_eta_sq_t / one_minus_phi_sq
    init_mean = mu_t

    # Geometric pseudo-time substeps
    q = 1.2
    epsilon_1 = (1.0 - q) / (1.0 - q ** n_lambda)
    epsilons = [epsilon_1 * q ** j for j in range(n_lambda)]

    use_exp = integrator == "exp"
    N_f = tf.cast(N, tf.float32)
    particles = init_mean + tf.random.normal([N]) * tf.sqrt(init_var)
    P = tf.fill([N], init_var)
    log_w = tf.fill([N], -tf.math.log(N_f))

    captured: list[dict] = []

    for t_int in range(1, T + 1):
        t_f = tf.cast(t_int, tf.float32)
        z_val = z_obs[t_int - 1]

        # Predict (only for t >= 2)
        if t_int >= 2:
            x_det = _safe_scalar(mu_t + phi_t * (particles - mu_t))
            particles = x_det + tf.random.normal([N]) * tf.sqrt(sigma_eta_sq_t)
            particles = _safe_scalar(particles)
            P = tf.clip_by_value(phi_t ** 2 * P + sigma_eta_sq_t, _EPS, _CLAMP)

        # LEDH flow
        eta = tf.identity(particles)
        log_det_jac = tf.zeros([N])
        lam_cum = 0.0
        for j in range(n_lambda):
            eps_j = epsilons[j]
            lam_k = lam_cum + eps_j / 2.0
            lam_cum += eps_j

            S = tf.maximum(lam_k * P + R_val, _EPS)
            A = tf.clip_by_value(-0.5 * P / S, -10.0, 0.0)
            lam_A = lam_k * A
            I_lam_A = 1.0 + lam_A
            I_2lam_A = 1.0 + 2.0 * lam_A
            innov = tf.clip_by_value(z_val - mu_z, -100.0, 100.0)
            b_vec = I_2lam_A * (I_lam_A * P * R_inv_val * innov + A * eta)
            b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)

            if use_exp:
                Az = A * eps_j
                exp_Az = tf.exp(Az)
                expm1_Az = tf.math.expm1(Az)
                A_safe = tf.where(tf.abs(A) > 1e-8, A, tf.fill(tf.shape(A), 1e-8))
                phi_A = expm1_Az / A_safe
                particles = _safe_scalar(particles * exp_Az + b_vec * phi_A)
                eta = _safe_scalar(eta * exp_Az + b_vec * phi_A)
                log_det_jac = log_det_jac + Az
            else:
                vel = tf.clip_by_value(A * particles + b_vec, -50.0, 50.0)
                particles = _safe_scalar(particles + eps_j * vel)
                vel_eta = tf.clip_by_value(A * eta + b_vec, -50.0, 50.0)
                eta = _safe_scalar(eta + eps_j * vel_eta)
                J_val = tf.maximum(tf.abs(1.0 + eps_j * A), _EPS)
                log_det_jac = log_det_jac + tf.math.log(J_val)

        # Weight update
        resid = z_val - (particles + mu_z)
        log_lik = -0.5 * R_inv_val * resid ** 2 + log_norm_const
        log_lik = _safe_scalar(log_lik)
        log_w_incr = log_lik + log_det_jac
        log_w_incr = tf.where(
            tf.math.is_finite(log_w_incr),
            log_w_incr,
            tf.constant(-100.0, dtype=tf.float32),
        )
        log_w_t = log_w + log_w_incr
        log_w_t = log_w_t - tf.reduce_logsumexp(log_w_t)

        # ===== Capture-and-resample =====
        p2d = particles[:, tf.newaxis]
        p_mean = tf.reduce_mean(p2d, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(p2d, axis=0, keepdims=True) + _EPS
        p_norm = (p2d - p_mean) / p_std

        w_normed = tf.nn.softmax(log_w_t, axis=0)
        ess = _compute_ess(w_normed)

        ctx = build_svssm_context_scalars(
            mu=mu_t, phi=phi_t, sigma_eta_sq=sigma_eta_sq_t,
            t=float(t_f.numpy()), z_t=float(z_val.numpy()),
            ess=float(ess.numpy()), epsilon=sinkhorn_epsilon, T_max=float(T),
        )

        # Sinkhorn target
        target_norm_2d, _ = det_resample(
            p_norm, log_w_t,
            epsilon=sinkhorn_epsilon, n_iters=sinkhorn_iters,
        )
        target_norm_2d = tf.cast(tf.math.real(target_norm_2d), tf.float32)

        captured.append({
            'particles_norm': p_norm[:, 0].numpy().astype(np.float32),
            'weights': w_normed.numpy().astype(np.float32),
            'ctx': ctx.numpy().astype(np.float32),
            'target_norm': target_norm_2d[:, 0].numpy().astype(np.float32),
        })

        # Continue filter with Sinkhorn output (so subsequent timesteps see
        # realistic state, not a wedged distribution)
        particles_new_norm = target_norm_2d[:, 0]
        particles = particles_new_norm * p_std[0, 0] + p_mean[0, 0]
        P_mean = tf.reduce_mean(P)
        P = tf.fill([N], P_mean)
        log_w = tf.fill([N], -tf.math.log(N_f))

    return captured


def generate_svssm_training_data(
    theta_grid: list[tuple[float, float, float]],
    T: int = 20,
    N: int = 64,
    n_lambda: int = 10,
    sinkhorn_epsilon: float = 1.0,
    sinkhorn_iters: int = 10,
    integrator: str = "exp",
    seeds_per_theta: int = 1,
    base_seed: int = 42,
    verbose: bool = True,
) -> SVSSMTrainingDataset:
    """Generate the full training set across a theta grid.

    Parameters
    ----------
    theta_grid
        List of (mu, phi, sigma_eta) tuples to sample from. Note this is
        the *constrained* parameterisation; the trainer stores the
        context vector in the unconstrained form (atanh phi, log sigma2).
    seeds_per_theta
        Number of independent observation series per theta point. Each
        seed contributes T training samples.

    Returns
    -------
    SVSSMTrainingDataset with M = len(theta_grid) * seeds_per_theta * T
    samples.
    """
    all_pn, all_w, all_ctx, all_tn, all_theta, all_t = [], [], [], [], [], []
    t_start = time.perf_counter()
    n_thetas = len(theta_grid)
    for ti, (mu, phi, sigma_eta) in enumerate(theta_grid):
        for s in range(seeds_per_theta):
            data_seed = base_seed + ti * 100 + s
            y_obs = gen_svssm_observations(T, mu, phi, sigma_eta, seed=data_seed)
            captured = _run_one_filter_capture(
                mu=mu, phi=phi, sigma_eta_sq=sigma_eta ** 2, y_obs=y_obs,
                N=N, n_lambda=n_lambda,
                sinkhorn_epsilon=sinkhorn_epsilon,
                sinkhorn_iters=sinkhorn_iters,
                integrator=integrator,
            )
            for t_int, sample in enumerate(captured):
                all_pn.append(sample['particles_norm'])
                all_w.append(sample['weights'])
                all_ctx.append(sample['ctx'])
                all_tn.append(sample['target_norm'])
                all_theta.append(ti)
                all_t.append(t_int)
        if verbose and (ti + 1) % max(1, n_thetas // 10) == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [data-gen] theta {ti + 1}/{n_thetas}  "
                  f"({elapsed:.1f}s elapsed)", flush=True)

    return SVSSMTrainingDataset(
        particles_norm=np.stack(all_pn, axis=0),
        weights=np.stack(all_w, axis=0),
        ctx=np.stack(all_ctx, axis=0),
        targets_norm=np.stack(all_tn, axis=0),
        theta_idx=np.asarray(all_theta, dtype=np.int32),
        t_idx=np.asarray(all_t, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Convergence-aware trainer
# ---------------------------------------------------------------------------


@dataclass
class TrainingHistory:
    """Loss curves and convergence diagnostics from one training run.

    `train_loss` and `val_loss` track the *configured* objective
    (MSE / MA / mixed). `val_mse_vs_sinkhorn` always tracks pure MSE
    against Sinkhorn targets so cross-mode comparison stays apples-to-apples.
    """
    train_loss: list[float]
    val_loss: list[float]
    val_mse_vs_sinkhorn: list[float]
    best_val_loss: float
    best_val_mse_vs_sinkhorn: float
    best_epoch: int
    converged_at_epoch: int   # -1 if did not converge (hit max epochs)
    elapsed_s: float
    loss_mode: str = "supervised"


class SVSSMNeuralOTTrainer:
    """Supervised / Monge-Ampère / mixed trainer with plateau-based early stop.

    Parameters
    ----------
    model
        Any tf.keras.Model with batched signature
        ``model(particles_BN, weights_BN, ctx_B7) -> particles_BN``.
        For ``monge_ampere`` and ``mixed`` modes the model must also expose
        ``log_det_jacobian(particles_BN, weights_BN, ctx_B7) -> (B, N)``.
    loss_mode
        One of ``"supervised"`` (MSE against Sinkhorn targets),
        ``"monge_ampere"`` (self-supervised volume-change residual),
        or ``"mixed"`` ((1 - ma_weight) * MSE + ma_weight * MA).
    ma_weight
        Weight of the Monge-Ampère term in ``"mixed"`` mode.  Ignored
        otherwise.  Default 0.5.
    learning_rate, batch_size, max_epochs
        Standard hyperparameters.
    patience
        Stop training if val loss has not improved for ``patience``
        consecutive epochs.
    min_improvement
        Minimum relative improvement in val loss to count as "improved"
        (default 1e-4, i.e. 0.01%).
    clip_grad_norm
        Per-tensor gradient clip norm (default 5.0, matches Kitagawa trainer).
    """

    def __init__(
        self,
        model: tf.keras.Model,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 10,
        min_improvement: float = 1e-4,
        loss_mode: str = "supervised",
        ma_weight: float = 0.5,
        clip_grad_norm: float = 5.0,
    ):
        if loss_mode not in ("supervised", "monge_ampere", "mixed"):
            raise ValueError(
                f"loss_mode must be 'supervised', 'monge_ampere', or 'mixed'; "
                f"got '{loss_mode}'"
            )
        self.model = model
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.min_improvement = float(min_improvement)
        self.loss_mode = loss_mode
        self.ma_weight = tf.constant(float(ma_weight), tf.float32)
        self.clip_grad_norm = float(clip_grad_norm)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    def _supervised_loss(self, particles, weights, ctx, target):
        pred = self.model(particles, weights, ctx)
        return tf.reduce_mean(tf.square(pred - target))

    def _ma_loss(self, particles, weights, ctx):
        """Monge-Ampère residual: log|det J_T(x_i)| should equal log(N * w_i)."""
        log_det = self.model.log_det_jacobian(particles, weights, ctx)  # (B, N)
        N_f = tf.cast(tf.shape(particles)[1], tf.float32)
        log_N_w = tf.math.log(N_f * weights + 1e-20)  # (B, N)
        residual = log_det - log_N_w
        # Weighted MSE under the source measure p = sum w_i delta(x_i)
        return tf.reduce_mean(tf.reduce_sum(weights * tf.square(residual), axis=1))

    @tf.function
    def _train_step(self, particles, weights, ctx, target):
        with tf.GradientTape() as tape:
            if self.loss_mode == "supervised":
                loss = self._supervised_loss(particles, weights, ctx, target)
            elif self.loss_mode == "monge_ampere":
                loss = self._ma_loss(particles, weights, ctx)
            else:  # mixed
                l_mse = self._supervised_loss(particles, weights, ctx, target)
                l_ma = self._ma_loss(particles, weights, ctx)
                loss = (1.0 - self.ma_weight) * l_mse + self.ma_weight * l_ma
        grads = tape.gradient(loss, self.model.trainable_variables)
        grads = [tf.clip_by_norm(g, self.clip_grad_norm) if g is not None else g
                 for g in grads]
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    @tf.function
    def _val_loss(self, particles, weights, ctx, target):
        if self.loss_mode == "supervised":
            return self._supervised_loss(particles, weights, ctx, target)
        elif self.loss_mode == "monge_ampere":
            return self._ma_loss(particles, weights, ctx)
        else:  # mixed
            l_mse = self._supervised_loss(particles, weights, ctx, target)
            l_ma = self._ma_loss(particles, weights, ctx)
            return (1.0 - self.ma_weight) * l_mse + self.ma_weight * l_ma

    @tf.function
    def _val_mse_against_sinkhorn(self, particles, weights, ctx, target):
        """MSE vs Sinkhorn target — same metric across all loss modes, for
        apples-to-apples cross-mode comparison."""
        return self._supervised_loss(particles, weights, ctx, target)

    def _iterate_batches(self, ds: SVSSMTrainingDataset, shuffle: bool = True,
                          seed: int = 0):
        M = len(ds)
        order = np.arange(M)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
        for i in range(0, M, self.batch_size):
            idx = order[i:i + self.batch_size]
            yield (
                tf.constant(ds.particles_norm[idx], tf.float32),
                tf.constant(ds.weights[idx], tf.float32),
                tf.constant(ds.ctx[idx], tf.float32),
                tf.constant(ds.targets_norm[idx], tf.float32),
            )

    def train(
        self,
        train_ds: SVSSMTrainingDataset,
        val_ds: SVSSMTrainingDataset,
        checkpoint_path: Optional[Path] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Run training with early stop on val plateau; save best weights."""
        train_curve: list[float] = []
        val_curve: list[float] = []
        val_mse_curve: list[float] = []
        best_val = float('inf')
        best_val_mse = float('inf')
        best_epoch = -1
        best_weights = None
        plateau_count = 0
        converged_at_epoch = -1
        t_start = time.perf_counter()

        for epoch in range(self.max_epochs):
            # === Train pass ===
            batch_losses = []
            for batch in self._iterate_batches(train_ds, shuffle=True,
                                                 seed=epoch):
                loss = self._train_step(*batch)
                batch_losses.append(float(loss.numpy()))
            train_loss = float(np.mean(batch_losses))
            train_curve.append(train_loss)

            # === Val pass (configured loss) ===
            val_batch_losses = []
            val_batch_mse = []
            for batch in self._iterate_batches(val_ds, shuffle=False):
                vl = self._val_loss(*batch)
                val_batch_losses.append(float(vl.numpy()))
                vmse = self._val_mse_against_sinkhorn(*batch)
                val_batch_mse.append(float(vmse.numpy()))
            val_loss = float(np.mean(val_batch_losses))
            val_mse = float(np.mean(val_batch_mse))
            val_curve.append(val_loss)
            val_mse_curve.append(val_mse)

            # === Early-stop logic ===
            # The very first epoch always counts as improvement; otherwise
            # require a relative drop > min_improvement vs the running best.
            if best_weights is None:
                rel_improvement = float("inf")
                improved = True
            else:
                rel_improvement = (best_val - val_loss) / max(abs(best_val), 1e-12)
                improved = rel_improvement > self.min_improvement
            if improved:
                best_val = val_loss
                best_val_mse = val_mse
                best_epoch = epoch
                best_weights = [v.numpy().copy() for v in self.model.trainable_variables]
                plateau_count = 0
            else:
                plateau_count += 1

            if verbose:
                elapsed = time.perf_counter() - t_start
                marker = " *" if improved else ""
                cross = ""
                if self.loss_mode != "supervised":
                    cross = f"  vmse={val_mse:.5f}"
                print(f"  epoch {epoch + 1:3d}/{self.max_epochs}: "
                      f"train={train_loss:.5f}  val={val_loss:.5f}{cross}  "
                      f"best={best_val:.5f}  plateau={plateau_count}/{self.patience}  "
                      f"({elapsed:.1f}s){marker}", flush=True)

            if plateau_count >= self.patience:
                converged_at_epoch = epoch + 1
                if verbose:
                    print(f"  Early stop at epoch {epoch + 1} "
                          f"(no improvement > {self.min_improvement} for {self.patience} epochs)")
                break

        elapsed = time.perf_counter() - t_start

        # Restore best weights
        if best_weights is not None:
            for v, w in zip(self.model.trainable_variables, best_weights):
                v.assign(w)
            if verbose:
                print(f"  Restored best weights from epoch {best_epoch + 1}  "
                      f"(val={best_val:.5f})")

        # Save checkpoint
        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_weights(str(checkpoint_path))
            if verbose:
                print(f"  Saved checkpoint -> {checkpoint_path}")

        return TrainingHistory(
            train_loss=train_curve,
            val_loss=val_curve,
            val_mse_vs_sinkhorn=val_mse_curve,
            best_val_loss=best_val,
            best_val_mse_vs_sinkhorn=best_val_mse,
            best_epoch=best_epoch,
            converged_at_epoch=converged_at_epoch,
            elapsed_s=elapsed,
            loss_mode=self.loss_mode,
        )
