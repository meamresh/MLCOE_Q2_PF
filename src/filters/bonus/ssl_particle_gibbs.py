"""
Particle Gibbs inference for Gaussian State-Space LSTM (Zheng et al. 2017).

Implements Algorithm 2 from Zheng et al.: conditional SMC (S-step) alternating
with gradient-based M-step updates on the Keras `GaussianSSL` parameters.

See also
--------
- `src.models.gaussian_ssl` — `GaussianSSL`, `GaussianSSLasSSM`, synthetic data
"""

from __future__ import annotations

from typing import NamedTuple

import tensorflow as tf

from src.models.gaussian_ssl import GaussianSSL


class PGResult(NamedTuple):
    """Result container for Particle Gibbs SSL inference."""

    z_samples: tf.Tensor       # (n_iter, T, state_dim)
    log_marginal: tf.Tensor    # (n_iter,) log p(x_{1:T})
    ssl_losses: list           # M-step losses per iteration


def particle_gibbs_ssl(
    ssl: GaussianSSL,
    x_obs: tf.Tensor,
    *,
    n_particles: int = 50,
    n_iterations: int = 100,
    n_m_steps: int = 5,
    m_step_lr: float = 1e-3,
    verbose: bool = True,
) -> PGResult:
    """Particle Gibbs inference for Gaussian SSL (Zheng(17) Algorithm 2).

    Alternates:
      S-step: Conditional SMC to sample z_{1:T} from posterior
      M-step: MLE of LSTM params (omega) and emission params (phi)

    Parameters
    ----------
    ssl        : GaussianSSL model (will be modified in-place)
    x_obs      : (T, obs_dim) observations
    n_particles: number of SMC particles
    n_iterations: number of PG iterations
    n_m_steps  : gradient steps per M-step
    m_step_lr  : learning rate for M-step

    Returns
    -------
    PGResult with sampled latent trajectories and log-marginals
    """
    T = x_obs.shape[0]
    d_z = ssl.state_dim_val
    P = n_particles

    optimizer = tf.keras.optimizers.Adam(learning_rate=m_step_lr)

    # Initialize reference path
    z_ref = tf.random.normal([T, d_z]) * 0.1

    z_samples = []
    log_marginals = []
    ssl_losses = []

    for it in range(n_iterations):
        # ============================================================
        # S-step: Conditional SMC (Particle Gibbs)
        # ============================================================
        # Particle storage
        z_particles = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)
        ancestors = tf.TensorArray(dtype=tf.int32, size=T, dynamic_size=False)
        log_weights = tf.TensorArray(dtype=tf.float32, size=T, dynamic_size=False)

        # Batched LSTM states: (P, lstm_units) for h and c
        lstm_state = ssl.get_initial_lstm_state(batch_size=P)

        # Previous z for each particle
        z_prev_all = tf.zeros([P, d_z])
        log_ml = 0.0

        for t in range(T):
            # Broadcast x_t to (P, d_x) for batched call
            x_t = tf.broadcast_to(x_obs[t:t + 1], [P, ssl.obs_dim_val])

            # Single batched forward_messages call for all P particles
            alphas_t, gamma_mus_t, gamma_vars_t, new_lstm_state = \
                ssl.forward_messages(lstm_state, z_prev_all, x_t)
            # alphas_t: (P,), gamma_mus_t: (P, d_z), gamma_vars_t: (P, d_z, d_z)

            # Step (a): Fix reference path (particle 0)
            if t > 0:
                log_w_norm = alphas_t - tf.reduce_logsumexp(alphas_t)
                anc = tf.random.categorical(
                    log_w_norm[tf.newaxis, :], P - 1
                )[0]
                anc = tf.cast(anc, tf.int32)
                anc = tf.concat([[0], anc], axis=0)  # particle 0 keeps reference
            else:
                anc = tf.range(P, dtype=tf.int32)

            ancestors = ancestors.write(t, anc)

            # Step (c): Sample z_t^p from gamma (batched), override particle 0
            # Gather gamma params for ancestor indices
            gmu_anc = tf.gather(gamma_mus_t, anc)      # (P, d_z)
            gvar_anc = tf.gather(gamma_vars_t, anc)    # (P, d_z, d_z)

            L = tf.linalg.cholesky(gvar_anc + tf.eye(d_z) * 1e-6)
            eps = tf.random.normal([P, d_z, 1])
            z_sampled = gmu_anc + tf.squeeze(tf.matmul(L, eps), axis=-1)  # (P, d_z)

            # Particle 0 uses reference (conditional SMC) after first iteration
            if it > 0:
                ref_mask = tf.concat(
                    [tf.ones([1, d_z]), tf.zeros([P - 1, d_z])], axis=0
                )
                z_new_t = ref_mask * z_ref[t:t + 1] + (1.0 - ref_mask) * z_sampled
            else:
                z_new_t = z_sampled

            z_particles = z_particles.write(t, z_new_t)
            log_weights = log_weights.write(t, alphas_t)

            # Log marginal likelihood increment
            log_ml += tf.reduce_logsumexp(alphas_t) - tf.math.log(tf.cast(P, tf.float32))

            # Resample LSTM states via tf.gather (batched)
            h, c = new_lstm_state
            lstm_state = (tf.gather(h, anc), tf.gather(c, anc))
            z_prev_all = z_new_t

        # Final resampling: pick a path
        final_log_w = log_weights.read(T - 1)
        final_w_norm = final_log_w - tf.reduce_logsumexp(final_log_w)
        chosen = tf.random.categorical(final_w_norm[tf.newaxis, :], 1)[0, 0]
        chosen = int(chosen.numpy())

        # Trace back the ancestry to get the full path
        all_z = z_particles.stack()  # (T, P, d_z)
        all_anc = ancestors.stack()  # (T, P)

        z_path = tf.TensorArray(dtype=tf.float32, size=T)
        idx = chosen
        z_path = z_path.write(T - 1, all_z[T - 1, idx])
        for t in range(T - 2, -1, -1):
            idx = int(all_anc[t + 1, idx].numpy())
            z_path = z_path.write(t, all_z[t, idx])

        z_ref = z_path.stack()  # (T, d_z) — new reference path
        z_samples.append(z_ref)
        log_marginals.append(float(log_ml))

        # ============================================================
        # M-step: update SSL parameters given z_ref
        # ============================================================
        m_loss_sum = 0.0
        for _ in range(n_m_steps):
            with tf.GradientTape() as tape:
                # Compute joint log-likelihood: log p(z_{1:T}) + log p(x_{1:T}|z_{1:T})
                lstm_st = ssl.get_initial_lstm_state(batch_size=1)
                z_prev = tf.zeros([1, d_z])
                total_ll = 0.0

                for t in range(T):
                    mu_t, sigma_t, lstm_st = ssl.transition_params(lstm_st, z_prev)
                    # Transition LL
                    total_ll += ssl.transition_log_prob(
                        z_ref[t:t+1], mu_t, sigma_t
                    )[0]
                    # Emission LL
                    total_ll += ssl.emission_log_prob(
                        z_ref[t:t+1], x_obs[t:t+1]
                    )[0]
                    z_prev = z_ref[t:t+1]

                loss = -total_ll  # minimize negative log-likelihood

            grads = tape.gradient(loss, ssl.trainable_variables)
            grads = [tf.clip_by_norm(g, 5.0) if g is not None else g
                     for g in grads]
            optimizer.apply_gradients(
                [(g, v) for g, v in zip(grads, ssl.trainable_variables) if g is not None]
            )
            m_loss_sum += float(loss)

        avg_m_loss = m_loss_sum / n_m_steps
        ssl_losses.append(avg_m_loss)

        if verbose and (it + 1) % max(1, n_iterations // 20) == 0:
            print(f"  PG iter {it+1:4d}/{n_iterations}  "
                  f"log_ml={log_ml:.2f}  M-loss={avg_m_loss:.2f}")

    return PGResult(
        z_samples=tf.stack(z_samples),
        log_marginal=tf.constant(log_marginals, dtype=tf.float32),
        ssl_losses=ssl_losses,
    )


__all__ = ["PGResult", "particle_gibbs_ssl"]
