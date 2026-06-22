"""Architecture bisect: train the SAME DeepONetMonotoneOT we use for
multivariate (n_basis=64, d_branch=64, d_trunk=64) on univariate data
and verify it reproduces Phase 6/9 univariate result.

If |Δ log p| within MC noise (Phase 2 target ≤ 0.5) → architecture is
fine, the bug is in the multivariate filter wrapper.

If |Δ log p| huge → the architecture itself has issues, regardless of
state_dim.
"""

from __future__ import annotations

import time
import numpy as np
import tensorflow as tf

from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    DifferentiableLEDHLogLikelihoodSVSSM,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm import (
    DifferentiableLEDHNeuralOTSVSSM,
)
from src.filters.bonus.extra_bonus.svssm_neural_ot_training import (
    generate_svssm_training_data, SVSSMNeuralOTTrainer,
)


def main():
    T, N = 20, 64
    n_basis = 64  # Same as my multivariate v2/v3 config

    # Phase 2 theta grid: 3x3x3 = 27 thetas
    mu_vals = np.linspace(-1.5, 1.5, 3)
    phi_raw_vals = np.linspace(1.5, 2.5, 3)
    phi_vals = np.tanh(phi_raw_vals)
    log_sig_vals = np.linspace(-3.5, -0.5, 3)
    sigma_eta_vals = np.exp(0.5 * log_sig_vals)
    grid = [(float(m), float(p), float(s))
            for m in mu_vals for p in phi_vals for s in sigma_eta_vals]
    print(f"[arch-bisect-d1] theta grid: {len(grid)} pts, T={T}, N={N}")

    print("\n[1] Generate training data")
    ds = generate_svssm_training_data(
        theta_grid=grid, T=T, N=N, n_lambda=10,
        sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        integrator="exp", seeds_per_theta=3, base_seed=42, verbose=False,
    )
    print(f"  total samples: {len(ds)}")

    print("\n[2] Build DeepONetMonotoneOT(state_dim=1, n_basis=64, "
          "d_branch=64, d_trunk=64) -- matches my multivariate config")
    model = DeepONetMonotoneOT(
        state_dim=1, n_basis=n_basis,
        d_branch=64, d_trunk=64,
        n_scalar_ctx=7,
    )
    # Allocate weights
    pn_first = tf.constant(ds.particles_norm[:1, :, None], tf.float32)  # (1, N, 1)
    w_first = tf.constant(ds.weights[:1], tf.float32)
    ctx_first = tf.constant(ds.ctx[:1], tf.float32)
    _ = model(pn_first, w_first, ctx_first)
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  trainable params: {n_params:,}")

    print("\n[3] Train (Phase 2 recipe: max_epochs=80, patience=12)")
    # The trainer expects 1D particles in (M, N) shape, not (M, N, 1).
    # Look at the univariate trainer signature to understand.
    # SVSSMNeuralOTTrainer's _supervised_loss does:
    #   pred = self.model(particles, weights, ctx)
    # The model needs (M, N, d) for trunk net. Need to wrap.

    # Easier: subclass the trainer's train method to add the d=1 axis
    # before calling the model. Or: hot-patch the ds to add the axis.
    # The cleanest: just monkey-patch the model's __call__ to handle 1-D.

    class WrappedModel(tf.keras.Model):
        def __init__(self, base):
            super().__init__()
            self.base = base
            # Mirror n_ridges for compatibility with trainer.
            self.n_ridges = base.n_ridges
        def call(self, particles, weights, ctx):
            # particles arrive as (M, N); we need (M, N, 1) for the operator
            if len(particles.shape) == 2:
                particles_3d = particles[..., tf.newaxis]
                out_3d = self.base(particles_3d, weights, ctx)
                return out_3d[..., 0]  # back to (M, N)
            return self.base(particles, weights, ctx)

    wrapped = WrappedModel(model)
    # Allocate wrapped weights
    _ = wrapped(tf.constant(ds.particles_norm[:1], tf.float32),
                  tf.constant(ds.weights[:1], tf.float32),
                  tf.constant(ds.ctx[:1], tf.float32))

    train_ds, val_ds = ds.split_train_val(val_frac=0.2, seed=0)
    trainer = SVSSMNeuralOTTrainer(
        model=wrapped, learning_rate=1e-3, batch_size=64,
        max_epochs=80, patience=12, loss_mode="supervised",
    )
    history = trainer.train(train_ds, val_ds, verbose=False)
    print(f"  best val_mse: {history.best_val_loss:.5f}  "
          f"epochs: {history.best_epoch + 1}/{trainer.max_epochs}  "
          f"wall: {history.elapsed_s:.1f}s")

    print("\n[4] Filter forward log-p comparison at TRUTH (mu=0, phi=0.95, sig=0.3)")
    mu_t, phi_t, sigma_t = 0.0, 0.95, 0.3
    rng = np.random.default_rng(42)
    h = np.zeros(T, dtype=np.float32)
    h[0] = mu_t + sigma_t * rng.standard_normal() / np.sqrt(1 - phi_t ** 2)
    for t in range(1, T):
        h[t] = mu_t + phi_t * (h[t-1] - mu_t) + sigma_t * rng.standard_normal()
    y_obs = tf.constant((np.exp(h / 2) * rng.standard_normal(T)).astype(np.float32),
                          dtype=tf.float32)

    sk = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        grad_window=4, jit_compile=True, integrator="exp",
    )
    nn = DifferentiableLEDHNeuralOTSVSSM(
        neural_ot_model=wrapped,
        num_particles=N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4, jit_compile=True,
        integrator="exp",
    )

    deltas = []
    for trial in range(5):
        tf.random.set_seed(trial)
        lp_sk = float(sk(mu_t, phi_t, sigma_t ** 2, y_obs))
        tf.random.set_seed(trial)
        lp_nn = float(nn(mu_t, phi_t, sigma_t ** 2, y_obs))
        deltas.append(abs(lp_sk - lp_nn))
        print(f"  trial {trial+1}: Sinkhorn={lp_sk:+.4f}  NN-OT={lp_nn:+.4f}  "
              f"|Δ|={abs(lp_sk-lp_nn):.4f}")
    print(f"  mean |Δ|={np.mean(deltas):.4f}  median={np.median(deltas):.4f}")
    print(f"  Phase 2 target: ≤ 0.5;  Phase 9 univariate result: 0.017")

    if np.median(deltas) <= 0.5:
        print("\n  VERDICT: architecture works at d=1 within Phase 2 target. "
              "If multivariate fails, the bug is in the multivariate wrapper.")
    else:
        print("\n  VERDICT: architecture FAILS at d=1. The DeepONet config "
              "we're using is too small / wrong / etc.")

    # Time them
    print("\n[5] Per-call wall (3 calls, warm)")
    sk(mu_t, phi_t, sigma_t**2, y_obs); nn(mu_t, phi_t, sigma_t**2, y_obs)
    t0 = time.time()
    for _ in range(3): sk(mu_t, phi_t, sigma_t**2, y_obs)
    sk_w = (time.time()-t0)/3
    t0 = time.time()
    for _ in range(3): nn(mu_t, phi_t, sigma_t**2, y_obs)
    nn_w = (time.time()-t0)/3
    print(f"  Sinkhorn: {sk_w*1000:.1f} ms")
    print(f"  NN-OT:    {nn_w*1000:.1f} ms")
    print(f"  Ratio:    {sk_w/nn_w:.3f}x  (Phase 6 univariate: 1.29x at T=20 N=64)")


if __name__ == "__main__":
    main()
