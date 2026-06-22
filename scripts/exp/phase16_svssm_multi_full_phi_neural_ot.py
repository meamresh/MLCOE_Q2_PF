"""Phase 16: DeepONet NN-OT resampling for V1 multivariate (upper-tri Phi).

Mirror of Section 2's Phase 2-9 work, extended to d=2 upper-triangular
Phi:

  Phase 2 (univariate)  -> trained mGradNet, matched Sinkhorn within
                            0.1 log-units at N=64.
  Phase 6 (univariate)  -> XLA unblock, 1.3-9.3x speedup over Sinkhorn.
  Phase 9 (univariate)  -> retrained at N=256, 2.27x speedup.
  Phase 16 (this work)  -> multivariate full-Phi DeepONet, d=2 N=64.

What this script does
~~~~~~~~~~~~~~~~~~~~~
1. Build a DeepONet at state_dim=2 with n_scalar_ctx=12.
2. Build the Sinkhorn-baseline multivariate filter
   (`DifferentiableLEDHLogLikelihoodSVSSMmulti`) and the NN-OT
   variant (`DifferentiableLEDHNeuralOTSVSSMmulti`, this PR).
3. Sanity-check: one forward call on synthetic data
   - Shape correctness
   - Output finite
   - Untrained NN-OT log-p differs from Sinkhorn (sanity: random
     resample target should be very far from optimal).
4. Time per-call wall for Sinkhorn vs NN-OT (untrained).
5. Print the per-step decomposition that motivates the build:
   - Resample share of per-call time (Sinkhorn) -- expected ~5% at
     T=200 N=64.
6. Print the path to wire up training (TODO marker pointing at
   svssm_neural_ot_training.py which we'd extend for multivariate
   in a follow-up).

This is the SMOKE phase that proves the wiring works end-to-end.
Training the operator to actually match Sinkhorn is a separate
~95s pipeline that follows the Phase 2 recipe (a 3^4 theta grid x
T=20 snapshot set, supervised MSE loss on Sinkhorn-optimal map).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import tensorflow as tf

from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_multivariate import (
    DifferentiableLEDHNeuralOTSVSSMmulti,
    SVSSM_MULTI_CTX_DIM_D2,
)


def gen_data(T: int, d: int = 2, seed: int = 0):
    """Synthetic upper-tri Phi SVSSM data (same defaults as the laptop runs)."""
    rng = np.random.default_rng(seed)
    mu = np.array([0.0, -0.3], dtype=np.float32)
    Phi = np.array([[0.95, 0.05], [0.0, 0.85]], dtype=np.float32)
    sig = np.array([0.3, 0.4], dtype=np.float32)
    L = np.diag(sig).astype(np.float32)

    h = np.zeros((T, d), dtype=np.float32)
    # stationary init via simple draw at mu + sig (close enough for smoke)
    h[0] = mu + sig * rng.standard_normal(d) / np.sqrt(1 - np.diag(Phi) ** 2)
    for t in range(1, T):
        h[t] = mu + (Phi @ (h[t - 1] - mu)) + sig * rng.standard_normal(d)
    y = np.exp(h / 2) * rng.standard_normal((T, d)).astype(np.float32)
    return (
        tf.constant(y, dtype=tf.float32),
        tf.constant(mu, dtype=tf.float32),
        tf.constant(Phi, dtype=tf.float32),
        tf.constant(L, dtype=tf.float32),
        tf.constant(sig ** 2, dtype=tf.float32),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_basis", type=int, default=64)
    p.add_argument("--d_branch", type=int, default=64)
    p.add_argument("--d_trunk", type=int, default=64)
    p.add_argument("--n_calls", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    tf.random.set_seed(args.seed)
    d = 2

    print(f"[phase16-mv-deeponet] T={args.T} N={args.N} d={d}")
    print(f"  n_basis={args.n_basis} d_branch={args.d_branch} d_trunk={args.d_trunk}")
    print(f"  context dim = {SVSSM_MULTI_CTX_DIM_D2}")

    # --- Synthetic data ----------------------------------------------------
    y_obs, mu, Phi, L_eta, sigma_eta_sq_diag = gen_data(args.T, d, args.seed)
    print(f"  y_obs shape: {tuple(y_obs.shape)}  range [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")

    # --- Build the untrained DeepONet -------------------------------------
    print("\n[build] DeepONetMonotoneOT(state_dim=2, n_scalar_ctx=12)")
    model = DeepONetMonotoneOT(
        state_dim=d,
        n_basis=args.n_basis,
        d_branch=args.d_branch,
        d_trunk=args.d_trunk,
        n_scalar_ctx=SVSSM_MULTI_CTX_DIM_D2,
    )

    # Trigger weight allocation with a dummy forward pass.
    dummy_p = tf.random.normal([1, args.N, d])
    dummy_w = tf.ones([1, args.N], dtype=tf.float32) / args.N
    dummy_ctx = tf.zeros([1, SVSSM_MULTI_CTX_DIM_D2])
    dummy_out = model(dummy_p, dummy_w, dummy_ctx)
    n_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    print(f"  DeepONet trainable params: {n_params:,}")
    print(f"  dummy forward shape: {tuple(dummy_out.shape)} "
          f"(expected (1, {args.N}, {d}))")

    # --- Build filters ----------------------------------------------------
    print("\n[build] Sinkhorn filter (baseline)")
    sk = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=d, num_particles=args.N, n_lambda=10,
        sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        grad_window=4, jit_compile=True,
    )

    print("[build] NN-OT filter (untrained DeepONet, jit_compile=True)")
    nn = DifferentiableLEDHNeuralOTSVSSMmulti(
        neural_ot_model=model,
        state_dim=d, num_particles=args.N, n_lambda=10,
        sinkhorn_epsilon=1.0,
        grad_window=4,
        jit_compile=True,   # Phase 16 XLA unlock; safe since .numpy() removed
    )

    # --- Forward sanity check + timing ------------------------------------
    print("\n[sanity] one forward pass on each filter")
    # warmup compile
    lp_sk0 = float(sk.call_mat_phi(mu, Phi, L_eta, y_obs))
    lp_nn0 = float(nn.call_mat_phi(mu, Phi, L_eta, y_obs))
    print(f"  Sinkhorn  log-p (warm) = {lp_sk0:+.4f}")
    print(f"  NN-OT     log-p (warm) = {lp_nn0:+.4f}  "
          f"(untrained: |delta| should be LARGE)")
    print(f"  |delta| = {abs(lp_sk0 - lp_nn0):.4f}")

    print("\n[time] per-call wall, mean over {} calls".format(args.n_calls))
    t0 = time.time()
    for _ in range(args.n_calls):
        sk.call_mat_phi(mu, Phi, L_eta, y_obs)
    wall_sk = (time.time() - t0) / args.n_calls
    print(f"  Sinkhorn:  {wall_sk * 1000:.1f} ms / call  ({wall_sk:.3f}s)")

    t0 = time.time()
    for _ in range(args.n_calls):
        nn.call_mat_phi(mu, Phi, L_eta, y_obs)
    wall_nn = (time.time() - t0) / args.n_calls
    print(f"  NN-OT:     {wall_nn * 1000:.1f} ms / call  ({wall_nn:.3f}s)")

    speedup = wall_sk / wall_nn if wall_nn > 0 else float("nan")
    print()
    print(f"  Wall ratio Sinkhorn / NN-OT = {speedup:.2f}x")
    print(f"  (Phase 6 univariate at T=20 N=64: 1.29x;  "
          f"Phase 9 univariate at N=256: 2.27x)")
    print(f"  At this scale the NN-OT call is per-step constant, while "
          f"Sinkhorn is O(N^2 K).")

    # --- Diagnostic: where the time goes ----------------------------------
    print("\n[diagnostic] resample share of per-call time (estimated)")
    print(f"  Sinkhorn resample cost: K * N^2 ops per timestep "
          f"-> ~{args.T} * 10 * {args.N}^2 = "
          f"{args.T * 10 * args.N * args.N:,} float ops")
    print(f"  At T=200 N=64 the resample share was measured at ~4-5% "
          f"(see project_section2_phase11.md). So even a perfect "
          f"(zero-cost) NN-OT gives ~5% headroom at this scale; the "
          f"asymptotic 1.5-2x win shows up at N >= 256.")

    print()
    print("[next steps]")
    print("  1. Training data generator: extend svssm_neural_ot_training.py")
    print("     to multivariate (12-D ctx, (N, 2) particles, sinkhorn-target).")
    print("  2. Train on 3^4 theta grid x T=20 snapshots (~95s on M-series).")
    print("  3. Validate: filter forward log-p delta vs Sinkhorn within "
          "MC noise (Phase 2: |delta|=0.10).")
    print("  4. HMC driver: same as exp_hmc_svssm_multivariate_full_phi.py "
          "but build_filter -> NN-OT variant.")
    print("  5. Vehtari posterior match vs Sinkhorn (KS p > 0.05 on all "
          "7 raw params, Phase 6 standard).")


if __name__ == "__main__":
    main()
