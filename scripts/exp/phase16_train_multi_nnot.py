"""Phase 16 training pipeline: train DeepONet on multivariate full-Phi
Sinkhorn-target data, then verify against the laptop Sinkhorn smoke 2.

Stages
~~~~~~
1. Build a small theta grid that brackets the truth used in smoke 2.
2. For each theta, run the multivariate full-Phi Sinkhorn filter and
   capture (particles_norm, weights, ctx12, target_norm) at every
   timestep. ``_capture_one`` is the multivariate analogue of
   ``_run_one_filter_capture`` in svssm_neural_ot_training.py.
3. Train DeepONet(state_dim=2, n_basis=64, n_scalar_ctx=12) with
   supervised MSE for a small number of epochs.
4. Validate:
   - filter forward |Δ log-p| at truth theta vs Sinkhorn baseline.
5. Save weights + summary; print the wall comparison for the same
   forward call (Sinkhorn vs NN-OT at training-matched config).

Defaults are scoped for a ~3-5 min total run; pass --max_epochs and
--n_theta to scale.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti, expm_2x2_batch, _safe_nd,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_multivariate import (
    DifferentiableLEDHNeuralOTSVSSMmulti,
    SVSSM_MULTI_CTX_DIM_D2,
    build_svssm_multi_context_scalars,
    _compute_ess,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    LOG_CHI2_MEAN, LOG_CHI2_VAR, _EPS,
)
from src.filters.dpf.resampling import det_resample


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

def gen_y_obs(T: int, mu: np.ndarray, Phi: np.ndarray, sigma: np.ndarray,
              seed: int) -> tf.Tensor:
    rng = np.random.default_rng(seed)
    d = mu.shape[0]
    h = np.zeros((T, d), dtype=np.float32)
    h[0] = mu + sigma * rng.standard_normal(d) / np.sqrt(
        np.maximum(1 - np.diag(Phi) ** 2, 1e-6))
    for t in range(1, T):
        h[t] = mu + (Phi @ (h[t - 1] - mu)) + sigma * rng.standard_normal(d)
    y = np.exp(h / 2) * rng.standard_normal((T, d)).astype(np.float32)
    return tf.constant(y, dtype=tf.float32)


def sample_thetas(n_theta: int, seed: int = 0) -> List[dict]:
    """Sample n_theta points from a distribution bracketing the smoke-2 truth.

    Truths: mu=(0.0,-0.3), phi_diag=(0.95,0.85), phi_off=0.05,
    sigma_eta=(0.3,0.4).
    """
    rng = np.random.default_rng(seed)
    thetas = []
    for _ in range(n_theta):
        mu = np.array([0.0, -0.3], dtype=np.float32) + \
             0.3 * rng.standard_normal(2).astype(np.float32)
        phi_diag = np.clip(
            np.array([0.95, 0.85], dtype=np.float32) +
            0.05 * rng.standard_normal(2).astype(np.float32),
            -0.99, 0.99,
        )
        phi_off = float(0.05 + 0.15 * rng.standard_normal())
        sigma = np.maximum(
            np.array([0.3, 0.4], dtype=np.float32) +
            0.1 * rng.standard_normal(2).astype(np.float32),
            0.05,
        )
        thetas.append({"mu": mu, "phi_diag": phi_diag,
                        "phi_off": float(phi_off), "sigma_eta": sigma})
    return thetas


# ---------------------------------------------------------------------------
# Capture one filter trajectory: (particles, weights, ctx, target) at each t
# ---------------------------------------------------------------------------

def _capture_one(theta: dict, y_obs: tf.Tensor, N: int, n_lambda: int,
                  sinkhorn_eps: float, sinkhorn_iters: int) -> list:
    """Multivariate full-Phi analogue of _run_one_filter_capture.

    Runs the full-Phi filter, captures (p_norm, w, ctx_12, target_norm) at
    each timestep using Sinkhorn as the target. Then continues with the
    Sinkhorn output (so subsequent timesteps see realistic state).
    """
    mu = tf.constant(theta["mu"], tf.float32)
    Phi_np = np.array([[theta["phi_diag"][0], theta["phi_off"]],
                          [0.0, theta["phi_diag"][1]]], dtype=np.float32)
    Phi = tf.constant(Phi_np, tf.float32)
    sigma = tf.constant(theta["sigma_eta"], tf.float32)
    sigma_sq_diag = sigma ** 2
    L_eta = tf.linalg.diag(sigma)
    Sigma_eta = tf.linalg.diag(sigma_sq_diag)

    T = int(y_obs.shape[0])
    d = 2
    R_val = tf.constant(LOG_CHI2_VAR, tf.float32)
    R_inv = 1.0 / R_val
    mu_z = tf.constant(LOG_CHI2_MEAN, tf.float32)
    log_y_sq_offset = 1e-8
    z_obs = tf.math.log(tf.square(tf.cast(y_obs, tf.float32)) + log_y_sq_offset)

    # Geometric pseudo-time
    q = 1.2
    eps1 = (1.0 - q) / (1.0 - q ** n_lambda)
    epsilons = [eps1 * q ** j for j in range(n_lambda)]

    # Stationary init via Smith doubling on (Phi, Sigma_eta)
    X = tf.identity(Sigma_eta); A_mat = tf.identity(Phi)
    for _ in range(15):
        X = X + A_mat @ X @ tf.transpose(A_mat)
        A_mat = A_mat @ A_mat
    Sigma_h0 = 0.5 * (X + tf.transpose(X))
    Sigma_h0 = tf.clip_by_value(Sigma_h0, -1e3, 1e3)

    L0 = tf.linalg.cholesky(Sigma_h0 + 1e-6 * tf.eye(d, dtype=tf.float32))
    particles = mu[tf.newaxis, :] + tf.einsum(
        "ij,nj->ni", L0, tf.random.normal([N, d]))
    P = tf.tile(Sigma_h0[tf.newaxis, :, :], [N, 1, 1])
    log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

    captured = []
    I_d = tf.eye(d, dtype=tf.float32)
    I_d_batch = tf.tile(I_d[tf.newaxis, :, :], [N, 1, 1])

    for t_int in range(1, T + 1):
        z_t = z_obs[t_int - 1]

        # Predict
        if t_int >= 2:
            x_det = mu[tf.newaxis, :] + tf.einsum(
                "nj,ij->ni", particles - mu[tf.newaxis, :], Phi)
            noise = tf.einsum("ij,nj->ni", L_eta, tf.random.normal([N, d]))
            particles = tf.clip_by_value(_safe_nd(x_det + noise), -50.0, 50.0)
            PhiP = tf.einsum("ik,nkl->nil", Phi, P)
            PhiPPhiT = tf.einsum("nil,jl->nij", PhiP, Phi)
            P = tf.clip_by_value(PhiPPhiT + Sigma_eta[tf.newaxis, :, :],
                                   -1e3, 1e3)

        # LEDH flow
        eta = tf.identity(particles)
        log_det_jac = tf.zeros([N])
        lam_cum = 0.0
        innov_const = tf.clip_by_value(z_t - mu_z, -100.0, 100.0)
        innov_const_batch = tf.tile(innov_const[tf.newaxis, :], [N, 1])
        P = tf.where(tf.math.is_finite(P), P, tf.zeros_like(P))
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))

        for j in range(n_lambda):
            eps_j = epsilons[j]
            lam_k = lam_cum + eps_j / 2.0
            lam_cum += eps_j
            S = lam_k * P + R_val * I_d_batch
            S_ridge = S + 1e-3 * I_d_batch
            S_T = tf.linalg.matrix_transpose(S_ridge)
            P_T = tf.linalg.matrix_transpose(P)
            A_T = tf.linalg.solve(S_T, P_T)
            A = -0.5 * tf.linalg.matrix_transpose(A_T)
            A = tf.clip_by_value(A, -10.0, 10.0)
            I_lam_A = I_d_batch + lam_k * A
            I_2lam_A = I_d_batch + 2.0 * lam_k * A
            P_innov = tf.einsum("nij,nj->ni", P, innov_const_batch)
            ILA_P_innov = tf.einsum("nij,nj->ni", I_lam_A, P_innov) * R_inv
            A_eta = tf.einsum("nij,nj->ni", A, eta)
            b_vec = tf.einsum("nij,nj->ni", I_2lam_A, ILA_P_innov + A_eta)
            b_vec = tf.clip_by_value(b_vec, -100.0, 100.0)
            A_eps = A * eps_j
            exp_Aeps = expm_2x2_batch(A_eps)
            A_ridge = A + 1e-3 * I_d_batch
            exp_minus_I_b = tf.einsum("nij,nj->ni",
                                        exp_Aeps - I_d_batch, b_vec)
            phi_Ab = tf.linalg.solve(
                A_ridge, exp_minus_I_b[..., tf.newaxis])[..., 0]
            particles = tf.clip_by_value(_safe_nd(
                tf.einsum("nij,nj->ni", exp_Aeps, particles) + phi_Ab),
                -50.0, 50.0)
            eta = tf.clip_by_value(_safe_nd(
                tf.einsum("nij,nj->ni", exp_Aeps, eta) + phi_Ab),
                -50.0, 50.0)
            log_det_jac = log_det_jac + tf.linalg.trace(A_eps)

        # Weights
        resid = z_t[tf.newaxis, :] - (particles + mu_z)
        log_lik = (
            -0.5 * R_inv * tf.reduce_sum(resid ** 2, axis=1)
            - 0.5 * float(d) * tf.math.log(R_val)
            - 0.5 * float(d) * tf.math.log(2.0 * 3.141592653589793)
        )
        log_lik = _safe_nd(log_lik)
        log_w_incr = log_lik + log_det_jac
        log_w_incr = tf.where(tf.math.is_finite(log_w_incr), log_w_incr,
                                tf.constant(-100.0, tf.float32))
        log_w_t = log_w + log_w_incr
        log_w_t = log_w_t - tf.reduce_logsumexp(log_w_t)

        # Capture + Sinkhorn target
        p_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(particles, axis=0, keepdims=True) + _EPS
        p_norm = (particles - p_mean) / p_std
        w_normed = tf.nn.softmax(log_w_t, axis=0)
        ess = _compute_ess(w_normed)

        ctx = build_svssm_multi_context_scalars(
            mu=mu, Phi=Phi, sigma_eta_sq_diag=sigma_sq_diag,
            t=tf.constant(0.0, tf.float32), z_t=z_t, ess=ess,
            epsilon=tf.constant(sinkhorn_eps, tf.float32),
            T_max=tf.constant(float(T), tf.float32), d=d,
        )

        target_norm, _ = det_resample(
            p_norm, log_w_t,
            epsilon=sinkhorn_eps, n_iters=sinkhorn_iters,
        )
        target_norm = tf.cast(tf.math.real(target_norm), tf.float32)

        captured.append({
            "particles_norm": p_norm.numpy().astype(np.float32),     # (N, d)
            "weights": w_normed.numpy().astype(np.float32),          # (N,)
            "ctx": ctx.numpy().astype(np.float32),                   # (12,)
            "target_norm": target_norm.numpy().astype(np.float32),   # (N, d)
        })

        # Continue with Sinkhorn output
        particles = target_norm * p_std + p_mean
        P_mean = tf.reduce_mean(P, axis=0, keepdims=True)
        P = tf.tile(P_mean, [N, 1, 1])
        log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

    return captured


# ---------------------------------------------------------------------------
# Generate the full dataset
# ---------------------------------------------------------------------------

def generate_dataset(thetas: List[dict], T: int, N: int, n_lambda: int,
                      sinkhorn_eps: float, sinkhorn_iters: int,
                      seeds_per_theta: int = 1, base_seed: int = 42,
                      verbose: bool = True):
    pn, w, ctx, tn = [], [], [], []
    t0 = time.time()
    for ti, theta in enumerate(thetas):
        for s in range(seeds_per_theta):
            data_seed = base_seed + ti * 100 + s
            y_obs = gen_y_obs(T, theta["mu"],
                               np.array([[theta["phi_diag"][0], theta["phi_off"]],
                                         [0.0, theta["phi_diag"][1]]],
                                         dtype=np.float32),
                               theta["sigma_eta"], seed=data_seed)
            caps = _capture_one(theta, y_obs, N, n_lambda,
                                  sinkhorn_eps, sinkhorn_iters)
            for c in caps:
                pn.append(c["particles_norm"])
                w.append(c["weights"])
                ctx.append(c["ctx"])
                tn.append(c["target_norm"])
        if verbose and (ti + 1) % max(1, len(thetas) // 10) == 0:
            print(f"  [data-gen] theta {ti+1}/{len(thetas)}  "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return (np.stack(pn), np.stack(w), np.stack(ctx), np.stack(tn))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_deeponet(model, ds, val_frac=0.2, batch_size=64, max_epochs=30,
                    lr=1e-3, patience=5, seed=0, verbose=True):
    pn, w, ctx, tn = ds
    M = pn.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    n_val = int(round(val_frac * M))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    pn_tr, w_tr, ctx_tr, tn_tr = pn[tr_idx], w[tr_idx], ctx[tr_idx], tn[tr_idx]
    pn_va, w_va, ctx_va, tn_va = pn[val_idx], w[val_idx], ctx[val_idx], tn[val_idx]

    opt = tf.keras.optimizers.Adam(lr)

    @tf.function
    def step(particles, weights, ctxv, target):
        with tf.GradientTape() as tape:
            pred = model(particles, weights, ctxv)
            loss = tf.reduce_mean(tf.square(pred - target))
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_loss(particles, weights, ctxv, target):
        return tf.reduce_mean(tf.square(model(particles, weights, ctxv) - target))

    best_val = float("inf")
    best_weights = None
    bad_epochs = 0
    history = {"train": [], "val": []}

    t_start = time.time()
    for epoch in range(max_epochs):
        # Train
        order = rng.permutation(len(pn_tr))
        epoch_losses = []
        for i in range(0, len(pn_tr), batch_size):
            idx = order[i:i + batch_size]
            l = step(tf.constant(pn_tr[idx]), tf.constant(w_tr[idx]),
                      tf.constant(ctx_tr[idx]), tf.constant(tn_tr[idx]))
            epoch_losses.append(float(l))
        # Val
        val_losses = []
        for i in range(0, len(pn_va), batch_size):
            vl = val_loss(
                tf.constant(pn_va[i:i+batch_size]),
                tf.constant(w_va[i:i+batch_size]),
                tf.constant(ctx_va[i:i+batch_size]),
                tf.constant(tn_va[i:i+batch_size]),
            )
            val_losses.append(float(vl))
        train_mean = float(np.mean(epoch_losses))
        val_mean = float(np.mean(val_losses))
        history["train"].append(train_mean)
        history["val"].append(val_mean)
        if verbose:
            print(f"  epoch {epoch+1:3d}/{max_epochs}  "
                  f"train_mse={train_mean:.5f}  val_mse={val_mean:.5f}",
                  flush=True)
        if val_mean < best_val * (1 - 1e-4):
            best_val = val_mean
            best_weights = [v.numpy() for v in model.trainable_variables]
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"  early stop at epoch {epoch+1}, "
                           f"no improvement for {patience} epochs")
                break

    if best_weights is not None:
        for v, bw in zip(model.trainable_variables, best_weights):
            v.assign(bw)
    history["wall_s"] = time.time() - t_start
    history["best_val_mse"] = best_val
    return history


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=20, help="training-data series length")
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_theta", type=int, default=32)
    p.add_argument("--max_epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--n_basis", type=int, default=64)
    p.add_argument("--T_eval", type=int, default=50,
                   help="filter forward validation series length")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/"
                            "HMC_vs_PMMH/phase16_multi_nnot_training")
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(args.seed)
    print(f"[phase16-train] T={args.T} N={args.N} d=2  "
          f"n_theta={args.n_theta}  max_epochs={args.max_epochs}")

    # --- Generate training data ---
    print("\n[1] Generate training data via Sinkhorn capture")
    thetas = sample_thetas(args.n_theta, seed=args.seed)
    ds = generate_dataset(
        thetas, T=args.T, N=args.N, n_lambda=10,
        sinkhorn_eps=1.0, sinkhorn_iters=10,
        seeds_per_theta=1, base_seed=args.seed,
    )
    print(f"  total samples: M={ds[0].shape[0]} "
          f"(shapes p={ds[0].shape} w={ds[1].shape} ctx={ds[2].shape} "
          f"target={ds[3].shape})")

    # --- Build DeepONet ---
    print("\n[2] Build DeepONet")
    model = DeepONetMonotoneOT(
        state_dim=2, n_basis=args.n_basis,
        d_branch=64, d_trunk=64,
        n_scalar_ctx=SVSSM_MULTI_CTX_DIM_D2,
    )
    _ = model(tf.constant(ds[0][:1]), tf.constant(ds[1][:1]),
                tf.constant(ds[2][:1]))
    n_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    print(f"  trainable params: {n_params:,}")

    # --- Train ---
    print("\n[3] Train supervised on Sinkhorn targets")
    history = train_deeponet(model, ds, max_epochs=args.max_epochs,
                              patience=args.patience, seed=args.seed)
    print(f"  best val_mse: {history['best_val_mse']:.5f}  "
          f"wall: {history['wall_s']:.1f}s")

    # --- Save weights ---
    weights_path = out_dir / "deeponet_multi.weights.h5"
    model.save_weights(str(weights_path))
    print(f"  saved {weights_path}")

    # --- Validate: filter forward log-p ---
    print(f"\n[4] Validate: filter forward log-p, T={args.T_eval} N={args.N}")
    truth = {
        "mu": np.array([0.0, -0.3], dtype=np.float32),
        "phi_diag": np.array([0.95, 0.85], dtype=np.float32),
        "phi_off": 0.05,
        "sigma_eta": np.array([0.3, 0.4], dtype=np.float32),
    }
    Phi_truth = np.array([[truth["phi_diag"][0], truth["phi_off"]],
                            [0.0, truth["phi_diag"][1]]], dtype=np.float32)
    y_obs = gen_y_obs(args.T_eval, truth["mu"], Phi_truth,
                       truth["sigma_eta"], seed=999)
    mu = tf.constant(truth["mu"], tf.float32)
    Phi = tf.constant(Phi_truth, tf.float32)
    L_eta = tf.linalg.diag(tf.constant(truth["sigma_eta"], tf.float32))

    sk_filter = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=2, num_particles=args.N, n_lambda=10,
        sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        grad_window=4, jit_compile=True,
    )
    nn_filter = DifferentiableLEDHNeuralOTSVSSMmulti(
        neural_ot_model=model,
        state_dim=2, num_particles=args.N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4, jit_compile=True,
    )

    tf.random.set_seed(0)
    lp_sk = float(sk_filter.call_mat_phi(mu, Phi, L_eta, y_obs))
    tf.random.set_seed(0)
    lp_nn = float(nn_filter.call_mat_phi(mu, Phi, L_eta, y_obs))
    print(f"  Sinkhorn  log-p = {lp_sk:+.4f}")
    print(f"  NN-OT     log-p = {lp_nn:+.4f}")
    print(f"  |delta| = {abs(lp_sk - lp_nn):.4f}  "
          f"(Phase 2 univariate target: <= 0.5)")

    # --- Time both, 3 calls each ---
    print("\n[5] Time per-call wall (3 calls each, warm)")
    t0 = time.time()
    for _ in range(3):
        sk_filter.call_mat_phi(mu, Phi, L_eta, y_obs)
    sk_wall = (time.time() - t0) / 3
    t0 = time.time()
    for _ in range(3):
        nn_filter.call_mat_phi(mu, Phi, L_eta, y_obs)
    nn_wall = (time.time() - t0) / 3
    print(f"  Sinkhorn:  {sk_wall*1000:.1f} ms/call")
    print(f"  NN-OT:     {nn_wall*1000:.1f} ms/call")
    print(f"  Ratio:     {sk_wall/nn_wall:.2f}x")

    # --- Save summary ---
    summary = {
        "T": args.T, "N": args.N, "n_theta": args.n_theta,
        "n_samples": int(ds[0].shape[0]),
        "max_epochs": args.max_epochs, "patience": args.patience,
        "best_val_mse": history["best_val_mse"],
        "training_wall_s": history["wall_s"],
        "filter_lp_sinkhorn": lp_sk,
        "filter_lp_nnot": lp_nn,
        "filter_lp_delta": abs(lp_sk - lp_nn),
        "filter_wall_sinkhorn_ms": sk_wall * 1000,
        "filter_wall_nnot_ms": nn_wall * 1000,
        "filter_wall_ratio": sk_wall / nn_wall,
        "history_train": history["train"],
        "history_val": history["val"],
    }
    (out_dir / "phase16_train_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\n  [save] {out_dir / 'phase16_train_summary.json'}")


if __name__ == "__main__":
    main()
