"""Phase 16 d=1 bisect: train DeepONet at state_dim=1 through the
SAME multivariate code path we use for d=2, and verify it matches a
univariate Sinkhorn baseline. If yes -> multivariate code is correct
at d=1; the d=2 KS gap is genuinely a d=2 / training-scope issue.

This script mirrors phase16_train_multi_nnot.py but at d=1: Phi is a
1x1 scalar in [-1,1], Sigma_eta is a 1-vector, the captured tuples
are (particles_norm: (N,1), weights: (N,), ctx: (7,), target: (N,1)).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti, expm_2x2_batch, _safe_nd,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_multivariate import (
    DifferentiableLEDHNeuralOTSVSSMmulti,
    svssm_multi_ctx_dim,
    build_svssm_multi_context_scalars,
    _compute_ess,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    LOG_CHI2_MEAN, LOG_CHI2_VAR, _EPS,
)
from src.filters.dpf.resampling import det_resample


def gen_y_obs_d1(T, mu, phi, sigma, seed):
    rng = np.random.default_rng(seed)
    h = np.zeros(T, dtype=np.float32)
    h[0] = mu + sigma * rng.standard_normal() / np.sqrt(max(1 - phi**2, 1e-3))
    for t in range(1, T):
        h[t] = mu + phi * (h[t-1] - mu) + sigma * rng.standard_normal()
    y = (np.exp(h/2) * rng.standard_normal(T)).astype(np.float32)
    return tf.constant(y.reshape(T, 1), tf.float32)


def sample_thetas_d1(n_theta, seed=0, wide=False):
    """Theta sampling for training-data generation.

    wide=False: tight around the bisect truth (mu=0, phi=0.95, sig=0.3).
    wide=True:  covers the posterior bulk of the wide-prior T=200
                reference run (mu CI [-2.1, 1.9], phi tail down to -0.7,
                sigma_eta_sq CI [0.004, 0.98]).
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_theta):
        if wide:
            mu = float(1.5 * rng.standard_normal())
            phi = float(np.clip(np.tanh(1.2 + 1.2 * rng.standard_normal()),
                                  -0.99, 0.99))
            sigma = float(np.clip(np.exp(np.log(0.3) +
                                           0.8 * rng.standard_normal()),
                                    0.05, 1.5))
        else:
            mu = float(0.0 + 0.5 * rng.standard_normal())
            phi = float(np.clip(0.95 + 0.05 * rng.standard_normal(), -0.99, 0.99))
            sigma = float(max(0.3 + 0.15 * rng.standard_normal(), 0.05))
        out.append({"mu": mu, "phi": phi, "sigma": sigma})
    return out


def capture_one_d1(theta, y_obs, N, n_lambda, sinkhorn_eps, sinkhorn_iters):
    mu = tf.constant([theta["mu"]], tf.float32)
    Phi = tf.constant([[theta["phi"]]], tf.float32)
    sigma = tf.constant([theta["sigma"]], tf.float32)
    sigma_sq_diag = sigma ** 2
    L_eta = tf.linalg.diag(sigma)
    Sigma_eta = tf.linalg.diag(sigma_sq_diag)

    T = int(y_obs.shape[0])
    d = 1
    R_val = tf.constant(LOG_CHI2_VAR, tf.float32)
    R_inv = 1.0 / R_val
    mu_z = tf.constant(LOG_CHI2_MEAN, tf.float32)
    z_obs = tf.math.log(tf.square(y_obs) + 1e-8)

    q = 1.2
    eps1 = (1.0 - q) / (1.0 - q ** n_lambda)
    epsilons = [eps1 * q ** j for j in range(n_lambda)]

    # Stationary init via Smith doubling (works at d=1 trivially)
    X = tf.identity(Sigma_eta); A_mat = tf.identity(Phi)
    for _ in range(15):
        X = X + A_mat @ X @ tf.transpose(A_mat)
        A_mat = A_mat @ A_mat
    Sigma_h0 = 0.5 * (X + tf.transpose(X))
    L0 = tf.linalg.cholesky(Sigma_h0 + 1e-6 * tf.eye(d, dtype=tf.float32))
    particles = mu[tf.newaxis, :] + tf.einsum("ij,nj->ni", L0,
                                                 tf.random.normal([N, d]))
    P = tf.tile(Sigma_h0[tf.newaxis, :, :], [N, 1, 1])
    log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

    captured = []
    I_d = tf.eye(d, dtype=tf.float32)
    I_d_batch = tf.tile(I_d[tf.newaxis, :, :], [N, 1, 1])

    for t_int in range(1, T + 1):
        z_t = z_obs[t_int - 1]
        if t_int >= 2:
            x_det = mu[tf.newaxis, :] + tf.einsum(
                "nj,ij->ni", particles - mu[tf.newaxis, :], Phi)
            noise = tf.einsum("ij,nj->ni", L_eta, tf.random.normal([N, d]))
            particles = tf.clip_by_value(_safe_nd(x_det + noise), -50.0, 50.0)
            PhiP = tf.einsum("ik,nkl->nil", Phi, P)
            PhiPPhiT = tf.einsum("nil,jl->nij", PhiP, Phi)
            P = tf.clip_by_value(PhiPPhiT + Sigma_eta[tf.newaxis, :, :],
                                   -1e3, 1e3)

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
            A_T = tf.linalg.solve(tf.linalg.matrix_transpose(S_ridge),
                                    tf.linalg.matrix_transpose(P))
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
            exp_Aeps = tf.linalg.expm(A_eps)   # d=1 falls back to expm
            A_ridge = A + 1e-3 * I_d_batch
            exp_minus_I_b = tf.einsum("nij,nj->ni",
                                        exp_Aeps - I_d_batch, b_vec)
            phi_Ab = tf.linalg.solve(A_ridge,
                                       exp_minus_I_b[..., tf.newaxis])[..., 0]
            particles = tf.clip_by_value(_safe_nd(
                tf.einsum("nij,nj->ni", exp_Aeps, particles) + phi_Ab),
                -50.0, 50.0)
            eta = tf.clip_by_value(_safe_nd(
                tf.einsum("nij,nj->ni", exp_Aeps, eta) + phi_Ab),
                -50.0, 50.0)
            log_det_jac = log_det_jac + tf.linalg.trace(A_eps)

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
            "particles_norm": p_norm.numpy().astype(np.float32),
            "weights": w_normed.numpy().astype(np.float32),
            "ctx": ctx.numpy().astype(np.float32),
            "target_norm": target_norm.numpy().astype(np.float32),
        })

        particles = target_norm * p_std + p_mean
        P_mean = tf.reduce_mean(P, axis=0, keepdims=True)
        P = tf.tile(P_mean, [N, 1, 1])
        log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

    return captured


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_theta", type=int, default=27)
    p.add_argument("--seeds_per_theta", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--n_basis", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wide", action="store_true",
                   help="Wide theta sampling covering the wide-prior "
                        "T=200 reference posterior bulk.")
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                            "phase16_d1_bisect_training")
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(args.seed)
    d = 1
    ctx_dim = svssm_multi_ctx_dim(d)
    print(f"[phase16-d1-bisect] T={args.T} N={args.N} d=1  "
          f"n_theta={args.n_theta}  seeds_per_theta={args.seeds_per_theta}  "
          f"ctx_dim={ctx_dim}")

    # ----- Generate training data
    print("\n[1] Generate training data (multivariate filter at d=1, "
          "Sinkhorn target)")
    thetas = sample_thetas_d1(args.n_theta, seed=args.seed, wide=args.wide)
    pn, w, ctx, tn = [], [], [], []
    t0 = time.time()
    for ti, theta in enumerate(thetas):
        for s in range(args.seeds_per_theta):
            ds = args.seed + ti*100 + s
            y = gen_y_obs_d1(args.T, theta["mu"], theta["phi"], theta["sigma"], ds)
            caps = capture_one_d1(theta, y, args.N, 10, 1.0, 10)
            for c in caps:
                pn.append(c["particles_norm"]); w.append(c["weights"])
                ctx.append(c["ctx"]); tn.append(c["target_norm"])
        if (ti+1) % max(1, len(thetas)//8) == 0:
            print(f"  [data-gen] theta {ti+1}/{len(thetas)}  ({time.time()-t0:.1f}s)", flush=True)
    pn = np.stack(pn); w = np.stack(w); ctx = np.stack(ctx); tn = np.stack(tn)
    print(f"  total samples M={pn.shape[0]}  shapes p={pn.shape} w={w.shape} "
          f"ctx={ctx.shape} target={tn.shape}")

    # ----- Build and train DeepONet
    print("\n[2] Build DeepONet(state_dim=1, ctx_dim=7) — same arch as d=2")
    model = DeepONetMonotoneOT(
        state_dim=1, n_basis=args.n_basis,
        d_branch=64, d_trunk=64, n_scalar_ctx=ctx_dim,
    )
    _ = model(tf.constant(pn[:1]), tf.constant(w[:1]), tf.constant(ctx[:1]))
    n_params = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    print(f"  trainable params: {n_params:,}")

    print("\n[3] Train")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(pn.shape[0])
    n_val = int(round(0.2 * pn.shape[0]))
    val_idx = perm[:n_val]; tr_idx = perm[n_val:]
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def step(p_b, w_b, c_b, t_b):
        with tf.GradientTape() as tape:
            pred = model(p_b, w_b, c_b)
            loss = tf.reduce_mean(tf.square(pred - t_b))
        grads = tape.gradient(loss, model.trainable_variables)
        grads = [tf.clip_by_norm(g, 5.0) if g is not None else g for g in grads]
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    @tf.function
    def val_loss(p_b, w_b, c_b, t_b):
        return tf.reduce_mean(tf.square(model(p_b, w_b, c_b) - t_b))

    best_val = float("inf"); best_w = None; bad = 0
    bs = 64
    t_train = time.time()
    for epoch in range(args.max_epochs):
        order = rng.permutation(len(tr_idx))
        train_losses = []
        for i in range(0, len(tr_idx), bs):
            idx = tr_idx[order[i:i+bs]]
            l = step(tf.constant(pn[idx]), tf.constant(w[idx]),
                      tf.constant(ctx[idx]), tf.constant(tn[idx]))
            train_losses.append(float(l))
        val_losses = []
        for i in range(0, len(val_idx), bs):
            idx = val_idx[i:i+bs]
            vl = val_loss(tf.constant(pn[idx]), tf.constant(w[idx]),
                           tf.constant(ctx[idx]), tf.constant(tn[idx]))
            val_losses.append(float(vl))
        tr_mean = float(np.mean(train_losses)); va_mean = float(np.mean(val_losses))
        print(f"  epoch {epoch+1:3d}/{args.max_epochs}  "
              f"train_mse={tr_mean:.5f}  val_mse={va_mean:.5f}", flush=True)
        if va_mean < best_val * (1 - 1e-4):
            best_val = va_mean
            best_w = [v.numpy() for v in model.trainable_variables]; bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop at epoch {epoch+1}"); break
    if best_w is not None:
        for v, bw in zip(model.trainable_variables, best_w):
            v.assign(bw)
    print(f"  best val_mse={best_val:.5f}  wall={time.time()-t_train:.1f}s")
    weights_path = out_dir / "deeponet_d1.weights.h5"
    model.save_weights(str(weights_path))
    print(f"  saved {weights_path}")

    # ----- Validate filter forward log-p at truth
    print("\n[4] Validate filter forward log-p at TRUTH (mu=0, phi=0.95, sig=0.3)")
    truth_mu, truth_phi, truth_sig = 0.0, 0.95, 0.3
    y_truth = gen_y_obs_d1(args.T, truth_mu, truth_phi, truth_sig, seed=999)
    mu_v = tf.constant([truth_mu], tf.float32)
    Phi_v = tf.constant([[truth_phi]], tf.float32)
    L_v = tf.constant([[truth_sig]], tf.float32)

    sk = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=1, num_particles=args.N, n_lambda=10,
        sinkhorn_epsilon=1.0, sinkhorn_iters=10,
        grad_window=4, jit_compile=True,
    )
    nn = DifferentiableLEDHNeuralOTSVSSMmulti(
        neural_ot_model=model, state_dim=1, num_particles=args.N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4, jit_compile=True,
    )

    deltas = []
    for trial in range(5):
        tf.random.set_seed(trial)
        lp_sk = float(sk.call_mat_phi(mu_v, Phi_v, L_v, y_truth))
        tf.random.set_seed(trial)
        lp_nn = float(nn.call_mat_phi(mu_v, Phi_v, L_v, y_truth))
        deltas.append(abs(lp_sk - lp_nn))
        print(f"  trial {trial+1}: SK={lp_sk:+.4f}  NN-OT={lp_nn:+.4f}  "
              f"|Δ|={abs(lp_sk-lp_nn):.4f}")
    print(f"  median |Δ|={np.median(deltas):.4f}  mean={np.mean(deltas):.4f}")
    print(f"  Phase 2 target: ≤ 0.5;  Phase 9 univariate result: 0.017")

    # ----- Timing
    print("\n[5] Per-call wall (3 calls each, warm)")
    sk.call_mat_phi(mu_v, Phi_v, L_v, y_truth)
    nn.call_mat_phi(mu_v, Phi_v, L_v, y_truth)
    t0 = time.time()
    for _ in range(3): sk.call_mat_phi(mu_v, Phi_v, L_v, y_truth)
    sw = (time.time()-t0)/3
    t0 = time.time()
    for _ in range(3): nn.call_mat_phi(mu_v, Phi_v, L_v, y_truth)
    nw = (time.time()-t0)/3
    print(f"  Sinkhorn: {sw*1000:.1f} ms  NN-OT: {nw*1000:.1f} ms  "
          f"ratio: {sw/nw:.3f}x")

    summary = {
        "d": 1, "T": args.T, "N": args.N,
        "n_theta": args.n_theta, "seeds_per_theta": args.seeds_per_theta,
        "n_samples": int(pn.shape[0]), "n_params": n_params,
        "best_val_mse": best_val, "training_wall_s": time.time()-t_train,
        "lp_delta_median": float(np.median(deltas)),
        "lp_delta_mean": float(np.mean(deltas)),
        "wall_sinkhorn_ms": float(sw*1000), "wall_nnot_ms": float(nw*1000),
        "wall_ratio": float(sw/nw),
    }
    (out_dir / "phase16_d1_train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  [save] {out_dir / 'phase16_d1_train_summary.json'}")


if __name__ == "__main__":
    main()
