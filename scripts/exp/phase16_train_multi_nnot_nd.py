"""Phase 16 v8: d-generic DeepONet NN-OT trainer for V1 multivariate
(upper-triangular Phi, diagonal Sigma_eta).

Generalises phase16_train_multi_nnot.py (d=2) and
phase16_train_multi_nnot_d1.py (d=1) to ANY d. Parameter block:
    mu (d), phi_diag (d), phi_off (d(d-1)/2 upper-tri), sigma_eta (d)
Operator context dim = svssm_multi_ctx_dim(d) = 3d + d(d-1)/2 + 3 + d.

The LEDH-flow capture mirrors the filter's _timestep_nd_mat_phi_impl
exactly (same expm dispatch: tf.exp at d=1, closed-form 2x2 at d=2,
fixed-s Pade at d>2; same Smith-doubling stationary init; same
clip/ridge guards). At each timestep it records
(particles_norm (N,d), weights (N,), ctx (ctx_dim,),
 sinkhorn_target_norm (N,d)) and continues with the Sinkhorn output.

Usage (d=4 example):
    python -m scripts.exp.phase16_train_multi_nnot_nd \
        --d 4 --T 100 --N 64 --n_theta 60 --seeds_per_theta 2 \
        --max_epochs 60 --out_dir reports/.../phase16_d4_training

Train-at-deployment: pass the SAME T and N you will run HMC at
(the OOD-T / OOD-N lesson from Phase 8/12).
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
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
    expm_2x2_batch, expm_pade_batch, pade_scaling_for_dim, _safe_nd,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_multivariate import (
    DifferentiableLEDHNeuralOTSVSSMmulti,
    svssm_multi_ctx_dim, build_svssm_multi_context_scalars, _compute_ess,
)
from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import (
    LOG_CHI2_MEAN, LOG_CHI2_VAR, _EPS,
)
from src.filters.dpf.resampling import det_resample


def off_indices(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def _dispatch_expm(A_eps, d):
    if d == 2:
        return expm_2x2_batch(A_eps)
    if d == 1:
        return tf.exp(A_eps)
    return expm_pade_batch(A_eps, s=pade_scaling_for_dim(d))


def build_phi(theta, d):
    P = np.diag(np.asarray(theta["phi_diag"], np.float32)).astype(np.float32)
    for k, (i, j) in enumerate(off_indices(d)):
        P[i, j] = theta["phi_off"][k]
    return P


def gen_y_obs(T, d, mu, Phi, sigma, seed):
    rng = np.random.default_rng(seed)
    h = np.zeros((T, d), dtype=np.float32)
    h[0] = mu + sigma * rng.standard_normal(d) / np.sqrt(
        np.maximum(1 - np.diag(Phi) ** 2, 1e-3))
    for t in range(1, T):
        h[t] = mu + Phi @ (h[t - 1] - mu) + sigma * rng.standard_normal(d)
    return tf.constant((np.exp(h / 2) * rng.standard_normal((T, d))).astype(np.float32))


def sample_thetas(d, n_theta, seed=0, mu0=None, phi0=None, sig0=None,
                   phi_off0=0.05, mu_spread=0.3, phi_spread=0.05,
                   sig_spread=0.1, off_spread=0.15):
    """Sample theta around a truth, bracketing the posterior bulk.

    Defaults: mu_i=0, phi_ii descending 0.95..0.8, sigma_i ascending
    0.3..0.4, phi_off ~ phi_off0. Tune via the *0 args.
    """
    rng = np.random.default_rng(seed)
    if mu0 is None:
        mu0 = np.zeros(d, np.float32)
    if phi0 is None:
        phi0 = np.linspace(0.95, 0.80, d).astype(np.float32)
    if sig0 is None:
        sig0 = np.linspace(0.3, 0.4, d).astype(np.float32)
    n_off = d * (d - 1) // 2
    out = []
    for _ in range(n_theta):
        mu = (mu0 + mu_spread * rng.standard_normal(d)).astype(np.float32)
        phi = np.clip(phi0 + phi_spread * rng.standard_normal(d), -0.99, 0.99).astype(np.float32)
        sig = np.maximum(sig0 + sig_spread * rng.standard_normal(d), 0.05).astype(np.float32)
        off = (phi_off0 + off_spread * rng.standard_normal(n_off)).astype(np.float32)
        out.append({"mu": mu, "phi_diag": phi, "phi_off": off, "sigma_eta": sig})
    return out


def capture_one(theta, y_obs, d, N, n_lambda, sinkhorn_eps, sinkhorn_iters):
    """Run the full-Phi LEDH flow, capture (p_norm, w, ctx, target) per t."""
    mu = tf.constant(theta["mu"], tf.float32)
    Phi = tf.constant(build_phi(theta, d), tf.float32)
    sigma = tf.constant(theta["sigma_eta"], tf.float32)
    sig_sq = sigma ** 2
    L_eta = tf.linalg.diag(sigma)
    Sigma_eta = tf.linalg.diag(sig_sq)

    T = int(y_obs.shape[0])
    R_val = tf.constant(LOG_CHI2_VAR, tf.float32)
    R_inv = 1.0 / R_val
    mu_z = tf.constant(LOG_CHI2_MEAN, tf.float32)
    z_obs = tf.math.log(tf.square(y_obs) + 1e-8)

    q = 1.2
    eps1 = (1.0 - q) / (1.0 - q ** n_lambda)
    epsilons = [eps1 * q ** j for j in range(n_lambda)]

    # Smith-doubling stationary init
    X = tf.identity(Sigma_eta); A = tf.identity(Phi)
    for _ in range(15):
        X = X + A @ X @ tf.transpose(A)
        A = A @ A
    Sigma_h0 = tf.clip_by_value(0.5 * (X + tf.transpose(X)), -1e3, 1e3)
    L0 = tf.linalg.cholesky(Sigma_h0 + 1e-6 * tf.eye(d, dtype=tf.float32))
    particles = mu[tf.newaxis, :] + tf.einsum("ij,nj->ni", L0,
                                                 tf.random.normal([N, d]))
    P = tf.tile(Sigma_h0[tf.newaxis], [N, 1, 1])
    log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

    I_d = tf.eye(d, dtype=tf.float32)
    I_db = tf.tile(I_d[tf.newaxis], [N, 1, 1])
    cap = []

    for t_int in range(1, T + 1):
        z_t = z_obs[t_int - 1]
        if t_int >= 2:
            x_det = mu[tf.newaxis] + tf.einsum("nj,ij->ni",
                                                  particles - mu[tf.newaxis], Phi)
            noise = tf.einsum("ij,nj->ni", L_eta, tf.random.normal([N, d]))
            particles = tf.clip_by_value(_safe_nd(x_det + noise), -50.0, 50.0)
            PhiP = tf.einsum("ik,nkl->nil", Phi, P)
            P = tf.clip_by_value(tf.einsum("nil,jl->nij", PhiP, Phi)
                                  + Sigma_eta[tf.newaxis], -1e3, 1e3)

        eta = tf.identity(particles)
        log_det_jac = tf.zeros([N]); lam_cum = 0.0
        innov = tf.tile(tf.clip_by_value(z_t - mu_z, -100, 100)[tf.newaxis], [N, 1])
        P = tf.where(tf.math.is_finite(P), P, tf.zeros_like(P))
        P = 0.5 * (P + tf.linalg.matrix_transpose(P))

        for j in range(n_lambda):
            eps_j = epsilons[j]; lam_k = lam_cum + eps_j / 2.0; lam_cum += eps_j
            S = lam_k * P + R_val * I_db + 1e-3 * I_db
            A_T = tf.linalg.solve(tf.linalg.matrix_transpose(S),
                                    tf.linalg.matrix_transpose(P))
            Amat = tf.clip_by_value(-0.5 * tf.linalg.matrix_transpose(A_T), -10, 10)
            I_lam = I_db + lam_k * Amat; I_2lam = I_db + 2.0 * lam_k * Amat
            P_innov = tf.einsum("nij,nj->ni", P, innov)
            ILA = tf.einsum("nij,nj->ni", I_lam, P_innov) * R_inv
            A_eta = tf.einsum("nij,nj->ni", Amat, eta)
            b_vec = tf.clip_by_value(
                tf.einsum("nij,nj->ni", I_2lam, ILA + A_eta), -100, 100)
            A_eps = Amat * eps_j
            expA = _dispatch_expm(A_eps, d)
            A_ridge = Amat + 1e-3 * I_db
            emib = tf.einsum("nij,nj->ni", expA - I_db, b_vec)
            phi_Ab = tf.linalg.solve(A_ridge, emib[..., tf.newaxis])[..., 0]
            particles = tf.clip_by_value(_safe_nd(
                tf.einsum("nij,nj->ni", expA, particles) + phi_Ab), -50, 50)
            eta = tf.clip_by_value(_safe_nd(
                tf.einsum("nij,nj->ni", expA, eta) + phi_Ab), -50, 50)
            log_det_jac = log_det_jac + tf.linalg.trace(A_eps)

        resid = z_t[tf.newaxis] - (particles + mu_z)
        log_lik = _safe_nd(
            -0.5 * R_inv * tf.reduce_sum(resid ** 2, axis=1)
            - 0.5 * float(d) * tf.math.log(R_val)
            - 0.5 * float(d) * tf.math.log(2.0 * 3.141592653589793))
        log_w_t = log_w + tf.where(tf.math.is_finite(log_lik + log_det_jac),
                                     log_lik + log_det_jac,
                                     tf.constant(-100.0, tf.float32))
        log_w_t = log_w_t - tf.reduce_logsumexp(log_w_t)

        p_mean = tf.reduce_mean(particles, axis=0, keepdims=True)
        p_std = tf.math.reduce_std(particles, axis=0, keepdims=True) + _EPS
        p_norm = (particles - p_mean) / p_std
        w_n = tf.nn.softmax(log_w_t, axis=0)
        ess = _compute_ess(w_n)
        ctx = build_svssm_multi_context_scalars(
            mu=mu, Phi=Phi, sigma_eta_sq_diag=sig_sq,
            t=tf.constant(0.0, tf.float32), z_t=z_t, ess=ess,
            epsilon=tf.constant(sinkhorn_eps, tf.float32),
            T_max=tf.constant(float(T), tf.float32), d=d)
        target, _ = det_resample(p_norm, log_w_t, epsilon=sinkhorn_eps,
                                   n_iters=sinkhorn_iters)
        target = tf.cast(tf.math.real(target), tf.float32)

        cap.append({"particles_norm": p_norm.numpy().astype(np.float32),
                     "weights": w_n.numpy().astype(np.float32),
                     "ctx": ctx.numpy().astype(np.float32),
                     "target_norm": target.numpy().astype(np.float32)})

        particles = target * p_std + p_mean
        P = tf.tile(tf.reduce_mean(P, axis=0, keepdims=True), [N, 1, 1])
        log_w = tf.fill([N], -tf.math.log(tf.cast(N, tf.float32)))

    return cap


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_theta", type=int, default=60)
    p.add_argument("--seeds_per_theta", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--n_basis", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    # theta-grid centers. Default = the trainer's built-in centers
    # (mu=0, phi=linspace(0.95,0.80,d), sigma=linspace(0.3,0.4,d),
    # phi_off=0.05). Pass these to RE-CENTER the operator's training
    # region on the actual truth -- the operator is accurate inside its
    # grid and degrades at the edges, so a mismatched grid shifts the
    # posterior on edge parameters.
    p.add_argument("--mu0", type=str, default=None,
                   help="comma-separated length-d grid center for mu.")
    p.add_argument("--phi0", type=str, default=None,
                   help="comma-separated length-d grid center for phi_diag.")
    p.add_argument("--sig0", type=str, default=None,
                   help="comma-separated length-d grid center for sigma_eta.")
    p.add_argument("--phi_off0", type=float, default=0.05,
                   help="scalar grid center for phi_off entries.")
    # grid WIDTHS (std of the per-theta jitter around the centers). Widen to
    # cover a broader posterior so the operator is not out-of-distribution.
    p.add_argument("--mu_spread", type=float, default=0.3)
    p.add_argument("--phi_spread", type=float, default=0.05)
    p.add_argument("--sig_spread", type=float, default=0.1)
    p.add_argument("--off_spread", type=float, default=0.15)
    p.add_argument("--out_dir", type=str, required=True)
    args = p.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tf.random.set_seed(args.seed)
    d = args.d
    ctx_dim = svssm_multi_ctx_dim(d)
    print(f"[phase16-train-nd] d={d} T={args.T} N={args.N} "
          f"n_theta={args.n_theta} ctx_dim={ctx_dim} n_params={3*d+d*(d-1)//2}")

    # 1. data
    print("\n[1] generate training data (Sinkhorn capture)")

    def _parse_vec(s, name):
        if s is None:
            return None
        v = np.array([float(x) for x in s.split(",")], np.float32)
        if len(v) != d:
            raise ValueError(f"--{name} must have length d={d}; got {len(v)}")
        return v

    mu0 = _parse_vec(args.mu0, "mu0")
    phi0 = _parse_vec(args.phi0, "phi0")
    sig0 = _parse_vec(args.sig0, "sig0")
    print(f"  grid centers: mu0={mu0 if mu0 is not None else 'default(0)'}  "
          f"phi0={phi0 if phi0 is not None else 'default(0.95..0.80)'}  "
          f"sig0={sig0 if sig0 is not None else 'default(0.3..0.4)'}  "
          f"phi_off0={args.phi_off0}")
    print(f"  grid spreads: mu={args.mu_spread} phi={args.phi_spread} "
          f"sig={args.sig_spread} off={args.off_spread}")
    thetas = sample_thetas(d, args.n_theta, seed=args.seed,
                            mu0=mu0, phi0=phi0, sig0=sig0,
                            phi_off0=args.phi_off0,
                            mu_spread=args.mu_spread, phi_spread=args.phi_spread,
                            sig_spread=args.sig_spread, off_spread=args.off_spread)
    pn, w, ctx, tn = [], [], [], []
    t0 = time.time()
    for ti, th in enumerate(thetas):
        for s in range(args.seeds_per_theta):
            y = gen_y_obs(args.T, d, th["mu"], build_phi(th, d),
                           th["sigma_eta"], seed=args.seed + ti * 100 + s)
            for c in capture_one(th, y, d, args.N, 10, 1.0, 10):
                pn.append(c["particles_norm"]); w.append(c["weights"])
                ctx.append(c["ctx"]); tn.append(c["target_norm"])
        if (ti + 1) % max(1, len(thetas) // 8) == 0:
            print(f"  theta {ti+1}/{len(thetas)} ({time.time()-t0:.0f}s)", flush=True)
    pn = np.stack(pn); w = np.stack(w); ctx = np.stack(ctx); tn = np.stack(tn)
    print(f"  M={pn.shape[0]}  p={pn.shape} ctx={ctx.shape} target={tn.shape}")

    # 2. model + train
    print("\n[2] build + train DeepONet")
    model = DeepONetMonotoneOT(state_dim=d, n_basis=args.n_basis,
                                d_branch=64, d_trunk=64, n_scalar_ctx=ctx_dim)
    _ = model(tf.constant(pn[:1]), tf.constant(w[:1]), tf.constant(ctx[:1]))
    npar = int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))
    print(f"  trainable params: {npar:,}")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(pn.shape[0]); nval = int(round(0.2 * pn.shape[0]))
    vi, tri = perm[:nval], perm[nval:]
    opt = tf.keras.optimizers.Adam(1e-3)

    @tf.function
    def step(pb, wb, cb, tb):
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(tf.square(model(pb, wb, cb) - tb))
        g = tape.gradient(loss, model.trainable_variables)
        g = [tf.clip_by_norm(x, 5.0) if x is not None else x for x in g]
        opt.apply_gradients(zip(g, model.trainable_variables))
        return loss

    @tf.function
    def vloss(pb, wb, cb, tb):
        return tf.reduce_mean(tf.square(model(pb, wb, cb) - tb))

    best, bestw, bad, bs = float("inf"), None, 0, 64
    t_tr = time.time()
    for ep in range(args.max_epochs):
        o = rng.permutation(len(tri))
        for i in range(0, len(tri), bs):
            idx = tri[o[i:i+bs]]
            step(tf.constant(pn[idx]), tf.constant(w[idx]),
                 tf.constant(ctx[idx]), tf.constant(tn[idx]))
        vl = np.mean([float(vloss(tf.constant(pn[vi[i:i+bs]]),
                                    tf.constant(w[vi[i:i+bs]]),
                                    tf.constant(ctx[vi[i:i+bs]]),
                                    tf.constant(tn[vi[i:i+bs]])))
                      for i in range(0, len(vi), bs)])
        print(f"  epoch {ep+1:3d}/{args.max_epochs}  val_mse={vl:.5f}", flush=True)
        if vl < best * (1 - 1e-4):
            best, bestw, bad = vl, [v.numpy() for v in model.trainable_variables], 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop @ {ep+1}"); break
    if bestw is not None:
        for v, b in zip(model.trainable_variables, bestw):
            v.assign(b)
    print(f"  best val_mse={best:.5f}  wall={time.time()-t_tr:.0f}s")
    wpath = out_dir / "deeponet_nd.weights.h5"
    model.save_weights(str(wpath)); print(f"  saved {wpath}")

    # 3. validate filter |Δ log-p| at truth
    print("\n[3] validate filter forward log-p at truth")
    truth = {"mu": np.zeros(d, np.float32),
             "phi_diag": np.linspace(0.95, 0.80, d).astype(np.float32),
             "phi_off": np.full(d*(d-1)//2, 0.05, np.float32),
             "sigma_eta": np.linspace(0.3, 0.4, d).astype(np.float32)}
    Phi_t = tf.constant(build_phi(truth, d), tf.float32)
    mu_t = tf.constant(truth["mu"], tf.float32)
    L_t = tf.constant(np.diag(truth["sigma_eta"]).astype(np.float32), tf.float32)
    y_t = gen_y_obs(args.T, d, truth["mu"], build_phi(truth, d),
                     truth["sigma_eta"], seed=999)
    sk = DifferentiableLEDHLogLikelihoodSVSSMmulti(
        state_dim=d, num_particles=args.N, n_lambda=10, sinkhorn_epsilon=1.0,
        sinkhorn_iters=10, grad_window=4, jit_compile=True)
    nn = DifferentiableLEDHNeuralOTSVSSMmulti(
        neural_ot_model=model, state_dim=d, num_particles=args.N, n_lambda=10,
        sinkhorn_epsilon=1.0, grad_window=4, jit_compile=True)
    deltas = []
    for tr in range(5):
        tf.random.set_seed(tr); lps = float(sk.call_mat_phi(mu_t, Phi_t, L_t, y_t))
        tf.random.set_seed(tr); lpn = float(nn.call_mat_phi(mu_t, Phi_t, L_t, y_t))
        deltas.append(abs(lps - lpn))
        print(f"  trial {tr+1}: SK={lps:+.3f} NN={lpn:+.3f} |Δ|={abs(lps-lpn):.4f}")
    print(f"  median |Δ|={np.median(deltas):.4f}  (Phase 2 target ≤ 0.5)")

    (out_dir / "phase16_nd_train_summary.json").write_text(json.dumps({
        "d": d, "T": args.T, "N": args.N, "ctx_dim": ctx_dim,
        "n_samples": int(pn.shape[0]), "n_params_model": npar,
        "best_val_mse": float(best),
        "lp_delta_median": float(np.median(deltas)),
        "lp_delta_mean": float(np.mean(deltas)),
    }, indent=2))
    print(f"\n  [save] {out_dir / 'phase16_nd_train_summary.json'}")


if __name__ == "__main__":
    main()
