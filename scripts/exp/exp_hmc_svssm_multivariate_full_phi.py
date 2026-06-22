"""
HMC parameter recovery for V1 multivariate SVSSM with UPPER-TRIANGULAR Phi
(cross-asset persistence). Sister script to exp_hmc_svssm_multivariate.py
which uses diagonal Phi.

Why upper-triangular Phi:
    The V1 multivariate identifiability discussion (\\S 3.f/3.g) lists three
    canonicalisations of the rotation indeterminacy in the latent dynamics:
      (i) diagonal Phi (Phase 14), simplest but no cross-asset persistence
      (ii) upper-triangular Phi (this script), allows one-directional
           spillover h_{t,i} ← h_{t-1,j} for j > i with positive diagonal
      (iii) Cholesky/lower-tri factorisation in (A, Phi) (V2)
    Choice (ii) is the standard factor-SV parameterisation
    (Chib-Nardari-Shephard 2006). For JPMC's risk-modelling context, this is
    the natural extension to capture vol spillovers across assets.

Vector parameterisation (for state dim d):
    theta_raw \\in R^{3d + d(d-1)/2}
        mu                  : d
        phi_diag_raw        : d   (diagonal of Phi, tanh-constrained to (-1,1))
        phi_off_raw         : d(d-1)/2   (upper off-diagonals, unconstrained)
        log_sigma_eta_sq    : d   (diagonal Sigma_eta, exp-constrained)

For d=2 that's 2+2+1+2 = 7 parameters. Phi looks like:
    [[ tanh(phi_diag_raw_0), phi_off_raw_0       ],
     [                    0, tanh(phi_diag_raw_1)]]

Stability: eigenvalues of upper-triangular = diagonal entries; |tanh(.)| < 1.

Per-component priors (independent across d, applied to mu/phi_diag/log_sigma):
    mu_i              ~ N(prior_mu_loc, prior_mu_scale^2)
    phi_diag_raw_i    ~ N(prior_phi_raw_loc, prior_phi_raw_scale^2)
    log_sigma_eta_sq_i ~ N(prior_log_sigma_eta_sq_loc, prior_log_sigma_eta_sq_scale^2)
Off-diagonal priors (weak, centred at 0):
    phi_off_raw_k     ~ N(0, prior_phi_off_scale^2)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm_multivariate import (
    DifferentiableLEDHLogLikelihoodSVSSMmulti,
)
from scripts.exp.exp_hmc_svssm import run_chain_windowed_proper
from scripts.exp.compare_svssm_hmc_methods import (
    rank_rhat as _rank_rhat,
    bulk_ess as _bulk_ess,
    tail_ess as _tail_ess,
)

tfd = tfp.distributions
tfm = tfp.mcmc

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAVE_MPL = True
except ImportError:
    plt = None
    _HAVE_MPL = False


# ---------------------------------------------------------------------------
# Phi-matrix construction
# ---------------------------------------------------------------------------

def num_off_diag(d: int) -> int:
    return d * (d - 1) // 2


def upper_tri_indices(d: int) -> list[tuple[int, int]]:
    """Return [(i, j), ...] for i < j in row-major order."""
    return [(i, j) for i in range(d) for j in range(i + 1, d)]


def build_phi_matrix_np(phi_diag: np.ndarray, phi_off: np.ndarray, d: int) -> np.ndarray:
    """Construct (d, d) upper-triangular Phi from constrained diagonal + off-diag."""
    P = np.zeros((d, d), dtype=np.float32)
    for i in range(d):
        P[i, i] = float(phi_diag[i])
    for k, (i, j) in enumerate(upper_tri_indices(d)):
        P[i, j] = float(phi_off[k])
    return P


def build_phi_matrix_tf(phi_diag: tf.Tensor, phi_off: tf.Tensor, d: int) -> tf.Tensor:
    """Construct (d, d) upper-triangular Phi from constrained diag + off-diag tensors.

    Uses tf.scatter_nd so gradients flow.
    """
    indices = [[i, i] for i in range(d)]
    values = [phi_diag[i] for i in range(d)]
    for k, (i, j) in enumerate(upper_tri_indices(d)):
        indices.append([i, j])
        values.append(phi_off[k])
    return tf.scatter_nd(
        tf.constant(indices, dtype=tf.int32),
        tf.stack(values),
        shape=[d, d],
    )


# ---------------------------------------------------------------------------
# Data generation (upper-triangular Phi, diagonal Sigma_eta)
# ---------------------------------------------------------------------------

def gen_svssm_multi_phi_mat(T: int, mu_vec: np.ndarray, Phi: np.ndarray,
                             sigma_eta_vec: np.ndarray, seed: int = 42) -> tf.Tensor:
    """Generate observations from V1 multivariate with matrix Phi."""
    tf.random.set_seed(seed)
    np.random.seed(seed)
    d = len(mu_vec)
    h = mu_vec.astype(np.float32).copy()
    ys = []
    for _ in range(T):
        eta = np.random.randn(d).astype(np.float32) * sigma_eta_vec.astype(np.float32)
        h = mu_vec + Phi @ (h - mu_vec) + eta
        eps = np.random.randn(d).astype(np.float32)
        ys.append(np.exp(h / 2.0) * eps)
    return tf.constant(np.stack(ys, axis=0), tf.float32)


# ---------------------------------------------------------------------------
# Target log-prob
# ---------------------------------------------------------------------------

def make_target_log_prob(ll, y_obs, crn_seed, d,
                          prior_mu_loc=0.0, prior_mu_scale=1.0,
                          prior_phi_raw_loc=2.0, prior_phi_raw_scale=0.5,
                          prior_log_sigma_eta_sq_loc=-2.0,
                          prior_log_sigma_eta_sq_scale=1.0,
                          prior_phi_off_scale=0.5,
                          no_likelihood: bool = False):
    """Build target_log_prob_fn for theta_raw of dim (3d + d(d-1)/2)."""
    num_off = num_off_diag(d)
    n_params = 3 * d + num_off

    prior_mu = tfd.Normal(loc=tf.constant(prior_mu_loc, tf.float32),
                           scale=tf.constant(prior_mu_scale, tf.float32))
    prior_phi_raw = tfd.Normal(loc=tf.constant(prior_phi_raw_loc, tf.float32),
                                 scale=tf.constant(prior_phi_raw_scale, tf.float32))
    prior_log_sig = tfd.Normal(loc=tf.constant(prior_log_sigma_eta_sq_loc, tf.float32),
                                 scale=tf.constant(prior_log_sigma_eta_sq_scale, tf.float32))
    # Tight prior on off-diagonals — centred at 0 (the diagonal-Phi case),
    # so the data has to actively push phi_off away from 0 to detect
    # spillover. Standard regularisation for factor-SV models.
    prior_phi_off = tfd.Normal(loc=tf.constant(0.0, tf.float32),
                                 scale=tf.constant(prior_phi_off_scale, tf.float32))

    def target_log_prob(theta_raw):
        # theta_raw: (3d + d(d-1)/2,)
        mu               = theta_raw[:d]
        phi_diag_raw     = theta_raw[d:2 * d]
        phi_off_raw      = theta_raw[2 * d:2 * d + num_off]
        log_sigma_eta_sq = theta_raw[2 * d + num_off:n_params]

        phi_diag = tf.tanh(phi_diag_raw)
        Phi = build_phi_matrix_tf(phi_diag, phi_off_raw, d)
        sigma_eta_sq = tf.exp(log_sigma_eta_sq)
        # Diagonal Sigma_eta Cholesky = diag(sqrt(sigma_eta_sq))
        L_eta = tf.linalg.diag(tf.sqrt(sigma_eta_sq))

        log_prior = (
            tf.reduce_sum(prior_mu.log_prob(mu))
            + tf.reduce_sum(prior_phi_raw.log_prob(phi_diag_raw))
            + tf.reduce_sum(prior_phi_off.log_prob(phi_off_raw))
            + tf.reduce_sum(prior_log_sig.log_prob(log_sigma_eta_sq))
        )

        if no_likelihood:
            # Prior-only target — for Phase-19-style prior-dominance
            # diagnostic. Skips the LEDH-flow filter entirely, so
            # ~5000x cheaper per HMC step.
            return log_prior

        tf.random.set_seed(crn_seed)
        log_lik = ll.call_mat_phi(mu, Phi, L_eta, y_obs)
        log_lik = tf.cast(tf.math.real(log_lik), tf.float32)
        log_lik = tf.where(tf.math.is_finite(log_lik), log_lik,
                            tf.constant(-np.inf, tf.float32))

        return log_prior + log_lik

    return target_log_prob


def unconstrain(mu_vec: np.ndarray, phi_diag: np.ndarray, phi_off: np.ndarray,
                sigma_eta_sq_vec: np.ndarray) -> np.ndarray:
    """Pack (mu, phi_diag, phi_off, sigma_eta_sq) into unconstrained theta_raw."""
    mu = np.asarray(mu_vec, dtype=np.float32)
    phi_clipped = np.clip(np.asarray(phi_diag), -0.9999, 0.9999)
    phi_diag_raw = np.arctanh(phi_clipped).astype(np.float32)
    phi_off_raw = np.asarray(phi_off, dtype=np.float32)
    log_sig = np.log(np.maximum(np.asarray(sigma_eta_sq_vec), 1e-8)).astype(np.float32)
    return np.concatenate([mu, phi_diag_raw, phi_off_raw, log_sig])


def constrain(theta_raw: np.ndarray, d: int) -> dict:
    """theta_raw (..., 3d + d(d-1)/2) -> dict with mu, phi_diag, phi_off,
    sigma_eta_sq each (..., d) or (..., d(d-1)/2)."""
    th = np.asarray(theta_raw)
    num_off = num_off_diag(d)
    n_params = 3 * d + num_off
    return {
        "mu":           th[..., :d],
        "phi_diag":     np.tanh(th[..., d:2 * d]),
        "phi_off":      th[..., 2 * d:2 * d + num_off],
        "sigma_eta_sq": np.exp(th[..., 2 * d + num_off:n_params]),
    }


# ---------------------------------------------------------------------------
# Chain driver (single chain)
# ---------------------------------------------------------------------------

def run_chain(target_log_prob_fn, init_raw, num_results, num_burnin,
              step_size, num_leapfrog, seed, progress_every=10,
              use_windowed_adaptive=True, dense_mass=True,
              target_accept_prob=None):
    """Multivariate full-Phi HMC. Defaults to v7 windowed-adaptive + dense mass."""
    if use_windowed_adaptive:
        tap = 0.65 if target_accept_prob is None else float(target_accept_prob)
        samples, acc, lp, step = run_chain_windowed_proper(
            target_log_prob_fn=target_log_prob_fn,
            init_raw=init_raw,
            num_results=int(num_results),
            num_burnin=int(num_burnin),
            step_size=float(step_size),
            num_leapfrog=int(num_leapfrog),
            seed=int(seed),
            target_accept_prob=tap,
            progress_every=int(progress_every),
            dense_mass=bool(dense_mass),
        )
        return samples, acc, lp, step

    # Vanilla HMC fallback path (not the production setup for multivariate).
    tap = 0.65 if target_accept_prob is None else float(target_accept_prob)
    kernel = tfm.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        step_size=tf.constant(step_size, tf.float32),
        num_leapfrog_steps=int(num_leapfrog),
    )
    kernel = tfm.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=int(num_burnin),
        target_accept_prob=tf.constant(tap, tf.float32),
    )

    def trace_fn(_, kr):
        inner = kr
        while hasattr(inner, "inner_results"):
            inner = inner.inner_results
        accepted = tf.cast(inner.is_accepted, tf.bool) \
            if hasattr(inner, "is_accepted") else tf.constant(False)
        lp = inner.target_log_prob if hasattr(inner, "target_log_prob") \
            else inner.accepted_results.target_log_prob
        st = kr.new_step_size if hasattr(kr, "new_step_size") \
            else tf.constant(float("nan"), tf.float32)
        return accepted, tf.cast(lp, tf.float32), tf.cast(st, tf.float32)

    init = tf.constant(init_raw, dtype=tf.float32)
    samples, (acc, lp, step) = tfm.sample_chain(
        num_results=int(num_results),
        num_burnin_steps=int(num_burnin),
        current_state=init,
        kernel=kernel,
        trace_fn=trace_fn,
        seed=int(seed),
    )
    return samples.numpy(), acc.numpy(), lp.numpy(), step.numpy()


# ---------------------------------------------------------------------------
# Posterior diagnostics
# ---------------------------------------------------------------------------

def split_rhat(samples: np.ndarray) -> np.ndarray:
    """Split-Rhat per parameter on (chains, draws, P) samples."""
    chains, draws, P = samples.shape
    if chains < 2:
        return np.full(P, float("nan"))
    half = draws // 2
    s = np.concatenate([samples[:, :half], samples[:, half:2 * half]], axis=0)
    N = s.shape[1]
    means = s.mean(axis=1)
    vars_ = s.var(axis=1, ddof=1)
    B = N * means.var(axis=0, ddof=1)
    W = vars_.mean(axis=0)
    var_hat = ((N - 1) / N) * W + B / N
    return np.sqrt(var_hat / np.maximum(W, 1e-12))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--d", type=int, default=2, dest="d", metavar="D")
    p.add_argument("--T", type=int, default=50)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--n_lambda", type=int, default=10)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--L", type=int, default=5)
    p.add_argument("--step_size", type=float, default=0.05)
    p.add_argument("--num_chains", type=int, default=4)
    p.add_argument("--num_burnin", type=int, default=200)
    p.add_argument("--num_results", type=int, default=1000)
    p.add_argument("--dispersion", type=float, default=0.10)
    p.add_argument("--data_seed", type=int, default=42)
    p.add_argument("--base_seed", type=int, default=300)
    p.add_argument("--progress_every", type=int, default=50)
    # Truth
    p.add_argument("--mu", type=str, default="0.0,-0.3",
                   help="comma-separated truth for mu (length d)")
    p.add_argument("--phi_diag", type=str, default="0.95,0.85",
                   help="comma-separated truth for Phi diagonal (length d)")
    p.add_argument("--phi_off", type=str, default="0.05",
                   help="comma-separated truth for Phi upper off-diagonals "
                        "(length d(d-1)/2; row-major: (0,1), (0,2), (1,2), ...)")
    p.add_argument("--sigma_eta", type=str, default="0.3,0.4",
                   help="comma-separated truth for sigma_eta_diag (length d)")
    # Priors
    p.add_argument("--prior_mu_loc", type=float, default=-0.2)
    p.add_argument("--prior_mu_scale", type=float, default=1.0)
    p.add_argument("--prior_phi_raw_loc", type=float, default=1.5)
    p.add_argument("--prior_phi_raw_scale", type=float, default=1.0)
    p.add_argument("--prior_log_sigma_eta_sq_loc", type=float, default=-2.0)
    p.add_argument("--prior_log_sigma_eta_sq_scale", type=float, default=1.0)
    p.add_argument("--prior_phi_off_scale", type=float, default=0.5,
                   help="Prior sd on off-diagonal phi entries (centred at 0). "
                        "Smaller value pulls toward diagonal Phi.")
    p.add_argument("--out_dir", type=str,
                   default="reports/6_BonusQ1_HMC_Invertible_Flows/HMC_vs_PMMH/"
                           "svssm_hmc_multivariate_full_phi")
    p.add_argument("--use_windowed_adaptive", action="store_true", default=True,
                   help="v7 windowed-adaptive PreconditionedHMC + dense mass.")
    p.add_argument("--diagonal_mass", action="store_true")
    p.add_argument("--nnot_weights", type=str, default=None,
                   help="Path to trained DeepONet weights (.h5). If set, "
                        "the Sinkhorn resampler is replaced by the operator "
                        "at every timestep. Phase 16 (d=2 only).")
    p.add_argument("--mat_phi_ridge", type=float, default=1e-3,
                   help="Ridge added to S and A solves in the full-Phi LEDH "
                        "flow. Default 1e-3 (heavy guards for large phi_off). "
                        "Lower to 1e-6 for cleaner gradients at d=1/near-diagonal.")
    p.add_argument("--mat_phi_clip_particle", type=float, default=50.0,
                   help="Particle clip bound in the full-Phi path (default 50).")
    p.add_argument("--mat_phi_clip_P", type=float, default=1e3,
                   help="State-cov clip bound in the full-Phi path (default 1e3).")
    p.add_argument("--nnot_n_basis", type=int, default=64,
                   help="DeepONet n_basis when loading --nnot_weights "
                        "(must match training-time architecture).")
    p.add_argument("--chain_id", type=int, default=-1,
                   help="If >=0, run ONLY chain_id (zero-indexed), with the "
                        "same chain_seed/init_raw that chain would have had "
                        "in a sequential --num_chains run. Use to launch "
                        "individual chains in parallel under srun while "
                        "preserving shared CRN (crn_seed = base_seed) "
                        "across all chains.")
    p.add_argument("--no_likelihood", action="store_true",
                   help="Prior-only target: skip the filter call entirely "
                        "and sample from the prior under HMC. Used as a "
                        "Phase-19-style prior-dominance diagnostic.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mu_truth        = np.array([float(x) for x in args.mu.split(",")], dtype=np.float32)
    phi_diag_truth  = np.array([float(x) for x in args.phi_diag.split(",")], dtype=np.float32)
    # phi_off has 0 entries at d=1; allow empty string.
    _phi_off_parts  = [x for x in args.phi_off.split(",") if x.strip() != ""]
    phi_off_truth   = np.array([float(x) for x in _phi_off_parts], dtype=np.float32)
    sigma_eta_truth = np.array([float(x) for x in args.sigma_eta.split(",")], dtype=np.float32)
    if not (len(mu_truth) == len(phi_diag_truth) == len(sigma_eta_truth) == args.d):
        raise ValueError(f"mu/phi_diag/sigma_eta must be length d={args.d}")
    if len(phi_off_truth) != num_off_diag(args.d):
        raise ValueError(f"phi_off must be length d(d-1)/2 = {num_off_diag(args.d)}; "
                         f"got {len(phi_off_truth)}")
    sigma_eta_sq_truth = sigma_eta_truth ** 2

    Phi_truth = build_phi_matrix_np(phi_diag_truth, phi_off_truth, args.d)

    print(f"[hmc-multi-full-phi] TF {tf.__version__}, TFP {tfp.__version__}")
    print(f"  d={args.d}  T={args.T}  N={args.N}  L={args.L}  "
          f"step_size={args.step_size}")
    print(f"  chains={args.num_chains}  burnin={args.num_burnin}  "
          f"results={args.num_results}")
    print(f"  truth mu             : {mu_truth.tolist()}")
    print(f"  truth phi_diag       : {phi_diag_truth.tolist()}")
    print(f"  truth phi_off (upper): {phi_off_truth.tolist()}")
    print(f"  truth sigma_eta      : {sigma_eta_truth.tolist()}")
    print(f"  Phi matrix:")
    for row in Phi_truth:
        print(f"    {row.tolist()}")
    # Stability check
    eigs = np.linalg.eigvals(Phi_truth)
    print(f"  eigvals(Phi)         : {[float(np.abs(e)) for e in eigs]}  "
          f"(must all be < 1 for stationarity)")

    # Data
    y_obs = gen_svssm_multi_phi_mat(args.T, mu_truth, Phi_truth, sigma_eta_truth,
                                     seed=args.data_seed)
    print(f"  y_obs shape: {tuple(y_obs.shape)}  range: "
          f"[{float(tf.reduce_min(y_obs)):.3f}, {float(tf.reduce_max(y_obs)):.3f}]")

    # Filter. With the closed-form expm_2x2_batch in place, d=2 now
    # runs under jit_compile=True; d>2 still falls back to eager
    # tf.linalg.expm internally.
    # Phase 16: --nnot_weights swaps the Sinkhorn resampler for a
    # trained DeepONet operator. Works at ANY d (v6 Pade expm made the
    # filter + context builder dimension-generic; the operator is
    # state_dim-parameterised). The checkpoint must have been trained
    # at this d (ctx_dim = svssm_multi_ctx_dim(d) and the (N, d) target
    # shape are d-specific).
    if args.nnot_weights:
        from src.filters.bonus.deeponet_ot import DeepONetMonotoneOT
        from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm_multivariate import (
            DifferentiableLEDHNeuralOTSVSSMmulti, svssm_multi_ctx_dim,
        )
        ctx_dim = svssm_multi_ctx_dim(args.d)
        print(f"\n  [nnot] loading trained DeepONet from {args.nnot_weights}  "
              f"(d={args.d}, ctx_dim={ctx_dim})")
        model = DeepONetMonotoneOT(
            state_dim=args.d, n_basis=args.nnot_n_basis,
            d_branch=64, d_trunk=64, n_scalar_ctx=ctx_dim,
        )
        _ = model(tf.zeros([1, args.N, args.d]),
                   tf.ones([1, args.N]) / args.N,
                   tf.zeros([1, ctx_dim]))
        model.load_weights(args.nnot_weights)
        ll = DifferentiableLEDHNeuralOTSVSSMmulti(
            neural_ot_model=model,
            state_dim=args.d, num_particles=args.N, n_lambda=args.n_lambda,
            sinkhorn_epsilon=1.0, grad_window=4, jit_compile=True,
        )
    else:
        ll = DifferentiableLEDHLogLikelihoodSVSSMmulti(
            state_dim=args.d, num_particles=args.N, n_lambda=args.n_lambda,
            sinkhorn_epsilon=1.0, sinkhorn_iters=args.K,
            grad_window=4, jit_compile=True,
            mat_phi_ridge=args.mat_phi_ridge,
            mat_phi_clip_particle=args.mat_phi_clip_particle,
            mat_phi_clip_P=args.mat_phi_clip_P,
        )

    # Target log-prob
    crn_seed = args.base_seed
    if args.no_likelihood:
        print("\n  [no_likelihood] sampling from the prior under HMC; "
              "filter call skipped.")
    target_log_prob_fn = make_target_log_prob(
        ll, y_obs, crn_seed=crn_seed, d=args.d,
        prior_mu_loc=args.prior_mu_loc, prior_mu_scale=args.prior_mu_scale,
        prior_phi_raw_loc=args.prior_phi_raw_loc,
        prior_phi_raw_scale=args.prior_phi_raw_scale,
        prior_log_sigma_eta_sq_loc=args.prior_log_sigma_eta_sq_loc,
        prior_log_sigma_eta_sq_scale=args.prior_log_sigma_eta_sq_scale,
        prior_phi_off_scale=args.prior_phi_off_scale,
        no_likelihood=args.no_likelihood,
    )

    # Sanity at truth
    truth_raw = unconstrain(mu_truth, phi_diag_truth, phi_off_truth, sigma_eta_sq_truth)
    print(f"  truth_raw ({len(truth_raw)}-vector): {truth_raw.tolist()}")
    lp_at_truth = float(target_log_prob_fn(tf.constant(truth_raw)).numpy())
    print(f"  target_log_prob(truth_raw) = {lp_at_truth:.4f}")

    # Initial states. CRN: every chain uses the SAME args.base_seed for
    # the filter's tf.random.set_seed inside target_log_prob so the
    # likelihood surface is identical across chains (standard HMC CRN
    # pattern). HMC momentum + init_raw dispersion differ per chain.
    n_params = len(truth_raw)

    # If --chain_id >= 0, we are running ONLY one chain (this srun task)
    # but we want its init_raw and momentum seed to match what it would
    # have been in a sequential --num_chains run. So we always generate
    # all `target_num_chains` init_raws/seeds, then index by chain_id.
    if args.chain_id >= 0:
        target_num_chains = max(args.num_chains, args.chain_id + 1)
    else:
        target_num_chains = args.num_chains
    rng = np.random.default_rng(args.base_seed)
    init_raws = [
        truth_raw + args.dispersion * rng.standard_normal(n_params).astype(np.float32)
        for _ in range(target_num_chains)
    ]
    chain_seeds = [args.base_seed + 1009 * (c + 1) for c in range(target_num_chains)]

    if args.chain_id >= 0:
        # Single-chain mode: only run the requested chain.
        chain_indices = [args.chain_id]
        print(f"  [chain_id={args.chain_id}] running ONE chain in this "
              f"task; CRN seed = base_seed = {args.base_seed} (shared).")
    else:
        chain_indices = list(range(args.num_chains))

    # Sequential chains (or just one if --chain_id was set)
    all_samples, all_accs, all_lps, all_steps = [], [], [], []
    t_total = time.perf_counter()
    for c in chain_indices:
        print(f"\n  [chain {c+1}/{args.num_chains}] init_raw = {init_raws[c].tolist()}")
        s, acc, lp, st = run_chain(
            target_log_prob_fn=target_log_prob_fn,
            init_raw=init_raws[c],
            num_results=args.num_results,
            num_burnin=args.num_burnin,
            step_size=args.step_size,
            num_leapfrog=args.L,
            seed=chain_seeds[c],
            progress_every=args.progress_every,
            use_windowed_adaptive=args.use_windowed_adaptive,
            dense_mass=(not args.diagonal_mass),
        )
        all_samples.append(s)
        all_accs.append(acc)
        all_lps.append(lp)
        all_steps.append(st)
    elapsed = time.perf_counter() - t_total

    samples_raw = np.stack(all_samples, axis=0)   # (chains, results, n_params)
    accs = np.stack(all_accs, axis=0)
    lps = np.stack(all_lps, axis=0)
    cons = constrain(samples_raw, args.d)

    # Per-parameter Vehtari diagnostics on the constrained samples:
    # rank-Rhat (Vehtari 2021), bulk-ESS, tail-ESS.
    def _vehtari(x: np.ndarray) -> tuple:
        # x: (chains, draws)
        x_clean = np.where(np.isfinite(x), x, float(np.nanmedian(x)))
        return (float(_rank_rhat(x_clean)),
                float(_bulk_ess(x_clean)),
                float(_tail_ess(x_clean)))

    print(f"\n  total wall: {elapsed:.1f}s")
    print(f"  overall accept rate: {float(accs.mean()):.3f}")
    print(f"\n{'param':<24} {'comp':>6} {'truth':>10} {'mean':>10} {'std':>10} "
          f"{'median':>10} {'2.5%':>10} {'97.5%':>10} {'rankR':>7} "
          f"{'bulkESS':>8} {'tailESS':>8} cov")

    rows = []

    def _emit(label: str, comp_index, tval: float, *,
              row_param_name: str, chain_x: np.ndarray):
        # chain_x: (chains, draws) for this scalar quantity
        samp = chain_x.ravel()
        med = float(np.median(samp))
        q025 = float(np.quantile(samp, 0.025))
        q975 = float(np.quantile(samp, 0.975))
        rhat, be, te = _vehtari(chain_x)
        cov = "OK" if q025 <= tval <= q975 else "OUT"
        print(f"{label:<24} {comp_index:>6} {tval:>10.4f} "
              f"{float(samp.mean()):>10.4f} {float(samp.std()):>10.4f} "
              f"{med:>10.4f} {q025:>10.4f} {q975:>10.4f} "
              f"{rhat:>7.3f} {be:>8.0f} {te:>8.0f} {cov:>3}")
        rows.append({
            "param": row_param_name,
            "component": (comp_index if isinstance(comp_index, int)
                           else int(comp_index)),
            "truth": tval, "median": med, "q025": q025, "q975": q975,
            "rhat": rhat, "bulk_ess": be, "tail_ess": te,
            "covered_95ci": (cov == "OK"),
        })

    # mu
    for i in range(args.d):
        _emit("mu", i, float(mu_truth[i]),
              row_param_name="mu",
              chain_x=cons["mu"][:, :, i])
    # phi_diag
    for i in range(args.d):
        _emit("phi_diag", i, float(phi_diag_truth[i]),
              row_param_name="phi_diag",
              chain_x=cons["phi_diag"][:, :, i])
    # phi_off (THE HEADLINE: did we recover cross-asset persistence?)
    for k, (i, j) in enumerate(upper_tri_indices(args.d)):
        _emit(f"phi_off ({i},{j})", k, float(phi_off_truth[k]),
              row_param_name=f"phi_off_{i}{j}",
              chain_x=cons["phi_off"][:, :, k])
    # sigma_eta_sq
    for i in range(args.d):
        _emit("sigma_eta_sq", i, float(sigma_eta_sq_truth[i]),
              row_param_name="sigma_eta_sq",
              chain_x=cons["sigma_eta_sq"][:, :, i])

    n_covered = sum(1 for r in rows if r["covered_95ci"])
    print(f"\n  coverage: {n_covered}/{len(rows)} params covered at 95% CI")
    valid_rhats = [r["rhat"] for r in rows if not np.isnan(r["rhat"])]
    if valid_rhats:
        max_rhat = max(valid_rhats)
        print(f"  max rank-Rhat: {max_rhat:.3f}")
    else:
        max_rhat = float("nan")
        print(f"  rank-Rhat: not computable (single chain or NaN)")

    # ---- Save ----
    np.savez_compressed(
        out_dir / "svssm_hmc_multi_full_phi_samples.npz",
        samples_raw=samples_raw,
        mu=cons["mu"], phi_diag=cons["phi_diag"], phi_off=cons["phi_off"],
        sigma_eta_sq=cons["sigma_eta_sq"],
        accept=accs, log_prob=lps,
        mu_truth=mu_truth, phi_diag_truth=phi_diag_truth,
        phi_off_truth=phi_off_truth, sigma_eta_sq_truth=sigma_eta_sq_truth,
        Phi_truth=Phi_truth,
    )

    (out_dir / "svssm_hmc_multi_full_phi_summary.json").write_text(json.dumps({
        "tf": tf.__version__, "tfp": tfp.__version__,
        "config": vars(args), "rows": rows,
        "elapsed_s": elapsed,
        "accept_rate_overall": float(accs.mean()),
        "max_rhat": max_rhat,
        "n_covered_95ci": n_covered,
        "n_total_params": len(rows),
    }, indent=2))
    print(f"\n  [save] {out_dir / 'svssm_hmc_multi_full_phi_samples.npz'}")
    print(f"  [save] {out_dir / 'svssm_hmc_multi_full_phi_summary.json'}")

    # Human-readable Vehtari diagnostics .txt (raw params + derived
    # stationary covariance). Same numbers as the JSON, formatted for
    # eyeballing without writing a one-shot script.
    try:
        import subprocess
        subprocess.run([
            "python", "scripts/save_diagnostics_multi_full_phi.py",
            "--out_dir", str(out_dir),
        ], check=True)
    except Exception as e:
        print(f"  [save_diagnostics warning] {e}")

    # ---- Plots ----
    if _HAVE_MPL and args.d == 2:
        # 7-panel posterior marginals
        fig, axes = plt.subplots(2, 4, figsize=(16, 7))
        param_specs = [
            ("mu_0",   cons["mu"][:, :, 0].ravel(),       float(mu_truth[0])),
            ("mu_1",   cons["mu"][:, :, 1].ravel(),       float(mu_truth[1])),
            ("phi_00", cons["phi_diag"][:, :, 0].ravel(), float(phi_diag_truth[0])),
            ("phi_11", cons["phi_diag"][:, :, 1].ravel(), float(phi_diag_truth[1])),
            ("phi_01", cons["phi_off"][:, :, 0].ravel(),  float(phi_off_truth[0])),
            ("sigma_eta_sq_0", cons["sigma_eta_sq"][:, :, 0].ravel(),
              float(sigma_eta_sq_truth[0])),
            ("sigma_eta_sq_1", cons["sigma_eta_sq"][:, :, 1].ravel(),
              float(sigma_eta_sq_truth[1])),
        ]
        for ax, (name, samp, tval) in zip(axes.flatten()[:7], param_specs):
            ax.hist(samp, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
            ax.axvline(tval, color="red", ls="--", lw=1.5, label=f"truth={tval:.3f}")
            ax.axvline(float(np.median(samp)), color="black", ls=":", lw=1.0,
                        label=f"med={float(np.median(samp)):.3f}")
            ax.set_title(name)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        # Hide unused 8th panel
        axes.flatten()[7].axis("off")
        fig.suptitle(f"V1 multivariate, upper-triangular Phi, T={args.T}, d={args.d}")
        fig.tight_layout()
        fig.savefig(out_dir / "svssm_hmc_multi_full_phi_marginals.png",
                     dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [save] {out_dir / 'svssm_hmc_multi_full_phi_marginals.png'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
