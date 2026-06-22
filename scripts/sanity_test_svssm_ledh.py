"""
Sanity tests for DifferentiableLEDHLogLikelihoodSVSSM.

Checks:
  1. Generates canonical-SVSSM data with known (mu, phi, sigma_eta).
  2. Confirms log p is finite and gradient is finite over a few CRN seeds.
  3. Confirms Section-1 properties: per-step XLA kernel traces once and only
     once across 10 distinct-theta evaluations.
  4. Quick benchmark: cold/warm forward and forward+grad timings.
  5. Mild sanity check: log p at the data-generating theta should be higher
     than at a nearby wrong-theta on average (likelihood peaks near truth).
"""

from __future__ import annotations

import statistics as stats
import time

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import DifferentiableLEDHLogLikelihoodSVSSM


def gen_svssm_data(T: int, mu: float, phi: float, sigma_eta: float,
                   h0: float | None = None, seed: int = 42) -> tuple[tf.Tensor, tf.Tensor]:
    """Sample T observations from y_t = exp(h_t/2) * eps_t with AR(1) h_t."""
    tf.random.set_seed(seed)
    sigma_eta_t = tf.constant(sigma_eta, tf.float32)
    if h0 is None:
        h0 = mu  # start at mean reversion level
    h = tf.constant(float(h0), tf.float32)
    hs, ys = [], []
    for t in range(T):
        eta = tf.random.normal([])
        h = mu + phi * (h - mu) + sigma_eta_t * eta
        eps = tf.random.normal([])
        y = tf.exp(h / 2.0) * eps
        hs.append(h)
        ys.append(y)
    return tf.stack(hs), tf.stack(ys)


def time_callable(fn, reps: int) -> float:
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        out = fn()
        if isinstance(out, tf.Tensor):
            _ = out.numpy()
        elif isinstance(out, (tuple, list)):
            for o in out:
                if isinstance(o, tf.Tensor):
                    _ = o.numpy()
        ts.append(time.perf_counter() - t0)
    return float(stats.median(ts))


def main() -> int:
    # True parameters
    mu_true, phi_true, sigma_eta_true = 0.0, 0.95, 0.3
    T, N = 50, 64

    print(f"[svssm-ledh sanity] TF {tf.__version__}")
    print(f"  Data-generating: mu={mu_true}, phi={phi_true}, sigma_eta={sigma_eta_true}, "
          f"T={T}, N={N}")

    h_true, y_obs = gen_svssm_data(T=T, mu=mu_true, phi=phi_true,
                                   sigma_eta=sigma_eta_true, seed=42)
    print(f"  h_true range: [{float(tf.reduce_min(h_true)):.2f}, "
          f"{float(tf.reduce_max(h_true)):.2f}]")
    print(f"  y_obs range:  [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")
    print(f"  |y_obs|>0:    {int(tf.reduce_sum(tf.cast(tf.abs(y_obs) > 0, tf.int32)))}/{T}")

    ll = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=10, sinkhorn_epsilon=1.0,
        sinkhorn_iters=10, grad_window=4, jit_compile=True, integrator="exp",
    )

    # === 1. log p + grad at truth, multiple seeds ===
    print("\n[1] Forward + gradient at (mu*, phi*, sigma_eta*) over 5 CRN seeds")
    lps, grads = [], []
    for s in (101, 102, 103, 104, 105):
        tf.random.set_seed(s)
        params = tf.constant([mu_true, phi_true, sigma_eta_true ** 2], dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(params)
            v = ll(params[0], params[1], params[2], y_obs)
        g = tape.gradient(v, params)
        lps.append(float(v.numpy()))
        grads.append(g.numpy().astype(np.float64))
        print(f"  seed={s}: log p = {lps[-1]:.4f}   grad = {grads[-1]}")
    lps_arr = np.asarray(lps)
    grads_arr = np.asarray(grads)
    print(f"  log p mean={lps_arr.mean():.4f} std={lps_arr.std(ddof=1):.4f}  "
          f"finite={int(np.all(np.isfinite(lps_arr)))}")
    print(f"  ||grad|| mean={float(np.mean(np.linalg.norm(grads_arr, axis=1))):.3e}  "
          f"finite_rate={int(np.all(np.isfinite(grads_arr)))}")

    # === 2. Retracing check ===
    print("\n[2] Retracing check over 10 distinct theta")
    kernel = ll._timestep_1d_xla
    rng = np.random.default_rng(0)
    thetas = [
        tf.constant(rng.normal(loc=[mu_true, phi_true, sigma_eta_true ** 2], scale=0.05),
                    dtype=tf.float32)
        for _ in range(10)
    ]
    c0 = int(kernel.experimental_get_tracing_count())
    _ = ll(thetas[0][0], thetas[0][1], tf.maximum(thetas[0][2], 1e-3), y_obs).numpy()
    c1 = int(kernel.experimental_get_tracing_count())
    for th in thetas[1:]:
        _ = ll(th[0], th[1], tf.maximum(th[2], 1e-3), y_obs).numpy()
    c2 = int(kernel.experimental_get_tracing_count())
    print(f"  trace count: pre={c0}  post-warmup={c1}  post-10evals={c2}")
    no_retrace = (c2 - c1) == 0
    print(f"  {'PASS' if no_retrace else 'FAIL'}: "
          f"{'zero retraces' if no_retrace else 'retraces detected'}")

    # === 3. Benchmark ===
    print("\n[3] Benchmark (warm forward and forward+grad)")
    mu_t = tf.constant(mu_true, tf.float32)
    phi_t = tf.constant(phi_true, tf.float32)
    sig_t = tf.constant(sigma_eta_true ** 2, tf.float32)

    def fwd():
        tf.random.set_seed(123)
        return ll(mu_t, phi_t, sig_t, y_obs)

    def fwd_grad():
        tf.random.set_seed(123)
        params = tf.identity(tf.stack([mu_t, phi_t, sig_t]))
        with tf.GradientTape() as tape:
            tape.watch(params)
            v = ll(params[0], params[1], params[2], y_obs)
        return tape.gradient(v, params)

    _ = fwd(); _ = fwd_grad()  # warm compile
    fwd_med = time_callable(fwd, 5)
    fg_med = time_callable(fwd_grad, 5)
    print(f"  warm forward      = {fwd_med * 1000:7.2f} ms")
    print(f"  warm forward+grad = {fg_med * 1000:7.2f} ms")
    print(f"  backward/forward  = {fg_med / fwd_med:5.2f}x")

    # === 4. Likelihood peaks near truth ===
    print("\n[4] log p at truth vs perturbed parameters (5 seeds each)")
    perturbations = [
        ("truth          ", (mu_true, phi_true, sigma_eta_true)),
        ("mu+1.0         ", (mu_true + 1.0, phi_true, sigma_eta_true)),
        ("phi=0.5        ", (mu_true, 0.5, sigma_eta_true)),
        ("sigma_eta*2    ", (mu_true, phi_true, sigma_eta_true * 2.0)),
        ("sigma_eta/2    ", (mu_true, phi_true, sigma_eta_true / 2.0)),
    ]
    results = {}
    for label, (m, p, s) in perturbations:
        vals = []
        for seed in (201, 202, 203, 204, 205):
            tf.random.set_seed(seed)
            v = ll(tf.constant(m, tf.float32),
                   tf.constant(p, tf.float32),
                   tf.constant(s ** 2, tf.float32),
                   y_obs)
            vals.append(float(v.numpy()))
        results[label] = float(np.mean(vals))
        print(f"  {label}: mean log p = {results[label]:>10.3f}")
    truth_lp = results["truth          "]
    higher_at_truth = all(truth_lp >= v - 5.0 for k, v in results.items())  # tolerant
    print(f"  {'OK' if higher_at_truth else 'CHECK'}: truth is among the higher (or "
          f"within 5 of) competing thetas (PF noise tolerated)")

    overall = no_retrace and bool(np.all(np.isfinite(lps_arr))) and bool(np.all(np.isfinite(grads_arr)))
    print("\n" + "=" * 60)
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
