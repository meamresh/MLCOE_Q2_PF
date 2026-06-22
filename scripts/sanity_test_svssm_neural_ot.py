"""
Phase-1 sanity test for the SVSSM neural-OT adapter.

Purpose: verify that DifferentiableLEDHNeuralOTSVSSM is wired correctly
end-to-end. Uses an UNTRAINED ConditionalMGradNet (n_scalar_ctx=7) so the
log-likelihood numbers themselves are meaningless; what we check is:

  1. The integration runs without shape errors.
  2. The forward call returns a finite scalar log-likelihood.
  3. The gradient flows through to (mu, phi, sigma_eta_sq).
  4. Two distinct theta values produce DIFFERENT outputs (the neural OT
     is being called per-call, not cached).
  5. The 7-D SVSSM context vector is constructed correctly.

This is a wiring test, not a quality test. Phase 2 trains the model.
"""

from __future__ import annotations

import time

import numpy as np
import tensorflow as tf

from src.filters.bonus.extra_bonus.differentiable_ledh_neural_ot_svssm import (
    DifferentiableLEDHNeuralOTSVSSM,
    build_svssm_context_scalars,
)
from src.filters.bonus.mgradnet_ot import ConditionalMGradNet


def gen_svssm(T, mu=0.0, phi=0.95, sigma_eta=0.3, seed=42):
    tf.random.set_seed(seed)
    sigma_eta_t = tf.constant(sigma_eta, tf.float32)
    h = tf.constant(float(mu), tf.float32)
    ys = []
    for _ in range(T):
        h = mu + phi * (h - mu) + sigma_eta_t * tf.random.normal([])
        ys.append(tf.exp(h / 2.0) * tf.random.normal([]))
    return tf.stack(ys)


def main() -> int:
    T = 20
    N = 64
    print(f"[svssm-neural-ot sanity] TF {tf.__version__}")
    print(f"  T={T}, N={N} (untrained mGradNet, n_scalar_ctx=7)\n")

    # --- Check 1: context-vector construction ---
    print("[1] build_svssm_context_scalars shape + content")
    ctx = build_svssm_context_scalars(
        mu=tf.constant(0.0), phi=tf.constant(0.95),
        sigma_eta_sq=tf.constant(0.09),
        t=10.0, z_t=-1.2, ess=42.0, epsilon=1.0, T_max=20.0,
    )
    expected_shape = (7,)
    assert tuple(ctx.shape) == expected_shape, \
        f"ctx shape {ctx.shape} != {expected_shape}"
    # Quick sanity on a couple of values
    expected_mu = 0.0
    expected_phi_raw = float(np.arctanh(0.95))  # ~1.832
    expected_log_sig = float(np.log(0.09))      # ~-2.408
    diffs = [
        ("mu",        float(ctx[0]) - expected_mu),
        ("phi_raw",   float(ctx[1]) - expected_phi_raw),
        ("log_sig",   float(ctx[2]) - expected_log_sig),
    ]
    ok = all(abs(d) < 1e-4 for _, d in diffs)
    print(f"  ctx={ctx.numpy()}")
    print(f"  expected: mu=0.0, phi_raw={expected_phi_raw:.4f}, "
          f"log_sig={expected_log_sig:.4f}, t/T=0.5, z_t=-1.2, ess=42.0, eps=1.0")
    print(f"  {'PASS' if ok else 'FAIL'}: parameter mapping correct\n")

    # --- Build untrained model + filter ---
    print("[2] Build untrained ConditionalMGradNet (n_scalar_ctx=7)")
    model = ConditionalMGradNet(state_dim=1, n_ridges=64, n_scalar_ctx=7)
    # Build the model by calling it once with dummy input
    dummy_p = tf.random.normal([N])
    dummy_w = tf.fill([N], 1.0 / N)
    dummy_ctx = tf.zeros([7], tf.float32)
    _ = model(dummy_p, dummy_w, dummy_ctx)
    n_params = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"  model built, trainable params: {n_params:,}\n")

    ll = DifferentiableLEDHNeuralOTSVSSM(
        neural_ot_model=model, num_particles=N,
        n_lambda=10, sinkhorn_epsilon=1.0,
        grad_window=4, jit_compile=False, integrator="exp",
    )
    print(f"  built DifferentiableLEDHNeuralOTSVSSM\n")

    # --- Check 3: end-to-end forward call ---
    print("[3] End-to-end forward call (untrained, log p value is meaningless)")
    y_obs = gen_svssm(T)
    print(f"  y_obs range: [{float(tf.reduce_min(y_obs)):.3f}, "
          f"{float(tf.reduce_max(y_obs)):.3f}]")
    t0 = time.perf_counter()
    val = ll(
        tf.constant(0.0, tf.float32),
        tf.constant(0.95, tf.float32),
        tf.constant(0.09, tf.float32),
        y_obs,
    )
    elapsed = time.perf_counter() - t0
    val_f = float(val.numpy())
    finite = np.isfinite(val_f)
    print(f"  log p = {val_f:.4f}  ({'finite' if finite else 'NOT finite'})  "
          f"({elapsed:.2f}s)")
    print(f"  {'PASS' if finite else 'FAIL'}: forward returned a finite scalar\n")

    # --- Check 4: gradient flow ---
    print("[4] Gradient flows through to (mu, phi, sigma_eta_sq)")
    params = tf.constant([0.0, 0.95, 0.09], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(params)
        v = ll(params[0], params[1], params[2], y_obs)
    g = tape.gradient(v, params)
    g_np = g.numpy()
    finite_g = bool(np.all(np.isfinite(g_np)))
    print(f"  grad = {g_np}")
    print(f"  {'PASS' if finite_g else 'FAIL'}: gradient is finite\n")

    # --- Check 5: two distinct thetas give two distinct outputs ---
    print("[5] Two distinct theta values produce two different outputs")
    tf.random.set_seed(123)
    v1 = float(ll(tf.constant(0.0, tf.float32), tf.constant(0.95, tf.float32),
                  tf.constant(0.09, tf.float32), y_obs).numpy())
    tf.random.set_seed(123)
    v2 = float(ll(tf.constant(0.5, tf.float32), tf.constant(0.80, tf.float32),
                  tf.constant(0.25, tf.float32), y_obs).numpy())
    diff = abs(v1 - v2)
    print(f"  theta=(0.00, 0.95, 0.09): log p = {v1:.4f}")
    print(f"  theta=(0.50, 0.80, 0.25): log p = {v2:.4f}")
    print(f"  |delta| = {diff:.4f}")
    print(f"  {'PASS' if diff > 0.01 else 'FAIL'}: outputs differ across theta "
          f"(the network is being called per-call)\n")

    # --- Compare to Sinkhorn baseline at the same theta (smoke compare) ---
    print("[6] Smoke compare vs Sinkhorn baseline (DifferentiableLEDHLogLikelihoodSVSSM)")
    from src.filters.bonus.extra_bonus.differentiable_ledh_svssm import DifferentiableLEDHLogLikelihoodSVSSM
    base = DifferentiableLEDHLogLikelihoodSVSSM(
        num_particles=N, n_lambda=10, sinkhorn_epsilon=1.0,
        sinkhorn_iters=10, grad_window=4, jit_compile=True, integrator="exp",
    )
    tf.random.set_seed(123)
    v_base = float(base(tf.constant(0.0), tf.constant(0.95),
                        tf.constant(0.09), y_obs).numpy())
    tf.random.set_seed(123)
    v_neural = float(ll(tf.constant(0.0), tf.constant(0.95),
                        tf.constant(0.09), y_obs).numpy())
    print(f"  Sinkhorn baseline:        log p = {v_base:.4f}")
    print(f"  Neural OT (UNTRAINED):    log p = {v_neural:.4f}")
    print(f"  |delta| = {abs(v_base - v_neural):.4f}  "
          f"(large is expected with an untrained model)\n")

    overall = ok and finite and finite_g and diff > 0.01
    print("=" * 60)
    print(f"OVERALL Phase-1 wiring: {'PASS' if overall else 'FAIL'}")
    print("=" * 60)
    print("  (Quality-vs-Sinkhorn test deferred to Phase 2: train the model.)")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
