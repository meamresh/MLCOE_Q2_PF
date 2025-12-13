
# src/models/ssm_lgssm.py
from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp
from dataclasses import dataclass

tfd = tfp.distributions

@dataclass
class LGSSM:
    """
    Linear Gaussian State-Space Model (ABCD notation).

    Discrete-time linear dynamics and linear measurements:
        x_n = A x_{n-1} + B v_n
        y_n = C x_n     + D w_n
    with v_n ~ N(0, I) and w_n ~ N(0, I).

    Parameters
    ----------
    A : tf.Tensor
        State transition matrix (n, n).
    B : tf.Tensor
        Process noise coupling (n, nv).
    C : tf.Tensor
        Observation matrix (m, n).
    D : tf.Tensor
        Measurement noise coupling (m, nw).
    m0 : tf.Tensor
        Initial mean (n,) or (n, 1).
    P0 : tf.Tensor
        Initial covariance (n, n).
    Q : tf.Tensor
        Process noise covariance (n, n), typically B B^T.
    R : tf.Tensor
        Measurement noise covariance (m, m), typically D D^T.
    nx, ny, nv, nw : int
        Dimensions for state, observation, and noise channels.
    """

    A: tf.Tensor
    B: tf.Tensor
    C: tf.Tensor
    D: tf.Tensor
    m0: tf.Tensor
    P0: tf.Tensor
    Q: tf.Tensor
    R: tf.Tensor
    nx: int
    ny: int
    nv: int
    nw: int

    @staticmethod
    def from_config(cfg: dict) -> "LGSSM":
        """
        Construct an LGSSM from a YAML-style configuration dictionary.

        Parameters
        ----------
        cfg : dict
            Keys include 'dimensions' (nx, ny, nv, nw), matrices ('A', 'B_raw',
            'C', 'D'), and 'params' (sigma_a, sigma_z, Sigma0_diag).

        Returns
        -------
        LGSSM
            Instantiated model.
        """
        dims = cfg["dimensions"]
        nx, ny, nv, nw = dims["nx"], dims["ny"], dims["nv"], dims["nw"]

        A = tf.constant(cfg["A"], dtype=tf.float32)
        C = tf.constant(cfg["C"], dtype=tf.float32)
        B_raw = tf.constant(cfg["B_raw"], dtype=tf.float32)
        D_raw = tf.constant(cfg["D"], dtype=tf.float32)

        sigma_a = float(cfg["params"]["sigma_a"])
        sigma_z = float(cfg["params"]["sigma_z"])
        P0 = tf.linalg.diag(tf.constant(cfg["params"]["Sigma0_diag"], dtype=tf.float32))
        m0 = tf.zeros(nx, dtype=tf.float32)

        B = B_raw * sigma_a
        D = D_raw * sigma_z

        Q = tf.matmul(B, B, transpose_b=True)
        R = tf.matmul(D, D, transpose_b=True)

        return LGSSM(A=A, B=B, C=C, D=D, m0=m0, P0=P0, Q=Q, R=R,
                     nx=nx, ny=ny, nv=nv, nw=nw)

    def sample(self, N: int, seed: int | None = None) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Simulate states and observations for N time steps.

        Parameters
        ----------
        N : int
            Number of time steps.
        seed : int or None
            Random seed.

        Returns
        -------
        X : tf.Tensor
            True states, shape (N, nx).
        Y : tf.Tensor
            Observations, shape (N, ny).
        """
        if seed is not None:
            tf.random.set_seed(seed)

        # Initialize distributions
        # Use MultivariateNormalTriL instead of deprecated MultivariateNormalFullCovariance
        P0_chol = tf.linalg.cholesky(self.P0)
        mvn_initial = tfd.MultivariateNormalTriL(
            loc=self.m0,
            scale_tril=P0_chol
        )
        mvn_process = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.nv, dtype=tf.float32),
            scale_diag=tf.ones(self.nv, dtype=tf.float32)
        )
        mvn_obs = tfd.MultivariateNormalDiag(
            loc=tf.zeros(self.nw, dtype=tf.float32),
            scale_diag=tf.ones(self.nw, dtype=tf.float32)
        )

        X_list = []
        Y_list = []

        # Initial state x1 ~ N(m0, P0)
        X_prev = mvn_initial.sample()
        X_list.append(X_prev)
        W1 = mvn_obs.sample()
        # Ensure X_prev and W1 are column vectors for matmul
        X_prev_col = tf.reshape(X_prev, [-1, 1])
        W1_col = tf.reshape(W1, [-1, 1])
        Y_0 = tf.squeeze(tf.matmul(self.C, X_prev_col)) + tf.squeeze(tf.matmul(self.D, W1_col))
        Y_list.append(Y_0)

        # Generate remaining time steps
        for t in range(1, N):
            Vt = mvn_process.sample()
            Wt = mvn_obs.sample()
            # Ensure vectors are column vectors for matmul
            X_prev_col = tf.reshape(X_prev, [-1, 1])
            Vt_col = tf.reshape(Vt, [-1, 1])
            X_t = tf.squeeze(tf.matmul(self.A, X_prev_col)) + tf.squeeze(tf.matmul(self.B, Vt_col))
            X_t_col = tf.reshape(X_t, [-1, 1])
            Wt_col = tf.reshape(Wt, [-1, 1])
            Y_t = tf.squeeze(tf.matmul(self.C, X_t_col)) + tf.squeeze(tf.matmul(self.D, Wt_col))
            X_list.append(X_t)
            Y_list.append(Y_t)
            X_prev = X_t

        X = tf.stack(X_list, axis=0)
        Y = tf.stack(Y_list, axis=0)

        return X, Y
