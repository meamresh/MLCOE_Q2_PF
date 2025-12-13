
# src/filters/kalman.py
from __future__ import annotations
import tensorflow as tf

class KalmanFilter:
    """
    Multidimensional Kalman Filter for linear Gaussian systems (ABCD notation).

    Discrete-time model:
        x_n = A x_{n-1} + B v_n
        y_n = C x_n     + D w_n
    with v_n ~ N(0, I) and w_n ~ N(0, I).
    The filter implements predict/update steps with either the canonical Riccati
    covariance update (default) or the Joseph-stabilized update.

    Parameters
    ----------
    A : tf.Tensor
        State transition matrix of shape (n, n).
    C : tf.Tensor
        Observation matrix of shape (m, n).
    x0 : tf.Tensor
        Initial state estimate of shape (n,) or (n, 1).
    P0 : tf.Tensor
        Initial covariance estimate of shape (n, n).
    Q : tf.Tensor | None
        Process noise covariance (n, n). If None, but B is given, uses Q = B B^T.
    R : tf.Tensor | None
        Measurement noise covariance (m, m). If None, but D is given, uses R = D D^T.
    B : tf.Tensor | None
        Process noise coupling (n, nv), used only to construct Q when Q is None.
    D : tf.Tensor | None
        Measurement noise coupling (m, nw), used only to construct R when R is None.

    Notes
    -----
    - Riccati update:
        P <- (I - K C) P_pred
      Joseph update:
        P <- (I - K C) P_pred (I - K C)^T + K R K^T
      These are algebraically equivalent in exact arithmetic, but Joseph is
      preferred in practice since it better preserves symmetry/PSD under rounding.  # noqa
      See: Welch & Bishop; Stengel MAE546; Joseph form notes.  # noqa
    - The gain uses a linear solve (LAPACK) by default, which is more accurate
      than explicit inversion.  # noqa
    """
    def __init__(self,
                 A: tf.Tensor,
                 C: tf.Tensor,
                 x0: tf.Tensor,
                 P0: tf.Tensor,
                 Q: tf.Tensor | None = None,
                 R: tf.Tensor | None = None,
                 B: tf.Tensor | None = None,
                 D: tf.Tensor | None = None) -> None:
        self.A = A
        self.C = C
        # Reshape x0 to (n, 1) if it's 1D
        try:
            if x0.shape.ndims == 1:
                self.x = tf.reshape(x0, [-1, 1])
            else:
                self.x = x0
        except (AttributeError, ValueError):
            # Fallback for dynamic shapes
            x0_rank = tf.rank(x0)
            self.x = tf.cond(tf.equal(x0_rank, 1), 
                            lambda: tf.reshape(x0, [-1, 1]),
                            lambda: x0)
        self.P = P0
        # Build Q/R if not supplied
        A_shape = tf.shape(A)
        C_shape = tf.shape(C)
        if Q is not None:
            self.Q = Q
        elif B is not None:
            self.Q = tf.matmul(B, B, transpose_b=True)
        else:
            self.Q = tf.zeros([A_shape[0], A_shape[0]], dtype=A.dtype)
        if R is not None:
            self.R = R
        elif D is not None:
            self.R = tf.matmul(D, D, transpose_b=True)
        else:
            self.R = tf.zeros([C_shape[0], C_shape[0]], dtype=C.dtype)

    def predict(self) -> tuple[tf.Tensor, tf.Tensor]:
        """
        Time update (prediction): (x_pred, P_pred).

        Returns
        -------
        x_pred : tf.Tensor
            Predicted mean, shape (n, 1).
        P_pred : tf.Tensor
            Predicted covariance, shape (n, n).
        """
        self.x = tf.matmul(self.A, self.x)
        self.P = tf.matmul(self.A, tf.matmul(self.P, self.A, transpose_b=True)) + self.Q
        return self.x, self.P

    def update(self,
               z: tf.Tensor,
               *,
               joseph: bool = False,
               use_solve: bool = True) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Measurement update (correction), Riccati default; Joseph optional.

        Parameters
        ----------
        z : tf.Tensor
            Measurement vector of shape (m,) or (m, 1).
        joseph : bool, optional
            If True, use Joseph-stabilized covariance update. Default False (Riccati).
        use_solve : bool, optional
            If True, compute K via solve(S, C P_pred); else use explicit inv(S).

        Returns
        -------
        x_filt : tf.Tensor
            Filtered mean (n, 1).
        P_filt : tf.Tensor
            Filtered covariance (n, n).
        K : tf.Tensor
            Kalman gain (n, m).
        y : tf.Tensor
            Innovation residual (m, 1).
        S : tf.Tensor
            Innovation covariance (m, m).
        """
        # Reshape z to (m, 1) if it's 1D
        try:
            if z.shape.ndims == 1:
                z = tf.reshape(z, [-1, 1])
        except (AttributeError, ValueError):
            # Fallback for dynamic shapes
            z_rank = tf.rank(z)
            z = tf.cond(tf.equal(z_rank, 1),
                       lambda: tf.reshape(z, [-1, 1]),
                       lambda: z)
        # Innovation and its covariance
        y = z - tf.matmul(self.C, self.x)
        S = tf.matmul(self.C, tf.matmul(self.P, self.C, transpose_b=True)) + self.R
        # Gain
        if use_solve:
            # K = P @ C^T @ S^{-1}
            # We can compute this as: solve(S^T, (C @ P)^T)^T = solve(S, C @ P)^T
            # Or more directly: K^T = S^{-1} @ (C @ P), so K = (S^{-1} @ (C @ P))^T
            CP = tf.matmul(self.C, self.P)  # (m, n) where m=obs_dim, n=state_dim
            # Solve S @ K^T = CP for K^T, giving K^T = S^{-1} @ CP
            K_T = tf.linalg.solve(S, CP)  # (m, n)
            K = tf.transpose(K_T)  # (n, m)
        else:
            K = tf.matmul(tf.matmul(self.P, self.C, transpose_b=True), tf.linalg.inv(S))
        # Mean update
        self.x = self.x + tf.matmul(K, y)
        # Covariance update
        P_shape = tf.shape(self.P)
        I = tf.eye(P_shape[0], dtype=self.P.dtype)
        if joseph:
            IKC = I - tf.matmul(K, self.C)
            self.P = tf.matmul(IKC, tf.matmul(self.P, IKC, transpose_b=True)) + tf.matmul(K, tf.matmul(self.R, K, transpose_b=True))
        else:
            self.P = tf.matmul(I - tf.matmul(K, self.C), self.P)
        return self.x, self.P, K, y, S

    def filter(self,
               Z: tf.Tensor,
               *,
               joseph: bool = False,
               use_solve: bool = True) -> dict[str, tf.Tensor]:
        """
        Run the Kalman Filter over a sequence of measurements.

        Parameters
        ----------
        Z : tf.Tensor
            Measurement sequence of shape (N, m) or (N, m, 1).
        joseph : bool, optional
            If True, use Joseph update each step; else Riccati. Default False.
        use_solve : bool, optional
            If True, use a linear solve for the gain; else explicit inverse.

        Returns
        -------
        result : dict
            Keys:
            - 'x_pred' : (N, n, 1) predicted means
            - 'P_pred' : (N, n, n)  predicted covariances
            - 'x_filt' : (N, n, 1) filtered means
            - 'P_filt' : (N, n, n)  filtered covariances
        """
        Z = tf.convert_to_tensor(Z)
        # Expand dims if Z is 2D
        try:
            if Z.shape.ndims == 2:
                Z = tf.expand_dims(Z, axis=-1)  # (N, m, 1)
        except (AttributeError, ValueError):
            # Fallback for dynamic shapes
            Z_rank = tf.rank(Z)
            Z = tf.cond(tf.equal(Z_rank, 2),
                       lambda: tf.expand_dims(Z, axis=-1),
                       lambda: Z)
        N = tf.shape(Z)[0]
        n = tf.shape(self.x)[0]

        x_pred_list = []
        P_pred_list = []
        x_filt_list = []
        P_filt_list = []

        for t in range(N):
            xp, Pp = self.predict()
            x_pred_list.append(xp)
            P_pred_list.append(Pp)

            xf, Pf, *_ = self.update(Z[t], joseph=joseph, use_solve=use_solve)
            x_filt_list.append(xf)
            P_filt_list.append(Pf)

        x_pred = tf.stack(x_pred_list, axis=0)
        P_pred = tf.stack(P_pred_list, axis=0)
        x_filt = tf.stack(x_filt_list, axis=0)
        P_filt = tf.stack(P_filt_list, axis=0)

        return {"x_pred": x_pred, "P_pred": P_pred,
                "x_filt": x_filt, "P_filt": P_filt}
