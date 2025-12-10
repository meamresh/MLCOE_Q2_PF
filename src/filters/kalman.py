import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R, x0, P0):
        """
        Multidimensional Kalman Filter

        Parameters
        ----------
        F : np.ndarray
            State transition matrix (n×n)
        H : np.ndarray
            Measurement matrix (m×n)
        Q : np.ndarray
            Process noise covariance (n×n)
        R : np.ndarray
            Measurement noise covariance (m×m)
        x0 : np.ndarray
            Initial state estimate (n×1)
        P0 : np.ndarray
            Initial covariance estimate (n×n)
        """

        self.F = F
        self.H = H
        self.Q = Q
        self.R = R

        self.x = x0        # state estimate
        self.P = P0        # covariance estimate

    def predict(self):
        """Prediction step"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x, self.P

    def update(self, z):
        """Correction step"""
        # Innovation
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Updated state
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.x, self.P
