"""
Multi-target acoustic state-space model.

Compatible with existing filter infrastructure (EKF, UKF, PF, PF-PF).
State: flattened [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...] for C targets.
Control: not used (constant velocity model); kept for interface compatibility.
Measurement: acoustic amplitudes from all sensors [z1, z2, ..., z_Ns].
"""

from __future__ import annotations

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


class MultiTargetAcousticSSM:
    """
    Multi-target acoustic tracking SSM compatible with filter interface.

    State: [x1, y1, vx1, vy1, x2, y2, vx2, vy2, ...] (flattened, 4*C dims).
    Control: not used (constant velocity); interface accepts it for compatibility.
    Measurement: [z1, z2, ..., z_Ns] (acoustic amplitudes from all sensors).
    """

    def __init__(
        self,
        num_targets: int = 4,
        num_sensors: int = 25,
        area_size: float = 40.0,
        dt: float = 1.0,
        psi: float = 10.0,
        d0: float = 0.1,
        sigma_w: float = 0.1,
        process_noise: tf.Tensor | None = None,
        meas_noise: tf.Tensor | None = None,
        process_noise_scale: float = 1.5,
        dtype: tf.DType = tf.float32,
        enforce_boundaries: bool = False,
        sensor_grid_size: int | None = None,
    ):
        """
        Initialize multi-target acoustic SSM.

        Parameters
        ----------
        num_targets : int
            Number of targets (C).
        num_sensors : int
            Number of sensors (N_s).
        area_size : float
            Size of tracking area (meters).
        dt : float
            Time step.
        psi : float
            Acoustic amplitude scaling factor.
        d0 : float
            Distance regularization parameter.
        sigma_w : float
            Measurement noise standard deviation (paper uses σ_w = 0.1).
        process_noise : tf.Tensor, optional
            Process noise matrix (4x4 per target), or None for default.
        meas_noise : tf.Tensor or float, optional
            Measurement noise (scalar or matrix), or None for default.
        process_noise_scale : float
            Multiplier for process noise (filter vs generation).
        dtype : tf.DType
            Data type.
        enforce_boundaries : bool
            Whether to enforce boundary conditions.
        sensor_grid_size : int, optional
            Size of sensor grid (sensor_grid_size x sensor_grid_size).
            If None, inferred from num_sensors (e.g. 5 for 25).
        """
        self.C = num_targets
        self.N_s = num_sensors
        self.area_size = float(area_size)
        self.dt = dt
        self.psi = psi
        self.d0 = d0
        self.sigma_w = sigma_w
        self.dtype = dtype
        self.enforce_boundaries = enforce_boundaries
        self.process_noise_scale = process_noise_scale

        # State dimension: 4 per target (x, y, vx, vy)
        self.state_dim = 4 * self.C
        self.meas_per_landmark = 1

        # Default process noise per target (4x4)
        if process_noise is None:
            Q0 = tf.constant([
                [1.0 / 3, 0.0, 0.5, 0.0],
                [0.0, 1.0 / 3, 0.0, 0.5],
                [0.5, 0.0, 1.0, 0.0],
                [0.0, 0.5, 0.0, 1.0],
            ], dtype=dtype)
            Q_gen_block = (1.0 / 20.0) * Q0

            Q_filter_block = tf.constant([
                [3.0, 0.0, 0.1, 0.0],
                [0.0, 3.0, 0.0, 0.1],
                [0.1, 0.0, 0.03, 0.0],
                [0.0, 0.1, 0.0, 0.03],
            ], dtype=dtype)

            self.V_gen_single = Q_gen_block
            self.V_filter_single = Q_filter_block
        else:
            V_single = tf.cast(process_noise, dtype)
            self.V_gen_single = V_single
            self.V_filter_single = V_single * process_noise_scale

        # Build process noise covariances eagerly to avoid graph-scope issues
        self._Q_gen_cache = self._build_block_diagonal(
            self.V_gen_single, self.C
        )
        self._Q_filter_cache = self._build_block_diagonal(
            self.V_filter_single, self.C
        )

        # Measurement noise
        if meas_noise is None:
            self.R = tf.eye(self.N_s, dtype=dtype) * (sigma_w ** 2)
        else:
            meas_noise = tf.cast(meas_noise, dtype)
            if meas_noise.shape.rank == 0 or (meas_noise.shape.rank == 1 and meas_noise.shape[0] is None):
                scalar = tf.reshape(meas_noise, [])
                self.R = tf.eye(self.N_s, dtype=dtype) * scalar
            else:
                self.R = meas_noise

        grid_size = (
            sensor_grid_size
            if sensor_grid_size is not None
            else tf.cast(
                tf.math.sqrt(tf.cast(self.N_s, tf.float32)), tf.int32
            )
        )
        self.sensor_positions = self._create_sensor_grid(grid_size)

        self.x0_true = self._create_default_initial_states()

        # Prior for particle-flow filters (Dai–Daum, PFPF-Dai): set by filter each step
        self._prior_mean = tf.Variable(
            tf.identity(self.x0_true),
            trainable=False,
            name="prior_mean",
        )
        self._prior_cov = tf.Variable(
            tf.eye(self.state_dim, dtype=self.dtype) * 100.0,
            trainable=False,
            name="prior_cov",
        )
        self._R_inv_meas = tf.linalg.inv(
            self.R + tf.eye(self.N_s, dtype=self.dtype) * 1e-8
        )

    @property
    def Q(self) -> tf.Tensor:
        """Process noise covariance for filtering (lazy construction)."""
        return self.Q_filter

    @property
    def Q_gen(self) -> tf.Tensor:
        """Process noise covariance for generation (precomputed)."""
        return self._Q_gen_cache

    @property
    def Q_filter(self) -> tf.Tensor:
        """Process noise covariance for filtering (precomputed)."""
        return self._Q_filter_cache

    @property
    def prior_mean(self) -> tf.Tensor:
        """Prior mean of length ``state_dim``; EKF/flow code may assign after predict."""
        return self._prior_mean.value()

    @prior_mean.setter
    def prior_mean(self, value: tf.Tensor) -> None:
        """Set the prior mean vector; reshaped to ``(state_dim,)`` in model dtype."""
        self._prior_mean.assign(tf.reshape(tf.cast(value, self.dtype), [self.state_dim]))

    @property
    def prior_cov(self) -> tf.Tensor:
        """Prior covariance of shape ``(state_dim, state_dim)``; set by the filter as needed."""
        return self._prior_cov.value()

    @prior_cov.setter
    def prior_cov(self, value: tf.Tensor) -> None:
        """Set the prior covariance; reshaped to ``(state_dim, state_dim)`` in model dtype."""
        self._prior_cov.assign(
            tf.reshape(tf.cast(value, self.dtype), [self.state_dim, self.state_dim])
        )

    def log_prior(self, x: tf.Tensor) -> tf.Tensor:
        """Log prior density log p0(x). x: (state_dim,) or (batch, state_dim)."""
        x = tf.cast(x, self.dtype)
        m = self.prior_mean
        P = self.prior_cov
        P_inv = tf.linalg.inv(P + tf.eye(self.state_dim, dtype=self.dtype) * 1e-8)
        if x.shape.rank == 1:
            d = x - m
            return -0.5 * tf.reduce_sum(d * tf.linalg.matvec(P_inv, d))
        d = x - m[tf.newaxis, :]
        return -0.5 * tf.reduce_sum(d * (d @ tf.transpose(P_inv)), axis=1)

    def gradient_log_prior(self, x: tf.Tensor) -> tf.Tensor:
        """Gradient of log prior w.r.t. x. x: (state_dim,)."""
        x = tf.cast(x, self.dtype)
        P_inv = tf.linalg.inv(
            self.prior_cov + tf.eye(self.state_dim, dtype=self.dtype) * 1e-8
        )
        return -tf.linalg.matvec(P_inv, x - self.prior_mean)

    def hessian_log_prior(self, x: tf.Tensor) -> tf.Tensor:
        """Hessian of log prior (constant). Returns (state_dim, state_dim)."""
        _ = tf.cast(x, self.dtype)
        return -tf.linalg.inv(
            self.prior_cov + tf.eye(self.state_dim, dtype=self.dtype) * 1e-8
        )

    def log_likelihood(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Log measurement likelihood log p(z|x). x: (state_dim,), z: (N_s,)."""
        x = tf.cast(x, self.dtype)
        z = tf.cast(z, self.dtype)
        h = self.measurement_model(x)
        if h.shape.rank == 0:
            h = h[tf.newaxis]
        res = z - h
        return -0.5 * tf.reduce_sum(res * tf.linalg.matvec(self._R_inv_meas, res))

    def gradient_log_likelihood(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Gradient of log p(z|x) w.r.t. x."""
        x = tf.cast(x, self.dtype)
        z = tf.cast(z, self.dtype)
        with tf.GradientTape() as tape:
            tape.watch(x)
            ll = self.log_likelihood(x, z)
        return tape.gradient(ll, x)

    def hessian_log_likelihood(self, x: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        """Hessian of log p(z|x) w.r.t. x. Returns (state_dim, state_dim)."""
        x = tf.cast(x, self.dtype)
        z = tf.cast(z, self.dtype)
        with tf.GradientTape() as outer_tape:
            outer_tape.watch(x)
            with tf.GradientTape() as inner_tape:
                inner_tape.watch(x)
                ll = self.log_likelihood(x, z)
            grad = inner_tape.gradient(ll, x)
        return outer_tape.jacobian(grad, x)

    def _build_block_diagonal(
        self, block: tf.Tensor, num_blocks: int
    ) -> tf.Tensor:
        """Build block diagonal matrix from repeated blocks."""
        block_size = int(block.shape[0])
        total_size = block_size * num_blocks

        result = tf.zeros([total_size, total_size], dtype=self.dtype)

        for i in range(num_blocks):
            start = i * block_size
            end = start + block_size

            rows = tf.range(start, end)[:, tf.newaxis]
            cols = tf.range(start, end)[tf.newaxis, :]
            row_indices = tf.tile(rows, [1, block_size])
            col_indices = tf.tile(cols, [block_size, 1])

            indices = tf.stack([
                tf.reshape(row_indices, [-1]),
                tf.reshape(col_indices, [-1]),
            ], axis=1)

            updates = tf.reshape(block, [-1])
            result = tf.tensor_scatter_nd_update(result, indices, updates)

        return result

    def _create_sensor_grid(
        self, grid_size: int | tf.Tensor
    ) -> tf.Tensor:
        """Create grid of sensor positions."""
        if isinstance(grid_size, tf.Tensor):
            n_side = grid_size
        else:
            if self.N_s != grid_size * grid_size:
                n_sqrt = tf.math.sqrt(tf.cast(self.N_s, tf.float32))
                n_side = tf.cast(n_sqrt, tf.int32)
                # If N_s is not a perfect square, use ceiling
                n_side = tf.cond(
                    tf.not_equal(n_side * n_side, self.N_s),
                    lambda: tf.cast(tf.math.ceil(n_sqrt), tf.int32),
                    lambda: n_side,
                )
            else:
                n_side = grid_size
        grid_points = tf.linspace(0.0, self.area_size, n_side)

        xx, yy = tf.meshgrid(grid_points, grid_points)
        positions = tf.stack([
            tf.reshape(xx, [-1])[: self.N_s],
            tf.reshape(yy, [-1])[: self.N_s],
        ], axis=1)
        return tf.cast(positions, self.dtype)

    def _create_default_initial_states(self) -> tf.Tensor:
        """Create default initial states for targets."""
        positions = [
            [12.0, 6.0],
            [32.0, 32.0],
            [20.0, 13.0],
            [15.0, 35.0],
        ]
        velocities = [
            [0.3, 0.2],
            [-0.25, -0.15],
            [-0.4, 0.3],
            [0.2, -0.35],
        ]

        while len(positions) < self.C:
            idx = len(positions)
            positions.append([
                self.area_size * 0.3 * (idx % 3 + 1),
                self.area_size * 0.3 * ((idx // 3) % 3 + 1),
            ])
            velocities.append([0.2, -0.2])

        positions = positions[: self.C]
        velocities = velocities[: self.C]

        state = []
        for pos, vel in zip(positions, velocities):
            state.extend([pos[0], pos[1], vel[0], vel[1]])

        return tf.constant(state, dtype=self.dtype)

    def _apply_boundary_conditions(self, x: tf.Tensor) -> tf.Tensor:
        """Apply reflecting boundary conditions to state."""
        if not self.enforce_boundaries:
            return x

        x_reshaped = tf.reshape(x, [-1, self.C, 4])
        positions = x_reshaped[..., :2]
        velocities = x_reshaped[..., 2:]

        below_lower = positions < 0.0
        positions = tf.where(below_lower, -positions, positions)
        velocities = tf.where(below_lower, -velocities, velocities)

        above_upper = positions > self.area_size
        positions = tf.where(
            above_upper,
            2.0 * self.area_size - positions,
            positions,
        )
        velocities = tf.where(above_upper, -velocities, velocities)

        x_bounded = tf.concat([positions, velocities], axis=-1)
        return tf.reshape(x_bounded, tf.shape(x))

    def motion_model(
        self, state: tf.Tensor, control: tf.Tensor | None = None
    ) -> tf.Tensor:
        """
        State transition: constant velocity model.

        Parameters
        ----------
        state : tf.Tensor
            State of shape (batch, 4*C) or (4*C).
        control : tf.Tensor, optional
            Not used; kept for interface compatibility.

        Returns
        -------
        next_state : tf.Tensor
            Next state, same shape as state.
        """
        state = tf.cast(state, self.dtype)

        if state.shape.rank == 1:
            state = state[tf.newaxis, :]

        batch_size = tf.shape(state)[0]
        x = tf.reshape(state, [batch_size, self.C, 4])

        positions = x[:, :, :2]
        velocities = x[:, :, 2:]

        positions_next = positions + velocities * self.dt
        velocities_next = velocities

        x_next = tf.concat([positions_next, velocities_next], axis=-1)
        x_next = tf.reshape(x_next, [batch_size, self.state_dim])
        x_next = self._apply_boundary_conditions(x_next)

        return x_next

    def measurement_model(
        self, state: tf.Tensor, landmarks: tf.Tensor | None = None
    ) -> tf.Tensor:
        """
        Acoustic measurement model: z = sum over targets of psi / (d^2 + d0).

        Parameters
        ----------
        state : tf.Tensor
            State of shape (batch, 4*C) or (4*C).
        landmarks : tf.Tensor, optional
            Sensor positions [N_s, 2]. If None, uses self.sensor_positions.

        Returns
        -------
        measurements : tf.Tensor
            Shape (batch, N_s) or (N_s).
        """
        state = tf.cast(state, self.dtype)

        sensors = (
            self.sensor_positions
            if landmarks is None
            else tf.cast(landmarks, self.dtype)
        )

        if state.shape.rank == 1:
            state = state[tf.newaxis, :]
            was_1d = True
        else:
            was_1d = False

        batch_size = tf.shape(state)[0]
        num_sensors = tf.shape(sensors)[0]

        x = tf.reshape(state, [batch_size, self.C, 4])
        positions = x[:, :, :2]

        pos_expanded = positions[:, :, tf.newaxis, :]
        sens_expanded = sensors[tf.newaxis, tf.newaxis, :, :]

        diff = pos_expanded - sens_expanded
        dist_sq = tf.reduce_sum(tf.square(diff), axis=-1)

        amplitudes = self.psi / (dist_sq + self.d0)
        z = tf.reduce_sum(amplitudes, axis=1)

        if was_1d:
            z = z[0]

        return z

    def motion_jacobian(
        self, state: tf.Tensor, control: tf.Tensor | None = None
    ) -> tf.Tensor:
        """
        Motion Jacobian: F = ∂f/∂x.

        Constant velocity: F is block diagonal with [I, dt*I; 0, I] per target.

        Parameters
        ----------
        state : tf.Tensor
            State of shape (batch, 4*C) or (4*C).
        control : tf.Tensor, optional
            Not used; kept for interface compatibility.

        Returns
        -------
        F : tf.Tensor
            Jacobian of shape (batch, 4*C, 4*C).
        """
        state = tf.cast(state, self.dtype)

        if state.shape.rank == 1:
            state = state[tf.newaxis, :]

        batch_size = tf.shape(state)[0]

        base_block = tf.constant([
            [1.0, 0.0, self.dt, 0.0],
            [0.0, 1.0, 0.0, self.dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=self.dtype)

        F_full = self._build_block_diagonal(base_block, self.C)
        F_batch = tf.tile(
            F_full[tf.newaxis, :, :],
            [batch_size, 1, 1],
        )
        return F_batch

    def measurement_jacobian(
        self, state: tf.Tensor, landmarks: tf.Tensor | None = None
    ) -> tf.Tensor:
        """
        Measurement Jacobian: H = ∂h/∂x.

        Parameters
        ----------
        state : tf.Tensor
            State of shape (batch, 4*C) or (4*C).
        landmarks : tf.Tensor, optional
            Sensor positions [N_s, 2]. If None, uses self.sensor_positions.

        Returns
        -------
        H : tf.Tensor
            Jacobian of shape (batch, N_s, 4*C) or (N_s, 4*C).
        """
        state = tf.cast(state, self.dtype)

        sensors = (
            self.sensor_positions
            if landmarks is None
            else tf.cast(landmarks, self.dtype)
        )

        if state.shape.rank == 1:
            state = state[tf.newaxis, :]
            was_1d = True
        else:
            was_1d = False

        batch_size = tf.shape(state)[0]
        num_sensors = tf.shape(sensors)[0]

        x = tf.reshape(state, [batch_size, self.C, 4])
        positions = x[:, :, :2]

        pos_expanded = positions[:, :, tf.newaxis, :]
        sens_expanded = sensors[tf.newaxis, tf.newaxis, :, :]

        diff = pos_expanded - sens_expanded
        dist_sq = (
            tf.reduce_sum(tf.square(diff), axis=-1) + self.d0
        )
        coeff = -2.0 * self.psi / (dist_sq ** 2)

        dx = diff[:, :, :, 0]
        dy = diff[:, :, :, 1]

        dh_dx = coeff * dx
        dh_dy = coeff * dy

        dh_dx_reshaped = tf.transpose(dh_dx, [0, 2, 1])
        dh_dy_reshaped = tf.transpose(dh_dy, [0, 2, 1])

        H_list = []
        for c in range(self.C):
            H_c = tf.stack([
                dh_dx_reshaped[:, :, c],
                dh_dy_reshaped[:, :, c],
                tf.zeros([batch_size, num_sensors], dtype=self.dtype),
                tf.zeros([batch_size, num_sensors], dtype=self.dtype),
            ], axis=-1)
            H_list.append(H_c)

        H = tf.concat(H_list, axis=-1)

        if was_1d:
            H = H[0]

        return H

    def full_measurement_cov(
        self, num_sensors: int | tf.Tensor | None = None
    ) -> tf.Tensor:
        """
        Full measurement covariance matrix.

        For acoustic model, R is already (N_s, N_s). Returns R or a diagonal
        matrix of matching size.

        Parameters
        ----------
        num_sensors : int or tf.Tensor, optional
            Number of sensors. If None, uses self.N_s.

        Returns
        -------
        R_full : tf.Tensor
            Shape (M, M) with M = num_sensors or N_s.
        """
        if num_sensors is None:
            return self.R

        static_n = (
            tf.get_static_value(num_sensors)
            if isinstance(num_sensors, tf.Tensor)
            else num_sensors
        )
        if static_n is not None:
            M = int(static_n)
        else:
            M = self.N_s

        if M == 0:
            return tf.zeros([0, 0], dtype=self.R.dtype)

        if tf.shape(self.R)[0] == M:
            return self.R
        return tf.eye(M, dtype=self.R.dtype) * (self.sigma_w ** 2)

    def sample_process_noise(
        self,
        shape: int | tuple[int, ...] = (),
        use_gen: bool = True,
    ) -> tf.Tensor:
        """Sample process noise for each target."""
        V = self.V_gen_single if use_gen else self.V_filter_single

        try:
            L = tf.linalg.cholesky(V)
            mvn = tfd.MultivariateNormalTriL(
                loc=tf.zeros(4, dtype=self.dtype),
                scale_tril=L,
            )
            if shape == () or (isinstance(shape, tuple) and len(shape) == 0):
                samples = mvn.sample([self.C])
            else:
                full_shape = (
                    (shape, self.C)
                    if isinstance(shape, int)
                    else shape + (self.C,)
                )
                samples = mvn.sample(full_shape)
            return samples
        except Exception:
            std = tf.sqrt(tf.linalg.diag_part(V))
            if shape == () or (isinstance(shape, tuple) and len(shape) == 0):
                samples = (
                    tf.random.normal([self.C, 4], dtype=self.dtype)
                    * std[tf.newaxis, :]
                )
            else:
                full_shape = (
                    (shape, self.C, 4)
                    if isinstance(shape, int)
                    else shape + (self.C, 4)
                )
                samples = (
                    tf.random.normal(full_shape, dtype=self.dtype)
                    * std[tf.newaxis, tf.newaxis, :]
                )
            return samples

    def sample_initial_state(
        self,
        num_samples: int = 1,
        pos_std: float = 2.0,
        vel_std: float = 0.1,
        seed: int | None = None,
    ) -> tf.Tensor:
        """Sample initial state from uncertain distribution."""
        if seed is not None:
            tf.random.set_seed(seed)

        std = tf.constant(
            [pos_std, pos_std, vel_std, vel_std],
            dtype=self.dtype,
        )
        std_full = tf.tile(std[tf.newaxis, :], [self.C, 1])
        std_full = tf.reshape(std_full, [-1])

        noise = (
            tf.random.normal(
                [num_samples, self.state_dim],
                dtype=self.dtype,
            )
            * std_full[tf.newaxis, :]
        )
        x0_samples = self.x0_true[tf.newaxis, :] + noise

        x0_reshaped = tf.reshape(x0_samples, [num_samples, self.C, 4])
        positions = tf.clip_by_value(
            x0_reshaped[:, :, :2],
            0.0,
            self.area_size,
        )
        x0_reshaped = tf.concat(
            [positions, x0_reshaped[:, :, 2:]],
            axis=-1,
        )
        x0_samples = tf.reshape(x0_reshaped, [num_samples, self.state_dim])

        if num_samples == 1:
            return x0_samples[0]
        return x0_samples
