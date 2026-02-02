#!/usr/bin/env python3
"""
Enhanced Filter Performance Analysis with Fixed Recommendation Algorithm.

Key Improvements:
1. Rank-based recommendations (not just win counting)
2. Proper hyperparameter tuning for PFF_MATRIX vs PFF_SCALAR
3. Correlation-stressing scenarios
4. Matrix vs Scalar detailed analysis
5. Fair computational budget comparison
6. Consistency metrics in addition to wins
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.ssm_multi_target_acoustic import MultiTargetAcousticSSM
from src.filters.ekf import ExtendedKalmanFilter
from src.filters.ukf import UnscentedKalmanFilter
from src.filters.particle_filter import ParticleFilter
from src.filters.pfpf_filter import PFPFLEDHFilter, PFPFEDHFilter
from src.filters.ledh import LEDH
from src.filters.edh import EDH
from src.filters.pff_kernel import ScalarPFF, MatrixPFF
from src.metrics.accuracy import compute_rmse as _compute_rmse_shared
from src.metrics.particle_filter_metrics import compute_effective_sample_size
from src.utils.linalg import compute_condition_number as _compute_cond_shared

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings(
    'ignore',
    message=".*'mode' parameter is deprecated.*",
    category=DeprecationWarning,
)

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    try:
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        pass
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 10
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# ============================================================================
# Enhanced Stability Diagnostics
# ============================================================================

@dataclass
class StabilityMetrics:
    """Container for stability diagnostic metrics."""
    flow_magnitude_mean: Optional[float] = None
    flow_magnitude_std: Optional[float] = None
    flow_magnitude_max: Optional[float] = None
    motion_jacobian_cond_mean: Optional[float] = None
    motion_jacobian_cond_max: Optional[float] = None
    measurement_jacobian_cond_mean: Optional[float] = None
    measurement_jacobian_cond_max: Optional[float] = None
    covariance_cond_mean: Optional[float] = None
    covariance_cond_max: Optional[float] = None
    particle_degeneracy_rate: Optional[float] = None
    ess_mean: Optional[float] = None
    ess_min: Optional[float] = None
    ess_std: Optional[float] = None
    numerical_instability_count: int = 0
    filter_divergence: bool = False
    flow_magnitude_series: Optional[List[float]] = None
    motion_jacobian_series: Optional[List[float]] = None
    measurement_jacobian_series: Optional[List[float]] = None
    covariance_cond_series: Optional[List[float]] = None
    ess_series: Optional[List[float]] = None
    rmse_series: Optional[List[float]] = None


def compute_condition_number(matrix: tf.Tensor) -> float:
    """Compute condition number of a matrix using shared utility."""
    return _compute_cond_shared(matrix)


def compute_flow_magnitude(particles_before: tf.Tensor, particles_after: tf.Tensor) -> Dict[str, float]:
    """Compute magnitude of particle flow."""
    try:
        flow = particles_after - particles_before
        magnitude = tf.norm(flow, axis=-1)
        return {
            'mean': float(tf.reduce_mean(magnitude).numpy()),
            'std': float(tf.math.reduce_std(magnitude).numpy()),
            'max': float(tf.reduce_max(magnitude).numpy()),
        }
    except Exception:
        return {'mean': float('nan'), 'std': float('nan'), 'max': float('nan')}


def compute_ess(weights: tf.Tensor) -> float:
    """Compute effective sample size using shared utility."""
    try:
        ess = compute_effective_sample_size(weights)
        return float(ess.numpy()) if hasattr(ess, 'numpy') else float(ess)
    except Exception:
        return float('nan')


# ============================================================================
# Filter Wrapper with Diagnostics
# ============================================================================

class FilterWithDiagnostics:
    """Wrapper that adds diagnostic tracking to any filter."""

    def __init__(self, filter_obj, ssm, track_flow: bool = True, verbose: bool = False):
        self.filter = filter_obj
        self.ssm = ssm
        self.track_flow = track_flow
        self.verbose = verbose
        self.flow_magnitudes: List[Dict[str, float]] = []
        self.motion_jacobian_conds: List[float] = []
        self.measurement_jacobian_conds: List[float] = []
        self.covariance_conds: List[float] = []
        self.ess_history: List[float] = []
        self.numerical_instabilities = 0
        self.flow_magnitude_series: List[float] = []
        self.motion_jacobian_series: List[float] = []
        self.measurement_jacobian_series: List[float] = []
        self.covariance_cond_series: List[float] = []
        self.ess_series: List[float] = []
        self.prev_particles: Optional[tf.Tensor] = None

    def _safe_jacobian_cond(self, jacobian_fn, state, name: str = "") -> float:
        """Safely compute Jacobian condition number."""
        try:
            state_tf = tf.constant(state, dtype=self.ssm.dtype) if not isinstance(state, tf.Tensor) else tf.cast(state, self.ssm.dtype)
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(state_tf)
                output = jacobian_fn(state_tf)
            jac = tape.jacobian(output, state_tf)
            if jac is None:
                return float('nan')
            if len(jac.shape) > 2:
                jac = jac[0] if jac.shape[0] == 1 else tf.reshape(jac, [jac.shape[-2], jac.shape[-1]])
            cond = compute_condition_number(jac)
            return cond
        except Exception as e:
            if self.verbose:
                print(f"Jacobian {name} computation failed: {e}")
            self.numerical_instabilities += 1
            return float('nan')

    def predict(self, control=None):
        """Predict step with diagnostics."""
        if self.track_flow and hasattr(self.filter, 'particles'):
            try:
                self.prev_particles = tf.identity(self.filter.particles)
            except Exception:
                self.prev_particles = None

        try:
            if hasattr(self.filter, 'state'):
                current_state = self.filter.state.numpy() if hasattr(self.filter.state, 'numpy') else self.filter.state
            elif hasattr(self.filter, 'x_hat'):
                current_state = self.filter.x_hat.numpy() if hasattr(self.filter.x_hat, 'numpy') else self.filter.x_hat
            else:
                current_state = None

            if current_state is not None:
                cond = self._safe_jacobian_cond(self.ssm.motion_model, current_state, "motion")
                if math.isfinite(cond):
                    self.motion_jacobian_conds.append(cond)
                    self.motion_jacobian_series.append(cond)
                else:
                    self.motion_jacobian_series.append(float('nan'))
        except Exception:
            self.motion_jacobian_series.append(float('nan'))

        try:
            if control is not None:
                self.filter.predict(control)
            else:
                try:
                    self.filter.predict(tf.zeros([self.ssm.state_dim], dtype=tf.float32))
                except TypeError:
                    self.filter.predict()
        except Exception as e:
            self.numerical_instabilities += 1
            if self.verbose:
                print(f"Prediction failed: {e}")
            raise

    def update(self, observation, sensor_positions=None):
        """Update step with diagnostics."""
        try:
            if hasattr(self.filter, 'state'):
                current_state = self.filter.state.numpy() if hasattr(self.filter.state, 'numpy') else self.filter.state
            elif hasattr(self.filter, 'x_hat'):
                current_state = self.filter.x_hat.numpy() if hasattr(self.filter.x_hat, 'numpy') else self.filter.x_hat
            else:
                current_state = None

            if current_state is not None:
                cond = self._safe_jacobian_cond(self.ssm.measurement_model, current_state, "measurement")
                if math.isfinite(cond):
                    self.measurement_jacobian_conds.append(cond)
                    self.measurement_jacobian_series.append(cond)
                else:
                    self.measurement_jacobian_series.append(float('nan'))
        except Exception:
            self.measurement_jacobian_series.append(float('nan'))

        particles_before = None
        if self.track_flow and hasattr(self.filter, 'particles'):
            try:
                particles_before = tf.identity(self.filter.particles)
            except Exception:
                pass

        try:
            if sensor_positions is not None:
                self.filter.update(observation, sensor_positions)
            else:
                self.filter.update(observation)
        except Exception as e:
            self.numerical_instabilities += 1
            if self.verbose:
                print(f"Update failed: {e}")
            raise

        if self.track_flow and hasattr(self.filter, 'particles') and particles_before is not None:
            try:
                particles_after = self.filter.particles
                flow_stats = compute_flow_magnitude(particles_before, particles_after)
                self.flow_magnitudes.append(flow_stats)
                self.flow_magnitude_series.append(flow_stats['mean'])
            except Exception:
                self.flow_magnitude_series.append(float('nan'))
        else:
            self.flow_magnitude_series.append(float('nan'))

        if hasattr(self.filter, 'weights'):
            try:
                ess = compute_ess(self.filter.weights)
                if math.isfinite(ess):
                    self.ess_history.append(ess)
                    self.ess_series.append(ess)
                else:
                    self.ess_series.append(float('nan'))
            except Exception:
                self.ess_series.append(float('nan'))
        else:
            self.ess_series.append(float('nan'))

        try:
            if hasattr(self.filter, 'P'):
                P = self.filter.P
                if hasattr(P, 'numpy'):
                    P = P.numpy()
                P_tf = tf.constant(P, dtype=self.ssm.dtype)
                cond = compute_condition_number(P_tf)
                if math.isfinite(cond):
                    self.covariance_conds.append(cond)
                    self.covariance_cond_series.append(cond)
                else:
                    self.covariance_cond_series.append(float('nan'))
            else:
                self.covariance_cond_series.append(float('nan'))
        except Exception:
            self.covariance_cond_series.append(float('nan'))

    def get_stability_metrics(self, rmse_series=None) -> StabilityMetrics:
        """Compute and return stability metrics."""
        metrics = StabilityMetrics()

        if self.flow_magnitudes:
            valid_flows = [f for f in self.flow_magnitudes if math.isfinite(f['mean'])]
            if valid_flows:
                all_means = [f['mean'] for f in valid_flows]
                all_maxs = [f['max'] for f in valid_flows]
                metrics.flow_magnitude_mean = sum(all_means) / len(all_means)
                metrics.flow_magnitude_std = (sum((x - metrics.flow_magnitude_mean) ** 2 for x in all_means) / len(all_means)) ** 0.5
                metrics.flow_magnitude_max = max(all_maxs)

        if self.motion_jacobian_conds:
            valid_conds = [c for c in self.motion_jacobian_conds if math.isfinite(c)]
            if valid_conds:
                metrics.motion_jacobian_cond_mean = sum(valid_conds) / len(valid_conds)
                metrics.motion_jacobian_cond_max = max(valid_conds)

        if self.measurement_jacobian_conds:
            valid_conds = [c for c in self.measurement_jacobian_conds if math.isfinite(c)]
            if valid_conds:
                metrics.measurement_jacobian_cond_mean = sum(valid_conds) / len(valid_conds)
                metrics.measurement_jacobian_cond_max = max(valid_conds)

        if self.covariance_conds:
            valid_conds = [c for c in self.covariance_conds if math.isfinite(c)]
            if valid_conds:
                metrics.covariance_cond_mean = sum(valid_conds) / len(valid_conds)
                metrics.covariance_cond_max = max(valid_conds)

        if self.ess_history:
            ess_list = [e for e in self.ess_history if math.isfinite(e)]
            if ess_list:
                metrics.ess_mean = sum(ess_list) / len(ess_list)
                metrics.ess_min = min(ess_list)
                metrics.ess_std = (sum((x - metrics.ess_mean) ** 2 for x in ess_list) / len(ess_list)) ** 0.5
                if hasattr(self.filter, 'num_particles'):
                    n_particles = self.filter.num_particles
                    low_ess_threshold = n_particles * 0.5
                    metrics.particle_degeneracy_rate = sum(1 for e in ess_list if e < low_ess_threshold) / len(ess_list)

        metrics.numerical_instability_count = self.numerical_instabilities
        metrics.flow_magnitude_series = self.flow_magnitude_series
        metrics.motion_jacobian_series = self.motion_jacobian_series
        metrics.measurement_jacobian_series = self.measurement_jacobian_series
        metrics.covariance_cond_series = self.covariance_cond_series
        metrics.ess_series = self.ess_series
        metrics.rmse_series = rmse_series if rmse_series is not None else []

        return metrics

    def __getattr__(self, name):
        """Delegate attribute access to wrapped filter."""
        return getattr(self.filter, name)


# ============================================================================
# Scenario Definitions
# ============================================================================

@dataclass
class ScenarioConfig:
    """Configuration for a test scenario."""
    name: str
    num_targets: int
    num_sensors: int
    area_size: float
    sensor_grid_size: int
    process_noise_scale: float
    obs_noise_scale: float
    observation_rate: float
    psi: float
    d0: float
    dt: float
    n_steps: int = 20
    n_runs: int = 2
    description: str = ""
    measurement_correlation: float = 0.0
    state_correlation: bool = False


def create_scenario_configs() -> List[ScenarioConfig]:
    """Create test scenarios including correlation-stressing scenarios."""
    scenarios: List[ScenarioConfig] = []

    scenarios.append(ScenarioConfig(
        name="baseline",
        num_targets=2, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Baseline: 2 targets, 16 sensors, good conditions"
    ))

    scenarios.append(ScenarioConfig(
        name="nonlin_mild",
        num_targets=2, num_sensors=16, area_size=50.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=5.0, d0=0.2, dt=1.0,
        description="Mild nonlinearity: sparse targets, weak coupling"
    ))
    scenarios.append(ScenarioConfig(
        name="nonlin_moderate",
        num_targets=4, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Moderate nonlinearity: 4 targets, standard acoustic model"
    ))
    scenarios.append(ScenarioConfig(
        name="nonlin_severe",
        num_targets=6, num_sensors=16, area_size=25.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=20.0, d0=0.05, dt=1.0,
        description="Severe nonlinearity: dense targets, strong coupling"
    ))

    for rate in [0.7, 0.5, 0.3]:
        scenarios.append(ScenarioConfig(
            name=f"sparse_obs_{int(rate*100)}pct",
            num_targets=3, num_sensors=16, area_size=30.0, sensor_grid_size=4,
            process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=rate,
            psi=10.0, d0=0.1, dt=1.0,
            description=f"Sparse observations: {int(rate*100)}% availability"
        ))

    scenarios.append(ScenarioConfig(
        name="sparse_sensors_9",
        num_targets=3, num_sensors=9, area_size=30.0, sensor_grid_size=3,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Sparse sensors: 3x3 grid"
    ))
    scenarios.append(ScenarioConfig(
        name="sparse_sensors_4",
        num_targets=2, num_sensors=4, area_size=30.0, sensor_grid_size=2,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Very sparse sensors: 2x2 grid"
    ))

    for n_targets in [1, 3, 5, 8]:
        scenarios.append(ScenarioConfig(
            name=f"dimension_{n_targets}targets",
            num_targets=n_targets, num_sensors=16, area_size=30.0, sensor_grid_size=4,
            process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
            psi=10.0, d0=0.1, dt=1.0,
            description=f"Dimension test: {n_targets} targets ({n_targets*4}D state)"
        ))

    scenarios.append(ScenarioConfig(
        name="illcond_high_process_noise",
        num_targets=3, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=3.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Ill-conditioned: high process noise"
    ))
    scenarios.append(ScenarioConfig(
        name="illcond_high_obs_noise",
        num_targets=3, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.5, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Ill-conditioned: high observation noise"
    ))
    scenarios.append(ScenarioConfig(
        name="illcond_poor_geometry",
        num_targets=4, num_sensors=16, area_size=15.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Ill-conditioned: poor sensor geometry"
    ))
    scenarios.append(ScenarioConfig(
        name="illcond_noise_ratio",
        num_targets=3, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=2.5, obs_noise_scale=0.3, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0,
        description="Ill-conditioned: high noise ratio"
    ))

    scenarios.append(ScenarioConfig(
        name="corr_measurement_moderate",
        num_targets=3, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0, measurement_correlation=0.3,
        description="Correlated measurements: 30% sensor correlation"
    ))
    scenarios.append(ScenarioConfig(
        name="corr_measurement_high",
        num_targets=3, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.15, observation_rate=1.0,
        psi=10.0, d0=0.1, dt=1.0, measurement_correlation=0.6,
        description="Correlated measurements: 60% sensor correlation"
    ))
    scenarios.append(ScenarioConfig(
        name="corr_state_formation",
        num_targets=4, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=1.0, obs_noise_scale=0.1, observation_rate=1.0,
        psi=15.0, d0=0.08, dt=1.0, state_correlation=True,
        description="State correlation: formation flight"
    ))

    scenarios.append(ScenarioConfig(
        name="stress_nonlin_sparse",
        num_targets=6, num_sensors=9, area_size=25.0, sensor_grid_size=3,
        process_noise_scale=1.5, obs_noise_scale=0.2, observation_rate=0.5,
        psi=15.0, d0=0.05, dt=1.0,
        description="Stress: severe nonlinearity + sparse observations"
    ))
    scenarios.append(ScenarioConfig(
        name="stress_highdim_illcond",
        num_targets=8, num_sensors=16, area_size=30.0, sensor_grid_size=4,
        process_noise_scale=2.0, obs_noise_scale=0.3, observation_rate=0.7,
        psi=10.0, d0=0.1, dt=1.0,
        description="Stress: high dimension + ill-conditioning"
    ))
    scenarios.append(ScenarioConfig(
        name="stress_extreme",
        num_targets=6, num_sensors=9, area_size=20.0, sensor_grid_size=3,
        process_noise_scale=2.5, obs_noise_scale=0.4, observation_rate=0.4,
        psi=20.0, d0=0.05, dt=1.0,
        description="Stress: extreme - all challenges combined"
    ))
    scenarios.append(ScenarioConfig(
        name="stress_corr_extreme",
        num_targets=5, num_sensors=12, area_size=25.0, sensor_grid_size=4,
        process_noise_scale=2.0, obs_noise_scale=0.3, observation_rate=0.5,
        psi=18.0, d0=0.06, dt=1.0, measurement_correlation=0.4,
        description="Stress: extreme with correlated measurements"
    ))

    return scenarios


# ============================================================================
# Experiment Runner
# ============================================================================

@dataclass
class ExperimentResult:
    """Results from running a filter on a scenario."""
    filter_name: str
    scenario_name: str
    rmse_mean: float
    rmse_std: float
    exec_time_mean: float
    exec_time_std: float
    success_rate: float
    stability_metrics: Dict
    rmse_per_run: List[float]
    time_series_data: Optional[Dict] = None


def generate_trajectory(ssm, n_steps: int, observation_rate: float = 1.0,
                        measurement_correlation: float = 0.0) -> Tuple[tf.Tensor, tf.Tensor, List[bool]]:
    """Generate ground truth trajectory and observations. Returns (states, measurements, obs_available)."""
    x0 = ssm.sample_initial_state(num_samples=1)
    x0 = tf.reshape(x0, [-1])

    states_list: List[tf.Tensor] = [x0]
    measurements_list: List[tf.Tensor] = []
    obs_available: List[bool] = []

    if measurement_correlation > 0:
        N_s = ssm.N_s
        pos = ssm.sensor_positions
        diff = pos[:, None, :] - pos[None, :, :]
        dist = tf.sqrt(tf.reduce_sum(diff ** 2, axis=-1) + 1e-12)
        corr_off = measurement_correlation * tf.exp(-dist / 10.0)
        corr_matrix = tf.linalg.set_diag(corr_off, tf.ones(N_s, dtype=tf.float32))
        L = tf.linalg.cholesky(corr_matrix + 1e-6 * tf.eye(N_s, dtype=tf.float32))
    else:
        L = None

    for t in range(n_steps):
        state_t = states_list[-1]
        state_t_batch = state_t[tf.newaxis, :]
        state_next = ssm.motion_model(state_t_batch)
        noise = ssm.sample_process_noise((), use_gen=True)
        noise_flat = tf.reshape(noise, [-1])
        state_next = state_next[0] + noise_flat
        states_list.append(state_next)

        should_observe = float(tf.random.uniform([]).numpy()) < observation_rate
        obs_available.append(should_observe)

        if should_observe:
            z = ssm.measurement_model(state_next[tf.newaxis, :])
            z = z[0]
            if L is not None:
                base_noise = tf.random.normal([ssm.N_s], dtype=ssm.dtype)
                corr_noise = tf.linalg.matvec(L, base_noise) * ssm.sigma_w
                meas_t = z + corr_noise
            else:
                noise_meas = tf.random.normal(tf.shape(z), mean=0.0, stddev=ssm.sigma_w, dtype=ssm.dtype)
                meas_t = z + noise_meas
            measurements_list.append(meas_t)
        else:
            measurements_list.append(tf.zeros(ssm.N_s, dtype=ssm.dtype))

    states = tf.stack(states_list)
    measurements = tf.stack(measurements_list)
    return states, measurements, obs_available


def compute_rmse(estimates: tf.Tensor, true_states: tf.Tensor) -> float:
    """Root Mean Square Error using shared utility."""
    return _compute_rmse_shared(estimates, true_states)


def run_filter_on_trajectory(filter_obj, ssm, states: tf.Tensor, measurements: tf.Tensor,
                             obs_available: List[bool], sensor_positions: tf.Tensor,
                             track_diagnostics: bool = True, verbose: bool = False,
                             show_step_progress: bool = False) -> Tuple[tf.Tensor, float, float, StabilityMetrics]:
    """Run filter on a single trajectory with diagnostics. Returns (estimates, rmse, exec_time, stability_metrics)."""
    n_steps = int(measurements.shape[0])

    if track_diagnostics:
        filter_diag = FilterWithDiagnostics(filter_obj, ssm, track_flow=True, verbose=verbose)
    else:
        filter_diag = filter_obj

    if hasattr(filter_obj, 'state'):
        s0 = filter_obj.state
    elif hasattr(filter_obj, 'x_hat'):
        s0 = filter_obj.x_hat
    else:
        raise ValueError("Filter has no state attribute")

    s0 = tf.reshape(tf.cast(s0, tf.float32), [-1])
    state_dim = int(s0.shape[0])

    estimates_list: List[tf.Tensor] = [s0]
    rmse_per_step: List[float] = []

    start_time = time.time()

    step_iter = range(n_steps)
    if show_step_progress and HAS_TQDM and n_steps > 10:
        step_iter = tqdm(range(n_steps), desc="      Steps", leave=False, unit="step")

    for t in step_iter:
        try:
            filter_diag.predict()
        except Exception as e:
            if verbose:
                print(f"Prediction failed at step {t}: {e}")
            raise

        if obs_available[t]:
            try:
                filter_diag.update(measurements[t], sensor_positions)
            except Exception as e:
                if verbose:
                    print(f"Update failed at step {t}: {e}")
                raise

        if hasattr(filter_obj, 'state'):
            st = filter_obj.state
        elif hasattr(filter_obj, 'x_hat'):
            st = filter_obj.x_hat
        else:
            st = states[t + 1]

        st = tf.reshape(tf.cast(st, tf.float32), [-1])
        estimates_list.append(st)
        step_rmse = float(tf.sqrt(tf.reduce_mean((st - states[t + 1]) ** 2)).numpy())
        rmse_per_step.append(step_rmse)

    exec_time = time.time() - start_time
    estimates = tf.stack(estimates_list)
    rmse = compute_rmse(estimates, states)

    if track_diagnostics:
        stability_metrics = filter_diag.get_stability_metrics(rmse_series=rmse_per_step)
        stability_metrics.filter_divergence = (rmse > 100.0)
    else:
        stability_metrics = StabilityMetrics()

    return estimates, rmse, exec_time, stability_metrics


def create_filter(filter_name: str, ssm, x0: tf.Tensor, P0: tf.Tensor, args):
    """Factory function to create filters with proper hyperparameter tuning."""
    x0 = tf.cast(x0, tf.float32)
    P0 = tf.cast(P0, tf.float32)
    state_dim = int(x0.shape[0])
    if P0.shape.rank == 0 or (P0.shape[0] != state_dim):
        P0 = tf.eye(state_dim, dtype=tf.float32) * 1.0

    if filter_name == 'ekf':
        return ExtendedKalmanFilter(ssm, x0, P0)
    elif filter_name == 'ukf':
        return UnscentedKalmanFilter(ssm, x0, P0, alpha=args.ukf_alpha, beta=args.ukf_beta, kappa=0.0)
    elif filter_name == 'pf':
        return ParticleFilter(ssm, x0, P0, num_particles=args.pf_particles)
    elif filter_name == 'pfpf_ledh':
        return PFPFLEDHFilter(ssm, x0, P0, num_particles=args.n_particles, n_lambda=args.n_lambda,
                              filter_type='ekf', ukf_alpha=args.ukf_alpha, ukf_beta=args.ukf_beta, show_progress=False)
    elif filter_name == 'pfpf_edh':
        return PFPFEDHFilter(ssm, x0, P0, num_particles=args.n_particles, n_lambda=args.n_lambda,
                             filter_type='ekf', ukf_alpha=args.ukf_alpha, ukf_beta=args.ukf_beta, show_progress=False)
    elif filter_name == 'ledh':
        return LEDH(ssm, x0, P0, num_particles=args.n_particles, n_lambda=args.n_lambda, filter_type='ekf',
                    ukf_alpha=args.ukf_alpha, show_progress=False, redraw_particles=False)
    elif filter_name == 'edh':
        return EDH(ssm, x0, P0, num_particles=args.n_particles, n_lambda=args.n_lambda, filter_type='ekf',
                   ukf_alpha=args.ukf_alpha, show_progress=False, redraw_particles=args.edh_redraw)
    elif filter_name == 'pff_scalar':
        return ScalarPFF(ssm, x0, P0, num_particles=args.n_particles, step_size=args.pff_step_size,
                         localization_radius=args.pff_localization_radius, max_steps=args.pff_max_steps)
    elif filter_name == 'pff_matrix':
        return MatrixPFF(ssm, x0, P0, num_particles=args.n_particles,
                         step_size=args.pff_step_size * args.pff_matrix_step_multiplier,
                         localization_radius=args.pff_localization_radius * args.pff_matrix_loc_multiplier,
                         max_steps=int(args.pff_max_steps * args.pff_matrix_steps_multiplier))
    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def _run_scenario_worker(scenario_dict: dict, filter_names: List[str], args_dict: dict, scenario_idx: int) -> List[ExperimentResult]:
    """Worker function for parallel execution."""
    import os
    import warnings
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tf.get_logger().setLevel('ERROR')
    warnings.filterwarnings('ignore', category=UserWarning)
    seed = args_dict['seed'] + scenario_idx * 1000
    tf.random.set_seed(seed)

    scenario = ScenarioConfig(**scenario_dict)

    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    args = Args(args_dict)
    args.show_progress = False
    args.verbose = False
    args.seed = seed

    from src.experiments.exp_filters_comparison_diagnostics import run_scenario_experiment
    return run_scenario_experiment(scenario, filter_names, args)


def run_scenario_experiment(scenario: ScenarioConfig, filter_names: List[str], args) -> List[ExperimentResult]:
    """Run all filters on a scenario multiple times."""
    ssm_gen = MultiTargetAcousticSSM(
        num_targets=scenario.num_targets, num_sensors=scenario.num_sensors,
        area_size=scenario.area_size, dt=scenario.dt, psi=scenario.psi, d0=scenario.d0,
        sigma_w=scenario.obs_noise_scale, process_noise_scale=1.0, sensor_grid_size=scenario.sensor_grid_size
    )
    ssm_filter = MultiTargetAcousticSSM(
        num_targets=scenario.num_targets, num_sensors=scenario.num_sensors,
        area_size=scenario.area_size, dt=scenario.dt, psi=scenario.psi, d0=scenario.d0,
        sigma_w=scenario.obs_noise_scale, process_noise_scale=scenario.process_noise_scale,
        sensor_grid_size=scenario.sensor_grid_size
    )

    sensor_positions = ssm_gen.sensor_positions
    results: List[ExperimentResult] = []

    if HAS_TQDM:
        print(f"\nScenario: {scenario.name}")
        print(f"  Description: {scenario.description}")
        print(f"  State: {scenario.num_targets * 4}D ({scenario.num_targets} targets), "
              f"Obs: {scenario.num_sensors} sensors ({scenario.observation_rate*100:.0f}% avail)")
        if scenario.measurement_correlation > 0:
            print(f"  Measurement correlation: {scenario.measurement_correlation:.1%}")

    filter_iter = tqdm(filter_names, desc="  Filters", leave=False, unit="filter") if HAS_TQDM else filter_names

    for filter_name in filter_iter:
        if HAS_TQDM and hasattr(filter_iter, 'set_description'):
            filter_iter.set_description(f"  Filter: {filter_name.upper()}")
        else:
            print(f"\n  Filter: {filter_name.upper()}")

        rmse_list: List[float] = []
        time_list: List[float] = []
        stability_metrics_list: List[StabilityMetrics] = []
        n_success = 0
        time_series_data = None

        iterator = tqdm(range(scenario.n_runs), desc="    Runs", leave=False, unit="run") if HAS_TQDM else range(scenario.n_runs)

        for run_idx in iterator:
            seed = args.seed + run_idx
            tf.random.set_seed(seed)

            try:
                states, measurements, obs_available = generate_trajectory(
                    ssm_gen, scenario.n_steps, scenario.observation_rate,
                    measurement_correlation=scenario.measurement_correlation
                )

                x0 = states[0]
                P0 = tf.eye(ssm_filter.state_dim, dtype=tf.float32) * 1.0
                filter_obj = create_filter(filter_name, ssm_filter, x0, P0, args)

                estimates, rmse, exec_time, stability_metrics = run_filter_on_trajectory(
                    filter_obj, ssm_filter, states, measurements, obs_available,
                    sensor_positions, track_diagnostics=True, verbose=args.verbose,
                    show_step_progress=args.show_progress
                )

                rmse_list.append(float(rmse))
                time_list.append(exec_time)
                stability_metrics_list.append(stability_metrics)
                n_success += 1

                if time_series_data is None and n_success == 1:
                    time_series_data = {
                        'rmse_series': stability_metrics.rmse_series,
                        'flow_magnitude_series': stability_metrics.flow_magnitude_series,
                        'motion_jacobian_series': stability_metrics.motion_jacobian_series,
                        'measurement_jacobian_series': stability_metrics.measurement_jacobian_series,
                        'covariance_cond_series': stability_metrics.covariance_cond_series,
                        'ess_series': stability_metrics.ess_series
                    }
            except Exception as e:
                if args.verbose:
                    print(f"    Run {run_idx} failed: {e}")
                continue

        if rmse_list:
            aggregated_stability: Dict = {}
            for key in asdict(stability_metrics_list[0]).keys():
                if key.endswith('_series'):
                    continue
                values = [getattr(m, key) for m in stability_metrics_list if getattr(m, key) is not None]
                if values:
                    if isinstance(values[0], (int, float)):
                        aggregated_stability[key + '_mean'] = sum(values) / len(values)
                        aggregated_stability[key + '_std'] = (sum((x - sum(values)/len(values))**2 for x in values) / len(values)) ** 0.5
                    elif isinstance(values[0], bool):
                        aggregated_stability[key + '_rate'] = sum(float(v) for v in values) / len(values)

            result = ExperimentResult(
                filter_name=filter_name,
                scenario_name=scenario.name,
                rmse_mean=sum(rmse_list) / len(rmse_list),
                rmse_std=(sum((x - sum(rmse_list)/len(rmse_list))**2 for x in rmse_list) / len(rmse_list)) ** 0.5,
                exec_time_mean=sum(time_list) / len(time_list),
                exec_time_std=(sum((x - sum(time_list)/len(time_list))**2 for x in time_list) / len(time_list)) ** 0.5,
                success_rate=float(n_success) / scenario.n_runs,
                stability_metrics=aggregated_stability,
                rmse_per_run=rmse_list,
                time_series_data=time_series_data
            )
            results.append(result)

            if not HAS_TQDM or args.verbose:
                print(f"    ✓ RMSE: {result.rmse_mean:.3f} ± {result.rmse_std:.3f}")
                print(f"      Time: {result.exec_time_mean:.3f}s, Success: {result.success_rate:.0%}")
        else:
            if not HAS_TQDM or args.verbose:
                print(f"    ✗ Filter failed on all runs")

    return results


# ============================================================================
# Analysis: rank-based recommendations and matrix vs scalar
# ============================================================================

def diagnose_failure_cause(result: ExperimentResult) -> str:
    """Diagnose the primary cause of filter failure."""
    metrics = result.stability_metrics
    MOTION_JAC_THRESHOLD = 1e5
    MEAS_JAC_THRESHOLD = 1e5
    COV_COND_THRESHOLD = 1e7
    ESS_THRESHOLD = 50
    FLOW_THRESHOLD = 10.0
    causes: List[Tuple[str, float]] = []

    if metrics.get('motion_jacobian_cond_max_mean', 0) > MOTION_JAC_THRESHOLD:
        causes.append(('ill_conditioned_motion', metrics.get('motion_jacobian_cond_max_mean', 0)))
    if metrics.get('measurement_jacobian_cond_max_mean', 0) > MEAS_JAC_THRESHOLD:
        causes.append(('ill_conditioned_measurement', metrics.get('measurement_jacobian_cond_max_mean', 0)))
    if metrics.get('covariance_cond_max_mean', 0) > COV_COND_THRESHOLD:
        causes.append(('covariance_collapse', metrics.get('covariance_cond_max_mean', 0)))
    if metrics.get('ess_min_mean', float('inf')) < ESS_THRESHOLD:
        causes.append(('particle_degeneracy', metrics.get('ess_min_mean', 0)))
    if metrics.get('flow_magnitude_max_mean', 0) > FLOW_THRESHOLD:
        causes.append(('excessive_flow', metrics.get('flow_magnitude_max_mean', 0)))
    if metrics.get('numerical_instability_count_mean', 0) > 5:
        causes.append(('numerical_instability', metrics.get('numerical_instability_count_mean', 0)))
    if not causes:
        return 'unknown'
    return causes[0][0]


def analyze_correlation(all_results: List[ExperimentResult]) -> Dict:
    """Analyze correlation between stability metrics and performance."""
    correlations: Dict[str, List[Tuple[float, float]]] = {
        'motion_jacobian_vs_rmse': [],
        'measurement_jacobian_vs_rmse': [],
        'covariance_cond_vs_rmse': [],
        'ess_vs_rmse': [],
        'flow_magnitude_vs_rmse': []
    }
    for result in all_results:
        rmse = result.rmse_mean
        metrics = result.stability_metrics
        if metrics.get('motion_jacobian_cond_mean_mean') is not None:
            correlations['motion_jacobian_vs_rmse'].append((metrics['motion_jacobian_cond_mean_mean'], rmse))
        if metrics.get('measurement_jacobian_cond_mean_mean') is not None:
            correlations['measurement_jacobian_vs_rmse'].append((metrics['measurement_jacobian_cond_mean_mean'], rmse))
        if metrics.get('covariance_cond_mean_mean') is not None:
            correlations['covariance_cond_vs_rmse'].append((metrics['covariance_cond_mean_mean'], rmse))
        if metrics.get('ess_mean_mean') is not None:
            correlations['ess_vs_rmse'].append((metrics['ess_mean_mean'], rmse))
        if metrics.get('flow_magnitude_mean_mean') is not None:
            correlations['flow_magnitude_vs_rmse'].append((metrics['flow_magnitude_mean_mean'], rmse))

    correlation_coeffs: Dict[str, float] = {}
    for key, values in correlations.items():
        if len(values) > 2:
            x_vals = [v[0] for v in values]
            y_vals = [v[1] for v in values]
            mx = sum(x_vals) / len(x_vals)
            my = sum(y_vals) / len(y_vals)
            std_x = (sum((a - mx) ** 2 for a in x_vals) / len(x_vals)) ** 0.5
            std_y = (sum((b - my) ** 2 for b in y_vals) / len(y_vals)) ** 0.5
            if std_x > 0 and std_y > 0:
                cov = sum((a - mx) * (b - my) for a, b in zip(x_vals, y_vals)) / len(x_vals)
                correlation_coeffs[key] = cov / (std_x * std_y)
    return {'raw_data': correlations, 'coefficients': correlation_coeffs}


def generate_recommendations_v2(all_results: List[ExperimentResult], scenarios: List[ScenarioConfig]) -> Dict:
    """Generate recommendations using rank-based analysis."""
    recommendations: Dict = defaultdict(lambda: {
        'best_for': [], 'consistent_in': [], 'avoid_for': [],
        'characteristics': {}, 'avg_rank_by_category': {}
    })
    scenario_categories = {
        'nonlinearity': ['nonlin_mild', 'nonlin_moderate', 'nonlin_severe'],
        'sparsity': ['sparse_obs_70pct', 'sparse_obs_50pct', 'sparse_obs_30pct', 'sparse_sensors_9', 'sparse_sensors_4'],
        'dimension': ['dimension_1targets', 'dimension_3targets', 'dimension_5targets', 'dimension_8targets'],
        'conditioning': ['illcond_high_process_noise', 'illcond_high_obs_noise', 'illcond_poor_geometry', 'illcond_noise_ratio'],
        'correlation': ['corr_measurement_moderate', 'corr_measurement_high', 'corr_state_formation'],
        'stress': ['stress_nonlin_sparse', 'stress_highdim_illcond', 'stress_extreme', 'stress_corr_extreme']
    }

    for category, scenario_names in scenario_categories.items():
        filter_ranks: Dict[str, List[int]] = defaultdict(list)
        for scenario_name in scenario_names:
            scenario_results = [r for r in all_results if r.scenario_name == scenario_name]
            if not scenario_results:
                continue
            successful_results = [r for r in scenario_results if r.success_rate > 0.5]
            if not successful_results:
                continue
            sorted_results = sorted(successful_results, key=lambda r: r.rmse_mean)
            for rank, result in enumerate(sorted_results, start=1):
                filter_ranks[result.filter_name].append(rank)

        avg_ranks: Dict[str, float] = {}
        for filter_name, ranks in filter_ranks.items():
            if ranks:
                avg_ranks[filter_name] = sum(ranks) / len(ranks)
                recommendations[filter_name]['avg_rank_by_category'][category] = avg_ranks[filter_name]

        if not avg_ranks:
            continue
        best_filter = min(avg_ranks, key=avg_ranks.get)
        recommendations[best_filter]['best_for'].append(category)
        for filter_name, avg_rank in avg_ranks.items():
            if avg_rank <= 2.0:
                recommendations[filter_name]['consistent_in'].append(category)
        for filter_name, avg_rank in avg_ranks.items():
            if avg_rank > 3.0:
                recommendations[filter_name]['avoid_for'].append(category)

    for filter_name in set(r.filter_name for r in all_results):
        filter_results = [r for r in all_results if r.filter_name == filter_name]
        if filter_results:
            successful_results = [r for r in filter_results if r.success_rate > 0.5]
            avg_rmse = sum(r.rmse_mean for r in successful_results) / len(successful_results) if successful_results else float('nan')
        else:
            avg_rmse = float('nan')
        avg_time = sum(r.exec_time_mean for r in filter_results) / len(filter_results) if filter_results else 0.0
        avg_success = sum(r.success_rate for r in filter_results) / len(filter_results) if filter_results else 0.0

        wins = 0
        for scenario_name in set(r.scenario_name for r in all_results):
            scenario_results = [r for r in all_results if r.scenario_name == scenario_name]
            if scenario_results:
                best_rmse = min(r.rmse_mean for r in scenario_results if r.success_rate > 0.5)
                filter_result = next((r for r in scenario_results if r.filter_name == filter_name), None)
                if filter_result and filter_result.rmse_mean == best_rmse and filter_result.success_rate > 0.5:
                    wins += 1

        all_ranks: List[int] = []
        for scenario_name in set(r.scenario_name for r in all_results):
            scenario_results = [r for r in all_results if r.scenario_name == scenario_name]
            successful_results = [r for r in scenario_results if r.success_rate > 0.5]
            if successful_results:
                sorted_results = sorted(successful_results, key=lambda r: r.rmse_mean)
                for rank, result in enumerate(sorted_results, start=1):
                    if result.filter_name == filter_name:
                        all_ranks.append(rank)

        recommendations[filter_name]['characteristics'] = {
            'avg_rmse': None if math.isnan(avg_rmse) else avg_rmse,
            'avg_time': avg_time,
            'success_rate': avg_success,
            'wins': wins,
            'avg_rank': sum(all_ranks) / len(all_ranks) if all_ranks else None,
            'consistency_score': len(recommendations[filter_name]['consistent_in'])
        }

    return dict(recommendations)


def analyze_matrix_vs_scalar(all_results: List[ExperimentResult]) -> Dict:
    """Detailed analysis of PFF_MATRIX vs PFF_SCALAR."""
    scalar_results = [r for r in all_results if r.filter_name == 'pff_scalar']
    matrix_results = [r for r in all_results if r.filter_name == 'pff_matrix']
    if not scalar_results or not matrix_results:
        return {}

    analysis: Dict = {'head_to_head': [], 'summary': {}, 'insights': []}
    scalar_wins = 0
    matrix_wins = 0
    close_races = 0

    for s_res, m_res in zip(scalar_results, matrix_results):
        scenario = s_res.scenario_name
        scalar_rmse = s_res.rmse_mean
        matrix_rmse = m_res.rmse_mean
        diff = scalar_rmse - matrix_rmse
        winner = "MATRIX" if diff > 0 else "SCALAR"
        margin = abs(diff)
        if winner == "MATRIX":
            matrix_wins += 1
        else:
            scalar_wins += 1
        is_close = margin < 0.5
        if is_close:
            close_races += 1
        scalar_time = s_res.exec_time_mean
        matrix_time = m_res.exec_time_mean
        time_ratio = matrix_time / scalar_time if scalar_time > 0 else 0
        likely_unconverged = time_ratio > 1.8
        analysis['head_to_head'].append({
            'scenario': scenario, 'scalar_rmse': scalar_rmse, 'matrix_rmse': matrix_rmse,
            'winner': winner, 'margin': margin, 'is_close': is_close,
            'scalar_time': scalar_time, 'matrix_time': matrix_time, 'time_ratio': time_ratio,
            'likely_unconverged': likely_unconverged
        })

    scalar_avg_rmse = sum(r.rmse_mean for r in scalar_results) / len(scalar_results)
    matrix_avg_rmse = sum(r.rmse_mean for r in matrix_results) / len(matrix_results)
    scalar_avg_time = sum(r.exec_time_mean for r in scalar_results) / len(scalar_results)
    matrix_avg_time = sum(r.exec_time_mean for r in matrix_results) / len(matrix_results)

    analysis['summary'] = {
        'scalar_wins': scalar_wins, 'matrix_wins': matrix_wins, 'close_races': close_races,
        'scalar_avg_rmse': scalar_avg_rmse, 'matrix_avg_rmse': matrix_avg_rmse,
        'rmse_improvement': (scalar_avg_rmse - matrix_avg_rmse) / scalar_avg_rmse * 100,
        'scalar_avg_time': scalar_avg_time, 'matrix_avg_time': matrix_avg_time,
        'time_overhead': (matrix_avg_time - scalar_avg_time) / scalar_avg_time * 100
    }

    if matrix_avg_rmse < scalar_avg_rmse:
        analysis['insights'].append(
            f"MATRIX is {analysis['summary']['rmse_improvement']:.1f}% more accurate on average"
        )
    if matrix_wins < scalar_wins:
        analysis['insights'].append(
            f"SCALAR wins more scenarios ({scalar_wins} vs {matrix_wins}) but MATRIX is more consistent"
        )
    if close_races > len(scalar_results) * 0.3:
        analysis['insights'].append(
            f"{close_races} scenarios are close races (margin < 0.5 RMSE) - differences may be noise"
        )
    unconverged_count = sum(1 for h in analysis['head_to_head'] if h['likely_unconverged'])
    if unconverged_count > 0:
        analysis['insights'].append(
            f"{unconverged_count} scenarios likely have unconverged MATRIX (time ratio > 1.8) - increase max_steps"
        )
    if analysis['summary']['time_overhead'] > 50:
        analysis['insights'].append(
            f"MATRIX has {analysis['summary']['time_overhead']:.0f}% time overhead - tuning needed for fair comparison"
        )
    corr_scenarios = [h for h in analysis['head_to_head'] if 'corr_' in h['scenario']]
    if corr_scenarios:
        matrix_better_in_corr = sum(1 for h in corr_scenarios if h['winner'] == 'MATRIX')
        if matrix_better_in_corr > len(corr_scenarios) * 0.6:
            analysis['insights'].append(
                f"MATRIX excels in correlated scenarios ({matrix_better_in_corr}/{len(corr_scenarios)})"
            )
    return analysis


def analyze_results(all_results: List[ExperimentResult], scenarios: List[ScenarioConfig], output_dir: Path) -> Dict:
    """Enhanced analysis with rank-based recommendations and matrix vs scalar."""
    scenarios_dict: Dict[str, List[ExperimentResult]] = {}
    for result in all_results:
        if result.scenario_name not in scenarios_dict:
            scenarios_dict[result.scenario_name] = []
        scenarios_dict[result.scenario_name].append(result)

    summary: Dict = {
        'scenarios': {}, 'filter_rankings': {}, 'failure_analysis': {},
        'correlations': {}, 'recommendations': {}, 'matrix_vs_scalar_analysis': {}, 'key_findings': {}
    }

    for scenario_name, results in scenarios_dict.items():
        scenario_summary: Dict = {'filters': {}, 'best_filter': None, 'worst_filter': None, 'ranking': []}
        successful_results = [r for r in results if r.success_rate > 0.5]
        if successful_results:
            sorted_results = sorted(successful_results, key=lambda r: r.rmse_mean)
            scenario_summary['ranking'] = [r.filter_name for r in sorted_results]
            scenario_summary['best_filter'] = sorted_results[0].filter_name
            scenario_summary['worst_filter'] = sorted_results[-1].filter_name
        for result in results:
            scenario_summary['filters'][result.filter_name] = {
                'rmse_mean': result.rmse_mean, 'rmse_std': result.rmse_std,
                'exec_time_mean': result.exec_time_mean, 'success_rate': result.success_rate,
                'stability': result.stability_metrics,
                'failure_cause': diagnose_failure_cause(result) if result.success_rate < 0.7 else None
            }
        summary['scenarios'][scenario_name] = scenario_summary

    failure_by_filter: Dict[str, List[Dict]] = defaultdict(list)
    for result in all_results:
        if result.success_rate < 0.7 or result.rmse_mean > 50:
            cause = diagnose_failure_cause(result)
            failure_by_filter[result.filter_name].append({
                'scenario': result.scenario_name, 'cause': cause,
                'rmse': result.rmse_mean, 'success_rate': result.success_rate
            })
    summary['failure_analysis'] = dict(failure_by_filter)
    summary['correlations'] = analyze_correlation(all_results)
    summary['recommendations'] = generate_recommendations_v2(all_results, scenarios)
    summary['matrix_vs_scalar_analysis'] = analyze_matrix_vs_scalar(all_results)

    filter_wins: Dict[str, List[str]] = {}
    for scenario_name, scenario_data in summary['scenarios'].items():
        best_filter = scenario_data['best_filter']
        if best_filter:
            if best_filter not in filter_wins:
                filter_wins[best_filter] = []
            filter_wins[best_filter].append(scenario_name)
    summary['filter_rankings'] = {'wins': filter_wins}

    findings: Dict[str, List[str]] = {
        'nonlinearity': [], 'sparsity': [], 'dimension': [], 'conditioning': [],
        'correlation': [], 'stress': [], 'overall': [], 'causal_insights': [], 'matrix_vs_scalar': []
    }
    for category in ['nonlinearity', 'sparsity', 'dimension', 'conditioning', 'correlation', 'stress']:
        best_filters = [f for f, rec in summary['recommendations'].items() if category in rec['best_for']]
        for f in best_filters:
            avg_rank = summary['recommendations'][f]['avg_rank_by_category'].get(category, 0)
            findings[category].append(f"{f.upper()} best in {category} (avg rank: {avg_rank:.2f})")
        consistent_filters = [f for f, rec in summary['recommendations'].items()
                            if category in rec['consistent_in'] and category not in rec['best_for']]
        for f in consistent_filters:
            avg_rank = summary['recommendations'][f]['avg_rank_by_category'].get(category, 0)
            findings[category].append(f"{f.upper()} consistent in {category} (avg rank: {avg_rank:.2f})")

    all_filters = list(summary['recommendations'].keys())
    sorted_by_rank = sorted(
        all_filters,
        key=lambda f: summary['recommendations'][f]['characteristics'].get('avg_rank', 999) or 999
    )
    for filter_name in sorted_by_rank:
        chars = summary['recommendations'][filter_name]['characteristics']
        avg_rank = chars.get('avg_rank')
        wins = chars.get('wins', 0)
        consistency = chars.get('consistency_score', 0)
        if avg_rank is not None:
            findings['overall'].append(
                f"{filter_name.upper()}: Avg rank {avg_rank:.2f}, {wins} wins, consistent in {consistency} categories"
            )
    if summary['correlations']['coefficients']:
        for metric, corr in summary['correlations']['coefficients'].items():
            if abs(corr) > 0.5:
                findings['causal_insights'].append(
                    f"{metric.replace('_', ' ').title()}: {corr:.3f} correlation with RMSE"
                )
    if summary['matrix_vs_scalar_analysis'] and 'insights' in summary['matrix_vs_scalar_analysis']:
        findings['matrix_vs_scalar'] = summary['matrix_vs_scalar_analysis']['insights']
    summary['key_findings'] = findings

    def _to_serializable(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_to_serializable(x) for x in obj]
        return str(obj)

    with open(output_dir / 'experiment_summary.json', 'w') as f:
        json.dump(_to_serializable(summary), f, indent=2)

    detailed_results = []
    for result in all_results:
        detailed_results.append({
            'filter': result.filter_name, 'scenario': result.scenario_name,
            'rmse_mean': result.rmse_mean, 'rmse_std': result.rmse_std,
            'exec_time': result.exec_time_mean, 'success_rate': result.success_rate,
            'stability': result.stability_metrics,
            'failure_cause': diagnose_failure_cause(result) if result.success_rate < 0.7 else None
        })
    with open(output_dir / 'detailed_results.json', 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Build experiment summary text (print to terminal and save to .txt)
    summary_lines: List[str] = []
    summary_lines.append("")
    summary_lines.append("=" * 80)
    summary_lines.append("EXPERIMENT SUMMARY")
    summary_lines.append("=" * 80)
    for category, finding_list in findings.items():
        if finding_list:
            summary_lines.append(f"\n{category.upper().replace('_', ' ')}:")
            for finding in finding_list:
                summary_lines.append(f"  • {finding}")

    summary_lines.append("\n" + "=" * 80)
    summary_lines.append("FILTER RECOMMENDATIONS (Rank-Based Analysis)")
    summary_lines.append("=" * 80)
    sorted_filters = sorted(
        summary['recommendations'].items(),
        key=lambda x: x[1]['characteristics'].get('avg_rank', 999) or 999
    )
    for filter_name, rec in sorted_filters:
        chars = rec['characteristics']
        summary_lines.append(f"\n{filter_name.upper()}:")
        if rec['best_for']:
            summary_lines.append(f"  ✓ Best for: {', '.join(rec['best_for'])}")
        if rec['consistent_in']:
            consistent_not_best = [c for c in rec['consistent_in'] if c not in rec['best_for']]
            if consistent_not_best:
                summary_lines.append(f"  ◉ Consistent in: {', '.join(consistent_not_best)}")
        if rec['avoid_for']:
            summary_lines.append(f"  ✗ Avoid for: {', '.join(rec['avoid_for'])}")
        avg_rmse = chars.get('avg_rmse')
        rmse_str = f"{avg_rmse:.3f}" if avg_rmse is not None else "N/A"
        avg_rank = chars.get('avg_rank')
        rank_str = f"{avg_rank:.2f}" if avg_rank is not None else "N/A"
        summary_lines.append(f"  Stats: RMSE={rmse_str}, Time={chars.get('avg_time', 0):.3f}s, Success={chars.get('success_rate', 0):.1%}, Wins={chars.get('wins', 0)}, Avg Rank={rank_str}")

    if summary['matrix_vs_scalar_analysis'] and 'summary' in summary['matrix_vs_scalar_analysis']:
        summary_lines.append("\n" + "=" * 80)
        summary_lines.append("PFF_MATRIX vs PFF_SCALAR DETAILED ANALYSIS")
        summary_lines.append("=" * 80)
        mvs_summary = summary['matrix_vs_scalar_analysis']['summary']
        summary_lines.append(f"\nHead-to-Head: SCALAR wins: {mvs_summary['scalar_wins']}, MATRIX wins: {mvs_summary['matrix_wins']}, Close races: {mvs_summary['close_races']}")
        summary_lines.append(f"Performance: SCALAR avg RMSE: {mvs_summary['scalar_avg_rmse']:.3f}, MATRIX avg RMSE: {mvs_summary['matrix_avg_rmse']:.3f}, MATRIX improvement: {mvs_summary['rmse_improvement']:.1f}%")
        summary_lines.append(f"Computational: SCALAR avg time: {mvs_summary['scalar_avg_time']:.3f}s, MATRIX avg time: {mvs_summary['matrix_avg_time']:.3f}s, MATRIX overhead: {mvs_summary['time_overhead']:.1f}%")
        summary_lines.append("\nKey Insights:")
        for insight in summary['matrix_vs_scalar_analysis']['insights']:
            summary_lines.append(f"  • {insight}")
    summary_lines.append("\n" + "=" * 80)

    for line in summary_lines:
        print(line)

    with open(output_dir / 'experiment_summary.txt', 'w') as f:
        f.write("\n".join(summary_lines))
        f.write("\n")
    print(f"\n✓ Experiment summary saved to: {output_dir / 'experiment_summary.txt'}")

    return summary


# ============================================================================
# Visualization (TF/Python only; .numpy() only for matplotlib; no NumPy)
# ============================================================================

def _mean(lst: List[float]) -> float:
    """Mean of list (no NumPy)."""
    if not lst:
        return float('nan')
    return sum(lst) / len(lst)


def _std(lst: List[float]) -> float:
    """Standard deviation of list (no NumPy)."""
    if len(lst) < 2:
        return 0.0
    m = _mean(lst)
    return (sum((x - m) ** 2 for x in lst) / len(lst)) ** 0.5


def plot_performance_radar(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Radar chart showing filter performance across challenge categories."""
    categories = {
        'Nonlinearity': ['nonlin_mild', 'nonlin_moderate', 'nonlin_severe'],
        'Sparsity': ['sparse_obs_70pct', 'sparse_obs_50pct', 'sparse_obs_30pct'],
        'High Dimension': ['dimension_5targets', 'dimension_8targets'],
        'Ill-Conditioning': [
            'illcond_high_process_noise',
            'illcond_high_obs_noise',
            'illcond_poor_geometry',
        ],
        'Correlation': ['corr_measurement_moderate', 'corr_measurement_high'],
        'Stress': ['stress_nonlin_sparse', 'stress_highdim_illcond', 'stress_extreme'],
    }
    filters = sorted(set(r.filter_name for r in all_results))
    scores: Dict[str, List[float]] = {}
    for filter_name in filters:
        scores[filter_name] = []
        for scen_list in categories.values():
            ranks: List[int] = []
            for scenario in scen_list:
                scenario_results = [r for r in all_results if r.scenario_name == scenario]
                if not scenario_results:
                    continue
                sorted_results = sorted(scenario_results, key=lambda r: r.rmse_mean)
                for rank, r in enumerate(sorted_results, 1):
                    if r.filter_name == filter_name:
                        ranks.append(rank)
                        break
            if ranks:
                avg_rank = _mean(ranks)
                max_rank = len(filters)
                score = (1.0 - (avg_rank - 1.0) / (max_rank - 1.0)) if max_rank > 1 else 1.0
                scores[filter_name].append(float(score))
            else:
                scores[filter_name].append(0.0)
    if not filters:
        return
    n_filters = len(filters)
    n_cols = 3
    n_rows = (n_filters + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(18, 6 * n_rows),
        subplot_kw=dict(projection="polar"),
    )
    axes_flat = axes.flatten() if n_filters > 1 else [axes]
    cat_labels = list(categories.keys())
    angles = [n / len(cat_labels) * 2 * math.pi for n in range(len(cat_labels))]
    angles += angles[:1]
    for idx, filter_name in enumerate(filters):
        ax = axes_flat[idx]
        vals = scores[filter_name] + scores[filter_name][:1]
        ax.plot(angles, vals, "o-", linewidth=2, label=filter_name.upper())
        ax.fill(angles, vals, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cat_labels, size=10)
        ax.set_ylim(0.0, 1.0)
        ax.set_title(filter_name.upper(), size=14, weight="bold", pad=20)
        ax.grid(True)
        for ang, v in zip(angles[:-1], scores[filter_name]):
            ax.text(ang, min(max(v + 0.05, 0.0), 1.05), f"{v:.2f}", ha="center", va="center", size=8, weight="bold")
    for idx in range(n_filters, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    plt.suptitle("Filter Performance Across Challenge Categories\n(1.0 = Best, 0.0 = Worst)", size=16, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "performance_radar.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_failure_diagnosis_heatmap(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Heatmap with failure cause annotations."""
    scenarios = sorted(set(r.scenario_name for r in all_results))
    filters = sorted(set(r.filter_name for r in all_results))
    n_s, n_f = len(scenarios), len(filters)
    rmse_data = [[float('nan')] * n_f for _ in range(n_s)]
    success_data = [[float('nan')] * n_f for _ in range(n_s)]
    for i, scenario in enumerate(scenarios):
        for j, filter_name in enumerate(filters):
            results = [r for r in all_results if r.scenario_name == scenario and r.filter_name == filter_name]
            if results:
                rmse_data[i][j] = results[0].rmse_mean
                success_data[i][j] = results[0].success_rate
    rmse_normalized = [[math.log10(v + 1.0) if math.isfinite(v) else 0.0 for v in row] for row in rmse_data]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    im1 = ax1.imshow(rmse_normalized, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xticks(range(n_f))
    ax1.set_yticks(range(n_s))
    ax1.set_xticklabels([f.upper() for f in filters], rotation=45, ha="right")
    ax1.set_yticklabels(scenarios, fontsize=8)
    for i in range(n_s):
        for j in range(n_f):
            if math.isfinite(rmse_data[i][j]):
                rmse_val = rmse_data[i][j]
                success_val = success_data[i][j]
                text_color = "darkgreen" if success_val > 0.7 else "darkred"
                weight = "bold" if success_val < 0.5 else "normal"
                marker = "X" if success_val < 0.5 else ""
                ax1.text(j, i, f"{rmse_val:.1f}{marker}", ha="center", va="center", color=text_color, fontsize=8, weight=weight)
    ax1.set_title("RMSE by Scenario and Filter\n(X = Failure, Green = Success, Red = Poor)", fontsize=12, pad=10)
    plt.colorbar(im1, ax=ax1, label="log10(RMSE + 1)")
    stability_scores = [[0] * n_f for _ in range(n_s)]
    for i, scenario in enumerate(scenarios):
        for j, filter_name in enumerate(filters):
            results = [r for r in all_results if r.scenario_name == scenario and r.filter_name == filter_name]
            if results:
                metrics = results[0].stability_metrics
                score = 0
                if metrics.get("motion_jacobian_cond_max_mean", 0.0) > 1e5:
                    score += 1
                if metrics.get("measurement_jacobian_cond_max_mean", 0.0) > 1e5:
                    score += 1
                if metrics.get("covariance_cond_max_mean", 0.0) > 1e7:
                    score += 1
                if metrics.get("ess_min_mean", 1e9) < 50.0:
                    score += 1
                if metrics.get("flow_magnitude_max_mean", 0.0) > 10.0:
                    score += 1
                stability_scores[i][j] = score
    im2 = ax2.imshow(stability_scores, cmap="YlOrRd", aspect="auto", vmin=0, vmax=5)
    ax2.set_xticks(range(n_f))
    ax2.set_yticks(range(n_s))
    ax2.set_xticklabels([f.upper() for f in filters], rotation=45, ha="right")
    ax2.set_yticklabels(scenarios, fontsize=8)
    for i in range(n_s):
        for j in range(n_f):
            score = stability_scores[i][j]
            if score > 0:
                ax2.text(j, i, str(score), ha="center", va="center", color="white" if score > 2 else "black", fontsize=10, weight="bold")
    ax2.set_title("Stability Issues Count\n(0 = Stable, 5 = Multiple Issues)", fontsize=12, pad=10)
    plt.colorbar(im2, ax=ax2, label="Number of Stability Issues")
    plt.tight_layout()
    plt.savefig(output_dir / "failure_diagnosis_heatmap.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_challenge_performance_profiles(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Bar charts showing filter performance on specific challenges."""
    challenge_groups = {
        "Nonlinearity Stress": {
            "scenarios": ["nonlin_mild", "nonlin_moderate", "nonlin_severe"],
            "labels": ["Mild", "Moderate", "Severe"],
        },
        "Observation Sparsity": {
            "scenarios": ["sparse_obs_70pct", "sparse_obs_50pct", "sparse_obs_30pct"],
            "labels": ["70%", "50%", "30%"],
        },
        "Dimensionality": {
            "scenarios": ["dimension_1targets", "dimension_3targets", "dimension_5targets", "dimension_8targets"],
            "labels": ["4D", "12D", "20D", "32D"],
        },
        "Ill-Conditioning": {
            "scenarios": ["illcond_high_process_noise", "illcond_high_obs_noise", "illcond_poor_geometry"],
            "labels": ["Proc Noise", "Obs Noise", "Geometry"],
        },
    }
    filters = sorted(set(r.filter_name for r in all_results))
    if not filters:
        return
    colors = plt.cm.tab10([i / max(len(filters) - 1, 1) for i in range(len(filters))])
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes_flat = axes.flatten()
    for idx, (challenge_name, config) in enumerate(challenge_groups.items()):
        ax = axes_flat[idx]
        scenarios = config["scenarios"]
        labels = config["labels"]
        x = list(range(len(scenarios)))
        width = 0.8 / len(filters)
        for f_idx, filter_name in enumerate(filters):
            rmses = []
            for scenario in scenarios:
                results = [r for r in all_results if r.scenario_name == scenario and r.filter_name == filter_name]
                if results and results[0].success_rate > 0.5:
                    rmses.append(results[0].rmse_mean)
                else:
                    rmses.append(float('nan'))
            offset = (f_idx - len(filters) / 2.0) * width + width / 2.0
            bars = ax.bar([xi + offset for xi in x], rmses, width, label=filter_name.upper(), alpha=0.8, color=colors[f_idx])
            for bar, rmse in zip(bars, rmses):
                if math.isfinite(rmse):
                    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{rmse:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("RMSE", fontsize=11)
        ax.set_xlabel("Challenge Level", fontsize=11)
        ax.set_title(challenge_name, fontsize=12, weight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_yscale("log")
    plt.suptitle("Filter Performance Across Challenge Dimensions", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "challenge_performance_profiles.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_stability_overview(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Create overview of stability metrics (box plots)."""
    filters = sorted(set(r.filter_name for r in all_results))
    metrics_to_plot = [
        ('motion_jacobian_cond_mean_mean', 'Motion Jacobian Cond.'),
        ('measurement_jacobian_cond_mean_mean', 'Meas. Jacobian Cond.'),
        ('covariance_cond_mean_mean', 'Covariance Cond.'),
        ('ess_mean_mean', 'ESS'),
        ('flow_magnitude_mean_mean', 'Flow Magnitude'),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes_flat = axes.flatten()
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        ax = axes_flat[idx]
        data_by_filter = defaultdict(list)
        for result in all_results:
            value = result.stability_metrics.get(metric_key)
            if value is not None and math.isfinite(value):
                data_by_filter[result.filter_name].append(value)
        plot_data = [data_by_filter[f] for f in filters if f in data_by_filter and data_by_filter[f]]
        plot_labels = [f.upper() for f in filters if f in data_by_filter and data_by_filter[f]]
        if plot_data:
            bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax.set_title(metric_name, fontsize=11)
            ax.set_ylabel('Value', fontsize=9)
            ax.tick_params(axis='x', rotation=45, labelsize=8)
            ax.grid(True, alpha=0.3)
            if 'cond' in metric_key.lower():
                ax.set_yscale('log')
    fig.delaxes(axes_flat[-1])
    plt.suptitle('Stability Metrics Overview Across All Scenarios', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'stability_overview.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_time_series(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Plot time series for selected scenarios."""
    representative_scenarios = ['baseline', 'nonlin_severe', 'sparse_obs_30pct', 'stress_extreme', 'corr_measurement_high']
    ts_dir = output_dir / 'time_series'
    ts_dir.mkdir(parents=True, exist_ok=True)
    for scenario_name in representative_scenarios:
        results_for_scenario = [
            r for r in all_results
            if r.scenario_name == scenario_name and getattr(r, 'time_series_data', None) is not None
        ]
        if not results_for_scenario:
            continue
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, :])
        for result in results_for_scenario:
            ts = getattr(result, 'time_series_data', None)
            if ts and ts.get('rmse_series'):
                ax1.plot(ts['rmse_series'], label=result.filter_name.upper(), alpha=0.7, linewidth=2)
        ax1.set_ylabel('RMSE', fontsize=10)
        ax1.set_title(f'Scenario: {scenario_name}', fontsize=12)
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2 = fig.add_subplot(gs[1, 0])
        for result in results_for_scenario:
            ts = getattr(result, 'time_series_data', None)
            if ts and ts.get('motion_jacobian_series'):
                series = [v for v in ts['motion_jacobian_series'] if math.isfinite(v)]
                if series:
                    ax2.plot(series, label=result.filter_name.upper(), alpha=0.7)
        ax2.set_ylabel('Motion Jac. Cond.', fontsize=9)
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=7)
        ax3 = fig.add_subplot(gs[1, 1])
        for result in results_for_scenario:
            ts = getattr(result, 'time_series_data', None)
            if ts and ts.get('measurement_jacobian_series'):
                series = [v for v in ts['measurement_jacobian_series'] if math.isfinite(v)]
                if series:
                    ax3.plot(series, label=result.filter_name.upper(), alpha=0.7)
        ax3.set_ylabel('Meas. Jac. Cond.', fontsize=9)
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=7)
        ax4 = fig.add_subplot(gs[2, 0])
        for result in results_for_scenario:
            ts = getattr(result, 'time_series_data', None)
            if ts and ts.get('ess_series'):
                series = [v for v in ts['ess_series'] if math.isfinite(v)]
                if series:
                    ax4.plot(series, label=result.filter_name.upper(), alpha=0.7)
        ax4.set_ylabel('ESS', fontsize=9)
        ax4.set_xlabel('Time Step', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=7)
        ax5 = fig.add_subplot(gs[2, 1])
        for result in results_for_scenario:
            ts = getattr(result, 'time_series_data', None)
            if ts and ts.get('flow_magnitude_series'):
                series = [v for v in ts['flow_magnitude_series'] if math.isfinite(v)]
                if series:
                    ax5.plot(series, label=result.filter_name.upper(), alpha=0.7)
        ax5.set_ylabel('Flow Magnitude', fontsize=9)
        ax5.set_xlabel('Time Step', fontsize=9)
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(ts_dir / f'timeseries_{scenario_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()


def plot_dimension_scaling(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Plot performance scaling with dimension."""
    dimension_results = [r for r in all_results if 'dimension' in r.scenario_name]
    if not dimension_results:
        return
    dimension_map = {}
    for r in dimension_results:
        parts = r.scenario_name.split('_')
        if len(parts) >= 2:
            dimension_map[r.scenario_name] = int(parts[1].replace('targets', ''))
    filters = sorted(set(r.filter_name for r in dimension_results))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for filter_name in filters:
        filter_results = [r for r in dimension_results if r.filter_name == filter_name]
        dims = [dimension_map.get(r.scenario_name, 0) for r in filter_results]
        rmses = [r.rmse_mean for r in filter_results]
        sorted_pairs = sorted(zip(dims, rmses))
        dims, rmses = [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]
        ax1.plot(dims, rmses, marker='o', label=filter_name.upper(), linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Targets (State Dimension / 4)', fontsize=11)
    ax1.set_ylabel('RMSE', fontsize=11)
    ax1.set_title('Performance Scaling with Dimension', fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    for filter_name in filters:
        filter_results = [r for r in dimension_results if r.filter_name == filter_name]
        dims = [dimension_map.get(r.scenario_name, 0) for r in filter_results]
        times = [r.exec_time_mean for r in filter_results]
        sorted_pairs = sorted(zip(dims, times))
        dims, times = [p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs]
        ax2.plot(dims, times, marker='s', label=filter_name.upper(), linewidth=2, markersize=8)
    ax2.set_xlabel('Number of Targets (State Dimension / 4)', fontsize=11)
    ax2.set_ylabel('Execution Time (s)', fontsize=11)
    ax2.set_title('Computational Scaling with Dimension', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'dimension_scaling.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_execution_times(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Plot execution time comparison."""
    filters = sorted(set(r.filter_name for r in all_results))
    times_by_filter = defaultdict(list)
    for result in all_results:
        times_by_filter[result.filter_name].append(result.exec_time_mean)
    plot_data = [times_by_filter[f] for f in filters]
    plot_labels = [f.upper() for f in filters]
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(plot_data, tick_labels=plot_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
    ax.set_ylabel('Execution Time (s)', fontsize=12)
    ax.set_xlabel('Filter', fontsize=12)
    ax.set_title('Computational Cost Comparison', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_yscale('log')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'execution_times.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_stability_diagnostic_dashboard(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Comprehensive stability diagnostic visualization."""
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    filters = sorted(set(r.filter_name for r in all_results))
    if not filters:
        return
    colors = {f: plt.cm.tab10(i) for i, f in enumerate(filters)}
    ax1 = fig.add_subplot(gs[0, 0])
    for filter_name in filters:
        f_results = [r for r in all_results if r.filter_name == filter_name]
        flow_mags = [r.stability_metrics.get("flow_magnitude_mean_mean", float('nan')) for r in f_results]
        rmses = [r.rmse_mean for r in f_results]
        data = [(f, rm) for f, rm in zip(flow_mags, rmses) if math.isfinite(f) and math.isfinite(rm)]
        if data:
            x, y = zip(*data)
            ax1.scatter(x, y, label=filter_name.upper(), alpha=0.6, color=colors[filter_name], s=50)
    ax1.set_xlabel("Flow Magnitude", fontsize=10)
    ax1.set_ylabel("RMSE", fontsize=10)
    ax1.set_title("Particle Flow vs Performance", fontsize=11, weight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax2 = fig.add_subplot(gs[0, 1])
    for filter_name in filters:
        f_results = [r for r in all_results if r.filter_name == filter_name]
        jac_conds = [r.stability_metrics.get("motion_jacobian_cond_mean_mean", float('nan')) for r in f_results]
        rmses = [r.rmse_mean for r in f_results]
        data = [(j, rm) for j, rm in zip(jac_conds, rmses) if math.isfinite(j) and math.isfinite(rm)]
        if data:
            x, y = zip(*data)
            ax2.scatter(x, y, label=filter_name.upper(), alpha=0.6, color=colors[filter_name], s=50)
    ax2.axvline(x=1e5, color="red", linestyle="--", alpha=0.5, label="Instability threshold")
    ax2.set_xlabel("Motion Jacobian Condition Number", fontsize=10)
    ax2.set_ylabel("RMSE", fontsize=10)
    ax2.set_title("Jacobian Conditioning vs Performance", fontsize=11, weight="bold")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax3 = fig.add_subplot(gs[0, 2])
    particle_filters = [f for f in filters if "pf" in f.lower() or "edh" in f.lower()]
    for filter_name in particle_filters:
        f_results = [r for r in all_results if r.filter_name == filter_name]
        ess_vals = [r.stability_metrics.get("ess_mean_mean", float('nan')) for r in f_results]
        rmses = [r.rmse_mean for r in f_results]
        data = [(e, rm) for e, rm in zip(ess_vals, rmses) if math.isfinite(e) and math.isfinite(rm)]
        if data:
            x, y = zip(*data)
            ax3.scatter(x, y, label=filter_name.upper(), alpha=0.6, color=colors[filter_name], s=50)
    ax3.set_xlabel("Effective Sample Size", fontsize=10)
    ax3.set_ylabel("RMSE", fontsize=10)
    ax3.set_title("Particle Degeneracy vs Performance", fontsize=11, weight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale("log")
    ax4 = fig.add_subplot(gs[1, :])
    categories = ["nonlinearity", "sparsity", "dimension", "conditioning", "correlation", "stress"]
    cat_labels = ["Nonlinearity", "Sparsity", "Dimension", "Conditioning", "Correlation", "Stress"]
    x = list(range(len(categories)))
    width = 0.8 / len(filters)
    for f_idx, filter_name in enumerate(filters):
        ranks = []
        for cat in categories:
            cat_scenarios = [r.scenario_name for r in all_results if cat in r.scenario_name]
            cat_results = [r for r in all_results if r.scenario_name in cat_scenarios and r.filter_name == filter_name]
            if cat_results:
                per_run_ranks = []
                for res in cat_results:
                    same_scenario = [rr for rr in all_results if rr.scenario_name == res.scenario_name]
                    same_sorted = sorted(same_scenario, key=lambda rr: rr.rmse_mean)
                    for rk, rr in enumerate(same_sorted, 1):
                        if rr.filter_name == filter_name and rr is res:
                            per_run_ranks.append(rk)
                            break
                ranks.append(_mean(per_run_ranks) if per_run_ranks else float(len(filters)))
            else:
                ranks.append(float(len(filters)))
        offset = (f_idx - len(filters) / 2.0) * width + width / 2.0
        ax4.bar([xi + offset for xi in x], ranks, width, label=filter_name.upper(), alpha=0.8, color=colors[filter_name])
    ax4.set_ylabel("Average Rank (lower = better)", fontsize=11)
    ax4.set_xlabel("Challenge Category", fontsize=11)
    ax4.set_title("Filter Rankings by Challenge Type", fontsize=12, weight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(cat_labels)
    ax4.legend(fontsize=8, ncol=max(1, len(filters) // 2))
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.invert_yaxis()
    ax5 = fig.add_subplot(gs[2, 0])
    success_rates = {}
    for filter_name in filters:
        f_results = [r for r in all_results if r.filter_name == filter_name]
        success_rates[filter_name] = _mean([r.success_rate for r in f_results]) * 100.0 if f_results else 0.0
    names = list(success_rates.keys())
    vals = [success_rates[n] for n in names]
    bars = ax5.barh(names, vals, color=[colors[n] for n in names], alpha=0.8)
    ax5.set_xlabel("Success Rate (%)", fontsize=10)
    ax5.set_title("Overall Success Rate", fontsize=11, weight="bold")
    ax5.grid(True, alpha=0.3, axis="x")
    for bar, rate in zip(bars, vals):
        ax5.text(rate + 1.0, bar.get_y() + bar.get_height() / 2.0, f"{rate:.1f}%", va="center", fontsize=9)
    ax6 = fig.add_subplot(gs[2, 1:])
    for filter_name in filters:
        f_results = [r for r in all_results if r.filter_name == filter_name]
        if not f_results:
            continue
        avg_time = _mean([r.exec_time_mean for r in f_results])
        good_rmse = [r.rmse_mean for r in f_results if r.success_rate > 0.5]
        if not good_rmse:
            continue
        avg_rmse = _mean(good_rmse)
        ax6.scatter(avg_time, avg_rmse, s=200, alpha=0.7, color=colors[filter_name], edgecolors="black", linewidth=2)
        ax6.annotate(filter_name.upper(), (avg_time, avg_rmse), fontsize=9, weight="bold", ha="center", va="center")
    ax6.set_xlabel("Average Execution Time (s)", fontsize=11)
    ax6.set_ylabel("Average RMSE", fontsize=11)
    ax6.set_title("Accuracy vs Computational Cost Trade-off", fontsize=12, weight="bold")
    ax6.grid(True, alpha=0.3)
    ax6.set_xscale("log")
    ax6.set_yscale("log")
    plt.suptitle("Comprehensive Stability Diagnostic Dashboard", fontsize=16, weight="bold")
    plt.savefig(output_dir / "stability_diagnostic_dashboard.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def plot_failure_mode_analysis(all_results: List[ExperimentResult], output_dir: Path) -> None:
    """Visualize WHY filters fail using stability diagnostics."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    filters = sorted(set(r.filter_name for r in all_results))
    ax1 = fig.add_subplot(gs[0, 0])
    for filter_name in filters:
        filter_results = [r for r in all_results if r.filter_name == filter_name]
        cov_conds = [r.stability_metrics.get("covariance_cond_mean_mean", float('nan')) for r in filter_results]
        rmses = [r.rmse_mean for r in filter_results]
        valid_data = [(c, rm) for c, rm in zip(cov_conds, rmses) if math.isfinite(c) and math.isfinite(rm)]
        if valid_data:
            x, y = zip(*valid_data)
            ax1.scatter(x, y, label=filter_name.upper(), alpha=0.6, s=60)
    ax1.axvline(x=1e7, color="red", linestyle="--", label="Danger threshold", linewidth=2)
    ax1.set_xlabel("Covariance Condition Number", fontsize=11)
    ax1.set_ylabel("RMSE", fontsize=11)
    ax1.set_xscale("log")
    ax1.set_title("Failure Mode: Covariance Ill-Conditioning", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(gs[0, 1])
    for filter_name in filters:
        filter_results = [r for r in all_results if r.filter_name == filter_name]
        flows = [r.stability_metrics.get("flow_magnitude_mean_mean", float('nan')) for r in filter_results]
        rmses = [r.rmse_mean for r in filter_results]
        valid_data = [(f, rm) for f, rm in zip(flows, rmses) if math.isfinite(f) and math.isfinite(rm)]
        if valid_data:
            x, y = zip(*valid_data)
            ax2.scatter(x, y, label=filter_name.upper(), alpha=0.6, s=60)
    ax2.axvline(x=15.0, color="red", linestyle="--", label="High flow threshold", linewidth=2)
    ax2.set_xlabel("Particle Flow Magnitude", fontsize=11)
    ax2.set_ylabel("RMSE", fontsize=11)
    ax2.set_title("Failure Mode: Excessive Particle Flow", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax3 = fig.add_subplot(gs[0, 2])
    cov_data = []
    labels = []
    for f in filters:
        f_results = [r for r in all_results if r.filter_name == f]
        conds = [r.stability_metrics.get("covariance_cond_mean_mean", float('nan')) for r in f_results]
        valid_conds = [c for c in conds if math.isfinite(c) and c > 0]
        if valid_conds:
            cov_data.append(valid_conds)
            labels.append(f.upper())
    if cov_data:
        bp = ax3.boxplot(cov_data, tick_labels=labels, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
    ax3.axhline(y=1e7, color="red", linestyle="--", linewidth=2, alpha=0.7)
    ax3.set_ylabel("Covariance Condition Number", fontsize=10)
    ax3.set_yscale("log")
    ax3.set_title("Covariance Stability by Filter", fontsize=12, fontweight="bold")
    ax3.tick_params(axis="x", rotation=45, labelsize=8)
    ax3.grid(True, alpha=0.3, axis="y")
    categories = {
        "Nonlinearity": ["nonlin_mild", "nonlin_moderate", "nonlin_severe"],
        "Conditioning": ["illcond_high_process_noise", "illcond_high_obs_noise", "illcond_poor_geometry"],
        "Stress": ["stress_nonlin_sparse", "stress_highdim_illcond", "stress_extreme"],
    }
    for cat_idx, (cat_name, scenarios) in enumerate(categories.items()):
        ax = fig.add_subplot(gs[1, cat_idx])
        failure_rates = defaultdict(list)
        for scenario in scenarios:
            for filter_name in filters:
                results = [r for r in all_results if r.scenario_name == scenario and r.filter_name == filter_name]
                if results:
                    failed = (results[0].rmse_mean > 10.0) or (results[0].success_rate < 0.7)
                    failure_rates[filter_name].append(1.0 if failed else 0.0)
        filter_names_list = []
        avg_failure_rates = []
        for f in filters:
            if failure_rates[f]:
                filter_names_list.append(f.upper())
                avg_failure_rates.append(_mean(failure_rates[f]) * 100.0)
        if filter_names_list:
            bars = ax.bar(
                range(len(filter_names_list)),
                avg_failure_rates,
                color=["red" if r > 30.0 else "orange" if r > 10.0 else "green" for r in avg_failure_rates],
                alpha=0.7, edgecolor="black",
            )
            ax.set_xticks(range(len(filter_names_list)))
            ax.set_xticklabels(filter_names_list, rotation=45, ha="right", fontsize=8)
            for bar, rate in zip(bars, avg_failure_rates):
                if rate > 5.0:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() / 2.0, f"{rate:.0f}%",
                            ha="center", va="center", fontsize=9, fontweight="bold", color="white")
        ax.set_ylabel("Failure Rate (%)", fontsize=10)
        ax.set_title(f"Failure Rate: {cat_name}", fontsize=11, fontweight="bold")
        ax.set_ylim([0, 100])
        ax.axhline(y=30.0, color="red", linestyle="--", alpha=0.3)
        ax.axhline(y=10.0, color="orange", linestyle="--", alpha=0.3)
        ax.grid(True, alpha=0.3, axis="y")
    metrics_to_analyze = [
        ("covariance_cond_mean_mean", "Covariance\nConditioning", 1e7, True),
        ("flow_magnitude_mean_mean", "Flow\nMagnitude", 15.0, False),
        ("ess_min_mean", "ESS Minimum", 100.0, False),
    ]
    for idx, (metric_key, metric_label, threshold, use_log) in enumerate(metrics_to_analyze):
        ax = fig.add_subplot(gs[2, idx])
        successful_values = []
        failed_values = []
        for result in all_results:
            value = result.stability_metrics.get(metric_key, float('nan'))
            if math.isfinite(value):
                if result.rmse_mean < 10.0 and result.success_rate > 0.7:
                    successful_values.append(float(value))
                else:
                    failed_values.append(float(value))
        if not successful_values and not failed_values:
            continue
        if use_log:
            all_vals = successful_values + failed_values
            vmin = max(min(all_vals), 1e-12)
            vmax = max(all_vals)
            if vmin <= 0 or vmax <= vmin:
                bins = 30
            else:
                log_min, log_max = math.log10(vmin), math.log10(vmax)
                bins = [10 ** (log_min + (log_max - log_min) * i / 29) for i in range(30)]
        else:
            bins = 30
        if successful_values:
            ax.hist(successful_values, bins=bins, alpha=0.6, label="Successful (RMSE<10)", color="green", edgecolor="black")
        if failed_values:
            ax.hist(failed_values, bins=bins, alpha=0.6, label="Failed (RMSE>10)", color="red", edgecolor="black")
        ax.axvline(x=threshold, color="blue", linestyle="--", linewidth=2, label=f"Threshold: {threshold}")
        ax.set_xlabel(metric_label.replace('\n', ' '), fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title(f"Distribution: {metric_label.replace(chr(10), ' ')}", fontsize=11, fontweight="bold")
        if use_log:
            ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle("Failure Mode Analysis: WHY Do Filters Fail?", fontsize=16, fontweight="bold")
    plt.savefig(output_dir / "failure_mode_analysis.pdf", dpi=300, bbox_inches="tight")
    plt.close()


def create_visualizations(all_results: List[ExperimentResult], summary: Dict, output_dir: Path) -> None:
    """Create full set of visualizations (no NumPy; Python/math only)."""
    if not HAS_PLOTTING:
        print("Skipping visualizations (matplotlib not available)")
        return
    print("\nGenerating visualizations...")
    scenarios = sorted(set(r.scenario_name for r in all_results))
    filters = sorted(set(r.filter_name for r in all_results))
    if not scenarios or not filters:
        return
    n_s, n_f = len(scenarios), len(filters)
    rmse_data = [[float('nan')] * n_f for _ in range(n_s)]
    for i, scenario in enumerate(scenarios):
        for j, filter_name in enumerate(filters):
            match = [r for r in all_results if r.scenario_name == scenario and r.filter_name == filter_name]
            if match:
                rmse_data[i][j] = match[0].rmse_mean
    fig, ax = plt.subplots(figsize=(max(10, n_f * 1.2), max(6, n_s * 0.4)))
    im = ax.imshow([[math.log10(v + 1) if math.isfinite(v) else 0 for v in row] for row in rmse_data],
                   cmap='RdYlGn_r', aspect='auto')
    ax.set_xticks(range(n_f))
    ax.set_yticks(range(n_s))
    ax.set_xticklabels([f.upper() for f in filters], rotation=45, ha='right')
    ax.set_yticklabels(scenarios, fontsize=8)
    ax.set_title('RMSE by Scenario and Filter (log10(RMSE+1))')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_dir / 'rmse_heatmap.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    plot_performance_radar(all_results, output_dir)
    plot_failure_diagnosis_heatmap(all_results, output_dir)
    plot_challenge_performance_profiles(all_results, output_dir)
    plot_stability_overview(all_results, output_dir)
    plot_failure_mode_analysis(all_results, output_dir)
    plot_time_series(all_results, output_dir)
    plot_dimension_scaling(all_results, output_dir)
    plot_execution_times(all_results, output_dir)
    plot_stability_diagnostic_dashboard(all_results, output_dir)
    print(f"✓ Visualizations saved to {output_dir}")


# ============================================================================
# Argument parsing and main
# ============================================================================

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Enhanced Filter Performance Analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--scenarios', nargs='+', default=['all'], help='Scenario names or "all"')
    parser.add_argument('--filters', nargs='+', default=['all'],
                        help='Filter names or "all". Options: ekf, ukf, pf, pfpf_ledh, pfpf_edh, ledh, edh, pff_scalar, pff_matrix')
    parser.add_argument('--output_dir', type=str, default='./reports/3_Deterministic_Kernel_Flow/Filters_Comparison_Diagnostics', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--ukf_alpha', type=float, default=1e-3, help='UKF alpha')
    parser.add_argument('--ukf_beta', type=float, default=2.0, help='UKF beta')
    parser.add_argument('--pf_particles', type=int, default=1000, help='PF particles')
    parser.add_argument('--n_particles', type=int, default=500, help='Particles for LEDH/EDH/PFF')
    parser.add_argument('--n_lambda', type=int, default=29, help='Lambda samples for LEDH/EDH')
    parser.add_argument('--pff_step_size', type=float, default=0.1, help='PFF step size')
    parser.add_argument('--pff_localization_radius', type=float, default=5.0, help='PFF localization radius')
    parser.add_argument('--pff_max_steps', type=int, default=50, help='PFF max steps')
    parser.add_argument('--pff_matrix_step_multiplier', type=float, default=1.5, help='PFF matrix step multiplier')
    parser.add_argument('--pff_matrix_loc_multiplier', type=float, default=1.2, help='PFF matrix localization multiplier')
    parser.add_argument('--pff_matrix_steps_multiplier', type=float, default=1.5, help='PFF matrix steps multiplier')
    parser.add_argument('--edh_redraw', action='store_true', help='EDH particle redraw')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--show_progress', action='store_true', help='Show step progress')
    parser.add_argument('--skip_plots', action='store_true', help='Skip plots')
    parser.add_argument('--n_jobs', type=int, default=1, help='Parallel workers (1=sequential, 0=all CPUs)')
    return parser.parse_args()


def main():
    """Main execution."""
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 80)
    print("ENHANCED FILTER PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Random seed: {args.seed}\n")

    all_scenarios = create_scenario_configs()
    if len(args.scenarios) == 1 and args.scenarios[0].lower() == 'all':
        scenarios = all_scenarios
    else:
        scenario_names = {s.strip().lower() for s in args.scenarios}
        scenarios = [s for s in all_scenarios if s.name.lower() in scenario_names]
        if not scenarios:
            print(f"Error: No matching scenarios for: {args.scenarios}")
            return
    print(f"Running {len(scenarios)} scenarios")

    available_filters = ['ekf', 'ukf', 'pf', 'pfpf_ledh', 'pfpf_edh', 'ledh', 'edh', 'pff_scalar', 'pff_matrix']
    if len(args.filters) == 1 and args.filters[0].lower() == 'all':
        filters = available_filters
    else:
        filters = [f.strip().lower() for f in args.filters]
        invalid = [f for f in filters if f not in available_filters]
        if invalid:
            print(f"Error: Invalid filters: {invalid}. Available: {available_filters}")
            return
    print(f"Testing filters: {', '.join(filters)}\n")

    n_jobs = (os.cpu_count() or 1) if args.n_jobs <= 0 else args.n_jobs
    all_results: List[ExperimentResult] = []

    if n_jobs > 1:
        args_dict = {
            'seed': args.seed, 'ukf_alpha': args.ukf_alpha, 'ukf_beta': args.ukf_beta,
            'pf_particles': args.pf_particles, 'n_particles': args.n_particles, 'n_lambda': args.n_lambda,
            'pff_step_size': args.pff_step_size, 'pff_localization_radius': args.pff_localization_radius,
            'pff_max_steps': args.pff_max_steps, 'pff_matrix_step_multiplier': args.pff_matrix_step_multiplier,
            'pff_matrix_loc_multiplier': args.pff_matrix_loc_multiplier,
            'pff_matrix_steps_multiplier': args.pff_matrix_steps_multiplier,
            'edh_redraw': args.edh_redraw, 'verbose': args.verbose, 'show_progress': args.show_progress,
        }
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_scenario = {
                executor.submit(_run_scenario_worker, asdict(s), filters, args_dict, idx): s
                for idx, s in enumerate(scenarios)
            }
            completed = tqdm(as_completed(future_to_scenario), total=len(scenarios), desc="Scenarios", unit="scenario") if HAS_TQDM else as_completed(future_to_scenario)
            for future in completed:
                try:
                    all_results.extend(future.result())
                except Exception as e:
                    print(f"\nError: {e}")
                    if args.verbose:
                        import traceback
                        traceback.print_exc()
    else:
        for scenario in (tqdm(scenarios, desc="Scenarios", unit="scenario") if HAS_TQDM else scenarios):
            try:
                all_results.extend(run_scenario_experiment(scenario, filters, args))
            except Exception as e:
                print(f"\nError in scenario {scenario.name}: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()

    if not all_results:
        print("\nNo results generated. Exiting.")
        return
    print(f"\nCollected {len(all_results)} experiment results")
    print("\nAnalyzing results...")
    summary = analyze_results(all_results, scenarios, output_dir)
    if not args.skip_plots and HAS_PLOTTING:
        create_visualizations(all_results, summary, output_dir)
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()

# python -m src.experiments.exp_filters_comparison_diagnostics --filters pfpf_ledh pfpf_edh ledh edh pff_scalar pff_matrix