"""Synthetic Temporal Sequence Data Generator.

Educational Software Simulation.
Generates multi-timestep physiological trajectory sequences using
Markov latent state dynamics and transition matrices. Each trajectory
represents a patient's vital-signs evolution over 60 time steps.

Physiological channels (6):
  0: heart_rate
  1: systolic_bp
  2: diastolic_bp
  3: spo2
  4: temperature
  5: respiratory_rate
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger("temporal_generator")


# ──────────────────────────────────────────────────────────────────────────────
# Latent physiological states (index → clinical interpretation)
# ──────────────────────────────────────────────────────────────────────────────
# State 0: Stable/Normal         → low HR, normal BP, good SpO2
# State 1: Mild Stress           → slightly elevated HR, mild BP drop
# State 2: Moderate Deterioration→ tachycardia, hypotension developing
# State 3: Severe/Critical       → high HR, very low BP, low SpO2

# Emission means and std per state × feature (6 features)
STATE_EMISSIONS: np.ndarray = np.array([
    # HR,   SBP,   DBP,  SpO2,  Temp,  RR
    [72.0, 122.0, 80.0, 98.5, 36.7, 15.0],   # State 0: Stable
    [92.0, 108.0, 70.0, 96.0, 37.5, 19.0],   # State 1: Mild Stress
    [115.0, 90.0, 58.0, 92.0, 38.5, 26.0],   # State 2: Moderate Deterioration
    [138.0, 68.0, 42.0, 83.0, 39.5, 33.0],   # State 3: Severe/Critical
], dtype=float)

STATE_STD: np.ndarray = np.array([
    [5.0,  8.0, 5.0, 0.6, 0.2, 1.5],   # State 0
    [8.0,  9.0, 6.0, 1.2, 0.4, 2.5],   # State 1
    [10.0, 10.0, 7.0, 1.8, 0.5, 3.5],  # State 2
    [14.0, 12.0, 8.0, 3.0, 0.6, 4.5],  # State 3
], dtype=float)

# Hard physiological clamps per feature
FEATURE_CLAMPS: List[Tuple[float, float]] = [
    (25.0, 240.0),   # heart_rate
    (45.0, 255.0),   # systolic_bp
    (25.0, 155.0),   # diastolic_bp
    (55.0, 100.0),   # spo2
    (30.5, 44.5),    # temperature
    (5.0,  58.0),    # respiratory_rate
]

FEATURE_NAMES = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2", "temperature", "respiratory_rate"]

# ──────────────────────────────────────────────────────────────────────────────
# Transition matrices per trajectory type (4×4 stochastic matrices)
# Rows = current state, Cols = next state
# ──────────────────────────────────────────────────────────────────────────────
TRANSITION_MATRICES: Dict[str, np.ndarray] = {
    # Stays in stable state with high probability
    "stable": np.array([
        [0.95, 0.04, 0.01, 0.00],
        [0.10, 0.85, 0.04, 0.01],
        [0.02, 0.10, 0.83, 0.05],
        [0.01, 0.03, 0.10, 0.86],
    ]),
    # Gradual drift toward worse states
    "gradual_deterioration": np.array([
        [0.70, 0.25, 0.04, 0.01],
        [0.05, 0.65, 0.25, 0.05],
        [0.01, 0.08, 0.62, 0.29],
        [0.00, 0.02, 0.10, 0.88],
    ]),
    # Fast transition into critical states
    "rapid_deterioration": np.array([
        [0.40, 0.35, 0.18, 0.07],
        [0.02, 0.38, 0.38, 0.22],
        [0.01, 0.04, 0.35, 0.60],
        [0.00, 0.01, 0.05, 0.94],
    ]),
    # Improvement trajectory (critical → stable)
    "recovery": np.array([
        [0.92, 0.06, 0.01, 0.01],
        [0.25, 0.68, 0.06, 0.01],
        [0.18, 0.28, 0.50, 0.04],
        [0.05, 0.20, 0.35, 0.40],
    ]),
    # High noise, unpredictable transitions
    "noisy": np.array([
        [0.50, 0.20, 0.20, 0.10],
        [0.15, 0.45, 0.25, 0.15],
        [0.10, 0.20, 0.45, 0.25],
        [0.08, 0.12, 0.25, 0.55],
    ]),
    # Mixed: normal → stress → partial recovery → worse
    "mixed": np.array([
        [0.65, 0.25, 0.08, 0.02],
        [0.15, 0.55, 0.25, 0.05],
        [0.08, 0.22, 0.55, 0.15],
        [0.03, 0.10, 0.28, 0.59],
    ]),
}

# Trajectory type → risk class label mapping
TRAJ_TO_LABEL: Dict[str, int] = {
    "stable":                0,  # NORMAL
    "gradual_deterioration": 1,  # MODERATE
    "recovery":              1,  # MODERATE (improving but was ill)
    "noisy":                 1,  # MODERATE (uncertain)
    "mixed":                 2,  # HIGH
    "rapid_deterioration":   3,  # CRITICAL
}

# Initial state probabilities per trajectory type
INITIAL_STATE_PROBS: Dict[str, np.ndarray] = {
    "stable":                np.array([0.90, 0.08, 0.02, 0.00]),
    "gradual_deterioration": np.array([0.55, 0.30, 0.12, 0.03]),
    "rapid_deterioration":   np.array([0.20, 0.30, 0.30, 0.20]),
    "recovery":              np.array([0.05, 0.15, 0.40, 0.40]),
    "noisy":                 np.array([0.40, 0.25, 0.20, 0.15]),
    "mixed":                 np.array([0.45, 0.30, 0.18, 0.07]),
}

ALL_TRAJECTORY_TYPES = list(TRANSITION_MATRICES.keys())


class TemporalDataGenerator:
    """Synthetic temporal trajectory generator for RNN/LSTM experiments.

    Uses Markov latent state dynamics to produce realistic multi-step
    physiological time series. Each generated trajectory is labeled by
    its trajectory type → risk class mapping.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize temporal trajectory generator.

        Args:
            config: Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.seed = config.get("random_seed", 42)
        self.seq_len = config.get("sequence_length", 60)
        set_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        logger.info(
            f"Initialized TemporalDataGenerator | seed={self.seed}, seq_len={self.seq_len}"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _next_state(self, current_state: int, transition_matrix: np.ndarray) -> int:
        """Sample next latent state from transition probabilities.

        Args:
            current_state: Current state index (0–3).
            transition_matrix: 4×4 row-stochastic transition matrix.

        Returns:
            Next state index.
        """
        row = transition_matrix[current_state]
        return int(self.rng.choice(4, p=row))

    def _emit_observation(self, state: int, extra_noise_scale: float = 1.0) -> np.ndarray:
        """Emit a single 6-dimensional observation from a latent state.

        Args:
            state: Latent state index (0–3).
            extra_noise_scale: Multiplier on state std (1.0 = normal, >1.0 = noisier).

        Returns:
            np.ndarray of shape (6,) with clamped physiological values.
        """
        means = STATE_EMISSIONS[state]
        stds  = STATE_STD[state] * extra_noise_scale
        obs = self.rng.normal(means, stds).astype(float)

        # Clamp each feature to physiological limits
        for i, (lo, hi) in enumerate(FEATURE_CLAMPS):
            obs[i] = max(lo, min(hi, obs[i]))

        # Enforce diastolic < systolic
        if obs[2] >= obs[1]:
            obs[2] = max(FEATURE_CLAMPS[2][0], obs[1] - self.rng.uniform(8, 20))

        return obs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_trajectory(self, trajectory_type: str = "stable") -> np.ndarray:
        """Generate a single patient physiological trajectory via Markov dynamics.

        Args:
            trajectory_type: Trajectory progression pattern. One of:
                'stable', 'gradual_deterioration', 'rapid_deterioration',
                'recovery', 'noisy', 'mixed'.

        Returns:
            np.ndarray of shape (seq_len, 6).
        """
        if trajectory_type not in TRANSITION_MATRICES:
            raise ValueError(f"Unknown trajectory type: {trajectory_type}. "
                             f"Must be one of {ALL_TRAJECTORY_TYPES}")

        trans_mat = TRANSITION_MATRICES[trajectory_type]
        init_probs = INITIAL_STATE_PROBS[trajectory_type]
        noise_scale = 1.5 if trajectory_type == "noisy" else 1.0

        # Sample initial state
        state = int(self.rng.choice(4, p=init_probs))

        trajectory = np.zeros((self.seq_len, 6), dtype=float)
        for t in range(self.seq_len):
            trajectory[t] = self._emit_observation(state, noise_scale)
            state = self._next_state(state, trans_mat)

        return trajectory

    def generate_batch(
        self,
        num_trajectories: Optional[int] = None,
        trajectory_types: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate a batch of patient trajectories with labels and IDs.

        Balances across trajectory types to produce class-balanced labels.

        Args:
            num_trajectories: Total number of trajectories. Defaults to
                config planned_dataset_sizes.temporal.
            trajectory_types: Explicit list of types to use (cycling). If None,
                uses all 6 types in balanced rotation.

        Returns:
            Tuple of:
                X  → np.ndarray (N, seq_len, 6)  — trajectory feature arrays
                y  → np.ndarray (N,)             — integer risk class labels
                ids → np.ndarray (N,)            — trajectory ID strings
        """
        if num_trajectories is None:
            num_trajectories = self.config.get("planned_dataset_sizes", {}).get("temporal", 20000)

        if trajectory_types is None:
            trajectory_types = ALL_TRAJECTORY_TYPES  # 6 types

        X_list:   List[np.ndarray] = []
        y_list:   List[int]        = []
        id_list:  List[str]        = []

        for i in range(num_trajectories):
            t_type = trajectory_types[i % len(trajectory_types)]
            traj   = self.generate_trajectory(t_type)
            label  = TRAJ_TO_LABEL[t_type]

            X_list.append(traj)
            y_list.append(label)
            id_list.append(f"TRAJ_{i+1:06d}")

            if (i + 1) % 5000 == 0:
                logger.info(f"  Generated {i+1}/{num_trajectories} trajectories...")

        X   = np.array(X_list,  dtype=np.float32)
        y   = np.array(y_list,  dtype=np.int64)
        ids = np.array(id_list, dtype=str)

        label_counts = dict(zip(*np.unique(y, return_counts=True)))
        logger.info(
            f"Generated temporal batch: X={X.shape}, y={y.shape}. "
            f"Class distribution: {label_counts}"
        )
        return X, y, ids
