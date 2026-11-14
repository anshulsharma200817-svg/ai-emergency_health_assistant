"""Synthetic Physiological Signal Data Generator.

Educational Software Simulation.
Generates realistic ECG-like waveforms using mathematical wave morphology:
  - P wave       : Gaussian pulse
  - QRS complex  : Differentiated Gaussian (narrow, high-amplitude spike)
  - T wave       : Gaussian pulse (broader, lower amplitude)
  - Baseline wander: Low-frequency sinusoidal drift
  - Powerline noise: 50 Hz additive sinusoid
  - Measurement noise: Gaussian white noise

Signal categories (4 rhythm types):
  NORMAL_SINUS      → HR ~70 bpm, regular RR intervals
  BRADY_PATTERN     → HR ~48 bpm, longer RR intervals
  TACHY_PATTERN     → HR ~120 bpm, shorter RR intervals
  IRREGULAR_PATTERN → HR ~85 bpm, variable RR intervals (AF-like)
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

from src.utils.seed import set_seed
from src.utils.logger import setup_logger

logger = setup_logger("signal_generator")


# ──────────────────────────────────────────────────────────────────────────────
# Signal pattern definitions
# ──────────────────────────────────────────────────────────────────────────────

PATTERN_CONFIG: Dict[str, Dict[str, Any]] = {
    "NORMAL_SINUS": {
        "hr_mean":   70.0,
        "hr_std":    4.0,
        "rr_jitter": 0.02,     # fraction of RR interval
        "p_amp":     0.20,
        "qrs_amp":   1.00,
        "t_amp":     0.35,
        "noise_std": 0.015,
        "wander_amp": 0.04,
        "label":     0,
    },
    "BRADY_PATTERN": {
        "hr_mean":   45.0,
        "hr_std":    4.0,
        "rr_jitter": 0.025,
        "p_amp":     0.22,
        "qrs_amp":   1.05,
        "t_amp":     0.38,
        "noise_std": 0.018,
        "wander_amp": 0.05,
        "label":     1,
    },
    "TACHY_PATTERN": {
        "hr_mean":   122.0,
        "hr_std":    8.0,
        "rr_jitter": 0.03,
        "p_amp":     0.15,     # P wave merges with preceding T
        "qrs_amp":   0.90,
        "t_amp":     0.28,
        "noise_std": 0.022,
        "wander_amp": 0.06,
        "label":     2,
    },
    "IRREGULAR_PATTERN": {
        "hr_mean":   85.0,
        "hr_std":    18.0,    # High std → irregular intervals
        "rr_jitter": 0.15,    # Very high jitter → AF-like
        "p_amp":     0.05,    # Absent/tiny P waves (AF)
        "qrs_amp":   0.85,
        "t_amp":     0.25,
        "noise_std": 0.025,
        "wander_amp": 0.08,
        "label":     3,
    },
}

ALL_PATTERNS = list(PATTERN_CONFIG.keys())
SAMPLING_RATE = 500  # Hz (samples per second) — standard ECG sampling rate


def _gaussian_pulse(t: np.ndarray, center: float, width: float, amplitude: float) -> np.ndarray:
    """Generate a Gaussian pulse.

    Args:
        t: Time array.
        center: Center time of the pulse (seconds).
        width: Width (sigma) of the Gaussian.
        amplitude: Peak amplitude.

    Returns:
        np.ndarray of same shape as t.
    """
    return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)


def _qrs_complex(t: np.ndarray, center: float, amplitude: float) -> np.ndarray:
    """Generate a QRS-like complex using a narrow Gaussian derivative.

    This produces the characteristic sharp spike of the QRS complex.

    Args:
        t: Time array.
        center: Center time of QRS peak (seconds).
        amplitude: Scaling amplitude.

    Returns:
        np.ndarray of same shape as t.
    """
    width = 0.025  # 25 ms QRS duration
    # Q dip
    q = -0.12 * amplitude * _gaussian_pulse(t, center - 0.02, width * 0.8, 1.0)
    # R peak (main spike)
    r = amplitude * _gaussian_pulse(t, center, width * 0.5, 1.0)
    # S dip
    s = -0.15 * amplitude * _gaussian_pulse(t, center + 0.022, width * 0.7, 1.0)
    return q + r + s


class SignalDataGenerator:
    """Synthetic ECG-like physiological signal generator for CNN experiments.

    Generates waveforms with realistic P-QRS-T morphology, baseline wander,
    powerline interference, and Gaussian measurement noise.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize signal generator.

        Args:
            config: Configuration dictionary loaded from config.yaml.
        """
        self.config = config
        self.seed = config.get("random_seed", 42)
        self.signal_length = config.get("signal_length", SAMPLING_RATE)  # default 500 samples = 1s
        set_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)
        logger.info(f"Initialized SignalDataGenerator | seed={self.seed}, signal_length={self.signal_length}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_rr_intervals(
        self,
        total_duration: float,
        hr_mean: float,
        hr_std: float,
        rr_jitter: float,
    ) -> List[float]:
        """Compute a list of RR intervals spanning total_duration.

        Args:
            total_duration: Total signal duration in seconds.
            hr_mean: Mean heart rate in BPM.
            hr_std: Standard deviation of heart rate.
            rr_jitter: Fractional jitter applied to each RR interval.

        Returns:
            List of RR intervals (seconds).
        """
        base_hr = float(self.rng.normal(hr_mean, hr_std))
        base_hr = max(25.0, min(220.0, base_hr))  # clamp
        base_rr = 60.0 / base_hr  # seconds per beat

        rr_intervals = []
        cumulative = 0.0
        while cumulative < total_duration:
            jitter = float(self.rng.normal(0, rr_jitter * base_rr))
            rr = max(0.25, base_rr + jitter)  # min RR = 0.25s (240 bpm max)
            rr_intervals.append(rr)
            cumulative += rr

        return rr_intervals

    def _generate_beat(
        self,
        t: np.ndarray,
        beat_center: float,
        rr: float,
        cfg: Dict[str, Any],
    ) -> np.ndarray:
        """Generate a single PQRST cardiac beat.

        Args:
            t: Full time array for the signal.
            beat_center: Center time of the QRS complex for this beat.
            rr: RR interval (seconds) — used to position P and T waves.
            cfg: Pattern configuration dict.

        Returns:
            np.ndarray signal contribution from this beat.
        """
        signal = np.zeros_like(t)

        # P wave: 160ms before QRS
        p_center = beat_center - 0.16
        if p_center > 0:
            signal += _gaussian_pulse(t, p_center, 0.04, cfg["p_amp"])

        # QRS complex
        signal += _qrs_complex(t, beat_center, cfg["qrs_amp"])

        # T wave: 200ms after QRS (wider than P)
        t_center = beat_center + 0.20
        signal += _gaussian_pulse(t, t_center, 0.07, cfg["t_amp"])

        return signal

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_single_signal(
        self,
        signal_length: Optional[int] = None,
        pattern_category: str = "NORMAL_SINUS",
    ) -> Tuple[np.ndarray, str]:
        """Generate a single synthetic ECG-like waveform signal sample.

        Args:
            signal_length: Number of time samples (default: self.signal_length).
            pattern_category: Signal morphology category.

        Returns:
            Tuple of (signal array of shape (signal_length,), pattern_category).
        """
        if signal_length is None:
            signal_length = self.signal_length

        if pattern_category not in PATTERN_CONFIG:
            raise ValueError(f"Unknown pattern: {pattern_category}. "
                             f"Must be one of {ALL_PATTERNS}")

        cfg = PATTERN_CONFIG[pattern_category]
        duration = signal_length / SAMPLING_RATE  # seconds

        # Time array
        t = np.linspace(0, duration, signal_length, endpoint=False)

        # Initialize signal
        signal = np.zeros(signal_length, dtype=float)

        # Generate RR intervals and beat positions
        rr_intervals = self._compute_rr_intervals(
            duration, cfg["hr_mean"], cfg["hr_std"], cfg["rr_jitter"]
        )

        # Place beats
        beat_time = rr_intervals[0] * 0.3  # first beat slightly offset
        for rr in rr_intervals:
            if beat_time < duration:
                signal += self._generate_beat(t, beat_time, rr, cfg)
            beat_time += rr

        # Baseline wander (low-frequency drift, 0.15–0.4 Hz)
        wander_freq = float(self.rng.uniform(0.15, 0.4))
        wander_phase = float(self.rng.uniform(0, 2 * np.pi))
        signal += cfg["wander_amp"] * np.sin(2 * np.pi * wander_freq * t + wander_phase)

        # Powerline interference (50 Hz)
        pl_amp = float(self.rng.uniform(0.005, 0.02))
        signal += pl_amp * np.sin(2 * np.pi * 50 * t)

        # Gaussian measurement noise
        signal += self.rng.normal(0.0, cfg["noise_std"], signal_length)

        return signal.astype(np.float32), pattern_category

    def generate_signal_batch(
        self,
        num_samples: Optional[int] = None,
        signal_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a batch of synthetic physiological signal samples.

        Args:
            num_samples: Number of signal samples. Defaults to
                config planned_dataset_sizes.signals.
            signal_length: Length of each signal sequence.

        Returns:
            Tuple of:
                X → np.ndarray shape (num_samples, 1, signal_length) — CNN input
                y → np.ndarray shape (num_samples,) — integer class labels
        """
        if num_samples is None:
            num_samples = self.config.get("planned_dataset_sizes", {}).get("signals", 20000)
        if signal_length is None:
            signal_length = self.signal_length

        X_list: List[np.ndarray] = []
        y_list: List[int]        = []

        for i in range(num_samples):
            # Balanced rotation across patterns
            cat = ALL_PATTERNS[i % len(ALL_PATTERNS)]
            sig, _ = self.generate_single_signal(signal_length, cat)
            X_list.append(sig)
            y_list.append(PATTERN_CONFIG[cat]["label"])

            if (i + 1) % 5000 == 0:
                logger.info(f"  Generated {i+1}/{num_samples} signals...")

        X = np.array(X_list, dtype=np.float32)[:, np.newaxis, :]  # (N, 1, L)
        y = np.array(y_list, dtype=np.int64)

        label_counts = dict(zip(*np.unique(y, return_counts=True)))
        logger.info(
            f"Generated signal batch: X={X.shape}, y={y.shape}. "
            f"Class distribution: {label_counts}"
        )
        return X, y
