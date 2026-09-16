"""Feature Engineering Module for Derived Physiological Features.

Educational Simulation System.
Computes domain-specific derived clinical features without data leakage.
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from src.data.schema import DERIVED_FEATURES
from src.utils.logger import setup_logger

logger = setup_logger("feature_engineering")


class FeatureEngineer:
    """Computes physiological derived features from raw tabular features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize FeatureEngineer.

        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived physiological features on DataFrame.

        Derived features:
        - pulse_pressure: systolic_bp - diastolic_bp
        - mean_arterial_pressure: diastolic_bp + (pulse_pressure / 3)
        - shock_index: heart_rate / systolic_bp
        - age_group: Categorical age bracket ('young', 'adult', 'senior')
        - spo2_severity: Categorical SpO2 severity ('normal', 'mild_hypoxia', 'severe_hypoxia')
        - tachycardia_flag: 1 if HR > 100 else 0
        - hypotension_flag: 1 if SBP < 90 else 0
        - fever_flag: 1 if Temp > 38.0 else 0
        - tachypnea_flag: 1 if RR > 20 else 0

        Args:
            df: Raw pandas DataFrame with physiological variables.

        Returns:
            pandas DataFrame enriched with derived feature columns.
        """
        out = df.copy()

        # 1. pulse_pressure
        if "systolic_bp" in out.columns and "diastolic_bp" in out.columns:
            out["pulse_pressure"] = out["systolic_bp"] - out["diastolic_bp"]
        else:
            out["pulse_pressure"] = np.nan

        # 2. mean_arterial_pressure
        if "diastolic_bp" in out.columns and "pulse_pressure" in out.columns:
            out["mean_arterial_pressure"] = out["diastolic_bp"] + (out["pulse_pressure"] / 3.0)
        else:
            out["mean_arterial_pressure"] = np.nan

        # 3. shock_index
        if "heart_rate" in out.columns and "systolic_bp" in out.columns:
            # Avoid division by zero
            sbp_safe = out["systolic_bp"].replace(0, np.nan)
            out["shock_index"] = out["heart_rate"] / sbp_safe
        else:
            out["shock_index"] = np.nan

        # 4. age_group
        if "age" in out.columns:
            out["age_group"] = pd.cut(
                out["age"],
                bins=[-np.inf, 18, 65, np.inf],
                labels=["child", "adult", "senior"]
            ).astype(str)
        else:
            out["age_group"] = "unknown"

        # 5. spo2_severity
        if "spo2" in out.columns:
            out["spo2_severity"] = pd.cut(
                out["spo2"],
                bins=[-np.inf, 88, 94, np.inf],
                labels=["severe_hypoxia", "mild_hypoxia", "normal"]
            ).astype(str)
        else:
            out["spo2_severity"] = "unknown"

        # 6. tachycardia_flag
        if "heart_rate" in out.columns:
            out["tachycardia_flag"] = (out["heart_rate"] > 100).astype(int)
        else:
            out["tachycardia_flag"] = 0

        # 7. hypotension_flag
        if "systolic_bp" in out.columns:
            out["hypotension_flag"] = (out["systolic_bp"] < 90).astype(int)
        else:
            out["hypotension_flag"] = 0

        # 8. fever_flag
        if "temperature" in out.columns:
            out["fever_flag"] = (out["temperature"] > 38.0).astype(int)
        else:
            out["fever_flag"] = 0

        # 9. tachypnea_flag
        if "respiratory_rate" in out.columns:
            out["tachypnea_flag"] = (out["respiratory_rate"] > 20).astype(int)
        else:
            out["tachypnea_flag"] = 0

        logger.info(f"Computed {len(DERIVED_FEATURES)} derived features. Columns: {out.columns.tolist()}")
        return out
