"""Preprocessing Pipeline.

Educational Simulation System.
Prevents data leakage by fitting scalers and encoders strictly on the
training set. Provides fit / transform / fit_transform / save / load API.
"""

from typing import Dict, Any, Tuple, Optional, List
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

from src.preprocessing.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing_pipeline")

# Risk class → integer mapping (alphabetical order ensures consistency)
RISK_CLASS_ORDER = ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]
RISK_CLASS_TO_INT = {c: i for i, c in enumerate(RISK_CLASS_ORDER)}
INT_TO_RISK_CLASS = {i: c for c, i in RISK_CLASS_TO_INT.items()}


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for tabular health simulation data.

    Steps:
      1. Feature engineering (derive pulse_pressure, shock_index, flags, etc.)
      2. Median imputation of missing numeric values
      3. Standard scaling of numeric features
      4. Label encoding of risk_class → integer (0–3)
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(RISK_CLASS_ORDER)
        self.is_fitted = False
        self.numeric_cols: List[str] = []

    # ------------------------------------------------------------------
    # Core fit / transform
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame, target_col: str = "risk_class") -> "PreprocessingPipeline":
        """Fit imputer and scaler parameters strictly on training DataFrame.

        Args:
            df: Training DataFrame.
            target_col: Target column name.

        Returns:
            Self instance (for chaining).
        """
        df_eng = self.feature_engineer.transform(df)

        # Identify numeric feature columns (exclude id + target)
        exclude = {"case_id", target_col}
        self.numeric_cols = [
            c for c in df_eng.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df_eng[c])
        ]

        # Fit imputer then scaler (no data leakage — on train only)
        imputed = self.imputer.fit_transform(df_eng[self.numeric_cols])
        self.scaler.fit(imputed)

        self.is_fitted = True
        logger.info(
            f"Fitted PreprocessingPipeline on {len(df)} training records. "
            f"Numeric features: {len(self.numeric_cols)}"
        )
        return self

    def transform(
        self, df: pd.DataFrame, target_col: str = "risk_class"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform DataFrame using fitted parameters.

        Args:
            df: DataFrame to transform.
            target_col: Target column name.

        Returns:
            Tuple of (X feature matrix as np.ndarray float32,
                      y integer target array or None).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform!")

        df_eng = self.feature_engineer.transform(df)

        imputed = self.imputer.transform(df_eng[self.numeric_cols])
        X = self.scaler.transform(imputed).astype(np.float32)

        y = None
        if target_col in df_eng.columns:
            y = df_eng[target_col].map(RISK_CLASS_TO_INT).values.astype(np.int64)

        return X, y

    def fit_transform(
        self, df: pd.DataFrame, target_col: str = "risk_class"
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit on df and immediately transform it.

        Args:
            df: Training DataFrame.
            target_col: Target column name.

        Returns:
            Tuple of (X, y) as numpy arrays.
        """
        self.fit(df, target_col)
        return self.transform(df, target_col)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist fitted pipeline to disk.

        Args:
            path: File path ending in .pkl.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved PreprocessingPipeline to {path}")

    @staticmethod
    def load(path: str) -> "PreprocessingPipeline":
        """Load a previously saved pipeline from disk.

        Args:
            path: File path to the pickled pipeline.

        Returns:
            Loaded PreprocessingPipeline instance.
        """
        with open(path, "rb") as f:
            pipeline = pickle.load(f)
        logger.info(f"Loaded PreprocessingPipeline from {path}")
        return pipeline

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def decode_labels(y: np.ndarray) -> List[str]:
        """Convert integer labels back to risk class strings.

        Args:
            y: Integer label array.

        Returns:
            List of risk class strings.
        """
        return [INT_TO_RISK_CLASS.get(int(i), "UNKNOWN") for i in y]

    @property
    def num_features(self) -> int:
        """Return number of numeric features in fitted pipeline."""
        return len(self.numeric_cols)
