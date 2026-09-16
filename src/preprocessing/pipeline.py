"""Preprocessing Pipeline Skeleton.

Educational Simulation System.
Prevents data leakage by fitting scalers and encoders strictly on the training set.
"""

from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.preprocessing.feature_engineering import FeatureEngineer
from src.utils.logger import setup_logger

logger = setup_logger("preprocessing_pipeline")


class PreprocessingPipeline:
    """End-to-end preprocessing pipeline for tabular health simulation data."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize pipeline with configuration.

        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.feature_engineer = FeatureEngineer(config)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []

    def fit(self, df: pd.DataFrame, target_col: str = "risk_class") -> "PreprocessingPipeline":
        """Fit imputer and scaler parameters strictly on training DataFrame to prevent leakage.

        Args:
            df: Training DataFrame.
            target_col: Target column name.

        Returns:
            Self instance.
        """
        # Feature engineering first
        df_engineered = self.feature_engineer.transform(df)

        feature_cols = [c for c in df_engineered.columns if c not in ["case_id", target_col]]
        self.numeric_cols = df_engineered[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df_engineered[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()

        if self.numeric_cols:
            self.imputer.fit(df_engineered[self.numeric_cols])
            imputed_num = self.imputer.transform(df_engineered[self.numeric_cols])
            self.scaler.fit(imputed_num)

        self.is_fitted = True
        logger.info(f"Fitted PreprocessingPipeline on {len(df)} training records.")
        return self

    def transform(self, df: pd.DataFrame, target_col: str = "risk_class") -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transform DataFrame using fitted parameters.

        Args:
            df: DataFrame to transform.
            target_col: Target column name.

        Returns:
            Tuple of (X feature matrix as np.ndarray, y target array or None).
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling transform!")

        df_engineered = self.feature_engineer.transform(df)

        if self.numeric_cols:
            imputed_num = self.imputer.transform(df_engineered[self.numeric_cols])
            scaled_num = self.scaler.transform(imputed_num)
        else:
            scaled_num = np.empty((len(df), 0))

        X = scaled_num

        y = None
        if target_col in df_engineered.columns:
            y = df_engineered[target_col].values

        return X, y
