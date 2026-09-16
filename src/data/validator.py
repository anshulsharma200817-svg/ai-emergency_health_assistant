"""Data Validation Module Skeleton.

Educational Software Simulation.
Provides schema validation, range checks, missing value reporting, outlier detection, and duplicate checks.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.data.schema import RAW_FEATURES, TARGET_RISK_CLASSES, PHYSIOLOGICAL_RANGES
from src.utils.logger import setup_logger

logger = setup_logger("data_validator")


class DataValidator:
    """Validator class for tabular health simulation dataset validation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize data validator with configuration.

        Args:
            config: Configuration dictionary loaded from config.yaml.
        """
        self.config = config

    def validate_schema(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate DataFrame schema against required raw features and target class.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Tuple of (is_valid, list_of_error_strings).
        """
        errors = []
        missing_cols = set(RAW_FEATURES) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns in dataset: {sorted(list(missing_cols))}")

        if "risk_class" in df.columns:
            invalid_classes = set(df["risk_class"].unique()) - set(TARGET_RISK_CLASSES)
            if invalid_classes:
                errors.append(f"Found invalid target risk classes: {invalid_classes}")

        is_valid = len(errors) == 0
        return is_valid, errors

    def check_physiological_ranges(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform boundary range validation on numerical physiological variables.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Dict mapping feature names to out-of-bounds counts and details.
        """
        range_report = {}
        for col, (min_v, max_v) in PHYSIOLOGICAL_RANGES.items():
            if col in df.columns:
                out_of_bounds = df[(df[col] < min_v) | (df[col] > max_v)]
                range_report[col] = {
                    "valid_min": min_v,
                    "valid_max": max_v,
                    "actual_min": float(df[col].min()) if not df[col].empty else None,
                    "actual_max": float(df[col].max()) if not df[col].empty else None,
                    "out_of_bounds_count": len(out_of_bounds)
                }
        return range_report

    def check_duplicates(self, df: pd.DataFrame, subset_cols: Optional[List[str]] = None) -> int:
        """Check for duplicate records in DataFrame.

        Args:
            df: Input DataFrame.
            subset_cols: Specific columns to evaluate duplicates on (default: all columns except case_id).

        Returns:
            Integer count of duplicate records.
        """
        if subset_cols is None:
            subset_cols = [c for c in df.columns if c != "case_id"]
        duplicate_count = int(df.duplicated(subset=subset_cols).sum())
        logger.info(f"Duplicate records detected: {duplicate_count}")
        return duplicate_count

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data quality report dictionary.

        Args:
            df: Input pandas DataFrame.

        Returns:
            Dict containing quality check metrics (missing values, duplicates, range checks, class distribution).
        """
        # TODO: Expand full quality metrics reporting for interim/processed datasets.
        is_valid, schema_errors = self.validate_schema(df)
        report = {
            "num_records": len(df),
            "num_features": len(df.columns),
            "schema_valid": is_valid,
            "schema_errors": schema_errors,
            "class_distribution": df["risk_class"].value_counts().to_dict() if "risk_class" in df.columns else {},
            "missing_values": df.isnull().sum().to_dict(),
            "duplicate_count": self.check_duplicates(df),
            "range_validation": self.check_physiological_ranges(df)
        }
        return report
