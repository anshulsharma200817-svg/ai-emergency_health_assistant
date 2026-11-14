"""Explainability module — SHAP-based feature importance for classical ML models.

Educational Simulation System.
Provides SHAP summary values and plots for trained Random Forest models.
"""

from typing import Dict, Any, List, Optional
import os
import json
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("explainability")


class SHAPExplainer:
    """SHAP-based feature importance explainer for classical ML models."""

    def __init__(self, model: Any, feature_names: List[str], config: Dict[str, Any]):
        """Initialize SHAP explainer.

        Args:
            model: Trained sklearn model (e.g. RandomForestClassifier).
            feature_names: List of feature column names.
            config: Configuration dictionary.
        """
        self.model = model
        self.feature_names = feature_names
        self.config = config
        self.shap_enabled = config.get("explainability", {}).get("shap_enabled", True)

    def compute_shap_values(
        self, X: np.ndarray, max_samples: int = 500
    ) -> Optional[Any]:
        """Compute SHAP values for feature importance analysis.

        Args:
            X: Feature matrix to explain (uses subset for speed).
            max_samples: Maximum background samples for SHAP TreeExplainer.

        Returns:
            SHAP values array or None if shap not installed.
        """
        if not self.shap_enabled:
            logger.info("SHAP explainability disabled in config.")
            return None

        try:
            import shap
        except ImportError:
            logger.warning("shap package not installed. Run: pip install shap")
            return None

        # Use subset for speed
        n = min(max_samples, X.shape[0])
        X_subset = X[:n]

        logger.info(f"Computing SHAP values for {n} samples...")
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_subset)
        logger.info("SHAP values computed.")
        return shap_values

    def feature_importance_dict(self, X: np.ndarray) -> Dict[str, float]:
        """Compute mean absolute SHAP value per feature.

        Args:
            X: Feature matrix.

        Returns:
            Dict mapping feature_name → mean |SHAP value| (class-averaged).
        """
        shap_values = self.compute_shap_values(X)
        if shap_values is None:
            # Fallback: use model's built-in feature importances
            if hasattr(self.model, "feature_importances_"):
                imps = self.model.feature_importances_
                return dict(zip(self.feature_names, [round(float(v), 6) for v in imps]))
            return {}

        # shap_values is list[class] or ndarray
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            mean_abs = np.mean(
                [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
            )
        else:
            mean_abs = np.abs(shap_values).mean(axis=0)

        importance = {
            name: round(float(val), 6)
            for name, val in zip(self.feature_names, mean_abs)
        }
        return dict(sorted(importance.items(), key=lambda x: -x[1]))

    def save_importance_report(
        self, X: np.ndarray, out_path: str
    ) -> Dict[str, float]:
        """Compute and save feature importance to JSON.

        Args:
            X: Feature matrix.
            out_path: Output JSON file path.

        Returns:
            Feature importance dict.
        """
        importance = self.feature_importance_dict(X)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(importance, f, indent=2)
        logger.info(f"Feature importance report saved → {out_path}")

        # Log top-10
        top10 = list(importance.items())[:10]
        logger.info("Top-10 features by importance:")
        for rank, (feat, val) in enumerate(top10, 1):
            logger.info(f"  {rank:>2}. {feat:<30} {val:.6f}")

        return importance
