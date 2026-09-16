"""Feature Importance and Coefficient Explainer Module.

Educational Simulation System.
Extracts tree feature importances and logistic regression coefficients.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger("feature_importance_explainer")


class FeatureImportanceExplainer:
    """Explainer for tabular classical ML models (Trees & Logistic Regression)."""

    def __init__(self, feature_names: List[str]):
        """Initialize explainer with feature names list.

        Args:
            feature_names: List of input feature names.
        """
        self.feature_names = feature_names

    def get_tree_importances(self, model: Any) -> pd.DataFrame:
        """Extract feature importances from Tree-based model (DecisionTree / RandomForest).

        Args:
            model: Trained tree model instance with `feature_importances_` attribute.

        Returns:
            pandas DataFrame sorted by importance score descending.
        """
        if not hasattr(model, "feature_importances_"):
            raise ValueError("Provided model does not have 'feature_importances_' attribute.")

        importances = model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": self.feature_names[:len(importances)],
            "importance": importances
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        logger.info(f"Extracted feature importances for tree model ({len(df_imp)} features).")
        return df_imp

    def get_logistic_coefficients(self, model: Any, class_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Extract feature coefficients from Logistic Regression model.

        Args:
            model: Trained LogisticRegression model instance with `coef_`.
            class_names: Target class names list.

        Returns:
            pandas DataFrame containing coefficients per feature per class.
        """
        if not hasattr(model, "coef_"):
            raise ValueError("Provided model does not have 'coef_' attribute.")

        coefs = model.coef_  # Shape: (num_classes, num_features)
        class_names = class_names or [f"class_{i}" for i in range(coefs.shape[0])]

        data = {"feature": self.feature_names[:coefs.shape[1]]}
        for idx, c_name in enumerate(class_names):
            data[f"coef_{c_name}"] = coefs[idx]

        df_coef = pd.DataFrame(data)
        logger.info(f"Extracted logistic regression coefficients across {len(class_names)} classes.")
        return df_coef
