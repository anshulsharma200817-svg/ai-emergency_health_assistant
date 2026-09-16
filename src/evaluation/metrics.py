"""Evaluation Metrics and Performance Analysis Module Skeleton.

Educational Simulation System.
Calculates Accuracy, Precision, Recall, Macro F1, Weighted F1, Confusion Matrix, and Error Analysis.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.utils.logger import setup_logger

logger = setup_logger("evaluation_metrics")


class ModelEvaluator:
    """Evaluator for classification model evaluation and error analysis."""

    def __init__(self, target_names: Optional[List[str]] = None):
        """Initialize Evaluator.

        Args:
            target_names: Class labels list ('NORMAL', 'MODERATE', 'HIGH', 'CRITICAL').
        """
        self.target_names = target_names or ["NORMAL", "MODERATE", "HIGH", "CRITICAL"]

    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics.

        Args:
            y_true: True ground truth risk class labels.
            y_pred: Model predicted risk class labels.

        Returns:
            Dict containing accuracy, precision (macro/weighted), recall (macro/weighted),
            macro F1, weighted F1, confusion matrix, per-class metrics, and error analysis.
        """
        # TODO: Expand automatic artifact logging to experiments/results/ upon training completion.

        acc = float(accuracy_score(y_true, y_pred))
        prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
        f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

        cm = confusion_matrix(y_true, y_pred).tolist()
        class_report = classification_report(
            y_true, y_pred, target_names=self.target_names, output_dict=True, zero_division=0
        )

        metrics = {
            "accuracy": acc,
            "precision_macro": prec_macro,
            "precision_weighted": prec_weighted,
            "recall_macro": rec_macro,
            "recall_weighted": rec_weighted,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "confusion_matrix": cm,
            "per_class_metrics": class_report,
            "error_analysis": self.analyze_errors(y_true, y_pred)
        }

        logger.info(f"Evaluation complete. Accuracy: {acc:.4f}, Macro F1: {f1_macro:.4f}")
        return metrics

    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform error analysis breaking down misclassification patterns.

        Args:
            y_true: True ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Dict containing error counts, misclassification rate, and confusion breakdown.
        """
        misclassified = (y_true != y_pred)
        total = len(y_true)
        errors_count = int(misclassified.sum())
        error_rate = float(errors_count / total) if total > 0 else 0.0

        return {
            "total_samples": total,
            "misclassified_count": errors_count,
            "error_rate": error_rate,
        }
