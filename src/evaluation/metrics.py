"""Evaluation Metrics and Performance Analysis Module.

Educational Simulation System.
Calculates Accuracy, Precision, Recall, Macro F1, Weighted F1,
Confusion Matrix, per-class breakdown, and error analysis.
"""

from typing import Dict, Any, List, Optional
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
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
            y_true: True ground truth risk class labels (integer encoded).
            y_pred: Model predicted risk class labels (integer encoded).

        Returns:
            Dict containing accuracy, precision, recall, F1, confusion matrix,
            per-class metrics, and error analysis.
        """
        acc           = float(accuracy_score(y_true, y_pred))
        prec_macro    = float(precision_score(y_true, y_pred, average="macro",    zero_division=0))
        prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
        rec_macro     = float(recall_score(   y_true, y_pred, average="macro",    zero_division=0))
        rec_weighted  = float(recall_score(   y_true, y_pred, average="weighted", zero_division=0))
        f1_macro      = float(f1_score(       y_true, y_pred, average="macro",    zero_division=0))
        f1_weighted   = float(f1_score(       y_true, y_pred, average="weighted", zero_division=0))

        cm           = confusion_matrix(y_true, y_pred).tolist()
        class_report = classification_report(
            y_true, y_pred,
            target_names=self.target_names,
            output_dict=True,
            zero_division=0,
        )

        metrics = {
            "accuracy":           acc,
            "precision_macro":    prec_macro,
            "precision_weighted": prec_weighted,
            "recall_macro":       rec_macro,
            "recall_weighted":    rec_weighted,
            "macro_f1":           f1_macro,
            "weighted_f1":        f1_weighted,
            "confusion_matrix":   cm,
            "per_class_metrics":  class_report,
            "error_analysis":     self.analyze_errors(y_true, y_pred),
        }

        logger.info(f"Evaluation: Accuracy={acc:.4f}  MacroF1={f1_macro:.4f}")
        return metrics

    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Perform error analysis breaking down misclassification patterns.

        Args:
            y_true: True ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Dict containing error counts, misclassification rate, and per-class breakdown.
        """
        misclassified = (y_true != y_pred)
        total         = len(y_true)
        errors_count  = int(misclassified.sum())
        error_rate    = float(errors_count / total) if total > 0 else 0.0

        # Per-class error counts
        per_class_errors: Dict[str, int] = {}
        unique_labels = np.unique(y_true)
        for lbl in unique_labels:
            mask = y_true == lbl
            class_errors = int((y_pred[mask] != lbl).sum())
            name = (
                self.target_names[int(lbl)]
                if int(lbl) < len(self.target_names)
                else str(lbl)
            )
            per_class_errors[name] = class_errors

        return {
            "total_samples":       total,
            "misclassified_count": errors_count,
            "error_rate":          error_rate,
            "per_class_errors":    per_class_errors,
        }

    def print_report(self, y_true: np.ndarray, y_pred: np.ndarray, title: str = "") -> None:
        """Print a nicely formatted classification report to the logger.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            title:  Optional title prefix.
        """
        metrics = self.evaluate(y_true, y_pred)
        header = f"{'─'*55}"
        logger.info(header)
        if title:
            logger.info(f"  {title}")
            logger.info(header)
        logger.info(f"  Accuracy     : {metrics['accuracy']:.4f}")
        logger.info(f"  Precision(M) : {metrics['precision_macro']:.4f}")
        logger.info(f"  Recall(M)    : {metrics['recall_macro']:.4f}")
        logger.info(f"  Macro F1     : {metrics['macro_f1']:.4f}")
        logger.info(f"  Weighted F1  : {metrics['weighted_f1']:.4f}")
        logger.info(f"  Error rate   : {metrics['error_analysis']['error_rate']:.4f}")
        logger.info(header)

    def save_report(self, metrics: Dict[str, Any], path: str) -> None:
        """Save evaluation metrics dict to a JSON file.

        Args:
            metrics: Metrics dict from evaluate().
            path:    Output file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # confusion_matrix is already a list; ensure JSON-serializable
        def _make_serializable(obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_make_serializable(x) for x in obj]
            return obj

        serializable = _make_serializable(metrics)
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info(f"Evaluation report saved → {path}")
