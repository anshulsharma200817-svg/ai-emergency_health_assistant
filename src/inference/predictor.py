"""Inference Predictor Module.

Educational Simulation System.
Executes inference on single cases or batch records using trained classical models.
"""

from typing import Dict, Any, List, Optional
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

from src.preprocessing.pipeline import PreprocessingPipeline, INT_TO_RISK_CLASS
from src.inference.validation import normalize_vitals
from src.utils.logger import setup_logger

logger = setup_logger("inference_predictor")

# Hard constraint learned from the training data. `unconsciousness = 1` occurs
# in 1,490 held-out records — 1,330 CRITICAL and 160 HIGH — and never in NORMAL
# or MODERATE. The linear model still gives that flag a positive NORMAL
# coefficient, so "healthy vitals + unconscious" came back as NORMAL 99.9%.
# Restricting the probabilities to the classes the data supports changes 0
# predictions on the full 15,000-row test set. Mirrored in web/lib/inference.ts.
UNCONSCIOUSNESS_PRIOR = {"HIGH": 0.123, "CRITICAL": 0.877}
CONSTRAINT_MIN_ALLOWED_MASS = 1e-3


def apply_data_constraints(
    probabilities: np.ndarray,
    record: Dict[str, Any],
    class_labels: List[str],
) -> tuple[np.ndarray, Optional[str]]:
    """Condition probabilities on classes the training data supports.

    Args:
        probabilities: Model probabilities in `class_labels` order.
        record: Raw input record (only `unconsciousness` is inspected).
        class_labels: Ordered class names matching `probabilities`.

    Returns:
        `(probabilities, constraint_name)`; constraint is `None` when no rule
        applies. The array always sums to 1.
    """
    flag = record.get("unconsciousness", 0)
    if flag is None or (isinstance(flag, float) and not np.isfinite(flag)):
        return probabilities, None
    try:
        flag = int(flag)
    except (TypeError, ValueError):
        return probabilities, None
    if flag != 1:
        return probabilities, None

    out = np.array(probabilities, dtype=float)
    normal_i = class_labels.index("NORMAL")
    moderate_i = class_labels.index("MODERATE")
    high_i = class_labels.index("HIGH")
    critical_i = class_labels.index("CRITICAL")

    allowed = out[high_i] + out[critical_i]
    out[normal_i] = 0.0
    out[moderate_i] = 0.0

    if allowed > CONSTRAINT_MIN_ALLOWED_MASS:
        out[high_i] = probabilities[high_i] / allowed
        out[critical_i] = probabilities[critical_i] / allowed
    else:
        # The model is extrapolating; use the empirical split for this symptom.
        out[high_i] = UNCONSCIOUSNESS_PRIOR["HIGH"]
        out[critical_i] = UNCONSCIOUSNESS_PRIOR["CRITICAL"]

    return out, "unconsciousness"


class HealthRiskPredictor:
    """Predictor engine for health risk inference using trained classical ML models."""

    def __init__(
        self,
        model: Any,
        pipeline: PreprocessingPipeline,
        config: Dict[str, Any],
        model_name: str = "unknown",
    ):
        """Initialize predictor.

        Args:
            model: Trained sklearn model instance.
            pipeline: Fitted PreprocessingPipeline instance.
            config: Configuration dictionary.
            model_name: Friendly name for logging/output.
        """
        self.model = model
        self.pipeline = pipeline
        self.config = config
        self.model_name = model_name
        self.class_labels = config.get("target", ["NORMAL", "MODERATE", "HIGH", "CRITICAL"])

    # ------------------------------------------------------------------

    def predict_record(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk classification inference on a single record dict.

        Args:
            record_dict: Dict containing raw physiological feature values.

        Returns:
            Dict with case_id, predicted_risk_class, confidence probabilities,
            and timestamp.

        Raises:
            VitalsValidationError: When a field is malformed or outside the
                accepted range. Absent fields take the healthy-adult default,
                exactly as the TypeScript API does.
        """
        vitals = normalize_vitals(record_dict)
        case_id = (record_dict or {}).get("case_id", "UNKNOWN")
        df_record = pd.DataFrame([vitals])
        X, _ = self.pipeline.transform(df_record)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            probs, constraint = apply_data_constraints(probs, vitals, self.class_labels)
            pred_idx = int(np.argmax(probs))
            prob_dict = {
                self.class_labels[i]: round(float(probs[i]), 4)
                for i in range(min(len(probs), len(self.class_labels)))
            }
        else:
            raw_pred = self.model.predict(X)[0]
            pred_idx = int(raw_pred)
            prob_dict = {self.class_labels[pred_idx]: 1.0}
            constraint = None

        pred_label = INT_TO_RISK_CLASS.get(pred_idx, "UNKNOWN")
        confidence = max(prob_dict.values())

        return {
            "case_id":               case_id,
            "predicted_risk_class":  pred_label,
            "confidence":            round(confidence, 4),
            "probabilities":         prob_dict,
            "model_used":            self.model_name,
            "timestamp":             datetime.now().isoformat(),
            "applied_constraint":    constraint,
        }

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict risk for a batch of records.

        Args:
            records: List of record dicts.

        Returns:
            List of prediction result dicts.

        Raises:
            VitalsValidationError: When any record is malformed or out of range.
                The error names the offending record index.
        """
        if not records:
            return []

        vitals_list: List[Dict[str, Any]] = []
        for index, record in enumerate(records):
            try:
                vitals_list.append(normalize_vitals(record))
            except ValueError as exc:
                raise ValueError(f"record {index}: {exc}") from exc

        df = pd.DataFrame(vitals_list)
        X, _ = self.pipeline.transform(df)

        results = []
        if hasattr(self.model, "predict_proba"):
            probs_all = self.model.predict_proba(X)
            for i, record in enumerate(records):
                probs, constraint = apply_data_constraints(
                    probs_all[i], vitals_list[i], self.class_labels
                )
                pred_idx = int(np.argmax(probs))
                pred_label = INT_TO_RISK_CLASS.get(pred_idx, "UNKNOWN")
                prob_dict = {
                    self.class_labels[j]: round(float(probs[j]), 4)
                    for j in range(min(len(probs), len(self.class_labels)))
                }
                results.append({
                    "case_id":              record.get("case_id", f"CASE_{i}"),
                    "predicted_risk_class": pred_label,
                    "confidence":           round(float(max(prob_dict.values())), 4),
                    "probabilities":        prob_dict,
                    "model_used":           self.model_name,
                    "applied_constraint":   constraint,
                })
        else:
            preds = self.model.predict(X)
            for i, record in enumerate(records):
                pred_label = INT_TO_RISK_CLASS.get(int(preds[i]), "UNKNOWN")
                results.append({
                    "case_id":              record.get("case_id", f"CASE_{i}"),
                    "predicted_risk_class": pred_label,
                    "confidence":           1.0,
                    "probabilities":        {pred_label: 1.0},
                    "model_used":           self.model_name,
                    "applied_constraint":   None,
                })

        logger.info(f"Batch prediction complete: {len(results)} records → model={self.model_name}")
        return results

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_saved_model(
        cls,
        model_path: str,
        pipeline_path: str,
        config: Dict[str, Any],
    ) -> "HealthRiskPredictor":
        """Load a saved model and pipeline and return a ready predictor.

        Args:
            model_path: Path to .joblib model file.
            pipeline_path: Path to .pkl pipeline file.
            config: Configuration dict.

        Returns:
            Configured HealthRiskPredictor instance.
        """
        model = joblib.load(model_path)
        pipeline = PreprocessingPipeline.load(pipeline_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        logger.info(f"Loaded model '{model_name}' from {model_path}")
        return cls(model, pipeline, config, model_name=model_name)
