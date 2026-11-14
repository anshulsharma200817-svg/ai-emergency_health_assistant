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
from src.utils.logger import setup_logger

logger = setup_logger("inference_predictor")


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
        """
        df_record = pd.DataFrame([record_dict])
        X, _ = self.pipeline.transform(df_record)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            prob_dict = {
                self.class_labels[i]: round(float(probs[i]), 4)
                for i in range(min(len(probs), len(self.class_labels)))
            }
        else:
            raw_pred = self.model.predict(X)[0]
            pred_idx = int(raw_pred)
            prob_dict = {self.class_labels[pred_idx]: 1.0}

        pred_label = INT_TO_RISK_CLASS.get(pred_idx, "UNKNOWN")
        confidence = max(prob_dict.values())

        return {
            "case_id":               record_dict.get("case_id", "UNKNOWN"),
            "predicted_risk_class":  pred_label,
            "confidence":            round(confidence, 4),
            "probabilities":         prob_dict,
            "model_used":            self.model_name,
            "timestamp":             datetime.now().isoformat(),
        }

    def predict_batch(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict risk for a batch of records.

        Args:
            records: List of record dicts.

        Returns:
            List of prediction result dicts.
        """
        df = pd.DataFrame(records)
        X, _ = self.pipeline.transform(df)

        results = []
        if hasattr(self.model, "predict_proba"):
            probs_all = self.model.predict_proba(X)
            preds = np.argmax(probs_all, axis=1)
            for i, record in enumerate(records):
                probs = probs_all[i]
                pred_idx = int(preds[i])
                pred_label = INT_TO_RISK_CLASS.get(pred_idx, "UNKNOWN")
                prob_dict = {
                    self.class_labels[j]: round(float(probs[j]), 4)
                    for j in range(min(len(probs), len(self.class_labels)))
                }
                results.append({
                    "case_id":              record.get("case_id", f"CASE_{i}"),
                    "predicted_risk_class": pred_label,
                    "confidence":           round(float(probs[pred_idx]), 4),
                    "probabilities":        prob_dict,
                    "model_used":           self.model_name,
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
