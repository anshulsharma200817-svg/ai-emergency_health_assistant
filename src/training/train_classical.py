"""Classical Machine Learning Training Pipeline.

Educational Simulation System.
Orchestrates training for KNN, Logistic Regression, Decision Tree,
and Random Forest. Saves trained models and evaluation results.
"""

from typing import Dict, Any, Optional
import os
import json
import time
import numpy as np
import joblib

from src.models.classical import KNNModel, LogisticRegressionModel, DecisionTreeModel, RandomForestModel
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("train_classical")

CLASSICAL_MODEL_NAMES = ["knn", "logistic_regression", "decision_tree", "random_forest"]


class ClassicalTrainer:
    """Trainer orchestrator for classical machine learning models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer with configuration.

        Args:
            config: Pipeline configuration dictionary.
        """
        self.config = config
        self.evaluator = ModelEvaluator(
            target_names=config.get("target", ["NORMAL", "MODERATE", "HIGH", "CRITICAL"])
        )
        self.models_dir = os.path.join(config.get("paths", {}).get("models", "models"), "classical")
        self.results_dir = os.path.join(
            config.get("paths", {}).get("experiments", "experiments"), "results"
        )
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    # ------------------------------------------------------------------

    def _build_models(self) -> Dict[str, Any]:
        """Instantiate all classical model objects from config.

        Returns:
            Dict mapping model name to model instance.
        """
        classical_cfg = self.config.get("models", {}).get("classical", {})
        return {
            "knn":                  KNNModel(classical_cfg.get("knn", {})),
            "logistic_regression":  LogisticRegressionModel(classical_cfg.get("logistic_regression", {})),
            "decision_tree":        DecisionTreeModel(classical_cfg.get("decision_tree", {})),
            "random_forest":        RandomForestModel(classical_cfg.get("random_forest", {})),
        }

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train all classical ML models, evaluate, and save artifacts.

        Args:
            X_train: Training feature matrix.
            y_train: Training integer target array.
            X_val: Validation feature matrix (optional).
            y_val: Validation integer target array (optional).
            X_test: Test feature matrix (optional).
            y_test: Test integer target array (optional).

        Returns:
            Dict mapping model name to dict with keys:
              'model', 'train_metrics', 'val_metrics', 'test_metrics'.
        """
        logger.info("=" * 60)
        logger.info("Starting Classical ML Training Pipeline")
        logger.info(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        if X_val is not None:
            logger.info(f"  Val:   {X_val.shape[0]} samples")
        if X_test is not None:
            logger.info(f"  Test:  {X_test.shape[0]} samples")
        logger.info("=" * 60)

        models = self._build_models()
        results: Dict[str, Any] = {}
        all_model_summaries = {}

        for name, model_obj in models.items():
            logger.info(f"\n[{name.upper()}] Training...")
            t0 = time.time()

            # ── Fit ──────────────────────────────────────────────────
            model_obj.fit(X_train, y_train)
            train_time = round(time.time() - t0, 2)
            logger.info(f"  Trained in {train_time}s")

            # ── Evaluate ─────────────────────────────────────────────
            y_pred_train = model_obj.predict(X_train)
            train_metrics = self.evaluator.evaluate(y_train, y_pred_train)

            val_metrics = None
            if X_val is not None and y_val is not None:
                y_pred_val = model_obj.predict(X_val)
                val_metrics = self.evaluator.evaluate(y_val, y_pred_val)
                logger.info(
                    f"  Val  Accuracy={val_metrics['accuracy']:.4f}  "
                    f"MacroF1={val_metrics['macro_f1']:.4f}"
                )

            test_metrics = None
            if X_test is not None and y_test is not None:
                y_pred_test = model_obj.predict(X_test)
                test_metrics = self.evaluator.evaluate(y_test, y_pred_test)
                logger.info(
                    f"  Test Accuracy={test_metrics['accuracy']:.4f}  "
                    f"MacroF1={test_metrics['macro_f1']:.4f}"
                )

            # ── Save model ───────────────────────────────────────────
            model_path = os.path.join(self.models_dir, f"{name}.joblib")
            joblib.dump(model_obj.model, model_path)
            logger.info(f"  Saved → {model_path}")

            # ── Collect ──────────────────────────────────────────────
            results[name] = {
                "model":          model_obj,
                "train_time_s":   train_time,
                "train_metrics":  train_metrics,
                "val_metrics":    val_metrics,
                "test_metrics":   test_metrics,
                "model_path":     model_path,
            }

            all_model_summaries[name] = {
                "train_accuracy":  round(train_metrics["accuracy"], 4),
                "train_macro_f1":  round(train_metrics["macro_f1"], 4),
                "val_accuracy":    round(val_metrics["accuracy"], 4)   if val_metrics  else None,
                "val_macro_f1":    round(val_metrics["macro_f1"], 4)   if val_metrics  else None,
                "test_accuracy":   round(test_metrics["accuracy"], 4)  if test_metrics else None,
                "test_macro_f1":   round(test_metrics["macro_f1"], 4)  if test_metrics else None,
                "train_time_s":    train_time,
            }

        # ── Save JSON results ─────────────────────────────────────────
        results_path = os.path.join(self.results_dir, "classical_results.json")
        with open(results_path, "w") as f:
            json.dump(all_model_summaries, f, indent=2)
        logger.info(f"\nClassical results saved → {results_path}")

        # ── Print summary table ───────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("CLASSICAL ML — RESULTS SUMMARY")
        logger.info("=" * 60)
        header = f"{'Model':<25} {'Val Acc':>9} {'Val F1':>9} {'Test Acc':>10} {'Test F1':>9}"
        logger.info(header)
        logger.info("-" * 65)
        for name, s in all_model_summaries.items():
            row = (
                f"{name:<25} "
                f"{str(s['val_accuracy']):>9} "
                f"{str(s['val_macro_f1']):>9} "
                f"{str(s['test_accuracy']):>10} "
                f"{str(s['test_macro_f1']):>9}"
            )
            logger.info(row)
        logger.info("=" * 60)

        return results
